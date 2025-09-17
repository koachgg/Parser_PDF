"""
PDF to JSON parser - Main entry point.
"""
import os
import sys
import logging
import click
import json
import datetime
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("WARNING: PyMuPDF (fitz) not available. PDF parsing will fail.")

try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("WARNING: Pillow (PIL) not available. Visual debugging will be disabled.")
    
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("WARNING: PyYAML not available. Configuration from YAML files will not be supported.")

try:
    import pkg_resources
    PKG_RESOURCES_AVAILABLE = True
except ImportError:
    PKG_RESOURCES_AVAILABLE = False
    print("WARNING: pkg_resources not available. Version information will be limited.")

from .parser.text_blocks import extract_text_blocks
from .parser.tables import extract_tables, TableExtractor
from .parser.charts import detect_charts
from .parser.sections import SectionTracker
from .parser.utils import is_ocr_needed
from .parser.content_overlap import remove_content_overlap
from .parser.visual_debug import visualize_page_content
from .ocr.ocr_pipeline import process_page_with_ocr
from .exporters.json_writer import JSONWriter
from .config import ConfigManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_SAMPLE_PATH = str(Path(__file__).parents[1] / "samples" / "input" / "[Fund Factsheet - May]360ONE-MF-May 2025.pdf")
DEFAULT_SCHEMA_PATH = str(Path(__file__).parents[1] / "schemas" / "output_schema.json")

def get_package_version(package_name: str) -> str:
    """
    Get installed package version.
    
    Args:
        package_name: Name of the package
        
    Returns:
        Version string or "unknown"
    """
    if not PKG_RESOURCES_AVAILABLE:
        return "unknown"
        
    try:
        return pkg_resources.get_distribution(package_name).version
    except (pkg_resources.DistributionNotFound, Exception):
        return "unknown"

def process_document(pdf_path: str, doc: fitz.Document, max_pages: Optional[int], enable_ocr: bool, table_mode: str, config: Optional[ConfigManager] = None) -> Dict[str, Any]:
    """
    Process the entire PDF document with preprocessing for better structure detection.
    
    Args:
        pdf_path: Path to PDF file
        doc: PDF document object
        max_pages: Maximum number of pages to process
        enable_ocr: Whether to use OCR for text extraction
        table_mode: Table extraction mode
        
    Returns:
        Structured document data
    """
    result = {
        "metadata": {
            "file_name": os.path.basename(pdf_path),
            "creation_date": datetime.datetime.now().isoformat(),
            "page_count": len(doc)
        },
        "pages": []
    }
    
    # Initialize a shared section tracker for the entire document
    section_tracker = SectionTracker()
    
    # Determine number of pages to process
    total_pages = len(doc)
    pages_to_process = min(total_pages, max_pages) if max_pages else total_pages
    
    # First pass: collect all blocks for document-level analysis
    logger.info("Performing document structure analysis...")
    all_doc_blocks = []
    
    for page_idx in range(pages_to_process):
        page = doc[page_idx]
        # Extract raw text blocks for structure analysis
        raw_blocks = page.get_text("dict")["blocks"]
        for block in raw_blocks:
            if block["type"] != 0:  # Skip non-text blocks
                continue
                
            block_text = ""
            font_info = {}
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    block_text += span.get("text", "")
                    # Use first span's font info as representative
                    if not font_info and "font" in span and "size" in span:
                        font_info = {
                            "font": span.get("font"),
                            "font_size": span.get("size"),
                            "is_bold": span.get("flags", 0) & 2 > 0,
                            "is_italic": span.get("flags", 0) & 1 > 0
                        }
            
            if block_text.strip():
                all_doc_blocks.append({
                    "text": block_text,
                    **font_info,
                    "page_num": page_idx + 1,
                    "bbox": block["bbox"]
                })
    
    # Preprocess document structure to learn patterns
    section_tracker.preprocess_document(all_doc_blocks)
    logger.info(f"Document analysis complete - detected {section_tracker.section_count} sections")
    
    # Process each page with the informed section tracker
    for page_idx in range(pages_to_process):
        page_num = page_idx + 1
        logger.info(f"Processing page {page_num}/{total_pages}")
        page = doc[page_idx]
        page_data = process_page(pdf_path, page, page_num, enable_ocr, table_mode, section_tracker, config)
        result["pages"].append(page_data)
    
    return result

def process_page(pdf_path: str, page: fitz.Page, page_num: int, enable_ocr: bool, table_mode: str, 
              section_tracker: Optional[SectionTracker] = None, config: Optional[ConfigManager] = None) -> Dict[str, Any]:
    """
    Process a single PDF page.
    
    Args:
        pdf_path: Path to PDF file
        page: PDF page object
        page_num: 1-based page number
        enable_ocr: Whether to use OCR for text extraction
        table_mode: Table extraction mode
        section_tracker: Optional pre-initialized section tracker
        config: Optional configuration manager
        
    Returns:
        Structured page data
    """
    # Initialize section tracker if not provided
    if section_tracker is None:
        section_tracker = SectionTracker()
    
    # Extract text blocks
    text_blocks = extract_text_blocks(page, section_tracker)
    
    # Extract tables
    table_blocks = extract_tables(page, pdf_path, page_num, table_mode)
    
    # Extract charts
    chart_blocks = detect_charts(page, use_ocr=enable_ocr)
    
    # Use OCR if enabled and needed
    ocr_blocks = []
    if enable_ocr and is_ocr_needed(page):
        logger.info(f"Using OCR for page {page_num}")
        ocr_blocks = process_page_with_ocr(page, pdf_path, page_num)
    
    # Combine all content blocks
    all_blocks = text_blocks + table_blocks + chart_blocks + ocr_blocks
    
    # Remove overlapping content (e.g., text inside tables) using config if available
    overlap_config = config.get_content_overlap_config() if config else None
    all_blocks = remove_content_overlap(all_blocks, overlap_config)
    
    # Attach section information to tables and charts based on nearest text block
    if text_blocks:
        # Find sections for non-text blocks
        for block in all_blocks:
            if block["type"] != "paragraph" and (block["section"] is None or block["sub_section"] is None):
                # Use the nearest text block's section information
                nearest_text_block = min(
                    [b for b in text_blocks if b["section"]], 
                    key=lambda b: abs(b["bbox"][1] - block["bbox"][1]) if b["bbox"] and block["bbox"] else float('inf'),
                    default=None
                )
                
                if nearest_text_block:
                    block["section"] = nearest_text_block["section"]
                    block["sub_section"] = nearest_text_block["sub_section"]
    
    # Generate visual debug output if enabled
    visual_debug_img = None
    if config and config.get("visual_debug", "enabled", False) and PIL_AVAILABLE:
        debug_config = {
            "output_path": config.get("visual_debug", "output_path", "debug_output"),
            "draw_bounding_boxes": config.get("visual_debug", "draw_bounding_boxes", True),
            "save_images": config.get("visual_debug", "save_images", False)
        }
        # Get debug configuration from the ConfigManager
        debug_config = config.get_visual_debug_config() if config else {
            "output_path": debug_config.get("output_path", "debug_output"),
            "draw_bounding_boxes": debug_config.get("draw_bounding_boxes", True),
            "save_images": debug_config.get("save_images", False)
        }
        visual_debug_img = visualize_page_content(page, all_blocks, debug_config)
        if visual_debug_img:
            logger.info(f"Visual debug output generated for page {page_num}: {visual_debug_img}")
    
    # Create page structure
    page_data = {
        "page_number": page_num,
        "content": all_blocks,
        "debug_image": visual_debug_img if visual_debug_img else None
    }
    
    return page_data

def check_dependencies():
    """
    Check if all required dependencies are available.
    
    Returns:
        bool: True if all dependencies are available, False otherwise
    """
    dependencies = {
        "PyMuPDF": PYMUPDF_AVAILABLE,
    }
    
    try:
        import camelot
        dependencies["camelot"] = True
    except ImportError:
        dependencies["camelot"] = False
    
    try:
        import pdfplumber
        dependencies["pdfplumber"] = True
    except ImportError:
        dependencies["pdfplumber"] = False
    
    try:
        import pytesseract
        dependencies["pytesseract"] = True
    except ImportError:
        dependencies["pytesseract"] = False
    
    # Print status of dependencies
    print("Dependency Status:")
    for dep, status in dependencies.items():
        status_str = "✓ Available" if status else "✗ Missing"
        print(f"  - {dep}: {status_str}")
    
    return all(dependencies.values())

@click.command()
@click.option("--input", "-i", "input_path", help="Path to input PDF file")
@click.option("--output", "-o", "output_path", help="Path to output JSON file")
@click.option("--config", "-c", "config_path", help="Path to YAML configuration file")
@click.option("--enable-ocr/--disable-ocr", default=None, help="Enable OCR for scanned pages")
@click.option("--table-mode", type=click.Choice(["auto", "lattice", "stream", "heuristic"]), help="Table extraction mode")
@click.option("--max-pages", type=int, help="Maximum number of pages to process")
@click.option("--debug/--no-debug", default=None, help="Enable debug output")
@click.option("--visual-debug/--no-visual-debug", default=None, help="Enable visual debugging output")
def main(input_path: Optional[str], output_path: Optional[str], config_path: Optional[str], 
         enable_ocr: Optional[bool], table_mode: Optional[str], max_pages: Optional[int], 
         debug: Optional[bool], visual_debug: Optional[bool]):
    """
    Parse PDF and convert to structured JSON.
    
    If input path is not provided, uses the default sample.
    If output path is not provided, uses the input filename with .json extension.
    If config path is provided, loads configuration from YAML file.
    """
    # Initialize configuration
    config_file = config_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml")
    config = ConfigManager(config_file if os.path.exists(config_file) else None)
    
    # Override config with command line arguments if provided
    if debug is not None:
        config.set("general", "debug", debug)
    if enable_ocr is not None:
        config.set("ocr", "enabled", enable_ocr)
    if table_mode is not None:
        config.set("tables", "mode", table_mode)
    if visual_debug is not None:
        config.set("visual_debug", "enabled", visual_debug)
    
    # Set up logging level
    if config.get("general", "debug", False):
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check dependencies
    print("\nChecking required dependencies...")
    if not check_dependencies():
        print("\nERROR: Some required dependencies are missing.")
        print("Please install all required dependencies using:")
        print("    pip install -r requirements.txt")
        print("\nNote: Some dependencies may require additional system packages.")
        print("For Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki")
        print("For Ghostscript: https://www.ghostscript.com/download.html")
        sys.exit(1)
    
    # Use default input if not specified
    if not input_path:
        input_path = DEFAULT_SAMPLE_PATH
        logger.info(f"Using default input file: {input_path}")
    
    # Create default output path if not specified
    if not output_path:
        input_basename = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = os.path.join(os.path.dirname(os.path.abspath(input_path)), "..", "output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{input_basename}.json")
        logger.info(f"Using default output path: {output_path}")
    
    # Check if input file exists
    if not os.path.isfile(input_path):
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    start_time = time.time()
    
    try:
        # Open the PDF
        doc = fitz.open(input_path)
        
        # Process the entire document with improved structure detection
        start_time = time.time()
        
        # Get configuration values
        use_ocr = config.get("ocr", "enabled", False)
        table_extraction_mode = config.get("tables", "mode", "auto")
        visual_debug_enabled = config.get("visual_debug", "enabled", False)
        
        # Setup visual debug directory if needed
        if visual_debug_enabled:
            debug_dir = config.get("visual_debug", "output_dir", "debug_output")
            os.makedirs(debug_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_output_dir = os.path.join(debug_dir, timestamp)
            os.makedirs(debug_output_dir, exist_ok=True)
            logger.info(f"Visual debug output will be saved to {debug_output_dir}")
            config.set("visual_debug", "output_path", debug_output_dir)
        
        result = process_document(input_path, doc, max_pages, use_ocr, table_extraction_mode, config)
        
        # Add version information to the metadata
        result["meta"] = {
            "source_file": os.path.basename(input_path),
            "parser_versions": {
                "text": get_package_version("PyMuPDF"),
                "tables": get_package_version("camelot-py"),
                "charts": "0.1.0"  # Custom chart detection
            },
            "created_at": datetime.datetime.now().isoformat()
        }
        
        # Write output JSON
        schema_path = DEFAULT_SCHEMA_PATH if os.path.isfile(DEFAULT_SCHEMA_PATH) else None
        json_writer = JSONWriter(schema_path)
        success = json_writer.write(output_path, result)
        
        if success:
            # Generate summary
            total_paragraphs = sum(len([c for c in page["content"] if c["type"] == "paragraph"]) for page in result["pages"])
            total_tables = sum(len([c for c in page["content"] if c["type"] == "table"]) for page in result["pages"])
            total_charts = sum(len([c for c in page["content"] if c["type"] == "chart"]) for page in result["pages"])
            processed_pages = len(result["pages"])
            total_pages = result["metadata"]["page_count"]
            
            logger.info(f"JSON output written to {output_path}")
            logger.info(f"Summary: {processed_pages} pages, {total_paragraphs} paragraphs, {total_tables} tables, {total_charts} charts")
            logger.info(f"Processing completed in {time.time() - start_time:.2f} seconds")
            
            # Print a summary to stdout
            print(f"PDF Parsing Complete:")
            print(f"- Pages processed: {processed_pages} of {total_pages}")
            print(f"- Paragraphs extracted: {total_paragraphs}")
            print(f"- Tables extracted: {total_tables}")
            print(f"- Charts detected: {total_charts}")
            print(f"- Output: {output_path}")
            print(f"- Time: {time.time() - start_time:.2f} seconds")
        else:
            logger.error("Failed to write JSON output")
            sys.exit(1)
            
    except Exception as e:
        logger.exception(f"Error processing PDF: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
