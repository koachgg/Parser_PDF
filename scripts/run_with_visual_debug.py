"""
Run a full demo of the PDF parser with visual debugging enabled.
"""
import os
import sys
import logging
import argparse
from pathlib import Path
import datetime

# Add parent directory to import path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    import fitz  # PyMuPDF
    from src.parser.text_blocks import extract_text_blocks
    from src.parser.tables import extract_tables
    from src.parser.charts import detect_charts
    from src.parser.sections import SectionTracker
    from src.parser.visual_debug import visualize_page_content
    from src.config import ConfigManager
    from src.main import process_document
except ImportError as e:
    print(f"Failed to import modules: {str(e)}")
    print("Make sure you have activated the virtual environment and installed all dependencies.")
    sys.exit(1)

def main():
    """Run a full demo of the PDF parser with visual debugging."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Demo PDF parsing with visual debugging")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--output", "-o", default=None, help="Output directory")
    parser.add_argument("--max-pages", "-m", type=int, default=None, help="Maximum pages to process")
    parser.add_argument("--enable-ocr", action="store_true", help="Enable OCR for text extraction")
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                      format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    
    # Check if PDF file exists
    if not os.path.isfile(args.pdf_path):
        logger.error(f"PDF file not found: {args.pdf_path}")
        sys.exit(1)
        
    # Create output directory
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                "../output", f"demo_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output will be saved to {output_dir}")
    
    # Create debug output directory
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Create configuration with visual debugging enabled
    config = ConfigManager()
    config.set("visual_debug", "enabled", True)
    config.set("visual_debug", "output_path", debug_dir)
    config.set("visual_debug", "draw_bounding_boxes", True)
    config.set("visual_debug", "save_images", True)
    config.set("ocr", "enabled", args.enable_ocr)
    
    # Process PDF
    try:
        logger.info(f"Processing PDF: {args.pdf_path}")
        doc = fitz.open(args.pdf_path)
        
        # Process the document
        result = process_document(
            args.pdf_path, 
            doc, 
            args.max_pages, 
            args.enable_ocr, 
            "auto",
            config
        )
        
        # Save result as JSON
        import json
        output_json = os.path.join(output_dir, "parsed_output.json")
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)
            
        logger.info(f"Parsing complete. Results saved to {output_json}")
        
        # Print summary
        total_pages = len(result["pages"])
        total_paragraphs = sum(sum(1 for c in page["content"] if c["type"] == "paragraph") for page in result["pages"])
        total_tables = sum(sum(1 for c in page["content"] if c["type"] == "table") for page in result["pages"])
        total_charts = sum(sum(1 for c in page["content"] if c["type"] == "chart") for page in result["pages"])
        
        print("\nParsing Results Summary:")
        print(f"- PDF: {os.path.basename(args.pdf_path)}")
        print(f"- Pages processed: {total_pages}")
        print(f"- Paragraphs extracted: {total_paragraphs}")
        print(f"- Tables extracted: {total_tables}")
        print(f"- Charts detected: {total_charts}")
        print(f"- Visual debug output: {debug_dir}")
        print(f"- JSON output: {output_json}")
        
    except Exception as e:
        logger.exception(f"Error processing PDF: {str(e)}")
        sys.exit(1)
        
if __name__ == "__main__":
    main()
