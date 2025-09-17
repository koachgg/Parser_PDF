"""
Demonstration of visual debugging in PDF parser.
"""
import os
import sys
import logging
from pathlib import Path

# Add parent directory to sys.path to import from src
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import fitz  # PyMuPDF
from src.parser.text_blocks import extract_text_blocks
from src.parser.tables import extract_tables
from src.parser.charts import detect_charts
from src.parser.visual_debug import visualize_page_content
from src.parser.sections import SectionTracker
from src.config import ConfigManager

def demo_visual_debug(pdf_path, output_dir='debug_output'):
    """
    Demonstrate visual debugging functionality.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save debug output
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                      format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    
    # Create configuration
    config = ConfigManager()
    config.set("visual_debug", "enabled", True)
    config.set("visual_debug", "output_path", output_dir)
    config.set("visual_debug", "draw_bounding_boxes", True)
    config.set("visual_debug", "save_images", True)
    
    # Open PDF
    doc = fitz.open(pdf_path)
    logger.info(f"Processing PDF: {pdf_path} with {len(doc)} pages")
    
    debug_config = {
        "output_path": output_dir,
        "draw_bounding_boxes": True,
        "save_images": True
    }
    
    section_tracker = SectionTracker()
    
    # Process each page
    for page_idx, page in enumerate(doc):
        page_num = page_idx + 1
        logger.info(f"Processing page {page_num}")
        
        # Extract content
        text_blocks = extract_text_blocks(page, section_tracker)
        table_blocks = extract_tables(page, pdf_path, page_num, 'auto')
        chart_blocks = detect_charts(page)
        
        # Combine all blocks
        all_blocks = text_blocks + table_blocks + chart_blocks
        
        # Generate visual debug output
        debug_img = visualize_page_content(page, all_blocks, debug_config)
        logger.info(f"Generated debug image: {debug_img}")
    
    logger.info(f"Visual debugging complete. Output saved to {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python demo_visual_debug.py <pdf_path> [output_dir]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'debug_output'
    
    demo_visual_debug(pdf_path, output_dir)
