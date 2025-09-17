"""
Script to extract tables from a PDF and update the existing JSON output.
"""
import os
import sys
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to sys.path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.parser.simple_table_extractor import SimpleTableExtractor
except ImportError:
    logger.error("Could not import SimpleTableExtractor. Make sure the path is correct.")
    sys.exit(1)

def update_json_with_tables(json_path: str, pdf_path: str, output_path: str = None) -> None:
    """
    Update an existing JSON file with tables extracted from the PDF.
    
    Args:
        json_path: Path to the existing JSON file
        pdf_path: Path to the PDF file
        output_path: Path for the updated JSON file (defaults to "_updated" suffix)
    """
    if not os.path.exists(json_path):
        logger.error(f"JSON file not found: {json_path}")
        return
        
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return
        
    # Set default output path if not provided
    if not output_path:
        base, ext = os.path.splitext(json_path)
        output_path = f"{base}_updated{ext}"
    
    # Load the existing JSON
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        logger.info(f"Successfully loaded JSON from {json_path}")
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in {json_path}")
        return
    except Exception as e:
        logger.error(f"Error loading JSON: {str(e)}")
        return
    
    # Extract tables from the PDF
    extractor = SimpleTableExtractor()
    
    # Get page numbers from the JSON
    page_numbers = [page["page_number"] for page in data.get("pages", [])]
    if not page_numbers:
        logger.error("No pages found in JSON")
        return
    
    logger.info(f"Extracting tables from {len(page_numbers)} pages")
    tables_by_page = extractor.extract_tables(pdf_path, page_numbers)
    
    # Update each page in the JSON with the extracted tables
    pages_updated = 0
    tables_added = 0
    
    for page in data.get("pages", []):
        page_number = page["page_number"]
        
        if page_number not in tables_by_page:
            logger.warning(f"No table data for page {page_number}")
            continue
            
        tables = tables_by_page[page_number]
        if not tables:
            continue
            
        # Add tables to the page content
        for table in tables:
            table_obj = {
                "type": "table",
                "section": None,  # Will need to be determined based on context
                "sub_section": None,
                "text": None,
                "table_data": table["table_data"],
                "chart_data": None,
                "description": None,
                "bbox": table["bbox"]
            }
            
            # Try to find the section from nearby paragraphs
            for content in page.get("content", []):
                if content["type"] == "paragraph" and content["section"]:
                    table_obj["section"] = content["section"]
                    table_obj["sub_section"] = content["sub_section"]
                    break
                    
            page["content"].append(table_obj)
            tables_added += 1
            
        pages_updated += 1
    
    logger.info(f"Added {tables_added} tables to {pages_updated} pages")
    
    # Write the updated JSON
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Updated JSON written to {output_path}")
    except Exception as e:
        logger.error(f"Error writing updated JSON: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python update_json_with_tables.py <json_path> <pdf_path> [output_path]")
        sys.exit(1)
        
    json_path = sys.argv[1]
    pdf_path = sys.argv[2]
    
    output_path = None
    if len(sys.argv) > 3:
        output_path = sys.argv[3]
        
    update_json_with_tables(json_path, pdf_path, output_path)
