"""
OCR module for extracting text from scanned PDF pages.
"""
import tempfile
import os
import logging
from typing import Dict, Any, List, Optional
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

from .utils import clean_text, is_ocr_needed

logger = logging.getLogger(__name__)

class OCRProcessor:
    """Process PDF pages with OCR when needed."""
    
    def __init__(self, dpi: int = 300):
        """
        Initialize OCR processor.
        
        Args:
            dpi: Image DPI for PDF rasterization
        """
        self.dpi = dpi
    
    def ocr_page(self, pdf_path: str, page_num: int) -> str:
        """
        Perform OCR on a specific PDF page.
        
        Args:
            pdf_path: Path to PDF file
            page_num: 1-based page number
            
        Returns:
            Extracted text from the page
        """
        # Convert page to image
        try:
            images = convert_from_path(
                pdf_path, 
                first_page=page_num, 
                last_page=page_num,
                dpi=self.dpi
            )
            
            if not images:
                logger.warning(f"Failed to convert page {page_num} to image")
                return ""
            
            # Process the page image
            img = images[0]
            
            # Run OCR
            text = pytesseract.image_to_string(img, lang='eng')
            
            return clean_text(text)
        except Exception as e:
            logger.error(f"OCR failed for page {page_num}: {str(e)}")
            return ""
    
    def get_ocr_blocks(self, pdf_path: str, page_num: int) -> List[Dict[str, Any]]:
        """
        Get text blocks from OCR results.
        
        Args:
            pdf_path: Path to PDF file
            page_num: 1-based page number
            
        Returns:
            List of text blocks with bounding box information
        """
        try:
            # Convert page to image
            images = convert_from_path(
                pdf_path, 
                first_page=page_num, 
                last_page=page_num,
                dpi=self.dpi
            )
            
            if not images:
                logger.warning(f"Failed to convert page {page_num} to image")
                return []
            
            img = images[0]
            
            # Get OCR data with bounding boxes
            ocr_data = pytesseract.image_to_data(img, lang='eng', output_type=pytesseract.Output.DICT)
            
            # Group by paragraph (text block)
            blocks = []
            current_block = {"text": "", "bbox": [float('inf'), float('inf'), 0, 0]}
            current_par = -1
            
            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i].strip()
                
                if not text:
                    continue
                
                par_num = ocr_data['par_num'][i]
                
                # New paragraph
                if par_num != current_par:
                    if current_block["text"]:
                        blocks.append(current_block.copy())
                    current_block = {"text": text, "bbox": [float('inf'), float('inf'), 0, 0]}
                    current_par = par_num
                else:
                    current_block["text"] += " " + text
                
                # Update bounding box
                x, y, w, h = (
                    ocr_data['left'][i], 
                    ocr_data['top'][i], 
                    ocr_data['width'][i], 
                    ocr_data['height'][i]
                )
                current_block["bbox"][0] = min(current_block["bbox"][0], x)
                current_block["bbox"][1] = min(current_block["bbox"][1], y)
                current_block["bbox"][2] = max(current_block["bbox"][2], x + w)
                current_block["bbox"][3] = max(current_block["bbox"][3], y + h)
            
            # Add the last block
            if current_block["text"]:
                blocks.append(current_block)
            
            # Clean and normalize text in blocks
            for block in blocks:
                block["text"] = clean_text(block["text"])
                # Scale bbox to PDF coordinates (approximate)
                doc = fitz.open(pdf_path)
                page = doc[page_num - 1]
                width_ratio = page.rect.width / img.width
                height_ratio = page.rect.height / img.height
                
                block["bbox"] = [
                    block["bbox"][0] * width_ratio,
                    block["bbox"][1] * height_ratio,
                    block["bbox"][2] * width_ratio,
                    block["bbox"][3] * height_ratio
                ]
            
            return blocks
        except Exception as e:
            logger.error(f"OCR block extraction failed for page {page_num}: {str(e)}")
            return []

def process_page_with_ocr(page: fitz.Page, pdf_path: str, page_num: int) -> List[Dict[str, Any]]:
    """
    Process a page with OCR if needed and return text blocks.
    
    Args:
        page: PDF page object
        pdf_path: Path to PDF file
        page_num: 1-based page number
        
    Returns:
        List of text blocks extracted via OCR
    """
    if not is_ocr_needed(page):
        return []
    
    processor = OCRProcessor()
    ocr_blocks = processor.get_ocr_blocks(pdf_path, page_num)
    
    # Convert OCR blocks to standard paragraph format
    structured_blocks = []
    for block in ocr_blocks:
        structured_block = {
            "type": "paragraph",
            "section": None,  # Will be updated based on context
            "sub_section": None,  # Will be updated based on context
            "text": block["text"],
            "bbox": block["bbox"],
            "table_data": None,
            "chart_data": None,
            "description": None,
            "ocr_generated": True
        }
        structured_blocks.append(structured_block)
    
    return structured_blocks
