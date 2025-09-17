"""
Unified table extraction module combining features from both tables.py and simple_table_extractor.py.
"""
import os
import tempfile
import logging
from typing import List, Dict, Any, Tuple, Optional
import re
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# Import necessary libraries, handling potential import errors gracefully
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not available. Some table extraction features will be limited.")

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
    logger.warning("Camelot not available. Lattice and stream table extraction will be disabled.")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("PDFPlumber not available. Some table extraction features will be disabled.")

from .utils import normalize_bbox, has_tabular_structure

class UnifiedTableExtractor:
    """Extract tables from PDFs using multiple strategies and libraries."""
    
    def __init__(self, table_mode: str = "auto", deduplication_threshold: float = 0.3, min_confidence: float = 50):
        """
        Initialize the unified table extractor.
        
        Args:
            table_mode: One of "auto", "lattice", "stream", or "heuristic"
            deduplication_threshold: IoU threshold for table deduplication
            min_confidence: Minimum confidence score (0-100) to accept a table
        """
        self.table_mode = table_mode
        self.deduplication_threshold = deduplication_threshold
        self.min_confidence = min_confidence
        
        # Available methods based on installed libraries
        self.available_methods = []
        
        if PYMUPDF_AVAILABLE:
            self.available_methods.append('pymupdf_blocks')
            
        if CAMELOT_AVAILABLE:
            self.available_methods.append('camelot_lattice')
            self.available_methods.append('camelot_stream')
            
        if PDFPLUMBER_AVAILABLE:
            self.available_methods.append('pdfplumber')
        
        logger.info(f"Available table extraction methods: {', '.join(self.available_methods)}")
        
        # Set up extraction strategy chain based on mode
        self.strategies = []
        
        if table_mode == "auto" or table_mode == "lattice":
            if 'camelot_lattice' in self.available_methods:
                self.strategies.append(self._extract_lattice_tables)
                
        if table_mode == "auto" or table_mode == "stream":
            if 'camelot_stream' in self.available_methods:
                self.strategies.append(self._extract_stream_tables)
                
        if table_mode == "auto" or table_mode == "heuristic":
            if 'pdfplumber' in self.available_methods:
                self.strategies.append(self._extract_pdfplumber_tables)
            
            if 'pymupdf_blocks' in self.available_methods:
                self.strategies.append(self._extract_pymupdf_tables)
    
    def extract_tables(self, pdf_path: str, page_num: int, page: Optional[fitz.Page] = None) -> List[Dict[str, Any]]:
        """
        Extract tables from a specific page using configured strategies.
        
        Args:
            pdf_path: Path to PDF file
            page_num: 1-based page number
            page: Optional PyMuPDF page object (to avoid reopening)
            
        Returns:
            List of extracted tables
        """
        all_tables = []
        
        # Try each strategy in order
        for strategy in self.strategies:
            tables = strategy(pdf_path, page_num, page)
            all_tables.extend(tables)
            
            # If auto mode and we found tables, no need to try other methods
            if self.table_mode == "auto" and tables:
                break
        
        # Deduplicate tables if we have multiple
        if len(all_tables) > 1:
            all_tables = self._deduplicate_tables(all_tables)
        
        return all_tables
    
    def _extract_lattice_tables(self, pdf_path: str, page_num: int, page: Optional[fitz.Page] = None) -> List[Dict[str, Any]]:
        """Extract tables using Camelot's lattice mode (tables with borders)."""
        if 'camelot_lattice' not in self.available_methods:
            return []
            
        try:
            tables = camelot.read_pdf(
                pdf_path, 
                pages=str(page_num), 
                flavor='lattice',
                suppress_stdout=True
            )
            
            if not tables:
                return []
            
            result = []
            for i, table in enumerate(tables):
                if table.parsing_report['accuracy'] < self.min_confidence:
                    continue
                    
                table_data = table.data
                # Convert to list of lists of strings
                table_data = [[str(cell).strip() for cell in row] for row in table_data]
                
                # Clean empty cells and rows
                table_data = [
                    [cell if cell else "" for cell in row]
                    for row in table_data
                    if any(cell.strip() for cell in row)
                ]
                
                if not table_data:
                    continue
                
                # Get the table bbox
                try:
                    bbox = normalize_bbox([
                        float(table.cells[0][0].x1), 
                        float(table.cells[0][0].y1), 
                        float(table.cells[-1][-1].x2), 
                        float(table.cells[-1][-1].y2)
                    ])
                except (IndexError, AttributeError):
                    # If we can't get bbox from cells, use the reported bbox
                    bbox = table.bbox
                
                result.append({
                    "table_data": table_data,
                    "bbox": bbox,
                    "strategy": "lattice",
                    "accuracy": table.parsing_report['accuracy'],
                    "confidence": table.parsing_report['accuracy'] / 100.0,
                })
                
            return result
        except Exception as e:
            logger.warning(f"Lattice table extraction failed: {str(e)}")
            return []
    
    def _extract_stream_tables(self, pdf_path: str, page_num: int, page: Optional[fitz.Page] = None) -> List[Dict[str, Any]]:
        """Extract tables using Camelot's stream mode (whitespace tables)."""
        if 'camelot_stream' not in self.available_methods:
            return []
            
        try:
            tables = camelot.read_pdf(
                pdf_path, 
                pages=str(page_num), 
                flavor='stream',
                suppress_stdout=True
            )
            
            if not tables:
                return []
                
            result = []
            for i, table in enumerate(tables):
                if table.parsing_report['accuracy'] < self.min_confidence:
                    continue
                    
                table_data = table.data
                # Convert to list of lists of strings
                table_data = [[str(cell).strip() for cell in row] for row in table_data]
                
                # Clean empty rows
                table_data = [row for row in table_data if any(cell.strip() for cell in row)]
                
                if not table_data:
                    continue
                    
                # Get the table bbox
                try:
                    bbox = normalize_bbox([
                        float(table.cells[0][0].x1), 
                        float(table.cells[0][0].y1), 
                        float(table.cells[-1][-1].x2), 
                        float(table.cells[-1][-1].y2)
                    ])
                except (IndexError, AttributeError):
                    # If we can't get bbox from cells, use the reported bbox
                    bbox = table.bbox
                
                result.append({
                    "table_data": table_data,
                    "bbox": bbox,
                    "strategy": "stream",
                    "accuracy": table.parsing_report['accuracy'],
                    "confidence": table.parsing_report['accuracy'] / 100.0
                })
                
            return result
        except Exception as e:
            logger.warning(f"Stream table extraction failed: {str(e)}")
            return []
    
    def _extract_pdfplumber_tables(self, pdf_path: str, page_num: int, page: Optional[fitz.Page] = None) -> List[Dict[str, Any]]:
        """Extract tables using pdfplumber."""
        if 'pdfplumber' not in self.available_methods:
            return []
            
        try:
            tables = []
            
            with pdfplumber.open(pdf_path) as pdf:
                # pdfplumber is 0-indexed
                pdf_page = pdf.pages[page_num - 1]
                raw_tables = pdf_page.extract_tables()
                
                if not raw_tables:
                    return []
                    
                # Get page dimensions for bbox normalization
                page_height = pdf_page.height
                page_width = pdf_page.width
                
                for idx, table_data in enumerate(raw_tables):
                    if not table_data:
                        continue
                        
                    # Clean empty cells and convert to strings
                    cleaned_data = []
                    for row in table_data:
                        if not row or all(cell is None or cell.strip() == "" for cell in row if cell is not None):
                            continue
                        cleaned_data.append([str(cell).strip() if cell is not None else "" for cell in row])
                    
                    if not cleaned_data:
                        continue
                    
                    # PDFPlumber doesn't provide bbox directly, but we can estimate from cell positions
                    # if available, otherwise use a default that spans most of the page
                    bbox = [50, 50, page_width - 50, page_height - 50]  # Default
                    
                    # Estimate table bounding box if we can get cell positions
                    try:
                        cells = pdf_page.find_tables(table_settings={"vertical_strategy": "text", 
                                                                   "horizontal_strategy": "text"})[idx].cells
                        if cells:
                            # Get min/max coordinates
                            x0 = min(cell[0] for cell in cells if cell[0] is not None)
                            y0 = min(cell[1] for cell in cells if cell[1] is not None)
                            x1 = max(cell[2] for cell in cells if cell[2] is not None)
                            y1 = max(cell[3] for cell in cells if cell[3] is not None)
                            bbox = [x0, y0, x1, y1]
                    except (IndexError, AttributeError, TypeError) as e:
                        logger.debug(f"Couldn't get precise bbox for pdfplumber table: {str(e)}")
                    
                    # Normalize bbox coordinates
                    bbox = normalize_bbox(bbox)
                    
                    # Fixed accuracy score for pdfplumber
                    accuracy = 70.0
                    
                    tables.append({
                        "table_data": cleaned_data,
                        "bbox": bbox,
                        "strategy": "pdfplumber",
                        "accuracy": accuracy,
                        "confidence": accuracy / 100.0
                    })
                
            return tables
        except Exception as e:
            logger.warning(f"PDFPlumber table extraction failed: {str(e)}")
            return []
    
    def _extract_pymupdf_tables(self, pdf_path: str, page_num: int, page: Optional[fitz.Page] = None) -> List[Dict[str, Any]]:
        """Extract tables using PyMuPDF's block detection."""
        if 'pymupdf_blocks' not in self.available_methods:
            return []
            
        try:
            # Open the page if not provided
            close_doc = False
            if page is None:
                doc = fitz.open(pdf_path)
                close_doc = True
                page = doc[page_num - 1]
            
            tables = []
            
            # Get blocks and identify potential tables based on block structure
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                # Skip non-text blocks
                if block["type"] != 0:
                    continue
                    
                # Check if this block has tabular structure
                lines = block.get("lines", [])
                if not lines or len(lines) < 2:  # Need at least 2 rows
                    continue
                    
                if has_tabular_structure(lines):
                    # Extract table data
                    table_data = []
                    
                    # Process each line as a row
                    for line in lines:
                        row = []
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            row.append(text)
                        
                        if row:
                            table_data.append(row)
                    
                    # Only add if we have valid data
                    if table_data and any(row for row in table_data):
                        accuracy = 60.0  # Fixed accuracy for PyMuPDF
                        
                        tables.append({
                            "table_data": table_data,
                            "bbox": normalize_bbox(block["bbox"]),
                            "strategy": "pymupdf",
                            "accuracy": accuracy,
                            "confidence": accuracy / 100.0
                        })
            
            # Close document if we opened it
            if close_doc:
                doc.close()
                
            return tables
        except Exception as e:
            logger.warning(f"PyMuPDF table extraction failed: {str(e)}")
            return []
    
    def _deduplicate_tables(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate tables based on IoU of bounding boxes.
        
        Args:
            tables: List of tables with bbox information
            
        Returns:
            Deduplicated list of tables
        """
        if not tables:
            return []
            
        # Sort tables by accuracy
        sorted_tables = sorted(tables, key=lambda x: x.get("accuracy", 0), reverse=True)
        
        # Function to calculate IoU between two bboxes
        def calculate_iou(bbox1, bbox2):
            if not bbox1 or not bbox2:
                return 0
                
            # Calculate intersection
            x_left = max(bbox1[0], bbox2[0])
            y_top = max(bbox1[1], bbox2[1])
            x_right = min(bbox1[2], bbox2[2])
            y_bottom = min(bbox1[3], bbox2[3])
            
            if x_right < x_left or y_bottom < y_top:
                return 0.0
                
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            
            # Calculate areas of both bboxes
            bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
            bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
            
            # Calculate IoU
            iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
            return iou
        
        # Keep track of tables to keep
        deduplicated_tables = []
        
        # Process each table in order of accuracy
        for table in sorted_tables:
            bbox = table.get("bbox")
            if not bbox:
                deduplicated_tables.append(table)
                continue
                
            # Check if this table overlaps significantly with any already-kept table
            is_duplicate = False
            for kept_table in deduplicated_tables:
                kept_bbox = kept_table.get("bbox")
                if not kept_bbox:
                    continue
                    
                iou = calculate_iou(bbox, kept_bbox)
                if iou > self.deduplication_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated_tables.append(table)
        
        return deduplicated_tables

# For backwards compatibility
TableExtractor = UnifiedTableExtractor

def extract_tables(page: fitz.Page, pdf_path: str, page_num: int, table_mode: str = "auto") -> List[Dict[str, Any]]:
    """
    Extract tables from a page.
    
    Args:
        page: PDF page object
        pdf_path: Path to PDF file
        page_num: 1-based page number
        table_mode: Table extraction mode
        
    Returns:
        List of table blocks
    """
    try:
        extractor = UnifiedTableExtractor(table_mode=table_mode)
        tables = extractor.extract_tables(pdf_path, page_num, page)
        
        # Convert to the expected format for the parser
        structured_tables = []
        for table in tables:
            structured_table = {
                "type": "table",
                "section": None,  # Will be filled in later based on context
                "sub_section": None,  # Will be filled in later based on context
                "text": None,
                "table_data": table["table_data"],
                "bbox": table["bbox"],
                "extraction_method": table["strategy"],
                "confidence": table["accuracy"] / 100.0 if "accuracy" in table else 0.7
            }
            structured_tables.append(structured_table)
            
        return structured_tables
    except Exception as e:
        logger.error(f"Error extracting tables: {str(e)}")
        return []
