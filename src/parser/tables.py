"""
Table extraction from PDF documents using multiple strategies.
"""
import os
import tempfile
import logging
from typing import List, Dict, Any, Tuple, Optional
import fitz  # PyMuPDF
import camelot
import pdfplumber
import numpy as np
from pathlib import Path
import re

from .utils import normalize_bbox, has_tabular_structure

logger = logging.getLogger(__name__)

class TableExtractor:
    """Extract tables from PDFs using multiple strategies."""
    
    def __init__(self, table_mode: str = "auto"):
        """
        Initialize table extractor.
        
        Args:
            table_mode: One of "auto", "lattice", "stream", or "heuristic"
        """
        self.table_mode = table_mode
        self.strategies = []
        
        # Set up extraction strategy chain based on mode
        if table_mode == "auto" or table_mode == "lattice":
            self.strategies.append(self.extract_lattice_tables)
            
        if table_mode == "auto" or table_mode == "stream":
            self.strategies.append(self.extract_stream_tables)
            
        if table_mode == "auto" or table_mode == "heuristic":
            self.strategies.append(self.extract_pdfplumber_tables)
            self.strategies.append(self.extract_heuristic_tables)
    
    def extract_lattice_tables(self, pdf_path: str, page_num: int) -> List[Dict[str, Any]]:
        """
        Extract tables using Camelot's lattice mode (tables with borders).
        
        Args:
            pdf_path: Path to PDF file
            page_num: 1-based page number
            
        Returns:
            List of extracted tables
        """
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
                if table.parsing_report['accuracy'] < 50:
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
                
                result.append({
                    "table_data": table_data,
                    "bbox": normalize_bbox(table.cells[0][0].bbox + table.cells[-1][-1].bbox),
                    "strategy": "lattice",
                    "accuracy": table.parsing_report['accuracy']
                })
                
            return result
        except Exception as e:
            logger.warning(f"Lattice table extraction failed: {str(e)}")
            return []
    
    def extract_stream_tables(self, pdf_path: str, page_num: int) -> List[Dict[str, Any]]:
        """
        Extract tables using Camelot's stream mode (whitespace tables).
        
        Args:
            pdf_path: Path to PDF file
            page_num: 1-based page number
            
        Returns:
            List of extracted tables
        """
        try:
            tables = camelot.read_pdf(
                pdf_path, 
                pages=str(page_num), 
                flavor='stream',
                suppress_stdout=True,
                edge_tol=500  # More tolerant of whitespace variations
            )
            
            if not tables:
                return []
            
            result = []
            for i, table in enumerate(tables):
                if table.parsing_report['accuracy'] < 50:
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
                
                # Attempt to determine table bbox
                try:
                    bbox = normalize_bbox(table.cells[0][0].bbox + table.cells[-1][-1].bbox)
                except:
                    # If bbox determination fails, use rough estimate
                    bbox = [0, 0, 0, 0]  # This will be refined later
                
                result.append({
                    "table_data": table_data,
                    "bbox": bbox,
                    "strategy": "stream",
                    "accuracy": table.parsing_report['accuracy']
                })
                
            return result
        except Exception as e:
            logger.warning(f"Stream table extraction failed: {str(e)}")
            return []
    
    def extract_pdfplumber_tables(self, pdf_path: str, page_num: int) -> List[Dict[str, Any]]:
        """
        Extract tables using pdfplumber.
        
        Args:
            pdf_path: Path to PDF file
            page_num: 1-based page number
            
        Returns:
            List of extracted tables
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # pdfplumber is 0-indexed
                page = pdf.pages[page_num - 1]
                tables = page.extract_tables()
                
                if not tables:
                    return []
                
                result = []
                for i, table_data in enumerate(tables):
                    # Clean empty cells and rows
                    table_data = [
                        [str(cell).strip() if cell else "" for cell in row]
                        for row in table_data
                        if any(cell for cell in row)
                    ]
                    
                    if not table_data:
                        continue
                    
                    # Get bounding box from the table's cells
                    # (approximation as pdfplumber doesn't provide exact bbox)
                    page_height = page.height
                    table_area = page.rects[i] if i < len(page.rects) else None
                    if table_area:
                        bbox = normalize_bbox([
                            table_area['x0'],
                            page_height - table_area['y1'],
                            table_area['x1'],
                            page_height - table_area['y0']
                        ])
                    else:
                        # If no rect available, use rough estimate
                        bbox = [0, 0, 0, 0]  # This will be refined later
                    
                    result.append({
                        "table_data": table_data,
                        "bbox": bbox,
                        "strategy": "pdfplumber",
                        "accuracy": 70  # Estimated accuracy
                    })
                
                return result
        except Exception as e:
            logger.warning(f"PDFPlumber table extraction failed: {str(e)}")
            return []
    
    def extract_heuristic_tables(self, pdf_path: str, page_num: int) -> List[Dict[str, Any]]:
        """
        Extract tables using heuristic analysis of text layout.
        
        Args:
            pdf_path: Path to PDF file
            page_num: 1-based page number
            
        Returns:
            List of extracted tables
        """
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num - 1]
            
            # Get text blocks
            blocks = page.get_text("dict")["blocks"]
            
            # Find blocks with tabular structure
            table_blocks = []
            for block in blocks:
                if block["type"] != 0:  # Skip non-text blocks
                    continue
                
                block_text = ""
                for line in block.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                    block_text += line_text + "\n"
                
                if has_tabular_structure(block_text):
                    table_blocks.append({
                        "text": block_text,
                        "bbox": normalize_bbox(block["bbox"])
                    })
            
            # Convert tabular text to structured tables
            result = []
            for block in table_blocks:
                table_data = self._parse_tabular_text(block["text"])
                if table_data and len(table_data) > 1:  # At least two rows
                    result.append({
                        "table_data": table_data,
                        "bbox": block["bbox"],
                        "strategy": "heuristic",
                        "accuracy": 50  # Lower confidence for heuristic method
                    })
            
            return result
        except Exception as e:
            logger.warning(f"Heuristic table extraction failed: {str(e)}")
            return []
    
    def _parse_tabular_text(self, text: str) -> List[List[str]]:
        """
        Parse text with tabular structure into a 2D array.
        
        Args:
            text: Text with potential tabular structure
            
        Returns:
            Table as list of lists of strings
        """
        lines = text.strip().split("\n")
        if not lines:
            return []
        
        # Analyze spaces to detect column positions
        space_positions = []
        for line in lines:
            for i, char in enumerate(line):
                if char == " " and (i == 0 or line[i-1] != " "):
                    space_positions.append(i)
        
        # Find most common space runs to detect column boundaries
        if not space_positions:
            return [lines]  # No structure detected, return as single column
        
        # Group spaces that are close to each other
        space_groups = []
        current_group = [space_positions[0]]
        
        for pos in space_positions[1:]:
            if pos - current_group[-1] <= 3:  # Spaces close to each other
                current_group.append(pos)
            else:
                if len(current_group) >= len(lines) / 3:  # Common space pattern
                    space_groups.append(sum(current_group) // len(current_group))
                current_group = [pos]
        
        if current_group and len(current_group) >= len(lines) / 3:
            space_groups.append(sum(current_group) // len(current_group))
        
        # Sort column positions
        space_groups.sort()
        
        # Parse lines into columns based on space positions
        table_data = []
        for line in lines:
            if not line.strip():
                continue
                
            row = []
            start_idx = 0
            
            for col_pos in space_groups:
                if col_pos < len(line):
                    cell = line[start_idx:col_pos].strip()
                    row.append(cell)
                    start_idx = col_pos
            
            # Add the last cell
            if start_idx < len(line):
                cell = line[start_idx:].strip()
                row.append(cell)
            
            if any(cell for cell in row):
                table_data.append(row)
        
        # Normalize the table (ensure all rows have same number of columns)
        max_cols = max(len(row) for row in table_data) if table_data else 0
        for row in table_data:
            while len(row) < max_cols:
                row.append("")
        
        return table_data
    
    def extract_tables(self, pdf_path: str, page_num: int) -> List[Dict[str, Any]]:
        """
        Extract tables from a PDF page using the configured strategy chain.
        
        Args:
            pdf_path: Path to PDF file
            page_num: 1-based page number
            
        Returns:
            List of extracted tables
        """
        all_tables = []
        
        # Try each strategy in sequence
        for strategy in self.strategies:
            tables = strategy(pdf_path, page_num)
            all_tables.extend(tables)
            
            # If auto mode and we found tables, stop trying more strategies
            if self.table_mode == "auto" and tables:
                break
        
        # Remove duplicate tables (similar content or overlapping)
        unique_tables = self._remove_duplicate_tables(all_tables)
        
        return unique_tables
    
    def _calculate_iou(self, boxA: List[float], boxB: List[float]) -> float:
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
        
        Args:
            boxA: First bounding box [x0, y0, x1, y1]
            boxB: Second bounding box [x0, y0, x1, y1]
            
        Returns:
            IoU value between 0 and 1
        """
        # Ensure boxes are in the correct format
        if len(boxA) != 4 or len(boxB) != 4:
            return 0.0
        
        try:
            # Determine the coordinates of the intersection rectangle
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            
            # Compute the area of intersection
            interArea = max(0, xB - xA) * max(0, yB - yA)
            
            # Compute the area of both bounding boxes
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            
            # Compute the intersection over union
            if boxAArea + boxBArea - interArea <= 0:
                return 0.0
                
            iou = interArea / float(boxAArea + boxBArea - interArea)
            return iou
        except Exception as e:
            logger.warning(f"IoU calculation failed: {str(e)}")
            return 0.0
            
    def _remove_duplicate_tables(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate tables based on content and position overlap.
        
        Args:
            tables: List of extracted tables
            
        Returns:
            List of unique tables
        """
        if not tables:
            return []
            
        # Sort by accuracy (highest first)
        tables = sorted(tables, key=lambda t: t.get("accuracy", 0), reverse=True)
        
        # Compare table content to detect duplicates
        unique_tables = [tables[0]]
        
        for table in tables[1:]:
            is_duplicate = False
            
            for unique_table in unique_tables:
                # Check content similarity
                table1 = table.get("table_data", [])
                table2 = unique_table.get("table_data", [])
                
                # Check bounding box overlap using IoU
                table_bbox = table.get("bbox", [0, 0, 0, 0])
                unique_bbox = unique_table.get("bbox", [0, 0, 0, 0])
                iou_score = self._calculate_iou(table_bbox, unique_bbox)
                
                # If bounding boxes overlap significantly or content is similar, consider it a duplicate
                if iou_score > 0.3:  # 30% overlap threshold
                    is_duplicate = True
                    logger.debug(f"Found duplicate table with IoU: {iou_score:.2f}")
                    break
                    
                # Simple size comparison as fallback
                if abs(len(table1) - len(table2)) <= 2:
                    # Check first row (header) similarity
                    if table1 and table2 and self._row_similarity(table1[0], table2[0]) > 0.7:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_tables.append(table)
        
        return unique_tables
    
    def _row_similarity(self, row1: List[str], row2: List[str]) -> float:
        """
        Calculate similarity between two table rows.
        
        Args:
            row1: First row as list of strings
            row2: Second row as list of strings
            
        Returns:
            Similarity score (0-1)
        """
        # Handle different length rows
        if not row1 or not row2:
            return 0.0
            
        # Join all cells and compare text
        text1 = " ".join(row1).lower()
        text2 = " ".join(row2).lower()
        
        # Simple similarity: common characters / total characters
        chars1 = set(text1)
        chars2 = set(text2)
        common = chars1.intersection(chars2)
        
        return len(common) / max(len(chars1) + len(chars2) - len(common), 1)


def extract_tables(page: fitz.Page, pdf_path: str, page_num: int, table_mode: str = "auto") -> List[Dict[str, Any]]:
    """
    Extract tables from a PDF page.
    
    Args:
        page: PDF page object
        pdf_path: Path to PDF file
        page_num: 1-based page number
        table_mode: Table extraction mode
        
    Returns:
        List of structured table blocks with section information
    """
    extractor = TableExtractor(table_mode=table_mode)
    tables = extractor.extract_tables(pdf_path, page_num)
    
    structured_tables = []
    for table in tables:
        structured_table = {
            "type": "table",
            "section": None,  # Will be updated based on context
            "sub_section": None,  # Will be updated based on context
            "text": None,
            "table_data": table["table_data"],
            "chart_data": None,
            "description": None,
            "bbox": table["bbox"]
        }
        structured_tables.append(structured_table)
    
    return structured_tables
