"""
Simplified table extraction module.
"""
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import json

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleTableExtractor:
    """A simplified table extractor that can use multiple methods based on available libraries."""
    
    def __init__(self):
        """Initialize the extractor with available methods."""
        self.available_methods = []
        
        # Check available methods
        try:
            import pdfplumber
            self.available_methods.append('pdfplumber')
        except ImportError:
            logger.warning("pdfplumber not available")
            
        try:
            import camelot
            self.available_methods.append('camelot_lattice')
            self.available_methods.append('camelot_stream')
        except ImportError:
            logger.warning("camelot not available")
            
        try:
            import fitz  # PyMuPDF
            self.available_methods.append('pymupdf_blocks')
        except ImportError:
            logger.warning("PyMuPDF not available")
            
        logger.info(f"Available table extraction methods: {', '.join(self.available_methods)}")
        
    def extract_tables(self, pdf_path: str, page_numbers: List[int] = None) -> Dict[int, List[Dict[str, Any]]]:
        """
        Extract tables from specified pages using all available methods.
        
        Args:
            pdf_path: Path to the PDF file
            page_numbers: List of page numbers to process (1-based, None = all pages)
            
        Returns:
            Dictionary with page numbers as keys and lists of tables as values
        """
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return {}
            
        result = {}
        
        # Try PyMuPDF first to get page count if available
        total_pages = 0
        try:
            import fitz
            with fitz.open(pdf_path) as doc:
                total_pages = len(doc)
                
            if page_numbers is None:
                page_numbers = list(range(1, total_pages + 1))
                
        except ImportError:
            logger.warning("PyMuPDF not available for page counting")
            
            # Try pdfplumber as fallback for page count
            try:
                import pdfplumber
                with pdfplumber.open(pdf_path) as pdf:
                    total_pages = len(pdf.pages)
                    
                if page_numbers is None:
                    page_numbers = list(range(1, total_pages + 1))
                    
            except ImportError:
                logger.error("No PDF library available to determine page count")
                return {}
                
        logger.info(f"Processing {len(page_numbers)} page(s) out of {total_pages} total pages")
        
        # Process each page
        for page_num in page_numbers:
            logger.info(f"Processing page {page_num}")
            tables = []
            
            # Try each available method
            if 'pdfplumber' in self.available_methods:
                tables.extend(self._extract_with_pdfplumber(pdf_path, page_num))
                
            if 'camelot_lattice' in self.available_methods:
                tables.extend(self._extract_with_camelot(pdf_path, page_num, flavor='lattice'))
                
            if 'camelot_stream' in self.available_methods and not tables:
                # Only try stream if lattice found nothing
                tables.extend(self._extract_with_camelot(pdf_path, page_num, flavor='stream'))
                
            if 'pymupdf_blocks' in self.available_methods and not tables:
                # Use PyMuPDF as last resort
                tables.extend(self._extract_with_pymupdf(pdf_path, page_num))
                
            result[page_num] = tables
            logger.info(f"Found {len(tables)} table(s) on page {page_num}")
            
        return result
    
    def _extract_with_pdfplumber(self, pdf_path: str, page_num: int) -> List[Dict[str, Any]]:
        """Extract tables using pdfplumber."""
        try:
            import pdfplumber
            tables = []
            
            with pdfplumber.open(pdf_path) as pdf:
                # pdfplumber is 0-indexed
                page = pdf.pages[page_num - 1]
                raw_tables = page.extract_tables()
                
                for idx, table_data in enumerate(raw_tables):
                    if not table_data:
                        continue
                        
                    # Clean empty cells and convert to strings
                    cleaned_data = []
                    for row in table_data:
                        if not row or all(cell is None for cell in row):
                            continue
                        cleaned_data.append([str(cell).strip() if cell is not None else "" for cell in row])
                    
                    if not cleaned_data:
                        continue
                        
                    tables.append({
                        "table_data": cleaned_data,
                        "method": "pdfplumber",
                        "confidence": 0.7,  # Fixed confidence for pdfplumber
                        "bbox": None  # pdfplumber doesn't provide bbox easily
                    })
            
            return tables
        except Exception as e:
            logger.error(f"Error in pdfplumber extraction: {str(e)}")
            return []
    
    def _extract_with_camelot(self, pdf_path: str, page_num: int, flavor: str) -> List[Dict[str, Any]]:
        """Extract tables using camelot with specified flavor."""
        try:
            import camelot
            tables = []
            
            # Extract tables with camelot
            raw_tables = camelot.read_pdf(
                pdf_path,
                pages=str(page_num),
                flavor=flavor,
                suppress_stdout=True
            )
            
            for idx, table in enumerate(raw_tables):
                if table.parsing_report['accuracy'] < 50:
                    continue
                    
                # Clean and convert data
                table_data = [[str(cell).strip() for cell in row] for row in table.data]
                
                # Remove empty rows
                table_data = [row for row in table_data if any(cell for cell in row)]
                
                if not table_data:
                    continue
                    
                # Try to get bbox
                try:
                    bbox = [
                        float(table.cells[0][0].x1),
                        float(table.cells[0][0].y1),
                        float(table.cells[-1][-1].x2),
                        float(table.cells[-1][-1].y2)
                    ]
                except (IndexError, AttributeError):
                    bbox = None
                
                tables.append({
                    "table_data": table_data,
                    "method": f"camelot_{flavor}",
                    "confidence": table.parsing_report['accuracy'] / 100.0,
                    "bbox": bbox
                })
            
            return tables
        except Exception as e:
            logger.error(f"Error in camelot extraction ({flavor}): {str(e)}")
            return []
    
    def _extract_with_pymupdf(self, pdf_path: str, page_num: int) -> List[Dict[str, Any]]:
        """Extract potential tables using PyMuPDF's block detection."""
        try:
            import fitz
            tables = []
            
            with fitz.open(pdf_path) as doc:
                page = doc[page_num - 1]
                
                # Get blocks with their coordinates
                blocks = page.get_text("dict")["blocks"]
                
                # Look for potential table blocks
                for block in blocks:
                    if block["type"] != 0:  # Skip non-text blocks
                        continue
                    
                    lines = []
                    for line in block.get("lines", []):
                        line_text = "".join(span.get("text", "") for span in line.get("spans", []))
                        if line_text.strip():
                            lines.append(line_text)
                    
                    if not lines:
                        continue
                        
                    # Check if this looks like tabular content
                    if self._is_likely_table(lines):
                        # Convert to table rows
                        table_data = self._lines_to_table(lines)
                        if len(table_data) > 1:  # At least two rows
                            tables.append({
                                "table_data": table_data,
                                "method": "pymupdf_blocks",
                                "confidence": 0.5,  # Lower confidence for heuristic detection
                                "bbox": block["bbox"]
                            })
            
            return tables
        except Exception as e:
            logger.error(f"Error in PyMuPDF extraction: {str(e)}")
            return []
    
    def _is_likely_table(self, lines: List[str]) -> bool:
        """Determine if a set of lines is likely to be a table."""
        if len(lines) < 2:
            return False
            
        # Check for consistent spacing patterns
        space_pattern_count = 0
        for line in lines:
            # Look for multiple spaces as column separators
            if line.count('  ') > 1:
                space_pattern_count += 1
                
        # If more than half the lines have consistent spacing, likely tabular
        return space_pattern_count > len(lines) / 2
    
    def _lines_to_table(self, lines: List[str]) -> List[List[str]]:
        """Convert a list of text lines to a table structure."""
        table_data = []
        
        # For each line, split by significant whitespace
        for line in lines:
            # Replace runs of spaces (2 or more) with a special marker
            marked = line.replace('  ', '§')
            
            # Split by the marker
            cells = marked.split('§')
            
            # Clean up cells
            cells = [cell.strip() for cell in cells if cell.strip()]
            
            if cells:
                table_data.append(cells)
        
        # Normalize the table (ensure all rows have same number of columns)
        max_cols = max(len(row) for row in table_data) if table_data else 0
        for row in table_data:
            while len(row) < max_cols:
                row.append("")
        
        return table_data

if __name__ == "__main__":
    # Simple command-line interface
    if len(sys.argv) < 2:
        print("Usage: python simple_table_extractor.py <pdf_path> [output_path] [page_numbers]")
        sys.exit(1)
        
    pdf_path = sys.argv[1]
    
    output_path = None
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        output_path = os.path.splitext(pdf_path)[0] + "_tables.json"
        
    page_numbers = None
    if len(sys.argv) > 3:
        try:
            # Handle comma-separated list of pages or ranges (e.g., "1,3-5,7")
            parts = sys.argv[3].split(',')
            page_numbers = []
            for part in parts:
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    page_numbers.extend(range(start, end + 1))
                else:
                    page_numbers.append(int(part))
        except ValueError:
            print("Error: Page numbers must be integers or ranges (e.g., '1,3-5,7')")
            sys.exit(1)
    
    extractor = SimpleTableExtractor()
    tables_by_page = extractor.extract_tables(pdf_path, page_numbers)
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tables_by_page, f, indent=2, ensure_ascii=False)
        
    print(f"Results written to {output_path}")
    
    # Print a summary
    total_tables = sum(len(tables) for tables in tables_by_page.values())
    print(f"Summary: {total_tables} tables found across {len(tables_by_page)} pages")
