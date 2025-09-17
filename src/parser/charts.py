"""
Chart detection and extraction from PDF documents.
"""
import fitz  # PyMuPDF
import re
from typing import List, Dict, Any, Tuple, Optional
import logging

from .utils import normalize_bbox

logger = logging.getLogger(__name__)

# Keywords that might indicate chart presence
CHART_KEYWORDS = [
    'figure', 'fig', 'chart', 'graph', 'plot', 'histogram', 'bar chart', 
    'pie chart', 'scatter plot', 'line graph', 'diagram'
]

def is_chart_caption(text: str) -> bool:
    """
    Check if text is likely a chart caption.
    
    Args:
        text: Text to analyze
        
    Returns:
        True if text appears to be a chart caption
    """
    text_lower = text.lower()
    
    # Check for common chart caption patterns
    for keyword in CHART_KEYWORDS:
        pattern = rf'(^|[^a-z]){keyword}(\s+\d+|\s*:|\s*\.|\s*\(|\s*-|\s*–|$)'
        if re.search(pattern, text_lower):
            return True
    
    # Check for "Figure X:" or "Fig. X:" patterns
    if re.search(r'(figure|fig)\.?\s*\d+\s*[:.-]', text_lower):
        return True
        
    return False

def extract_chart_description(text: str) -> str:
    """
    Extract a clean description from chart caption text.
    
    Args:
        text: Chart caption text
        
    Returns:
        Cleaned chart description
    """
    text = text.strip()
    
    # Remove common prefixes like "Figure 1:", "Chart 2:" etc.
    for keyword in CHART_KEYWORDS:
        pattern = rf'^{keyword}\s*\d*\s*[:.-]\s*'
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Also try with "Fig."
    text = re.sub(r'^fig\.\s*\d*\s*[:.-]\s*', '', text, flags=re.IGNORECASE)
    
    return text.strip()

def analyze_vector_graphics_density(page: fitz.Page, block_bbox: List[float]) -> Dict[str, Any]:
    """
    Analyze the density and patterns of vector graphics in a region to identify charts.
    
    Args:
        page: PDF page object
        block_bbox: Region to check [x0, y0, x1, y1]
        
    Returns:
        Dictionary with analysis results including:
        - has_graphics: Boolean indicating presence of vector graphics
        - density: Number of vector paths per square inch
        - horizontal_lines: Number of horizontal lines detected
        - vertical_lines: Number of vertical lines detected
        - is_likely_chart: Boolean indicating if the pattern suggests a chart
    """
    paths = page.get_drawings()
    if not paths:
        return {
            "has_graphics": False,
            "density": 0,
            "horizontal_lines": 0,
            "vertical_lines": 0,
            "is_likely_chart": False
        }
    
    # Calculate region size in points
    region_width = block_bbox[2] - block_bbox[0]
    region_height = block_bbox[3] - block_bbox[1]
    region_area = region_width * region_height / (72 * 72)  # Convert to square inches
    
    # Initialize counters
    paths_in_region = 0
    horizontal_lines = 0
    vertical_lines = 0
    
    for path in paths:
        # Get path components
        items = path.get("items", [])
        rect = path.get("rect")
        
        if not rect:
            continue
            
        # Check if path intersects with block
        if not (rect[0] < block_bbox[2] and rect[2] > block_bbox[0] and
                rect[1] < block_bbox[3] and rect[3] > block_bbox[1]):
            continue
        
        paths_in_region += 1
        
        # Analyze path segments for horizontal and vertical lines
        for item in items:
            if item[0] == "l":  # Line segment
                if len(item) >= 3:
                    # Check if horizontal line
                    if abs(item[2] - item[0]) < 2:
                        horizontal_lines += 1
                    # Check if vertical line
                    if abs(item[3] - item[1]) < 2:
                        vertical_lines += 1
    
    # Calculate density (paths per square inch)
    density = paths_in_region / max(region_area, 0.001)  # Avoid division by zero
    
    # Chart patterns typically have both horizontal and vertical lines (axes)
    # and a relatively high density of paths
    is_likely_chart = (density > 10 and  # More than 10 paths per square inch
                      horizontal_lines >= 1 and 
                      vertical_lines >= 1)
    
    return {
        "has_graphics": paths_in_region > 0,
        "density": density,
        "horizontal_lines": horizontal_lines,
        "vertical_lines": vertical_lines,
        "is_likely_chart": is_likely_chart
    }

def has_images(page: fitz.Page, block_bbox: List[float]) -> bool:
    """
    Check if a region of the page has images.
    
    Args:
        page: PDF page object
        block_bbox: Region to check [x0, y0, x1, y1]
        
    Returns:
        True if images detected
    """
    images = page.get_images()
    if not images:
        return False
        
    for img in page.get_image_info():
        # Convert image bbox to page coordinates
        bbox = img.get("bbox")
        if not bbox:
            continue
            
        # Check for overlap between image bbox and block_bbox
        if (bbox[0] < block_bbox[2] and bbox[2] > block_bbox[0] and
            bbox[1] < block_bbox[3] and bbox[3] > block_bbox[1]):
            return True
            
    return False

def find_chart_regions(page: fitz.Page) -> List[Dict[str, Any]]:
    """
    Find regions in the page that are likely to contain charts.
    
    Args:
        page: PDF page object
        
    Returns:
        List of potential chart regions with their descriptions
    """
    chart_regions = []
    
    # Get text blocks
    blocks = page.get_text("dict")["blocks"]
    
    # First, identify caption blocks
    caption_blocks = []
    for block in blocks:
        if block["type"] != 0:  # Skip non-text blocks
            continue
            
        block_text = ""
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                block_text += span.get("text", "")
                
        if is_chart_caption(block_text):
            caption_blocks.append({
                "text": block_text,
                "bbox": normalize_bbox(block["bbox"])
            })
    
    # For each caption, look above and below for chart content
    for caption in caption_blocks:
        caption_bbox = caption["bbox"]
        caption_y = (caption_bbox[1] + caption_bbox[3]) / 2
        
        # Determine search region (larger area around the caption)
        page_width = page.rect.width
        page_height = page.rect.height
        
        # Search above the caption
        search_up = [
            0,  # x0
            max(0, caption_bbox[1] - page_height * 0.2),  # y0
            page_width,  # x1
            caption_bbox[1]  # y1
        ]
        
        # Search below the caption
        search_down = [
            0,  # x0
            caption_bbox[3],  # y0
            page_width,  # x1
            min(page_height, caption_bbox[3] + page_height * 0.2)  # y1
        ]
        
        # Check both regions for vector graphics, vector density, or images
        for search_region in [search_up, search_down]:
            graphics_analysis = analyze_vector_graphics_density(page, search_region)
            has_chart_elements = graphics_analysis["is_likely_chart"] or has_images(page, search_region)
            
            if has_chart_elements:
                description = extract_chart_description(caption["text"])
                
                # Combine the search region with caption region for a complete chart bbox
                chart_bbox = [
                    min(search_region[0], caption_bbox[0]),
                    min(search_region[1], caption_bbox[1]),
                    max(search_region[2], caption_bbox[2]),
                    max(search_region[3], caption_bbox[3])
                ]
                
                chart_regions.append({
                    "description": description,
                    "bbox": chart_bbox,
                    "chart_type": "line_chart" if graphics_analysis["horizontal_lines"] > graphics_analysis["vertical_lines"] else
                                 "bar_chart" if graphics_analysis["vertical_lines"] > graphics_analysis["horizontal_lines"] else
                                 "scatter_plot" if graphics_analysis["density"] > 20 else
                                 "generic_chart"
                })
                break
    
    return chart_regions

def extract_chart_data(page: fitz.Page, chart_bbox: List[float], use_ocr: bool = True) -> Optional[List[List[str]]]:
    """
    Extract data from chart using both embedded text and OCR.
    
    Args:
        page: PDF page object
        chart_bbox: Chart region [x0, y0, x1, y1]
        use_ocr: Whether to use OCR for text extraction
        
    Returns:
        Table data if found, None otherwise
    """
    chart_data = []
    
    # Step 1: Try to extract structured text from the chart
    chart_text = page.get_textbox(chart_bbox)
    lines = chart_text.strip().split("\n")
    
    # Look for patterns like "Series 1: 42.5%"
    data_lines = []
    for line in lines:
        # Skip lines that are likely not data
        if len(line) < 5 or any(keyword in line.lower() for keyword in ["figure", "chart", "source"]):
            continue
            
        # Look for label-value pairs (e.g. "Category: 42.5%")
        if re.search(r'[a-zA-Z].*[:=]\s*\d+', line):
            data_lines.append(line)
            
        # Look for standalone numeric entries with % or units
        elif re.search(r'\d+(\.\d+)?%', line):
            data_lines.append(line)
            
        # Look for other numeric patterns that might be chart data
        elif re.search(r'\d+(\.\d+)?', line) and not re.search(r'(figure|fig|page)', line.lower()):
            data_lines.append(line)
    
    # If we found data lines using direct text extraction
    if data_lines:
        # Add header
        chart_data.append(["Label", "Value"])
        
        # Add data rows
        for line in data_lines:
            parts = re.split(r'[:=]', line, 1)
            if len(parts) == 2:
                label = parts[0].strip()
                value = parts[1].strip()
                chart_data.append([label, value])
            else:
                # Try to split by whitespace to separate label and value
                parts = line.strip().rsplit(maxsplit=1)
                if len(parts) == 2 and re.search(r'\d', parts[1]):
                    chart_data.append([parts[0], parts[1]])
                else:
                    # Fall back to the full line
                    chart_data.append([line, ""])
    
    # Step 2: If direct extraction yielded no results and OCR is enabled, try OCR
    if not data_lines and use_ocr:
        try:
            from ..ocr.ocr_pipeline import extract_text_with_ocr
            
            # Generate an image of the chart region
            pix = page.get_pixmap(clip=chart_bbox)
            ocr_text = extract_text_with_ocr(pix)
            
            # Process OCR text similarly to direct text
            if ocr_text:
                ocr_lines = ocr_text.strip().split("\n")
                ocr_data_lines = []
                
                for line in ocr_lines:
                    if len(line) < 5:
                        continue
                        
                    # Same patterns as before
                    if (re.search(r'[a-zA-Z].*[:=]\s*\d+', line) or
                        re.search(r'\d+(\.\d+)?%', line) or
                        (re.search(r'\d+(\.\d+)?', line) and not re.search(r'(figure|fig|page)', line.lower()))):
                        ocr_data_lines.append(line)
                
                if ocr_data_lines:
                    # Only add header if we didn't already add one
                    if not chart_data:
                        chart_data.append(["Label", "Value"])
                        
                    # Process OCR lines
                    for line in ocr_data_lines:
                        parts = re.split(r'[:=]', line, 1)
                        if len(parts) == 2:
                            label = parts[0].strip()
                            value = parts[1].strip()
                            chart_data.append([label, value])
                        else:
                            # Try to split by whitespace
                            parts = line.strip().rsplit(maxsplit=1)
                            if len(parts) == 2 and re.search(r'\d', parts[1]):
                                chart_data.append([parts[0], parts[1]])
                            else:
                                chart_data.append([line, ""])
        except Exception as e:
            logger.warning(f"OCR for chart data extraction failed: {str(e)}")
    
    return chart_data if chart_data else None

def detect_charts(page: fitz.Page, use_ocr: bool = True) -> List[Dict[str, Any]]:
    """
    Detect and extract chart information from a PDF page.
    
    Args:
        page: PDF page object
        use_ocr: Whether to use OCR for text extraction
        
    Returns:
        List of structured chart blocks
    """
    chart_regions = find_chart_regions(page)
    
    structured_charts = []
    for chart in chart_regions:
        # Try to extract any tabular data from the chart
        chart_data = extract_chart_data(page, chart["bbox"], use_ocr=use_ocr)
        
        structured_chart = {
            "type": "chart",
            "chart_type": chart.get("chart_type", "generic_chart"),
            "section": None,  # Will be updated based on context
            "sub_section": None,  # Will be updated based on context
            "text": None,
            "table_data": None,
            "chart_data": chart_data,
            "description": chart["description"],
            "bbox": chart["bbox"]
        }
        structured_charts.append(structured_chart)
    
    # Heuristic-based chart detection for cases without captions
    # Look for regions with high vector graphics density
    page_width = page.rect.width
    page_height = page.rect.height
    
    # Divide page into a 3x3 grid and analyze each cell
    cell_width = page_width / 3
    cell_height = page_height / 3
    
    for i in range(3):
        for j in range(3):
            cell_bbox = [
                i * cell_width,
                j * cell_height,
                (i + 1) * cell_width,
                (j + 1) * cell_height
            ]
            
            # Skip cells that already contain detected charts
            skip_cell = False
            for chart in chart_regions:
                chart_bbox = chart["bbox"]
                # If significant overlap
                if (chart_bbox[0] < cell_bbox[2] and chart_bbox[2] > cell_bbox[0] and
                    chart_bbox[1] < cell_bbox[3] and chart_bbox[3] > cell_bbox[1]):
                    area_chart = (chart_bbox[2] - chart_bbox[0]) * (chart_bbox[3] - chart_bbox[1])
                    area_cell = (cell_bbox[2] - cell_bbox[0]) * (cell_bbox[3] - cell_bbox[1])
                    overlap_ratio = min(area_chart, area_cell) / max(area_chart, area_cell)
                    if overlap_ratio > 0.3:  # 30% overlap
                        skip_cell = True
                        break
            
            if skip_cell:
                continue
                
            # Analyze cell for chart-like content
            graphics_analysis = analyze_vector_graphics_density(page, cell_bbox)
            if graphics_analysis["is_likely_chart"]:
                # Try to extract any text data from the potential chart
                chart_data = extract_chart_data(page, cell_bbox, use_ocr=use_ocr)
                
                chart_type = "line_chart" if graphics_analysis["horizontal_lines"] > graphics_analysis["vertical_lines"] else \
                             "bar_chart" if graphics_analysis["vertical_lines"] > graphics_analysis["horizontal_lines"] else \
                             "scatter_plot" if graphics_analysis["density"] > 20 else "generic_chart"
                
                structured_chart = {
                    "type": "chart",
                    "chart_type": chart_type,
                    "section": None,
                    "sub_section": None,
                    "text": None,
                    "table_data": None,
                    "chart_data": chart_data,
                    "description": f"Detected {chart_type.replace('_', ' ')}",
                    "bbox": cell_bbox
                }
                structured_charts.append(structured_chart)
    
    return structured_charts
