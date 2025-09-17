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

def has_vector_graphics(page: fitz.Page, block_bbox: List[float]) -> bool:
    """
    Check if a region of the page has vector graphics.
    
    Args:
        page: PDF page object
        block_bbox: Region to check [x0, y0, x1, y1]
        
    Returns:
        True if vector graphics detected
    """
    # Extract paths and check if any are in the region
    paths = page.get_drawings()
    if not paths:
        return False
        
    for path in paths:
        # Check if path intersects with block
        rect = path.get("rect")
        if not rect:
            continue
            
        # Check for overlap between path rect and block_bbox
        if (rect[0] < block_bbox[2] and rect[2] > block_bbox[0] and
            rect[1] < block_bbox[3] and rect[3] > block_bbox[1]):
            return True
            
    return False

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
        
        # Check both regions for vector graphics or images
        for search_region in [search_up, search_down]:
            if has_vector_graphics(page, search_region) or has_images(page, search_region):
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
                    "bbox": chart_bbox
                })
                break
    
    return chart_regions

def extract_chart_data(page: fitz.Page, chart_bbox: List[float]) -> Optional[List[List[str]]]:
    """
    Attempt to extract tabular data from chart legend or labels.
    
    Args:
        page: PDF page object
        chart_bbox: Chart region [x0, y0, x1, y1]
        
    Returns:
        Table data if found, None otherwise
    """
    # Simple approach: look for structured text within chart region
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
    
    # If we found data lines, convert to a simple two-column format
    if data_lines:
        chart_data = []
        
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
                # Fall back to the full line
                chart_data.append([line, ""])
                
        return chart_data
    
    return None

def detect_charts(page: fitz.Page) -> List[Dict[str, Any]]:
    """
    Detect and extract chart information from a PDF page.
    
    Args:
        page: PDF page object
        
    Returns:
        List of structured chart blocks
    """
    chart_regions = find_chart_regions(page)
    
    structured_charts = []
    for chart in chart_regions:
        # Try to extract any tabular data from the chart
        chart_data = extract_chart_data(page, chart["bbox"])
        
        structured_chart = {
            "type": "chart",
            "section": None,  # Will be updated based on context
            "sub_section": None,  # Will be updated based on context
            "text": None,
            "table_data": None,
            "chart_data": chart_data,
            "description": chart["description"],
            "bbox": chart["bbox"]
        }
        structured_charts.append(structured_chart)
    
    return structured_charts
