"""
Chart detection and extraction from PDF documents.
"""
import fitz  # PyMuPDF
import re
import os
import numpy as np
import io
from typing import List, Dict, Any, Tuple, Optional
import logging

from .utils import normalize_bbox

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("WARNING: OpenCV not available. Chart data extraction will be limited.")

try:
    from .chart_data import extract_data_from_chart, render_chart_data_visualization
    CHART_DATA_MODULE_AVAILABLE = True
except ImportError:
    CHART_DATA_MODULE_AVAILABLE = False
    print("WARNING: chart_data module not available. Chart data extraction will be disabled.")

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

def detect_pie_charts(page: fitz.Page, block_bbox: List[float]) -> Dict[str, Any]:
    """
    Detect pie charts in a region using simpler vector graphics analysis.
    
    Args:
        page: PDF page object
        block_bbox: Region to check [x0, y0, x1, y1]
        
    Returns:
        Dictionary with detection results
    """
    if not OPENCV_AVAILABLE:
        return {"is_pie_chart": False, "confidence": 0}
    
    # First try to detect using vector graphics - faster and more reliable when available
    paths = page.get_drawings()
    if paths:
        # Look for curved paths and arc segments, which are common in pie charts
        curved_paths = 0
        line_paths = 0
        filled_wedges = 0
        
        for path in paths:
            # Get path components
            items = path.get("items", [])
            rect = path.get("rect")
            fill = path.get("fill", False)
            
            if not rect:
                continue
                
            # Check if path intersects with block
            if not (rect[0] < block_bbox[2] and rect[2] > block_bbox[0] and
                    rect[1] < block_bbox[3] and rect[3] > block_bbox[1]):
                continue
            
            # Count curved path segments
            has_curves = False
            has_lines = False
            for item in items:
                try:
                    if item[0] == "c":  # Curve
                        has_curves = True
                    elif item[0] == "l":  # Line
                        has_lines = True
                except (IndexError, TypeError):
                    pass
            
            if has_curves:
                curved_paths += 1
            if has_lines:
                line_paths += 1
            
            # Check for filled wedge shapes (common in pie charts)
            if fill and has_curves:
                filled_wedges += 1
        
        # Pie charts typically have more curved paths than straight lines
        # and often have filled wedges
        if curved_paths >= 3 and filled_wedges >= 2:
            return {
                "is_pie_chart": True,
                "confidence": min(100, filled_wedges * 15),
                "detection_method": "vector_analysis",
                "curved_paths": curved_paths,
                "filled_wedges": filled_wedges
            }
    
    # If vector analysis didn't detect a pie chart, use a simpler image-based approach
    try:
        # Extract text in the region and look for keywords
        text = page.get_textbox(block_bbox).lower()
        pie_keywords = ["pie chart", "pie", "allocation", "breakdown", "distribution", "sector", "composition"]
        
        # Check if any keywords are present
        keyword_matches = sum(1 for keyword in pie_keywords if keyword in text)
        
        # Also check for percentage symbols, common in pie chart labels
        percentage_count = text.count("%")
        
        # If we have keywords related to pie charts and multiple percentage values,
        # it's likely a pie chart or related explanation
        if (keyword_matches > 0 and percentage_count >= 2) or percentage_count >= 4:
            return {
                "is_pie_chart": True,
                "confidence": min(100, keyword_matches * 15 + percentage_count * 5),
                "detection_method": "text_analysis",
                "keyword_matches": keyword_matches,
                "percentage_count": percentage_count
            }
        
        # Look for color-based clues in the image data
        # This is a simplified version that runs faster
        pix = page.get_pixmap(clip=block_bbox, dpi=100)  # Lower DPI for speed
        
        # Check if the image is colorful (pie charts usually are)
        # Simple heuristic: count pixels with significant RGB differences
        samples = pix.samples
        colorful_pixels = 0
        step = 12  # Sample every 12th pixel for speed
        
        for i in range(0, len(samples), step * pix.n):
            if i + 2 < len(samples):
                r, g, b = samples[i], samples[i+1], samples[i+2]
                # Check if RGB values differ significantly
                if max(abs(r-g), abs(r-b), abs(g-b)) > 40:
                    colorful_pixels += 1
        
        colorfulness = colorful_pixels / (pix.width * pix.height / step)
        
        # If the region is colorful, it might be a pie chart
        if colorfulness > 0.3:  # More than 30% colorful pixels
            return {
                "is_pie_chart": True,
                "confidence": min(100, colorfulness * 100),
                "detection_method": "color_analysis",
                "colorfulness": colorfulness
            }
        
        return {"is_pie_chart": False, "confidence": 0}
    
    except Exception as e:
        logger.warning(f"Error in pie chart detection: {str(e)}")
        return {"is_pie_chart": False, "confidence": 0, "error": str(e)}

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
            "is_likely_chart": False,
            "curved_paths": 0
        }
    
    # Calculate region size in points
    region_width = block_bbox[2] - block_bbox[0]
    region_height = block_bbox[3] - block_bbox[1]
    region_area = region_width * region_height / (72 * 72)  # Convert to square inches
    
    # Initialize counters
    paths_in_region = 0
    horizontal_lines = 0
    vertical_lines = 0
    curved_paths = 0
    
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
            try:
                if item[0] == "l":  # Line segment
                    if len(item) >= 5:  # Make sure we have enough elements
                        # Check if horizontal line (y coords are similar)
                        if abs(item[2] - item[4]) < 2:
                            horizontal_lines += 1
                        # Check if vertical line (x coords are similar)
                        if abs(item[1] - item[3]) < 2:
                            vertical_lines += 1
                elif item[0] == "c":  # Curved segment (Bézier curve)
                    curved_paths += 1
            except (IndexError, TypeError) as e:
                # Skip items with wrong format
                logger.debug(f"Skipping invalid path item: {e}")
    
    # Calculate density (paths per square inch)
    density = paths_in_region / max(region_area, 0.001)  # Avoid division by zero
    
    # Check for pie chart indicators: high curved path count, few straight lines
    is_pie_chart = (curved_paths > 3 and horizontal_lines < 2 and vertical_lines < 2)
    
    # Chart patterns typically have both horizontal and vertical lines (axes)
    # and a relatively high density of paths
    is_likely_chart = (density > 10 and  # More than 10 paths per square inch
                      ((horizontal_lines >= 1 and vertical_lines >= 1) or  # Line/bar charts
                       is_pie_chart))  # Pie charts
    
    return {
        "has_graphics": paths_in_region > 0,
        "density": density,
        "horizontal_lines": horizontal_lines,
        "vertical_lines": vertical_lines,
        "curved_paths": curved_paths,
        "is_likely_chart": is_likely_chart,
        "is_pie_chart": is_pie_chart
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
            
            # Check for pie charts if OpenCV is available
            pie_chart_info = {"is_pie_chart": False}
            if OPENCV_AVAILABLE:
                pie_chart_info = detect_pie_charts(page, search_region)
                
            # If any chart detection method succeeded
            if has_chart_elements or pie_chart_info["is_pie_chart"]:
                description = extract_chart_description(caption["text"])
                
                # Combine the search region with caption region for a complete chart bbox
                chart_bbox = [
                    min(search_region[0], caption_bbox[0]),
                    min(search_region[1], caption_bbox[1]),
                    max(search_region[2], caption_bbox[2]),
                    max(search_region[3], caption_bbox[3])
                ]
                
                # Determine chart type
                chart_type = "generic_chart"
                
                # Check if it's a pie chart first
                if pie_chart_info["is_pie_chart"]:
                    chart_type = "pie_chart"
                # If not a pie chart, use the vector analysis
                elif graphics_analysis["is_pie_chart"]:
                    chart_type = "pie_chart"
                elif graphics_analysis["horizontal_lines"] > graphics_analysis["vertical_lines"]:
                    chart_type = "line_chart"
                elif graphics_analysis["vertical_lines"] > graphics_analysis["horizontal_lines"]:
                    chart_type = "bar_chart"
                elif graphics_analysis["density"] > 20:
                    chart_type = "scatter_plot"
                
                # Check if the caption text gives hints about chart type
                caption_text_lower = caption["text"].lower()
                if "pie" in caption_text_lower:
                    chart_type = "pie_chart"
                elif "bar" in caption_text_lower:
                    chart_type = "bar_chart"
                elif "line" in caption_text_lower:
                    chart_type = "line_chart"
                
                chart_regions.append({
                    "description": description,
                    "bbox": chart_bbox,
                    "chart_type": chart_type,
                    "pie_chart_info": pie_chart_info if pie_chart_info["is_pie_chart"] else None
                })
                break
    
    return chart_regions

def extract_chart_data(page: fitz.Page, chart_bbox: List[float], use_ocr: bool = True, config=None) -> Dict[str, Any]:
    """
    Extract data from a chart, including any text elements and advanced data extraction if available.
    
    Args:
        page: PDF page object
        chart_bbox: Chart region [x0, y0, x1, y1]
        use_ocr: Whether to use OCR for text extraction
        config: Optional configuration manager
        
    Returns:
        Dictionary containing extracted chart data
    """
    # Extract text from the chart region
    chart_text = page.get_textbox(chart_bbox)
    lines = chart_text.strip().split("\n")
    
    # Extract vector graphics info
    graphics_analysis = analyze_vector_graphics_density(page, chart_bbox)
    
    # Try to infer chart type based on vector graphics patterns
    chart_type = "unknown"
    if graphics_analysis["is_likely_chart"]:
        if graphics_analysis["horizontal_lines"] > 0 and graphics_analysis["vertical_lines"] > 0:
            # Both axes present
            if graphics_analysis["horizontal_lines"] > graphics_analysis["vertical_lines"] * 2:
                chart_type = "line_chart"
            elif graphics_analysis["vertical_lines"] > graphics_analysis["horizontal_lines"] * 2:
                chart_type = "bar_chart"
            else:
                chart_type = "mixed_chart"
        else:
            # Other types of charts (pie, etc.)
            chart_type = "other_chart"
    
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
    
    # Process data lines to extract values
    tabular_data = []
    if data_lines:
        # Add header
        tabular_data.append(["Label", "Value"])
        
        # Add data rows
        for line in data_lines:
            parts = re.split(r'[:=]', line, 1)
            if len(parts) == 2:
                label = parts[0].strip()
                value = parts[1].strip()
                tabular_data.append([label, value])
            else:
                # Try to split by whitespace to separate label and value
                parts = line.strip().rsplit(maxsplit=1)
                if len(parts) == 2 and re.search(r'\d', parts[1]):
                    tabular_data.append([parts[0], parts[1]])
                else:
                    # Fall back to the full line
                    tabular_data.append([line, ""])
    
    # Find potential axis labels
    x_axis_label = ""
    y_axis_label = ""
    legend_items = []
    
    for line in lines:
        # Check for potential axis labels
        if re.search(r'(axis|scale|units|values|percent)', line.lower()):
            if not x_axis_label and chart_bbox[3] - chart_bbox[1] > 0:
                # If text is closer to bottom of chart, likely an X-axis label
                x_axis_label = line
            elif not y_axis_label:
                # Otherwise, could be Y-axis
                y_axis_label = line
        
        # Check for legend items (often have patterns like "■ Series 1")
        if re.search(r'^[■□●○▲▼◆◇♦♢]|^\*|^-|^–|^\+|^\|', line):
            legend_items.append(line)
    
    # Process OCR text if direct extraction yielded limited results and OCR is enabled
    ocr_data = None
    if (not data_lines or len(data_lines) < 3) and use_ocr:
        try:
            from ..ocr.ocr_pipeline import extract_text_with_ocr
            
            # Generate an image of the chart region
            pix = page.get_pixmap(clip=chart_bbox)
            ocr_text = extract_text_with_ocr(pix)
            
            ocr_data = {
                "status": "success" if ocr_text else "no_text_detected",
                "text": ocr_text,
                "data_points": []
            }
            
            # Process OCR text to extract structured data
            if ocr_text:
                ocr_lines = ocr_text.strip().split("\n")
                for line in ocr_lines:
                    if len(line) < 5:
                        continue
                        
                    # Look for patterns that indicate data points
                    if (re.search(r'[a-zA-Z].*[:=]\s*\d+', line) or
                        re.search(r'\d+(\.\d+)?%', line) or
                        (re.search(r'\d+(\.\d+)?', line) and not re.search(r'(figure|fig|page)', line.lower()))):
                        
                        # Try to parse the data point
                        parts = re.split(r'[:=]', line, 1)
                        if len(parts) == 2:
                            ocr_data["data_points"].append({
                                "label": parts[0].strip(),
                                "value": parts[1].strip()
                            })
                        else:
                            # Try to split by whitespace
                            parts = line.strip().rsplit(maxsplit=1)
                            if len(parts) == 2 and re.search(r'\d', parts[1]):
                                ocr_data["data_points"].append({
                                    "label": parts[0],
                                    "value": parts[1]
                                })
                            else:
                                ocr_data["data_points"].append({
                                    "text": line,
                                    "value": None
                                })
        except Exception as e:
            logger.warning(f"OCR for chart data extraction failed: {str(e)}")
            ocr_data = {"status": f"error: {str(e)}", "text": "", "data_points": []}
    
    # Advanced chart data extraction using OpenCV if available
    extracted_data = None
    if CHART_DATA_MODULE_AVAILABLE and OPENCV_AVAILABLE:
        try:
            # Render chart region to image
            pix = page.get_pixmap(clip=chart_bbox, dpi=300)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            
            # Convert to BGR format for OpenCV
            if pix.n == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 1:  # Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            # Extract data using chart_data module
            chart_data_type = None
            if chart_type == "bar_chart":
                chart_data_type = "bar"
            elif chart_type == "line_chart":
                chart_data_type = "line"
            elif chart_type == "pie_chart":
                chart_data_type = "pie"
                
            extracted_data = extract_data_from_chart(img, chart_data_type, config)
            
            # Generate visualization if visual debugging is enabled
            if config and config.get("visual_debug", "enabled", False):
                debug_dir = config.get("visual_debug", "output_path", "debug_output")
                os.makedirs(debug_dir, exist_ok=True)
                
                # Create unique filename based on page number and chart position
                filename = f"chart_data_p{page.number+1}_x{int(chart_bbox[0])}_y{int(chart_bbox[1])}.png"
                viz_path = os.path.join(debug_dir, filename)
                
                # Render visualization
                render_chart_data_visualization(img, extracted_data, viz_path)
                logger.debug(f"Chart data visualization saved to {viz_path}")
                
        except Exception as e:
            logger.warning(f"Error extracting chart data with OpenCV: {str(e)}")
            extracted_data = {"error": f"Failed to extract chart data: {str(e)}"}
            
    # Check if there are images in the chart region
    has_image = has_images(page, chart_bbox)
    
    return {
        "chart_type": chart_type,
        "has_vector_graphics": graphics_analysis["has_graphics"],
        "has_image": has_image,
        "graphics_density": graphics_analysis["density"],
        "extracted_text": chart_text,
        "tabular_data": tabular_data if tabular_data else None,
        "potential_axis_labels": {
            "x_axis": x_axis_label,
            "y_axis": y_axis_label
        },
        "potential_legend_items": legend_items,
        "potential_data_points": data_lines,
        "ocr_data": ocr_data,
        "extracted_data": extracted_data
    }

def detect_charts(page: fitz.Page, use_ocr: bool = True, config=None) -> List[Dict[str, Any]]:
    """
    Detect and extract chart information from a PDF page.
    
    Args:
        page: PDF page object
        use_ocr: Whether to use OCR for text extraction
        config: Optional configuration manager
        
    Returns:
        List of structured chart blocks
    """
    chart_regions = find_chart_regions(page)
    
    structured_charts = []
    for chart in chart_regions:
        # Try to extract any tabular data from the chart
        chart_data = extract_chart_data(page, chart["bbox"], use_ocr=use_ocr, config=config)
        
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
            
            chart_type = None
                
            # Check for pie charts using OpenCV
            if OPENCV_AVAILABLE:
                try:
                    pie_chart_info = detect_pie_charts(page, cell_bbox)
                    if pie_chart_info["is_pie_chart"]:
                        chart_type = "pie_chart"
                        chart_data = extract_chart_data(page, cell_bbox, use_ocr=use_ocr, config=config)
                        
                        structured_chart = {
                            "type": "chart",
                            "chart_type": chart_type,
                            "section": None,
                            "sub_section": None,
                            "text": None,
                            "table_data": None,
                            "chart_data": chart_data,
                            "description": f"Detected pie chart (confidence: {pie_chart_info['confidence']}%)",
                            "bbox": cell_bbox
                        }
                        structured_charts.append(structured_chart)
                        continue  # Skip vector analysis for this cell
                except Exception as e:
                    logger.warning(f"Error during pie chart detection: {str(e)}")
            
            # Analyze cell for other chart types using vector graphics
            try:
                graphics_analysis = analyze_vector_graphics_density(page, cell_bbox)
                
                if graphics_analysis["is_likely_chart"]:
                    # Try to extract any text data from the potential chart
                    chart_data = extract_chart_data(page, cell_bbox, use_ocr=use_ocr, config=config)
                    
                    # Determine chart type based on vector analysis
                    if graphics_analysis.get("is_pie_chart", False):
                        chart_type = "pie_chart"
                    elif graphics_analysis["horizontal_lines"] > graphics_analysis["vertical_lines"]:
                        chart_type = "line_chart"
                    elif graphics_analysis["vertical_lines"] > graphics_analysis["horizontal_lines"]:
                        chart_type = "bar_chart"
                    elif graphics_analysis["density"] > 20:
                        chart_type = "scatter_plot"
                    else:
                        chart_type = "generic_chart"
                    
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
            except Exception as e:
                logger.warning(f"Error analyzing cell for charts: {str(e)}")
                continue
    
    # Perform a full-page scan for pie charts if few charts detected
    if len(structured_charts) == 0 and OPENCV_AVAILABLE:
        try:
            # Try to detect pie charts in the whole page
            pie_chart_info = detect_pie_charts(page, [0, 0, page_width, page_height])
            if pie_chart_info["is_pie_chart"]:
                # We found a pie chart, but need to determine its proper boundaries
                # Use the detected circle center and radius to define a more precise bounding box
                center_x, center_y = pie_chart_info.get("center", (page_width/2, page_height/2))
                radius = pie_chart_info.get("radius", page_width/6)
                
                # Create a bounding box around the detected pie chart with some margin
                margin = radius * 0.5
                chart_bbox = [
                    max(0, center_x - radius - margin),
                    max(0, center_y - radius - margin),
                    min(page_width, center_x + radius + margin),
                    min(page_height, center_y + radius + margin)
                ]
                
                # Extract data
                chart_data = extract_chart_data(page, chart_bbox, use_ocr=use_ocr, config=config)
                
                structured_chart = {
                    "type": "chart",
                    "chart_type": "pie_chart",
                    "section": None,
                    "sub_section": None,
                    "text": None,
                    "table_data": None,
                    "chart_data": chart_data,
                    "description": f"Detected pie chart (confidence: {pie_chart_info['confidence']}%)",
                    "bbox": chart_bbox
                }
                structured_charts.append(structured_chart)
        except Exception as e:
            logger.warning(f"Error during full-page pie chart detection: {str(e)}")
    
    return structured_charts
