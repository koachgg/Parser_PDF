"""
Content overlap handling for PDF parser.
"""
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        bbox1: First bounding box [x0, y0, x1, y1]
        bbox2: Second bounding box [x0, y0, x1, y1]
        
    Returns:
        IoU value between 0 and 1
    """
    # Ensure boxes are in the correct format
    if len(bbox1) != 4 or len(bbox2) != 4:
        return 0.0
    
    try:
        # Determine the coordinates of the intersection rectangle
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])
        
        # Check if there is no intersection
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        # Compute the area of intersection rectangle
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Compute the area of both bounding boxes
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        # Compute the intersection over union
        iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
        return iou
    except Exception as e:
        logger.warning(f"IoU calculation failed: {str(e)}")
        return 0.0

def is_contained(bbox1: List[float], bbox2: List[float], threshold: float = 0.9) -> bool:
    """
    Check if bbox1 is largely contained within bbox2.
    
    Args:
        bbox1: First bounding box [x0, y0, x1, y1]
        bbox2: Second bounding box [x0, y0, x1, y1]
        threshold: Minimum containment ratio (0-1)
        
    Returns:
        True if bbox1 is largely contained within bbox2
    """
    # Ensure boxes are in the correct format
    if len(bbox1) != 4 or len(bbox2) != 4:
        return False
    
    try:
        # Determine the coordinates of the intersection rectangle
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])
        
        # Check if there is no intersection
        if x_right < x_left or y_bottom < y_top:
            return False
            
        # Compute the area of intersection rectangle
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Compute the area of the first bounding box
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        
        # Check if bbox1 is largely contained within bbox2
        containment_ratio = intersection_area / bbox1_area if bbox1_area > 0 else 0
        return containment_ratio >= threshold
    except Exception as e:
        logger.warning(f"Containment check failed: {str(e)}")
        return False

def is_table_caption(block: Dict[str, Any], table_block: Dict[str, Any]) -> bool:
    """
    Determine if a text block is likely a caption for a table.
    
    Args:
        block: Text block to check
        table_block: Table block to check against
        
    Returns:
        True if the text block appears to be a caption for the table
    """
    # If the block isn't a paragraph, it's not a caption
    if block.get("type", "") != "paragraph":
        return False
    
    text = block.get("text", "").lower()
    if not text:
        return False
    
    # Check for caption indicators
    caption_indicators = ["table", "tbl", "tab.", "figure", "fig", "chart"]
    has_indicator = any(indicator in text.lower() for indicator in caption_indicators)
    
    # Check position relative to table
    bbox = block.get("bbox", [0, 0, 0, 0])
    table_bbox = table_block.get("bbox", [0, 0, 0, 0])
    
    # Caption is typically above or below the table, with similar horizontal positioning
    horizontal_overlap = min(bbox[2], table_bbox[2]) - max(bbox[0], table_bbox[0])
    if horizontal_overlap <= 0:
        return False
    
    horizontal_overlap_ratio = horizontal_overlap / min(bbox[2] - bbox[0], table_bbox[2] - table_bbox[0])
    
    # Above table
    is_above = bbox[3] <= table_bbox[1] + 5 and bbox[3] >= table_bbox[1] - 50
    
    # Below table
    is_below = bbox[1] >= table_bbox[3] - 5 and bbox[1] <= table_bbox[3] + 50
    
    return has_indicator and (is_above or is_below) and horizontal_overlap_ratio > 0.5

def remove_content_overlap(blocks: List[Dict[str, Any]], config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Remove overlap between different content types.
    
    Prioritizes tables and charts over text blocks that overlap with them.
    Removes text blocks that are entirely contained within tables or charts.
    
    Args:
        blocks: List of content blocks
        config: Configuration dictionary with overlap settings
        
    Returns:
        List of content blocks with overlaps removed
    """
    if not blocks:
        return []
    
    # Default configuration
    default_config = {
        "enabled": True,
        "containment_threshold": 0.7,
        "overlap_threshold": 0.3,
        "remove_headers_in_tables": True,
        "remove_footers_in_tables": True,
        "preserve_table_captions": True,
    }
    
    # Use provided config or default
    if config is None:
        config = default_config
    
    # If overlap removal is disabled, return all blocks
    if not config.get("enabled", True):
        return blocks
    
    # Get thresholds from config
    containment_threshold = config.get("containment_threshold", 0.7)
    overlap_threshold = config.get("overlap_threshold", 0.3)
    preserve_captions = config.get("preserve_table_captions", True)
    remove_headers = config.get("remove_headers_in_tables", True)
    remove_footers = config.get("remove_footers_in_tables", True)
    
    # Sort blocks by type priority: tables and charts take precedence over paragraphs
    type_priority = {"table": 1, "chart": 2, "paragraph": 3, "header": 4, "footer": 5}
    sorted_blocks = sorted(blocks, key=lambda b: type_priority.get(b.get("type", ""), 99))
    
    # Check for overlaps
    filtered_blocks = []
    excluded_indices = set()
    overlap_count = 0
    
    for i, block in enumerate(sorted_blocks):
        if i in excluded_indices:
            continue
            
        block_bbox = block.get("bbox", [0, 0, 0, 0])
        block_type = block.get("type", "")
        
        # If this is a table or chart, check for text blocks that overlap significantly
        if block_type in ["table", "chart"]:
            for j, other_block in enumerate(sorted_blocks):
                if j == i or j in excluded_indices:
                    continue
                    
                other_type = other_block.get("type", "")
                other_bbox = other_block.get("bbox", [0, 0, 0, 0])
                
                # Check if it's a table caption we want to preserve
                is_caption = False
                if preserve_captions and block_type == "table" and other_type == "paragraph":
                    is_caption = is_table_caption(other_block, block)
                
                # Only check for overlap with paragraphs, headers, and footers
                if other_type in ["paragraph", "header", "footer"]:
                    # Skip headers if removal is disabled
                    if other_type == "header" and not remove_headers:
                        continue
                    
                    # Skip footers if removal is disabled
                    if other_type == "footer" and not remove_footers:
                        continue
                    
                    # Skip captions if we want to preserve them
                    if is_caption:
                        logger.debug(f"Preserving table caption: {other_block.get('text', '')[:30]}...")
                        continue
                    
                    # If paragraph is mostly contained within table/chart, exclude it
                    if is_contained(other_bbox, block_bbox, threshold=containment_threshold):
                        excluded_indices.add(j)
                        overlap_count += 1
                        logger.debug(f"Removed contained {other_type} from {block_type}")
                    # If there's significant overlap but not containment, keep both
                    # but note the overlap in logs
                    elif calculate_iou(block_bbox, other_bbox) > overlap_threshold:
                        logger.debug(f"Partial overlap between {block_type} and {other_type}")
        
        filtered_blocks.append(block)
    
    logger.info(f"Removed {overlap_count} overlapping content blocks")
    return filtered_blocks
