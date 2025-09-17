"""
Utility functions for PDF parsing.
"""
import re
from typing import List, Dict, Any, Tuple


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing extra whitespace and normalizing line breaks.
    
    Args:
        text: Raw text extracted from PDF
        
    Returns:
        Cleaned text string
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Trim whitespace
    text = text.strip()
    # Remove soft hyphens
    text = text.replace('\u00AD', '')
    return text


def unhyphenate_text(text: str) -> str:
    """
    Join hyphenated words at end of lines.
    
    Args:
        text: Text that may contain hyphenated words
        
    Returns:
        Text with hyphenated words joined
    """
    # Match pattern where a word ends with hyphen followed by whitespace
    return re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)


def sort_blocks_by_position(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort blocks by position (top-to-bottom, left-to-right).
    
    Args:
        blocks: List of text blocks with bbox coordinates
        
    Returns:
        Sorted blocks list
    """
    # Sort by y0 (top position) first, then by x0 (left position)
    return sorted(blocks, key=lambda b: (b['bbox'][1], b['bbox'][0]))


def has_tabular_structure(text: str) -> bool:
    """
    Check if text has tabular structure based on whitespace patterns.
    
    Args:
        text: Text content to analyze
        
    Returns:
        True if tabular structure is detected
    """
    lines = text.splitlines()
    if len(lines) < 3:
        return False
    
    # Count consistent spacing patterns
    spacing_patterns = 0
    for i in range(1, len(lines)):
        if re.search(r'\s{3,}', lines[i]) and re.search(r'\s{3,}', lines[i-1]):
            # Same spacing pattern in consecutive lines
            spacing_patterns += 1
    
    # If more than half the lines have consistent spacing patterns, likely tabular
    return spacing_patterns > len(lines) / 3


def is_ocr_needed(page, min_text_length: int = 50) -> bool:
    """
    Determine if OCR is needed for a page.
    
    Args:
        page: PDF page object
        min_text_length: Minimum text length threshold
        
    Returns:
        True if OCR is recommended
    """
    text = page.get_text()
    # If page has images and little text, OCR might be needed
    return len(text.strip()) < min_text_length and len(page.get_images()) > 0


def normalize_bbox(bbox: List[float]) -> List[float]:
    """
    Normalize bounding box coordinates.
    
    Args:
        bbox: Bounding box [x0, y0, x1, y1]
        
    Returns:
        Normalized bounding box
    """
    return [float(coord) for coord in bbox]


def merge_overlapping_bboxes(bbox1: List[float], bbox2: List[float]) -> List[float]:
    """
    Merge two overlapping bounding boxes.
    
    Args:
        bbox1: First bounding box [x0, y0, x1, y1]
        bbox2: Second bounding box [x0, y0, x1, y1]
        
    Returns:
        Merged bounding box
    """
    return [
        min(bbox1[0], bbox2[0]),  # x0
        min(bbox1[1], bbox2[1]),  # y0
        max(bbox1[2], bbox2[2]),  # x1
        max(bbox1[3], bbox2[3])   # y1
    ]
