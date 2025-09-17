"""
Text block extraction and paragraph assembly from PDF documents.
"""
import fitz  # PyMuPDF
from typing import List, Dict, Any, Tuple, Optional
import re
import logging

from .utils import clean_text, unhyphenate_text, sort_blocks_by_position, normalize_bbox
from .sections import SectionTracker

logger = logging.getLogger(__name__)

def extract_font_info(page: fitz.Page, span: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract font information from text span.
    
    Args:
        page: PDF page object
        span: Text span dictionary
        
    Returns:
        Dict with font name, size, and flags information
    """
    try:
        font_info = {
            'font': span['font'],
            'font_size': span['size'],
            'is_bold': span.get('flags', 0) & 2 > 0,  # Check if bold flag is set
            'is_italic': span.get('flags', 0) & 1 > 0  # Check if italic flag is set
        }
        return font_info
    except (KeyError, IndexError):
        # If font extraction fails, return defaults
        return {
            'font': None,
            'font_size': 0,
            'is_bold': False,
            'is_italic': False
        }

def get_enhanced_blocks(page: fitz.Page) -> List[Dict[str, Any]]:
    """
    Get text blocks with enhanced metadata from a PDF page.
    
    Args:
        page: PDF page object
        
    Returns:
        List of text blocks with additional metadata
    """
    blocks = []
    
    # Get text blocks with detailed info including font and style
    page_dict = page.get_text("dict")
    
    for block in page_dict.get("blocks", []):
        if block.get("type") == 0:  # Text block
            for line in block.get("lines", []):
                line_text = ""
                line_spans = []
                
                # Process each span (text fragment with uniform styling)
                for span in line.get("spans", []):
                    span_text = span.get("text", "")
                    if not span_text.strip():
                        continue
                        
                    font_info = extract_font_info(page, span)
                    span_info = {
                        "text": span_text,
                        "bbox": normalize_bbox(span["bbox"]),
                        **font_info
                    }
                    line_text += span_text
                    line_spans.append(span_info)
                
                if line_text.strip():
                    # Combine spans into a single line entry
                    combined_bbox = [
                        min(s["bbox"][0] for s in line_spans),
                        min(s["bbox"][1] for s in line_spans),
                        max(s["bbox"][2] for s in line_spans),
                        max(s["bbox"][3] for s in line_spans)
                    ]
                    
                    # Determine dominant font characteristics
                    # (use the characteristics of the longest span)
                    dominant_span = max(line_spans, key=lambda s: len(s["text"]))
                    
                    blocks.append({
                        "text": line_text,
                        "bbox": normalize_bbox(combined_bbox),
                        "font": dominant_span["font"],
                        "font_size": dominant_span["font_size"],
                        "is_bold": dominant_span["is_bold"],
                        "is_italic": dominant_span["is_italic"],
                        "spans": line_spans
                    })
    
    return sort_blocks_by_position(blocks)

def should_merge_blocks(block1: Dict[str, Any], block2: Dict[str, Any]) -> bool:
    """
    Determine if two blocks should be merged into a paragraph.
    
    Args:
        block1: First text block
        block2: Second text block (follows block1)
        
    Returns:
        True if blocks should be merged
    """
    # Don't merge if font properties are significantly different
    if block1.get("font") != block2.get("font"):
        return False
        
    # Don't merge if font size differs by more than 1 point
    if abs(block1.get("font_size", 0) - block2.get("font_size", 0)) > 1:
        return False
    
    # Check vertical distance (should be close but not overlapping)
    y_gap = block2["bbox"][1] - block1["bbox"][3]
    if y_gap < 0:  # Overlapping vertically
        return False
        
    # Too far apart vertically (more than 1.5 times the font size)
    if y_gap > 1.5 * block1.get("font_size", 10):
        return False
    
    # If first block ends with a period, question mark, etc.
    if re.search(r'[.!?:]$', block1["text"].strip()):
        # It's likely the end of a sentence, but could still be in same paragraph
        # Check for capital letter at start of next block as additional signal
        if re.match(r'^[A-Z]', block2["text"].strip()):
            # Likely start of new sentence in new paragraph
            return False
    
    # Default to merge
    return True

def assemble_paragraphs(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Assemble text blocks into paragraphs.
    
    Args:
        blocks: List of text blocks
        
    Returns:
        List of paragraph blocks
    """
    if not blocks:
        return []
        
    paragraphs = []
    current_paragraph = blocks[0].copy()
    
    for i in range(1, len(blocks)):
        if should_merge_blocks(current_paragraph, blocks[i]):
            # Merge blocks
            current_text = current_paragraph["text"]
            next_text = blocks[i]["text"]
            
            # Handle hyphenation
            if current_text.strip().endswith("-"):
                merged_text = current_text.strip()[:-1] + next_text
            else:
                merged_text = current_text + " " + next_text
                
            current_paragraph["text"] = merged_text
            
            # Expand the bounding box
            current_paragraph["bbox"] = [
                min(current_paragraph["bbox"][0], blocks[i]["bbox"][0]),
                min(current_paragraph["bbox"][1], blocks[i]["bbox"][1]),
                max(current_paragraph["bbox"][2], blocks[i]["bbox"][2]),
                max(current_paragraph["bbox"][3], blocks[i]["bbox"][3])
            ]
        else:
            # Start a new paragraph
            paragraphs.append(current_paragraph)
            current_paragraph = blocks[i].copy()
    
    # Add the last paragraph
    paragraphs.append(current_paragraph)
    
    # Clean paragraph text
    for para in paragraphs:
        para["text"] = clean_text(unhyphenate_text(para["text"]))
        
    return paragraphs

def extract_text_blocks(page: fitz.Page, section_tracker: SectionTracker) -> List[Dict[str, Any]]:
    """
    Extract structured text blocks from a PDF page.
    
    Args:
        page: PDF page object
        section_tracker: SectionTracker object to track section context
        
    Returns:
        List of structured text blocks with section information
    """
    # Extract raw blocks with font information
    raw_blocks = get_enhanced_blocks(page)
    
    # Process blocks for section information
    for block in raw_blocks:
        section_tracker.update_section_state(block)
    
    # Assemble paragraphs from raw blocks
    paragraphs = assemble_paragraphs(raw_blocks)
    
    # Add section information to each paragraph
    structured_blocks = []
    for para in paragraphs:
        # Check if the paragraph itself is a heading
        is_heading, _ = section_tracker.is_heading(para)
        
        # Skip blocks that are pure headings since they'll be captured as section/subsection
        if is_heading:
            continue
            
        sections = section_tracker.get_current_sections()
        
        structured_block = {
            "type": "paragraph",
            "section": sections["section"],
            "sub_section": sections["sub_section"],
            "text": para["text"],
            "bbox": para["bbox"],
            "table_data": None,
            "chart_data": None,
            "description": None
        }
        structured_blocks.append(structured_block)
    
    return structured_blocks
