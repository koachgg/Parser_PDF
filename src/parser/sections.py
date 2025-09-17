"""
Section tracker for maintaining section hierarchy during PDF parsing.
"""
import re
from typing import Dict, List, Optional, Tuple, Any

class SectionTracker:
    """
    Track current section and subsection context while parsing PDF content.
    Uses heuristics to identify headings and maintain section hierarchy.
    """
    
    def __init__(self):
        """Initialize empty section tracker."""
        self.current_section = None
        self.current_subsection = None
        self.section_pattern = re.compile(r'^(?:\d+\.?\s+|[A-Z]+\.?\s+|[IVXLCDM]+\.?\s+)(.+)$')
        self.subsection_pattern = re.compile(r'^(?:\d+\.\d+\.?\s+|[a-z]\.?\s+|\(\w+\)\s+)(.+)$')
        
        # Store font characteristics of detected section headings to improve detection
        self.section_font_sizes = []
        self.section_fonts = set()
        
    def is_heading(self, block: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Determine if a text block is a heading based on formatting and content.
        
        Args:
            block: Text block with text content and font information
            
        Returns:
            (is_heading, heading_level) where heading_level is "section" or "subsection"
        """
        text = block.get('text', '').strip()
        font_size = block.get('font_size', 0)
        is_bold = block.get('is_bold', False)
        
        # Skip empty blocks or very long text (likely not a heading)
        if not text or len(text) > 100:
            return False, ""
            
        # Check for numbering patterns indicating sections
        if self.section_pattern.match(text):
            return True, "section"
            
        # Check for numbering patterns indicating subsections
        if self.subsection_pattern.match(text):
            return True, "subsection"
            
        # Check for font characteristics
        if font_size > 0:
            # If we have seen section fonts before, compare
            if self.section_font_sizes and font_size >= max(self.section_font_sizes):
                return True, "section"
            
            # If bold and relatively large font, likely a heading
            if is_bold and font_size > 12:
                return True, "section"
                
            # If capitalized and bold
            if text.isupper() and is_bold:
                return True, "section"
                
        return False, ""
    
    def update_section_state(self, block: Dict[str, Any]) -> None:
        """
        Update section tracking based on current block.
        
        Args:
            block: Text block with potential heading information
        """
        is_heading, heading_level = self.is_heading(block)
        
        if is_heading:
            text = block.get('text', '').strip()
            font_size = block.get('font_size', 0)
            font_name = block.get('font', '')
            
            if font_size > 0:
                self.section_font_sizes.append(font_size)
            
            if font_name:
                self.section_fonts.add(font_name)
                
            # Clean heading text (remove numbering)
            if heading_level == "section":
                match = self.section_pattern.match(text)
                if match:
                    text = match.group(1).strip()
                else:
                    text = text
                self.current_section = text
                self.current_subsection = None
            else:  # subsection
                match = self.subsection_pattern.match(text)
                if match:
                    text = match.group(1).strip()
                else:
                    text = text
                self.current_subsection = text
    
    def get_current_sections(self) -> Dict[str, Optional[str]]:
        """
        Get current section and subsection.
        
        Returns:
            Dictionary with current section and subsection
        """
        return {
            "section": self.current_section,
            "sub_section": self.current_subsection
        }
