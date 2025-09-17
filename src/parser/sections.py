"""
Section tracker for maintaining section hierarchy during PDF parsing.
"""
import re
from typing import Dict, List, Optional, Tuple, Any

class SectionTracker:
    """
    Track current section and subsection context while parsing PDF content.
    Uses machine learning-inspired adaptive techniques to identify headings and maintain section hierarchy.
    """
    
    def __init__(self):
        """Initialize empty section tracker with learning capabilities."""
        self.current_section = None
        self.current_subsection = None
        
        # Regular expression patterns for common heading formats
        self.section_pattern = re.compile(r'^(?:\d+\.?\s+|[A-Z]+\.?\s+|[IVXLCDM]+\.?\s+)(.+)$')
        self.subsection_pattern = re.compile(r'^(?:\d+\.\d+\.?\s+|[a-z]\.?\s+|\(\w+\)\s+)(.+)$')
        
        # Learning structures for font characteristics
        self.section_font_sizes = []
        self.subsection_font_sizes = []
        self.body_font_sizes = []
        self.section_fonts = set()
        self.subsection_fonts = set()
        
        # Track all font sizes to establish document baseline
        self.all_font_sizes = []
        self.font_size_counts = {}  # For detecting the most common (body) font size
        
        # Track sequential context for better heading detection
        self.prev_block_font_size = None
        self.prev_block_is_heading = False
        self.heading_confidence_threshold = 0.6  # Adaptive threshold
        
        # Document-level statistics
        self.has_analyzed_document = False
        self.section_count = 0
        self.subsection_count = 0
        
        # Section numbering style detection
        self.detected_section_style = None  # Will be 'numeric', 'roman', 'alpha', or None
        
    def analyze_document_structure(self, blocks: List[Dict[str, Any]]) -> None:
        """
        Analyze document structure to learn heading styles and patterns.
        Call this method once with all blocks from the document before processing.
        
        Args:
            blocks: List of all text blocks from the document
        """
        if not blocks:
            return
            
        # Collect font size information
        for block in blocks:
            font_size = block.get('font_size', 0)
            if font_size > 0:
                self.all_font_sizes.append(font_size)
                
                # Count occurrences for finding the most common (body) font size
                if font_size not in self.font_size_counts:
                    self.font_size_counts[font_size] = 0
                self.font_size_counts[font_size] += 1
        
        # Determine body text font size (most common)
        if self.font_size_counts:
            self.body_font_size = max(self.font_size_counts, key=self.font_size_counts.get)
        else:
            self.body_font_size = 11  # Default assumption
            
        # First pass: detect section numbering style
        section_numbering = {
            'numeric': 0,    # e.g., "1. Introduction"
            'roman': 0,      # e.g., "I. Introduction" 
            'alpha': 0       # e.g., "A. Introduction"
        }
        
        for block in blocks:
            text = block.get('text', '').strip()
            if re.match(r'^\d+\.?\s+\w+', text):
                section_numbering['numeric'] += 1
            elif re.match(r'^[IVXLCDM]+\.?\s+\w+', text):
                section_numbering['roman'] += 1
            elif re.match(r'^[A-Z]\.?\s+\w+', text):
                section_numbering['alpha'] += 1
                
        # Determine predominant numbering style if any
        if section_numbering:
            max_style = max(section_numbering, key=section_numbering.get)
            if section_numbering[max_style] >= 2:  # Need at least two occurrences
                self.detected_section_style = max_style
                
        # Second pass: identify potential headings based on font size
        if self.all_font_sizes:
            # Sort font sizes from largest to smallest
            sorted_sizes = sorted(set(self.all_font_sizes), reverse=True)
            
            # Assume largest font sizes are headings
            if len(sorted_sizes) >= 3:
                # Title/section/body hierarchy
                self.section_font_sizes = [size for size in sorted_sizes 
                                          if size > self.body_font_size * 1.2]
                self.subsection_font_sizes = [size for size in sorted_sizes 
                                             if size > self.body_font_size 
                                             and size < max(self.section_font_sizes or [float('inf')])]
            elif len(sorted_sizes) >= 2:
                # Section/body hierarchy
                self.section_font_sizes = [sorted_sizes[0]]
                
        self.has_analyzed_document = True
    
    def calculate_heading_confidence(self, block: Dict[str, Any], level: str) -> float:
        """
        Calculate confidence score for a block being a heading of specified level.
        
        Args:
            block: Text block to evaluate
            level: "section" or "subsection"
            
        Returns:
            Confidence score from 0.0 to 1.0
        """
        text = block.get('text', '').strip()
        font_size = block.get('font_size', 0)
        is_bold = block.get('is_bold', False)
        is_italic = block.get('is_italic', False)
        
        score = 0.0
        
        # Text length factor (headings are usually short)
        text_length = len(text)
        if text_length <= 10:
            score += 0.15
        elif text_length <= 50:
            score += 0.1
        elif text_length > 100:
            score -= 0.2  # Penalty for very long text
            
        # Numbering patterns
        if level == "section":
            if self.section_pattern.match(text):
                score += 0.3
            if self.detected_section_style == 'numeric' and re.match(r'^\d+\.?\s+\w+', text):
                score += 0.15
            elif self.detected_section_style == 'roman' and re.match(r'^[IVXLCDM]+\.?\s+\w+', text):
                score += 0.15
            elif self.detected_section_style == 'alpha' and re.match(r'^[A-Z]\.?\s+\w+', text):
                score += 0.15
        elif level == "subsection":
            if self.subsection_pattern.match(text):
                score += 0.3
        
        # Font characteristics
        if is_bold:
            score += 0.2
            
        if text.isupper():
            score += 0.15
            
        # Font size compared to learned document structure
        if font_size > 0:
            if level == "section" and self.section_font_sizes:
                if font_size in self.section_font_sizes:
                    score += 0.25
                elif font_size > self.body_font_size * 1.2:
                    score += 0.15
            elif level == "subsection" and self.subsection_font_sizes:
                if font_size in self.subsection_font_sizes:
                    score += 0.25
                
        # Context from previous blocks
        if self.prev_block_font_size and font_size > self.prev_block_font_size:
            score += 0.1  # Larger font than previous block suggests heading
            
        # Adjust for section vs subsection
        if level == "section" and is_bold and not is_italic:
            score += 0.05  # Sections are more likely to be bold but not italic
        elif level == "subsection" and is_italic:
            score += 0.05  # Subsections might be italic
            
        return min(max(score, 0.0), 1.0)  # Ensure score is between 0 and 1
    
    def is_heading(self, block: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Determine if a text block is a heading based on formatting and content.
        Uses adaptive learning from document patterns for better accuracy.
        
        Args:
            block: Text block with text content and font information
            
        Returns:
            (is_heading, heading_level) where heading_level is "section" or "subsection"
        """
        text = block.get('text', '').strip()
        font_size = block.get('font_size', 0)
        
        # Skip empty blocks
        if not text:
            return False, ""
            
        # Calculate confidence scores for section and subsection
        section_confidence = self.calculate_heading_confidence(block, "section")
        subsection_confidence = self.calculate_heading_confidence(block, "subsection")
        
        # Store context for next block
        self.prev_block_font_size = font_size
        
        # Determine if it's a heading and what type
        if section_confidence > self.heading_confidence_threshold:
            self.prev_block_is_heading = True
            return True, "section"
        elif subsection_confidence > self.heading_confidence_threshold:
            self.prev_block_is_heading = True
            return True, "subsection"
                
        self.prev_block_is_heading = False
        return False, ""
    
    def update_section_state(self, block: Dict[str, Any]) -> None:
        """
        Update section tracking based on current block.
        Uses enhanced pattern detection and learning algorithms.
        
        Args:
            block: Text block with potential heading information
        """
        is_heading, heading_level = self.is_heading(block)
        
        if is_heading:
            text = block.get('text', '').strip()
            font_size = block.get('font_size', 0)
            font_name = block.get('font', '')
            
            # Update our font learning data
            if font_size > 0:
                if heading_level == "section":
                    self.section_font_sizes.append(font_size)
                else:  # subsection
                    self.subsection_font_sizes.append(font_size)
            
            if font_name:
                if heading_level == "section":
                    self.section_fonts.add(font_name)
                else:  # subsection
                    self.subsection_fonts.add(font_name)
            
            # Track section counting to refine our pattern detection
            if heading_level == "section":
                self.section_count += 1
            else:
                self.subsection_count += 1
                
            # Clean heading text (remove numbering) using multi-pattern approach
            cleaned_text = text
            
            # Try multiple patterns for different numbering styles
            if heading_level == "section":
                # Numeric: "1. Title"
                numeric_match = re.match(r'^\d+\.?\s+(.*?)$', text)
                # Roman: "IV. Title"
                roman_match = re.match(r'^[IVXLCDM]+\.?\s+(.*?)$', text)
                # Alphabetic: "A. Title"
                alpha_match = re.match(r'^[A-Z]\.?\s+(.*?)$', text)
                # Other pattern: "Section 1: Title"
                section_match = re.match(r'^(section|chapter)\s+\w+[:.]\s+(.*?)$', text, re.IGNORECASE)
                
                if numeric_match:
                    cleaned_text = numeric_match.group(1).strip()
                elif roman_match:
                    cleaned_text = roman_match.group(1).strip()
                elif alpha_match:
                    cleaned_text = alpha_match.group(1).strip()
                elif section_match:
                    cleaned_text = section_match.group(2).strip()
                
                self.current_section = cleaned_text
                self.current_subsection = None
                
                # Update section style detection based on what matched
                if numeric_match:
                    self.detected_section_style = 'numeric'
                elif roman_match:
                    self.detected_section_style = 'roman'
                elif alpha_match:
                    self.detected_section_style = 'alpha'
                
            else:  # subsection
                # Multiple subsection patterns
                # Decimal: "1.2 Title"
                decimal_match = re.match(r'^\d+\.\d+\.?\s+(.*?)$', text)
                # Alphabetic: "(a) Title" or "a. Title"
                alpha_match = re.match(r'^(\([a-z]\)|\[a-z\]|[a-z]\.)\s+(.*?)$', text)
                # Bullet point: "• Title" or "- Title"
                bullet_match = re.match(r'^[•\-\*]\s+(.*?)$', text)
                
                if decimal_match:
                    cleaned_text = decimal_match.group(1).strip()
                elif alpha_match:
                    cleaned_text = alpha_match.group(2).strip()
                elif bullet_match:
                    cleaned_text = bullet_match.group(1).strip()
                
                self.current_subsection = cleaned_text
    
    def get_current_sections(self) -> Dict[str, Optional[str]]:
        """
        Get current section and subsection with confidence information.
        
        Returns:
            Dictionary with current section and subsection
        """
        return {
            "section": self.current_section,
            "sub_section": self.current_subsection,
            "section_count": self.section_count,
            "subsection_count": self.subsection_count
        }
        
    def preprocess_document(self, blocks: List[Dict[str, Any]]) -> None:
        """
        Preprocess document to learn structure before individual block processing.
        Call this method once at the beginning of document processing.
        
        Args:
            blocks: List of all text blocks from the document
        """
        self.analyze_document_structure(blocks)
        
        # Adjust threshold based on document characteristics
        if self.section_count > 10:
            # Documents with many sections may need a higher threshold
            self.heading_confidence_threshold = 0.7
        elif not self.all_font_sizes:
            # If we couldn't learn font sizes, be more conservative
            self.heading_confidence_threshold = 0.75
