"""
Test text block extraction.
"""
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import json
from pathlib import Path

# Add the src directory to path for imports
sys.path.append(str(Path(__file__).parents[1]))

from src.parser.utils import clean_text, unhyphenate_text, has_tabular_structure
from src.parser.sections import SectionTracker


class TestTextBlockUtils(unittest.TestCase):
    """Test text block utility functions."""
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        # Test whitespace normalization
        self.assertEqual(clean_text("  Multiple    spaces  "), "Multiple spaces")
        
        # Test trimming
        self.assertEqual(clean_text("\n Text with newlines \n"), "Text with newlines")
        
        # Test soft hyphen removal
        self.assertEqual(clean_text("word\u00ADbreak"), "wordbreak")
        
    def test_unhyphenate_text(self):
        """Test unhyphenation of words split across lines."""
        # Test basic hyphenation
        self.assertEqual(unhyphenate_text("word- break"), "wordbreak")
        
        # Test multiple hyphenations
        self.assertEqual(
            unhyphenate_text("test- ing multi- line hyph- enation"),
            "testing multiline hyphenation"
        )
        
        # Test non-hyphenated text
        self.assertEqual(unhyphenate_text("normal text"), "normal text")
        
    def test_has_tabular_structure(self):
        """Test tabular structure detection."""
        # Test text with tabular structure (aligned columns)
        tabular_text = """
        Name        Age     Location
        John        25      New York
        Sarah       32      Chicago
        """
        self.assertTrue(has_tabular_structure(tabular_text))
        
        # Test regular paragraph text
        paragraph_text = """
        This is a normal paragraph with no tabular structure.
        It continues on multiple lines but doesn't have aligned columns.
        """
        self.assertFalse(has_tabular_structure(paragraph_text))


class TestSectionTracker(unittest.TestCase):
    """Test section tracking functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tracker = SectionTracker()
    
    def test_section_detection(self):
        """Test section heading detection."""
        # Test numeric section
        self.assertTrue(self.tracker.is_heading({
            "text": "1. Introduction",
            "font_size": 14,
            "is_bold": True
        })[0])
        
        # Test roman numeral section
        self.assertTrue(self.tracker.is_heading({
            "text": "IV. Results",
            "font_size": 14,
            "is_bold": True
        })[0])
        
        # Test capitalized section
        self.assertTrue(self.tracker.is_heading({
            "text": "SUMMARY",
            "font_size": 12,
            "is_bold": True
        })[0])
        
        # Test non-heading
        self.assertFalse(self.tracker.is_heading({
            "text": "This is just normal paragraph text.",
            "font_size": 11,
            "is_bold": False
        })[0])
    
    def test_update_section_state(self):
        """Test section state updates."""
        # Update with main section
        self.tracker.update_section_state({
            "text": "1. Main Section",
            "font_size": 14,
            "is_bold": True
        })
        self.assertEqual(self.tracker.current_section, "Main Section")
        self.assertIsNone(self.tracker.current_subsection)
        
        # Update with subsection
        self.tracker.update_section_state({
            "text": "1.1 Sub Section",
            "font_size": 12,
            "is_bold": False
        })
        self.assertEqual(self.tracker.current_section, "Main Section")
        self.assertEqual(self.tracker.current_subsection, "Sub Section")


if __name__ == "__main__":
    unittest.main()
