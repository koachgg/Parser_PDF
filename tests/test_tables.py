"""
Test table extraction functionality.
"""
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
from pathlib import Path

# Add the src directory to path for imports
sys.path.append(str(Path(__file__).parents[1]))

from src.parser.tables import TableExtractor


class TestTableExtraction(unittest.TestCase):
    """Test table extraction functionality."""
    
    def test_parse_tabular_text(self):
        """Test parsing tabular text into structured data."""
        extractor = TableExtractor()
        
        # Test basic tabular text
        tabular_text = """
        Name        Age     Location
        John        25      New York
        Sarah       32      Chicago
        """
        
        table_data = extractor._parse_tabular_text(tabular_text)
        
        # Should have 3 rows (header + 2 data rows)
        self.assertEqual(len(table_data), 3)
        
        # Each row should have 3 columns
        for row in table_data:
            self.assertEqual(len(row), 3)
        
        # Check header
        self.assertEqual(table_data[0][0].strip(), "Name")
        self.assertEqual(table_data[0][1].strip(), "Age")
        self.assertEqual(table_data[0][2].strip(), "Location")
        
        # Check data
        self.assertEqual(table_data[1][0].strip(), "John")
        self.assertEqual(table_data[1][1].strip(), "25")
        self.assertEqual(table_data[1][2].strip(), "New York")
    
    def test_row_similarity(self):
        """Test similarity detection between table rows."""
        extractor = TableExtractor()
        
        # Test identical rows
        row1 = ["Name", "Age", "Location"]
        row2 = ["Name", "Age", "Location"]
        self.assertGreater(extractor._row_similarity(row1, row2), 0.9)
        
        # Test similar rows
        row3 = ["Name", "Age", "City"]
        self.assertGreater(extractor._row_similarity(row1, row3), 0.7)
        
        # Test dissimilar rows
        row4 = ["Product", "Price", "Quantity"]
        self.assertLess(extractor._row_similarity(row1, row4), 0.5)
        
        # Test empty row handling
        self.assertEqual(extractor._row_similarity(row1, []), 0.0)
        self.assertEqual(extractor._row_similarity([], []), 0.0)


if __name__ == "__main__":
    unittest.main()
