"""
Test JSON schema validation.
"""
import unittest
import sys
import os
import json
from pathlib import Path
import jsonschema

# Add the src directory to path for imports
sys.path.append(str(Path(__file__).parents[1]))

from src.exporters.json_writer import JSONWriter


class TestJSONSchema(unittest.TestCase):
    """Test JSON schema validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        schema_path = str(Path(__file__).parents[1] / "schemas" / "output_schema.json")
        self.writer = JSONWriter(schema_path)
        
        # Create a minimal valid JSON structure
        self.valid_json = {
            "pages": [
                {
                    "page_number": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "section": "Introduction",
                            "sub_section": None,
                            "text": "This is a sample paragraph.",
                            "table_data": None,
                            "chart_data": None,
                            "description": None,
                            "bbox": [0, 0, 100, 20]
                        }
                    ]
                }
            ],
            "meta": {
                "source_file": "test.pdf",
                "parser_versions": {
                    "text": "1.0",
                    "tables": "1.0",
                    "charts": "1.0"
                },
                "created_at": "2023-05-01T12:00:00"
            }
        }
    
    def test_valid_json(self):
        """Test validation of valid JSON."""
        self.assertTrue(self.writer.validate_json(self.valid_json))
    
    def test_missing_required_field(self):
        """Test validation fails with missing required field."""
        # Remove required field
        invalid_json = self.valid_json.copy()
        del invalid_json["pages"]
        
        self.assertFalse(self.writer.validate_json(invalid_json))
    
    def test_invalid_type(self):
        """Test validation fails with invalid type."""
        # Change paragraph type to invalid value
        invalid_json = self.valid_json.copy()
        invalid_json["pages"][0]["content"][0]["type"] = "invalid_type"
        
        self.assertFalse(self.writer.validate_json(invalid_json))
    
    def test_paragraph_missing_text(self):
        """Test validation fails with paragraph missing text."""
        # Remove required text for paragraph
        invalid_json = self.valid_json.copy()
        invalid_json["pages"][0]["content"][0]["text"] = None
        
        self.assertFalse(self.writer.validate_json(invalid_json))
    
    def test_table_structure(self):
        """Test valid table structure validation."""
        # Add a valid table
        valid_table = {
            "type": "table",
            "section": "Results",
            "sub_section": None,
            "text": None,
            "table_data": [["Header1", "Header2"], ["Data1", "Data2"]],
            "chart_data": None,
            "description": None,
            "bbox": [0, 0, 100, 100]
        }
        
        self.valid_json["pages"][0]["content"].append(valid_table)
        self.assertTrue(self.writer.validate_json(self.valid_json))


if __name__ == "__main__":
    unittest.main()
