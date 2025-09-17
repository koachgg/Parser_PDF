"""
Unit tests for visual debugging functionality.
"""
import os
import sys
import unittest
from pathlib import Path
import shutil
import tempfile

# Add parent directory to import path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import fitz  # PyMuPDF
from src.parser.visual_debug import draw_bounding_boxes, save_content_images, visualize_page_content
from src.config import ConfigManager

class TestVisualDebug(unittest.TestCase):
    """Test cases for visual debugging functionality."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test output
        self.test_dir = tempfile.mkdtemp()
        
        # Path to test PDF
        self.test_pdf_path = str(Path(__file__).parent / "data" / "test_document.pdf")
        if not os.path.exists(self.test_pdf_path):
            # If test document doesn't exist, create a simple one
            self.create_test_pdf()

    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)

    def create_test_pdf(self):
        """Create a simple test PDF if none exists."""
        os.makedirs(os.path.dirname(self.test_pdf_path), exist_ok=True)
        
        doc = fitz.open()
        page = doc.new_page()
        
        # Add some text
        page.insert_text((50, 50), "This is a test document")
        page.insert_text((50, 100), "Second paragraph")
        
        # Add a simple rectangle to simulate a table
        page.draw_rect([50, 150, 200, 200], color=(1, 0, 0))
        
        # Save the document
        doc.save(self.test_pdf_path)
        doc.close()

    def test_draw_bounding_boxes(self):
        """Test drawing bounding boxes."""
        # Open test PDF
        doc = fitz.open(self.test_pdf_path)
        page = doc[0]
        
        # Create test blocks
        blocks = [
            {"type": "paragraph", "bbox": [50, 50, 200, 70]},
            {"type": "paragraph", "bbox": [50, 100, 200, 120]},
            {"type": "table", "bbox": [50, 150, 200, 200]}
        ]
        
        # Test with default colors
        img_path = draw_bounding_boxes(page, blocks, self.test_dir)
        self.assertIsNotNone(img_path)
        self.assertTrue(os.path.exists(img_path))
        
        # Test with custom colors
        custom_config = {
            "colors": {
                "paragraph": [255, 0, 0],
                "table": [0, 255, 0]
            }
        }
        img_path = draw_bounding_boxes(page, blocks, self.test_dir, custom_config)
        self.assertIsNotNone(img_path)
        self.assertTrue(os.path.exists(img_path))
        
        doc.close()

    def test_save_content_images(self):
        """Test saving individual content block images."""
        # Open test PDF
        doc = fitz.open(self.test_pdf_path)
        page = doc[0]
        
        # Create test blocks
        blocks = [
            {"type": "paragraph", "bbox": [50, 50, 200, 70]},
            {"type": "table", "bbox": [50, 150, 200, 200]}
        ]
        
        # Save content images
        save_content_images(page, blocks, self.test_dir)
        
        # Check that images were created
        page_dir = os.path.join(self.test_dir, "page_1")
        self.assertTrue(os.path.exists(page_dir))
        self.assertTrue(os.path.exists(os.path.join(page_dir, "paragraph_0.png")))
        self.assertTrue(os.path.exists(os.path.join(page_dir, "table_1.png")))
        
        doc.close()

    def test_visualize_page_content(self):
        """Test full page visualization."""
        # Open test PDF
        doc = fitz.open(self.test_pdf_path)
        page = doc[0]
        
        # Create test blocks
        blocks = [
            {"type": "paragraph", "bbox": [50, 50, 200, 70]},
            {"type": "paragraph", "bbox": [50, 100, 200, 120]},
            {"type": "table", "bbox": [50, 150, 200, 200]}
        ]
        
        # Test with various config settings
        debug_config = {
            "output_path": self.test_dir,
            "draw_bounding_boxes": True,
            "save_images": True
        }
        
        img_path = visualize_page_content(page, blocks, debug_config)
        self.assertIsNotNone(img_path)
        self.assertTrue(os.path.exists(img_path))
        
        # Check individual images
        page_dir = os.path.join(self.test_dir, "page_1")
        self.assertTrue(os.path.exists(page_dir))
        
        # Test with bounding boxes disabled
        debug_config["draw_bounding_boxes"] = False
        img_path = visualize_page_content(page, blocks, debug_config)
        self.assertIsNone(img_path)  # Should be None when bounding boxes disabled
        
        doc.close()

    def test_integration_with_config_manager(self):
        """Test integration with ConfigManager."""
        # Create ConfigManager with custom settings
        config = ConfigManager()
        config.set("visual_debug", "enabled", True)
        config.set("visual_debug", "output_path", self.test_dir)
        config.set("visual_debug", "draw_bounding_boxes", True)
        config.set("visual_debug", "save_images", True)
        config.set("visual_debug", "colors", {
            "paragraph": [0, 0, 255],
            "table": [255, 0, 0]
        })
        
        # Get debug config
        debug_config = config.get_visual_debug_config()
        self.assertEqual(debug_config["output_path"], self.test_dir)
        
        # Open test PDF
        doc = fitz.open(self.test_pdf_path)
        page = doc[0]
        
        # Create test blocks
        blocks = [
            {"type": "paragraph", "bbox": [50, 50, 200, 70]},
            {"type": "table", "bbox": [50, 150, 200, 200]}
        ]
        
        # Test visualization with config
        img_path = visualize_page_content(page, blocks, debug_config)
        self.assertIsNotNone(img_path)
        self.assertTrue(os.path.exists(img_path))
        
        doc.close()

if __name__ == "__main__":
    unittest.main()
