"""
Unit tests for content overlap handling.
"""
import os
import sys
import unittest
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to import path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.parser.content_overlap import calculate_iou, is_contained, is_table_caption, remove_content_overlap

class TestContentOverlap(unittest.TestCase):
    """Test cases for content overlap functionality."""

    def test_calculate_iou(self):
        """Test IoU calculation."""
        # No overlap
        bbox1 = [0, 0, 10, 10]
        bbox2 = [20, 20, 30, 30]
        self.assertEqual(calculate_iou(bbox1, bbox2), 0.0)
        
        # Complete overlap (same box)
        bbox1 = [0, 0, 10, 10]
        bbox2 = [0, 0, 10, 10]
        self.assertEqual(calculate_iou(bbox1, bbox2), 1.0)
        
        # Partial overlap
        bbox1 = [0, 0, 10, 10]
        bbox2 = [5, 5, 15, 15]
        # Intersection: [5, 5, 10, 10] = 25
        # Union: 100 + 100 - 25 = 175
        # IoU = 25/175 = 0.1429
        self.assertAlmostEqual(calculate_iou(bbox1, bbox2), 25/175, places=4)
        
        # Invalid boxes
        bbox1 = [0, 0, 10]  # Invalid box (not enough coords)
        bbox2 = [0, 0, 10, 10]
        self.assertEqual(calculate_iou(bbox1, bbox2), 0.0)

    def test_is_contained(self):
        """Test containment check."""
        # Complete containment
        bbox1 = [5, 5, 15, 15]  # Inner box
        bbox2 = [0, 0, 20, 20]  # Outer box
        self.assertTrue(is_contained(bbox1, bbox2, threshold=0.9))
        
        # Partial containment
        bbox1 = [0, 0, 15, 15]
        bbox2 = [5, 5, 20, 20]
        # Intersection: [5, 5, 15, 15] = 100
        # bbox1 area: 225
        # Ratio = 100/225 = 0.444
        self.assertFalse(is_contained(bbox1, bbox2, threshold=0.5))
        self.assertTrue(is_contained(bbox1, bbox2, threshold=0.4))
        
        # No containment
        bbox1 = [0, 0, 10, 10]
        bbox2 = [20, 20, 30, 30]
        self.assertFalse(is_contained(bbox1, bbox2, threshold=0.1))
        
        # Invalid boxes
        bbox1 = [0, 0]  # Invalid box
        bbox2 = [0, 0, 10, 10]
        self.assertFalse(is_contained(bbox1, bbox2))
    
    def test_is_table_caption(self):
        """Test table caption detection."""
        # Caption above table
        table_block = {"type": "table", "bbox": [100, 200, 500, 300]}
        
        # Good caption above
        caption_block = {
            "type": "paragraph",
            "bbox": [150, 180, 450, 195],
            "text": "Table 1: Sample data for analysis"
        }
        self.assertTrue(is_table_caption(caption_block, table_block))
        
        # Good caption below
        caption_block = {
            "type": "paragraph",
            "bbox": [150, 305, 450, 320],
            "text": "Table 1: Sample data for analysis"
        }
        self.assertTrue(is_table_caption(caption_block, table_block))
        
        # Not a caption (wrong type)
        caption_block = {
            "type": "header",
            "bbox": [150, 180, 450, 195],
            "text": "Table 1: Sample data for analysis"
        }
        self.assertFalse(is_table_caption(caption_block, table_block))
        
        # Not a caption (wrong text)
        caption_block = {
            "type": "paragraph",
            "bbox": [150, 180, 450, 195],
            "text": "This is just some text"
        }
        self.assertFalse(is_table_caption(caption_block, table_block))
        
        # Not a caption (wrong position)
        caption_block = {
            "type": "paragraph",
            "bbox": [150, 100, 450, 120],  # Too far above
            "text": "Table 1: Sample data for analysis"
        }
        self.assertFalse(is_table_caption(caption_block, table_block))
        
        # Not a caption (no horizontal overlap)
        caption_block = {
            "type": "paragraph",
            "bbox": [600, 180, 700, 195],  # No horizontal overlap
            "text": "Table 1: Sample data for analysis"
        }
        self.assertFalse(is_table_caption(caption_block, table_block))

    def test_remove_content_overlap(self):
        """Test content overlap removal."""
        # Create test blocks
        blocks = [
            {
                "type": "table",
                "bbox": [100, 100, 500, 300],
                "section": "Results"
            },
            {
                "type": "paragraph",
                "bbox": [150, 150, 450, 250],  # Contained within table
                "text": "This text is inside the table",
                "section": "Results"
            },
            {
                "type": "paragraph",
                "bbox": [150, 75, 450, 95],  # Caption above table
                "text": "Table 1: Sample data",
                "section": "Results"
            },
            {
                "type": "paragraph",
                "bbox": [150, 310, 450, 330],  # Caption below table
                "text": "Note: Data shown in Table 1",
                "section": "Results"
            },
            {
                "type": "header",
                "bbox": [0, 50, 600, 70],
                "text": "Header text",
                "section": "Header"
            },
            {
                "type": "footer",
                "bbox": [0, 500, 600, 520],
                "text": "Footer text",
                "section": "Footer"
            },
            {
                "type": "chart",
                "bbox": [100, 350, 500, 450],
                "section": "Results"
            },
            {
                "type": "paragraph",
                "bbox": [150, 370, 450, 400],  # Inside chart
                "text": "This text is inside the chart",
                "section": "Results"
            }
        ]
        
        # Default config - should remove overlapping text
        filtered_blocks = remove_content_overlap(blocks)
        self.assertEqual(len(filtered_blocks), 6)  # Removed 2 blocks
        
        # Test with preserve captions enabled
        config = {
            "enabled": True,
            "containment_threshold": 0.7,
            "preserve_table_captions": True
        }
        filtered_blocks = remove_content_overlap(blocks, config)
        self.assertEqual(len(filtered_blocks), 6)
        
        # Find captions in filtered blocks
        captions = [b for b in filtered_blocks if b.get("type") == "paragraph" and "Table" in b.get("text", "")]
        self.assertEqual(len(captions), 2)
        
        # Test with preserve captions disabled
        config = {
            "enabled": True,
            "containment_threshold": 0.7,
            "preserve_table_captions": False
        }
        filtered_blocks = remove_content_overlap(blocks, config)
        self.assertEqual(len(filtered_blocks), 6)
        
        # Test with overlap disabled
        config = {"enabled": False}
        filtered_blocks = remove_content_overlap(blocks, config)
        self.assertEqual(len(filtered_blocks), 8)  # All blocks retained
        
        # Test with different containment threshold
        config = {
            "enabled": True,
            "containment_threshold": 0.9  # Higher threshold
        }
        filtered_blocks = remove_content_overlap(blocks, config)
        self.assertGreater(len(filtered_blocks), 6)  # Fewer blocks removed

if __name__ == "__main__":
    unittest.main()
