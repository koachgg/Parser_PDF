"""
Unit tests for the parallel processing functionality.
"""
import os
import unittest
import tempfile
import pytest
import fitz
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.parser.parallel import process_pages_in_parallel, is_parallel_processing_supported, get_optimal_worker_count

class TestParallelProcessing(unittest.TestCase):
    """Test cases for parallel processing functionality."""
    
    def setUp(self):
        # Create a simple test PDF with multiple pages
        self.test_pdf_path = os.path.join(tempfile.gettempdir(), "test_parallel.pdf")
        self._create_test_pdf(self.test_pdf_path, 10)
        
    def tearDown(self):
        # Clean up the test PDF
        if os.path.exists(self.test_pdf_path):
            os.remove(self.test_pdf_path)
    
    def _create_test_pdf(self, pdf_path, num_pages):
        """Create a simple test PDF with specified number of pages."""
        doc = fitz.open()
        for i in range(num_pages):
            page = doc.new_page()
            # Add some text to the page
            page.insert_text((50, 50), f"Page {i+1}")
            
            # Add a rectangle to test content extraction
            page.draw_rect((100, 100, 200, 200))
        
        doc.save(pdf_path)
        doc.close()
    
    def _simple_page_processor(self, pdf_path, page, page_num, doc, **kwargs):
        """A simple page processor function for testing."""
        text = page.get_text("text")
        return {
            "page_number": page_num,
            "text_length": len(text),
            "has_text": "Page" in text,
            "test_param": kwargs.get("test_param", None)
        }
    
    def test_is_parallel_processing_supported(self):
        """Test if parallel processing is properly detected as supported."""
        # This should be True on most systems
        self.assertTrue(is_parallel_processing_supported())
    
    def test_get_optimal_worker_count(self):
        """Test that optimal worker count is calculated correctly."""
        # Default should be CPU count - 1 (at least 1)
        import multiprocessing
        expected = max(1, multiprocessing.cpu_count() - 1)
        self.assertEqual(get_optimal_worker_count(), expected)
        
        # Test with explicit max_workers
        self.assertEqual(get_optimal_worker_count(2), min(2, expected))
        self.assertEqual(get_optimal_worker_count(1), 1)
    
    def test_parallel_processing_basic(self):
        """Test basic parallel processing functionality."""
        doc = fitz.open(self.test_pdf_path)
        results = process_pages_in_parallel(
            self.test_pdf_path,
            doc,
            self._simple_page_processor
        )
        
        # Check that we got the expected number of results
        self.assertEqual(len(results), 10)
        
        # Check that results are sorted by page number
        self.assertEqual([r["page_number"] for r in results], list(range(1, 11)))
        
        # Check that each result has the expected structure
        for result in results:
            self.assertIn("page_number", result)
            self.assertIn("text_length", result)
            self.assertIn("has_text", result)
            self.assertTrue(result["has_text"])
        
        doc.close()
    
    def test_parallel_processing_with_batch_size(self):
        """Test parallel processing with different batch sizes."""
        doc = fitz.open(self.test_pdf_path)
        
        # Test with batch size of 2
        results = process_pages_in_parallel(
            self.test_pdf_path,
            doc,
            self._simple_page_processor,
            batch_size=2
        )
        
        self.assertEqual(len(results), 10)
        self.assertEqual([r["page_number"] for r in results], list(range(1, 11)))
        
        # Test with batch size of 5
        results = process_pages_in_parallel(
            self.test_pdf_path,
            doc,
            self._simple_page_processor,
            batch_size=5
        )
        
        self.assertEqual(len(results), 10)
        self.assertEqual([r["page_number"] for r in results], list(range(1, 11)))
        
        doc.close()
    
    def test_parallel_processing_with_max_workers(self):
        """Test parallel processing with different worker counts."""
        doc = fitz.open(self.test_pdf_path)
        
        # Test with 1 worker
        results = process_pages_in_parallel(
            self.test_pdf_path,
            doc,
            self._simple_page_processor,
            max_workers=1
        )
        
        self.assertEqual(len(results), 10)
        
        # Test with 2 workers
        results = process_pages_in_parallel(
            self.test_pdf_path,
            doc,
            self._simple_page_processor,
            max_workers=2
        )
        
        self.assertEqual(len(results), 10)
        
        doc.close()
    
    def test_parallel_processing_with_kwargs(self):
        """Test passing additional kwargs to the page processor."""
        doc = fitz.open(self.test_pdf_path)
        
        # Pass a test parameter
        results = process_pages_in_parallel(
            self.test_pdf_path,
            doc,
            self._simple_page_processor,
            test_param="test_value"
        )
        
        # Check that the parameter was passed to each page processor
        for result in results:
            self.assertEqual(result["test_param"], "test_value")
        
        doc.close()
    
    def test_parallel_processing_error_handling(self):
        """Test that errors in page processing are handled gracefully."""
        doc = fitz.open(self.test_pdf_path)
        
        def failing_processor(pdf_path, page, page_num, doc, **kwargs):
            if page_num == 5:
                raise Exception("Test error")
            return self._simple_page_processor(pdf_path, page, page_num, doc, **kwargs)
        
        # Process with a processor that fails on page 5
        results = process_pages_in_parallel(
            self.test_pdf_path,
            doc,
            failing_processor
        )
        
        # Should still get 10 results, but page 5 should have an error
        self.assertEqual(len(results), 10)
        
        # Check page 5 specifically
        page5 = next(r for r in results if r["page_number"] == 5)
        self.assertIn("error", page5)
        self.assertEqual(page5["content"], [])
        
        doc.close()

if __name__ == "__main__":
    unittest.main()
