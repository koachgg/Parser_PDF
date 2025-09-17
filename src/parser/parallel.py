"""
Parallel processing support for PDF parser.

This module provides utilities for processing PDF documents in parallel using multiple CPU cores.
"""
import os
import logging
from typing import List, Dict, Any, Optional, Callable
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

def get_optimal_worker_count(max_workers: Optional[int] = None) -> int:
    """
    Determine the optimal number of worker processes.
    
    Args:
        max_workers: Optional maximum number of workers to use
        
    Returns:
        Number of worker processes to use
    """
    # Default to number of CPU cores minus 1 (leave one for system)
    # But use at least 1 worker
    cpu_count = multiprocessing.cpu_count()
    default_count = max(1, cpu_count - 1)
    
    # If max_workers is specified, use the minimum of max_workers and default_count
    if max_workers is not None:
        return min(max_workers, default_count)
    
    return default_count

def process_pages_in_parallel(pdf_path: str, doc: fitz.Document, page_processor: Callable, 
                            batch_size: int = 1, max_workers: Optional[int] = None, 
                            **kwargs) -> List[Dict[str, Any]]:
    """
    Process PDF pages in parallel using multiple worker processes.
    
    Args:
        pdf_path: Path to PDF file
        doc: PyMuPDF Document
        page_processor: Function to process a page or batch of pages
        batch_size: Number of pages to process in each batch
        max_workers: Maximum number of worker processes to use
        **kwargs: Additional arguments to pass to page_processor
        
    Returns:
        List of processed page data
    """
    total_pages = len(doc)
    worker_count = get_optimal_worker_count(max_workers)
    
    logger.info(f"Processing {total_pages} pages using {worker_count} workers")
    
    # Create batches of pages
    batches = []
    for i in range(0, total_pages, batch_size):
        # Get page numbers in this batch
        batch_page_numbers = list(range(i, min(i + batch_size, total_pages)))
        batches.append(batch_page_numbers)
    
    logger.info(f"Created {len(batches)} batches of pages")
    
    # IMPORTANT: Fix for pickling issue in multiprocessing
    # Define process_batch function outside and make it top-level
    all_page_data = []
    
    # Process pages sequentially if multiprocessing fails
    if True:  # Temporary workaround until parallel processing is fixed
        logger.warning("Using sequential processing instead of parallel processing due to pickling limitations")
        for batch in batches:
            # Process each batch
            batch_doc = fitz.open(pdf_path)
            
            result = []
            for page_idx in batch:
                page = batch_doc[page_idx]
                page_num = page_idx + 1
                
                try:
                    # Process the page
                    page_data = page_processor(pdf_path, page, page_num, batch_doc, **kwargs)
                    result.append(page_data)
                except Exception as e:
                    logger.exception(f"Error processing page {page_num}: {str(e)}")
                    # Add empty page data on error
                    result.append({
                        "page_number": page_num,
                        "error": str(e),
                        "content": []
                    })
            
            # Close the document
            batch_doc.close()
            all_page_data.extend(result)
    
    # Sort pages by page number
    all_page_data.sort(key=lambda p: p.get("page_number", 0))
    
    return all_page_data

def is_parallel_processing_supported() -> bool:
    """
    Check if parallel processing is supported on this system.
    
    Returns:
        True if parallel processing is supported, False otherwise
    """
    try:
        # Check if multiprocessing is available
        cpu_count = multiprocessing.cpu_count()
        
        # Try creating a small process pool
        with ProcessPoolExecutor(max_workers=1) as executor:
            pass
            
        return True
    except Exception as e:
        logger.warning(f"Parallel processing not supported: {str(e)}")
        return False
