"""
PDF Parser module for extracting structured content from PDFs.
"""
from .text_blocks import extract_text_blocks
from .unified_tables import extract_tables, UnifiedTableExtractor
from .charts import detect_charts, extract_chart_data
from .sections import SectionTracker
from .chart_data import extract_data_from_chart, render_chart_data_visualization
from .visual_debug import visualize_page_content
from .content_overlap import remove_content_overlap
from .parallel import process_pages_in_parallel, is_parallel_processing_supported
