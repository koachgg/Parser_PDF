"""
PDF Parser module for extracting structured content from PDFs.
"""
from .text_blocks import extract_text_blocks
from .tables import extract_tables
from .charts import detect_charts
from .sections import SectionTracker
