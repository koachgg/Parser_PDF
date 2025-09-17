"""
Script to test table extraction from PDFs.
"""
import os
import sys
import json
from pathlib import Path

# Add project path to sys.path for imports
project_path = str(Path(__file__).parent.parent)
if project_path not in sys.path:
    sys.path.insert(0, project_path)

def test_table_extraction():
    """Test table extraction and print results."""
    try:
        print("Attempting to import PyMuPDF...")
        import fitz
        print("✓ PyMuPDF successfully imported")
        
        print("\nAttempting to import Camelot...")
        import camelot
        print("✓ Camelot successfully imported")
        
        print("\nAttempting to import pdfplumber...")
        import pdfplumber
        print("✓ pdfplumber successfully imported")
        
        # Try to load the PDF
        pdf_path = str(Path(__file__).parent.parent / "samples" / "input" / "[Fund Factsheet - May]360ONE-MF-May 2025.pdf")
        print(f"\nTrying to open PDF at: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            print(f"❌ Error: PDF file not found at {pdf_path}")
            return
            
        # Try PyMuPDF first
        try:
            doc = fitz.open(pdf_path)
            print(f"✓ PyMuPDF successfully opened the PDF ({len(doc)} pages)")
            
            # Check first page for content
            page = doc[0]
            text = page.get_text()
            print(f"First page text length: {len(text)} characters")
            print("First 100 characters of text:", text[:100])
            
            # Close the document
            doc.close()
        except Exception as e:
            print(f"❌ Error with PyMuPDF: {str(e)}")
        
        # Try Camelot extraction
        try:
            print("\nTrying Camelot lattice extraction on page 1...")
            tables = camelot.read_pdf(pdf_path, pages='1', flavor='lattice')
            print(f"Found {len(tables)} tables with lattice mode")
            
            if len(tables) > 0:
                print("Sample of first table data:")
                print(tables[0].df.head().to_string())
                
            print("\nTrying Camelot stream extraction on page 1...")
            tables = camelot.read_pdf(pdf_path, pages='1', flavor='stream')
            print(f"Found {len(tables)} tables with stream mode")
            
            if len(tables) > 0:
                print("Sample of first table data:")
                print(tables[0].df.head().to_string())
        except Exception as e:
            print(f"❌ Error with Camelot: {str(e)}")
            
        # Try pdfplumber extraction
        try:
            print("\nTrying pdfplumber extraction on page 1...")
            with pdfplumber.open(pdf_path) as pdf:
                page = pdf.pages[0]
                tables = page.extract_tables()
                print(f"Found {len(tables)} tables with pdfplumber")
                
                if len(tables) > 0:
                    print("Sample of first table data:")
                    print(tables[0])
        except Exception as e:
            print(f"❌ Error with pdfplumber: {str(e)}")
            
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print("Make sure all required dependencies are installed:")
        print("pip install PyMuPDF camelot-py pdfplumber")
        
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        
if __name__ == "__main__":
    print("Testing table extraction functionality...\n")
    test_table_extraction()
