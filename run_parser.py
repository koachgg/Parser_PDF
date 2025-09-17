"""
Simple wrapper script to run the PDF parser with graceful error handling.
"""
import sys
import os
import traceback
from pathlib import Path

def main():
    """Run the PDF parser with error handling."""
    try:
        # Add the project root to the path
        script_dir = Path(__file__).resolve().parent
        sys.path.insert(0, str(script_dir))
        
        # Try to run the main parser module
        from src.main import main as run_parser
        
        # Run the parser with CLI arguments
        sys.exit(run_parser())
    except ImportError as e:
        print(f"\nERROR: Failed to import required module: {e}")
        print("\nMake sure all dependencies are installed:")
        print("    pip install -r requirements.txt")
        print("\nSome packages require system dependencies:")
        print("  - Tesseract OCR: For text recognition")
        print("  - Ghostscript: For PDF processing")
        print("  - OpenCV: For image processing")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nStack trace:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
