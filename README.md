# PDF Parsing → Structured JSON

A Python tool to parse PDF files into structured JSON, preserving page hierarchy, sections/sub-sections, and content types (paragraphs, tables, charts).

## Features

- **Text extraction**: Detects paragraphs, headings, and text blocks with precise positioning
- **Section detection**: Identifies document structure with section/subsection tracking
- **Table extraction**: Multiple table detection strategies (lattice, stream, heuristic)
- **Chart detection**: Identifies charts and extracts descriptions and data when available
- **OCR support**: Falls back to OCR for scanned pages or images
- **Output validation**: Validates JSON against a schema

## Installation

### Prerequisites

- Python 3.8+
- System dependencies for Camelot and Tesseract:
  - **Windows**: Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH
  - **macOS**: `brew install tesseract ghostscript`
  - **Linux**: `apt-get install tesseract-ocr ghostscript`

### Setup with virtual environment

```bash
# Clone the repository
git clone https://github.com/username/pdf-parser.git
cd pdf-parser

# Create and activate virtual environment
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Command-line interface

```bash
# Basic usage
python -m src.main --input samples/input/sample.pdf --output samples/output/out.json

# With OCR enabled
python -m src.main --input samples/input/sample.pdf --output samples/output/out.json --enable-ocr

# Specify table extraction method
python -m src.main --input samples/input/sample.pdf --table-mode lattice

# Process only first few pages
python -m src.main --input samples/input/sample.pdf --max-pages 5

# Debug mode
python -m src.main --input samples/input/sample.pdf --debug
```

### Optional flags

- `--input`: Path to input PDF file (default: uses sample file)
- `--output`: Path to output JSON file (default: generated from input filename)
- `--enable-ocr/--disable-ocr`: Enable/disable OCR for scanned pages (default: disabled)
- `--table-mode`: Table extraction mode, one of `auto`, `lattice`, `stream`, `heuristic` (default: auto)
- `--max-pages`: Maximum number of pages to process (default: all)
- `--debug/--no-debug`: Enable/disable debug output (default: disabled)

## Output Schema

The parser produces a JSON file with the following structure:

```json
{
  "pages": [
    {
      "page_number": 1,
      "content": [
        {
          "type": "paragraph",
          "section": "Introduction",
          "sub_section": null,
          "text": "This is a sample paragraph.",
          "table_data": null,
          "chart_data": null,
          "description": null,
          "bbox": [0, 0, 100, 20]
        },
        {
          "type": "table",
          "section": "Results",
          "sub_section": null,
          "text": null,
          "table_data": [["Header1", "Header2"], ["Data1", "Data2"]],
          "chart_data": null,
          "description": null,
          "bbox": [0, 30, 100, 100]
        }
      ]
    }
  ],
  "meta": {
    "source_file": "sample.pdf",
    "parser_versions": {
      "text": "1.23.22",
      "tables": "0.11.0",
      "charts": "0.1.0"
    },
    "created_at": "2023-05-01T12:00:00"
  }
}
```

## Development

### Running tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_tables.py

# Run with coverage
pytest --cov=src tests/
```

### Troubleshooting

- **OCR is slow**: OCR processing is CPU-intensive. Use `--max-pages` for testing.
- **Table detection issues**: Try different table modes with `--table-mode`.
- **Missing dependencies**: Ensure all system dependencies are installed.

## License

MIT
