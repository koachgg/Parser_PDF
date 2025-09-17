# Visual Debugging for PDF Parser

This document explains how to use the visual debugging features of the PDF parser.

## Overview

The visual debugging module provides:

- Visual representation of detected content (text blocks, tables, charts)
- Color-coded bounding boxes for different content types
- Individual image extraction of each content block
- Integration with the main parsing pipeline

## Enabling Visual Debugging

### Via Command Line

```
python -m src.main --input sample.pdf --visual-debug
```

### Via Configuration File

In `config.yaml`:

```yaml
visual_debug:
  enabled: true
  output_dir: debug_output
  draw_bounding_boxes: true
  save_images: true
```

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `enabled` | Enable/disable visual debugging | `false` |
| `output_dir` | Directory to save debug output | `debug_output` |
| `draw_bounding_boxes` | Draw bounding boxes around content | `true` |
| `save_images` | Extract individual content blocks as images | `true` |

## Color Coding

Different content types are color-coded:

- **Blue**: Paragraphs (text blocks)
- **Red**: Tables
- **Green**: Charts
- **Orange**: Headers
- **Purple**: Footers
- **Yellow**: OCR-detected text

## Running the Demo

A demonstration script is provided to showcase the visual debugging capabilities:

```
python scripts/demo_visual_debug.py path/to/document.pdf [output_directory]
```

## Output Structure

The visual debug output is organized as follows:

```
debug_output/
  └── TIMESTAMP/
      ├── page_1_debug.png        # Full page with bounding boxes
      ├── page_1/                 # Individual content blocks
      │   ├── paragraph_1.png
      │   ├── paragraph_2.png
      │   ├── table_1.png
      │   └── ...
      ├── page_2_debug.png
      └── ...
```

## Requirements

Visual debugging requires:
- Pillow (PIL) for image manipulation
- PyMuPDF for PDF rendering

## Use Cases

- Troubleshooting parser issues
- Understanding document structure
- Validating content extraction
- Improving extraction accuracy
