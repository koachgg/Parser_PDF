# Content Overlap Handling

This document explains how the PDF parser handles overlapping content in documents.

## Overview

In PDFs, different content elements (text, tables, charts) often overlap, which can lead to duplicated data in the extracted output. The content overlap handling system identifies and resolves these overlaps to produce cleaner output.

## Key Features

- **Overlap Detection**: Uses Intersection over Union (IoU) calculations to identify when content overlaps
- **Intelligent Content Prioritization**: Prioritizes tables and charts over text blocks
- **Table Caption Preservation**: Special handling for text that appears to be table captions
- **Header/Footer Handling**: Options for removing or preserving headers and footers that overlap with content
- **Configurable Thresholds**: Customizable thresholds for overlap and containment detection

## Configuration Options

Content overlap handling can be configured in the `config.yaml` file:

```yaml
content_overlap:
  enabled: true                     # Enable/disable overlap handling
  containment_threshold: 0.7        # Threshold for determining if one block is inside another
  overlap_threshold: 0.3            # Threshold for reporting significant overlaps
  remove_headers_in_tables: true    # Remove headers that overlap with tables
  remove_footers_in_tables: true    # Remove footers that overlap with tables
  preserve_table_captions: true     # Preserve text blocks that appear to be table captions
```

## How It Works

### 1. Content Prioritization

Content is prioritized in the following order:
1. Tables (highest priority)
2. Charts
3. Paragraphs
4. Headers
5. Footers (lowest priority)

This ensures that more structured content types (tables, charts) take precedence.

### 2. Containment Detection

A block is considered "contained" within another if:
- A significant portion (default: 70%) of its area overlaps with the other block
- The block is of lower priority than the containing block

### 3. Table Caption Detection

Text blocks are identified as table captions if they:
- Contain keywords like "table", "tbl", etc.
- Are positioned directly above or below a table
- Have significant horizontal alignment with the table

### 4. Overlap Resolution

When resolving overlaps:
1. Tables and charts are always preserved
2. Text inside tables or charts is removed (unless it's a caption and caption preservation is enabled)
3. Headers and footers that overlap with tables can be optionally preserved

## Examples

### Example 1: Text Inside Table

If a paragraph is detected inside a table, the paragraph will be removed since tables take priority.

### Example 2: Table Caption

```
[Text Block: "Table 1: Financial Results"]  <-- Preserved as caption
+--------------------------------------+
| Revenue | Expenses | Profit          |
| $100    | $80      | $20             |  <-- Table preserved
+--------------------------------------+
```

### Example 3: Partial Overlap

When content partially overlaps but is not fully contained, both elements are preserved by default, but the overlap is logged for debugging.

## Visual Debugging

Enable visual debugging to see how content overlap is being handled:

```bash
python -m src.main --input sample.pdf --visual-debug
```

This will generate images showing the bounding boxes of detected content, making it easier to identify and troubleshoot overlap issues.
