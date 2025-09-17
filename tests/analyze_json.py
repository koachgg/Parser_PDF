"""
Analyze the JSON output to check for issues with table extraction.
"""
import os
import sys
import json
from pathlib import Path
from collections import Counter

def analyze_json(json_path: str):
    """
    Analyze a JSON file for content types and table data.
    
    Args:
        json_path: Path to the JSON file
    """
    if not os.path.exists(json_path):
        print(f"Error: File not found: {json_path}")
        return
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Successfully loaded JSON from {json_path}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_path}")
        return
    except Exception as e:
        print(f"Error loading JSON: {str(e)}")
        return
    
    # Check pages
    pages = data.get("pages", [])
    print(f"\nTotal pages: {len(pages)}")
    
    # Analyze content types
    content_types = Counter()
    table_data_present = 0
    chart_data_present = 0
    text_present = 0
    
    # Analyze sections
    sections = set()
    subsections = set()
    
    for page in pages:
        page_num = page.get("page_number", "unknown")
        content = page.get("content", [])
        
        for item in content:
            item_type = item.get("type", "unknown")
            content_types[item_type] += 1
            
            if item.get("section"):
                sections.add(item.get("section"))
            
            if item.get("sub_section"):
                subsections.add(item.get("sub_section"))
            
            if item.get("text"):
                text_present += 1
                
            if item.get("table_data"):
                table_data_present += 1
                
            if item.get("chart_data"):
                chart_data_present += 1
    
    print("\nContent Type Analysis:")
    for content_type, count in content_types.items():
        print(f"  - {content_type}: {count}")
    
    print("\nData Presence:")
    print(f"  - Items with text: {text_present}")
    print(f"  - Items with table_data: {table_data_present}")
    print(f"  - Items with chart_data: {chart_data_present}")
    
    print("\nSection Analysis:")
    print(f"  - Unique sections: {len(sections)}")
    print(f"  - Example sections: {list(sections)[:5]}")
    print(f"  - Unique subsections: {len(subsections)}")
    if subsections:
        print(f"  - Example subsections: {list(subsections)[:5]}")
    
    # Analyze tables specifically
    if content_types.get("table", 0) > 0:
        print("\nTable Analysis:")
        
        tables_by_page = {}
        for page_idx, page in enumerate(pages):
            page_num = page.get("page_number", page_idx + 1)
            page_tables = [item for item in page.get("content", []) if item.get("type") == "table"]
            if page_tables:
                tables_by_page[page_num] = len(page_tables)
        
        print(f"  - Pages with tables: {len(tables_by_page)}")
        print(f"  - Table distribution: {tables_by_page}")
        
        # Sample the first table we can find
        for page in pages:
            for item in page.get("content", []):
                if item.get("type") == "table" and item.get("table_data"):
                    print("\nSample Table Data:")
                    table_data = item.get("table_data")
                    print(f"  - Rows: {len(table_data)}")
                    print(f"  - Columns: {len(table_data[0]) if table_data and table_data[0] else 0}")
                    print("\nFirst 3 rows:")
                    for i, row in enumerate(table_data[:3]):
                        print(f"  Row {i}: {row}")
                    break
            else:
                continue
            break
    
    # Check for any issues in the JSON structure
    print("\nPotential Issues:")
    
    if content_types.get("table", 0) > 0 and table_data_present == 0:
        print("  - WARNING: Tables found but no table_data present")
    
    if content_types.get("chart", 0) > 0 and chart_data_present == 0:
        print("  - WARNING: Charts found but no chart_data present")
    
    # Check bbox values
    missing_bbox = 0
    for page in pages:
        for item in page.get("content", []):
            if not item.get("bbox"):
                missing_bbox += 1
    
    if missing_bbox > 0:
        print(f"  - WARNING: {missing_bbox} items have missing or empty bbox values")
    
    # Check for metadata
    meta = data.get("meta", {})
    if not meta:
        print("  - WARNING: Missing metadata section")
    else:
        print("\nMetadata:")
        for key, value in meta.items():
            print(f"  - {key}: {value}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_json.py <json_path>")
        sys.exit(1)
    
    json_path = sys.argv[1]
    analyze_json(json_path)
