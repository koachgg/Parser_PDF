"""
Visual debugging utilities for PDF parser.
"""
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

def draw_bounding_boxes(page: fitz.Page, blocks: List[Dict[str, Any]], output_path: str, 
                       debug_config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Draw bounding boxes around detected content elements and save as an image.
    
    Args:
        page: PDF page
        blocks: List of content blocks
        output_path: Directory to save output images
        debug_config: Optional debug configuration with color settings
        
    Returns:
        Path to the saved image or None if failed
    """
    try:
        # Render the page to an image
        pix = page.get_pixmap(alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        draw = ImageDraw.Draw(img)
        
        # Try to get a font - default to system font if PIL font not available
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except IOError:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        # Get color configuration if provided
        debug_config = debug_config or {}
        config_colors = debug_config.get("colors", {})
        
        # Color mapping for different block types
        colors = {
            "paragraph": tuple(config_colors.get("paragraph", [0, 0, 255])),   # Blue
            "table": tuple(config_colors.get("table", [255, 0, 0])),           # Red
            "chart": tuple(config_colors.get("chart", [0, 255, 0])),           # Green
            "header": tuple(config_colors.get("header", [255, 165, 0])),       # Orange
            "footer": tuple(config_colors.get("footer", [128, 0, 128])),       # Purple
            "ocr": tuple(config_colors.get("ocr", [255, 255, 0]))              # Yellow
        }
        
        # Draw boxes for each block
        for i, block in enumerate(blocks):
            bbox = block.get("bbox", [0, 0, 0, 0])
            block_type = block.get("type", "unknown")
            
            # Get color for this block type
            color = colors.get(block_type, (100, 100, 100))
            
            # Draw rectangle
            draw.rectangle(
                [(bbox[0], bbox[1]), (bbox[2], bbox[3])],
                outline=color,
                width=2
            )
            
            # Draw label
            label = f"{block_type} {i+1}"
            try:
                if font:
                    draw.text((bbox[0], bbox[1] - 15), label, fill=color, font=font)
                else:
                    draw.text((bbox[0], bbox[1] - 15), label, fill=color)
            except Exception as e:
                logger.warning(f"Failed to draw text label: {str(e)}")
        
        # Save the image
        page_number = page.number + 1
        img_path = os.path.join(output_path, f"page_{page_number}_debug.png")
        img.save(img_path)
        logger.info(f"Saved debug visualization to {img_path}")
        return img_path
    except Exception as e:
        logger.warning(f"Failed to create debug visualization: {str(e)}")
        return None

def save_content_images(page: fitz.Page, blocks: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save individual images of each content block.
    
    Args:
        page: PDF page
        blocks: List of content blocks
        output_path: Directory to save output images
    """
    try:
        page_number = page.number + 1
        pix = page.get_pixmap(alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Create a directory for this page
        page_dir = os.path.join(output_path, f"page_{page_number}")
        os.makedirs(page_dir, exist_ok=True)
        
        # Save an image for each block
        for i, block in enumerate(blocks):
            bbox = block.get("bbox", [0, 0, 0, 0])
            block_type = block.get("type", "unknown")
            
            # Skip blocks without valid bounding boxes
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
                
            # Crop the image to the block's bounding box
            try:
                crop_img = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                img_path = os.path.join(page_dir, f"{block_type}_{i+1}.png")
                crop_img.save(img_path)
            except Exception as e:
                logger.warning(f"Failed to save content image for block {i}: {str(e)}")
    except Exception as e:
        logger.warning(f"Failed to save content images: {str(e)}")

def visualize_page_content(page: fitz.Page, blocks: List[Dict[str, Any]], debug_config: Dict[str, Any]) -> Optional[str]:
    """
    Create visual debug output for a page.
    
    Args:
        page: PDF page
        blocks: List of content blocks
        debug_config: Configuration for visual debugging
        
    Returns:
        Path to main debug image if successful, None otherwise
    """
    output_dir = debug_config.get("output_path")
    if not output_dir:
        logger.warning("No output directory specified for visual debugging")
        return None
    
    # Main visualization with bounding boxes
    img_path = None
    if debug_config.get("draw_bounding_boxes", True):
        img_path = draw_bounding_boxes(page, blocks, output_dir, debug_config)
    
    # Individual content block images
    if debug_config.get("save_images", True):
        save_content_images(page, blocks, output_dir)
        
    return img_path
