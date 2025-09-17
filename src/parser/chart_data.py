"""
Chart data extraction for PDF parser.

This module provides utilities for extracting data from detected charts in PDF documents.
"""
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import fitz  # PyMuPDF

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("WARNING: OpenCV not available. Chart data extraction will be limited.")

try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("WARNING: Pillow (PIL) not available. Chart visualization will be disabled.")

logger = logging.getLogger(__name__)

class ChartDataExtractor:
    """Extract data points from chart images."""
    
    def __init__(self, config=None):
        """
        Initialize the chart data extractor.
        
        Args:
            config: Optional configuration manager
        """
        self.config = config
        
        # Check if OpenCV is available
        if not OPENCV_AVAILABLE:
            logger.warning("OpenCV not available. Chart data extraction will be limited.")
        
        # Configure extraction parameters from config
        self.min_contour_area = 50
        self.line_thickness_threshold = 2
        self.edge_detection_threshold1 = 50
        self.edge_detection_threshold2 = 150
        
        if config:
            self.min_contour_area = config.get("charts", "min_contour_area", 50)
            self.line_thickness_threshold = config.get("charts", "line_thickness_threshold", 2)
            self.edge_detection_threshold1 = config.get("charts", "edge_detection_threshold1", 50)
            self.edge_detection_threshold2 = config.get("charts", "edge_detection_threshold2", 150)
    
    def extract_chart_data(self, chart_image: np.ndarray, chart_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract data points from a chart image.
        
        Args:
            chart_image: NumPy array containing the chart image
            chart_type: Optional type of chart (e.g., "bar", "line", "pie")
            
        Returns:
            Dictionary containing extracted chart data
        """
        if not OPENCV_AVAILABLE:
            return {"error": "OpenCV not available for chart data extraction"}
        
        # Auto-detect chart type if not provided
        if chart_type is None:
            chart_type = self._detect_chart_type(chart_image)
        
        # Extract data based on chart type
        if chart_type == "bar":
            return self._extract_bar_chart_data(chart_image)
        elif chart_type == "line":
            return self._extract_line_chart_data(chart_image)
        elif chart_type == "pie":
            return self._extract_pie_chart_data(chart_image)
        else:
            return self._extract_generic_chart_data(chart_image)
    
    def _detect_chart_type(self, image: np.ndarray) -> str:
        """
        Detect the type of chart in the image.
        
        Args:
            image: NumPy array containing the chart image
            
        Returns:
            Detected chart type as string
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, self.edge_detection_threshold1, self.edge_detection_threshold2)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count rectangles (potential bars)
        rectangle_count = 0
        for contour in contours:
            if len(contour) >= 4:  # Minimum points for a rectangle
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                if len(approx) == 4 and cv2.contourArea(approx) > self.min_contour_area:
                    rectangle_count += 1
        
        # Check for circles (potential pie chart)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
            param1=50, param2=30, minRadius=50, maxRadius=300
        )
        
        # Detect lines for line charts
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi/180, threshold=100,
            minLineLength=50, maxLineGap=10
        )
        
        # Determine chart type based on detected elements
        if rectangle_count > 5:
            return "bar"
        elif circles is not None and len(circles[0]) > 0:
            return "pie"
        elif lines is not None and len(lines) > 10:
            return "line"
        else:
            return "unknown"
    
    def _extract_bar_chart_data(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract data from a bar chart.
        
        Args:
            image: NumPy array containing the bar chart image
            
        Returns:
            Dictionary containing extracted chart data
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, self.edge_detection_threshold1, self.edge_detection_threshold2)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract bar positions and heights
        bars = []
        for contour in contours:
            if len(contour) >= 4:  # Minimum points for a rectangle
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                if len(approx) == 4 and cv2.contourArea(approx) > self.min_contour_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    # Store bar position and dimensions
                    bars.append({
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h),
                        "value": int(h)  # Using height as value (relative)
                    })
        
        # Sort bars by x-position
        bars.sort(key=lambda bar: bar["x"])
        
        return {
            "chart_type": "bar",
            "data_points": bars,
            "x_axis": {"min": 0, "max": image.shape[1]},
            "y_axis": {"min": 0, "max": image.shape[0]},
            "bar_count": len(bars)
        }
    
    def _extract_line_chart_data(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract data from a line chart.
        
        Args:
            image: NumPy array containing the line chart image
            
        Returns:
            Dictionary containing extracted chart data
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, self.edge_detection_threshold1, self.edge_detection_threshold2)
        
        # Detect lines
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi/180, threshold=50,
            minLineLength=20, maxLineGap=10
        )
        
        # Extract line segments
        line_segments = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Store line segment
                line_segments.append({
                    "start": {"x": int(x1), "y": int(y1)},
                    "end": {"x": int(x2), "y": int(y2)},
                    "length": int(np.sqrt((x2-x1)**2 + (y2-y1)**2))
                })
        
        # Sort line segments by x-position
        line_segments.sort(key=lambda line: line["start"]["x"])
        
        return {
            "chart_type": "line",
            "line_segments": line_segments,
            "segment_count": len(line_segments),
            "x_axis": {"min": 0, "max": image.shape[1]},
            "y_axis": {"min": 0, "max": image.shape[0]}
        }
    
    def _extract_pie_chart_data(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract data from a pie chart.
        
        Args:
            image: NumPy array containing the pie chart image
            
        Returns:
            Dictionary containing extracted chart data
        """
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Try direct circle detection first with Hough transform
        min_dimension = min(height, width)
        min_radius = int(min_dimension * 0.15)  # Min radius is 15% of image dimension
        max_radius = int(min_dimension * 0.45)  # Max radius is 45% of image dimension
        
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1,  # Resolution ratio of accumulator to image
            minDist=min_dimension//2,  # Min distance between circle centers
            param1=100,  # Higher threshold for edge detection
            param2=30,   # Threshold for circle detection
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        # Find the main pie circle
        pie_circle = None
        if circles is not None:
            # Convert circles to integers
            circles = np.uint16(np.around(circles))
            
            # Find the largest circle (likely the pie chart)
            largest_radius = 0
            for circle in circles[0, :]:
                center_x, center_y, radius = circle
                if radius > largest_radius:
                    largest_radius = radius
                    pie_circle = {
                        "center": {"x": int(center_x), "y": int(center_y)},
                        "radius": int(radius)
                    }
        else:
            # If no circles found, assume center of image with reasonable radius
            pie_circle = {
                "center": {"x": width // 2, "y": height // 2},
                "radius": min(width, height) // 3
            }
        
        # Apply color segmentation to detect pie segments
        segments = []
        if pie_circle is not None:
            # Convert to HSV for better color segmentation
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Create a mask for the pie chart area
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            cv2.circle(
                mask, 
                (pie_circle["center"]["x"], pie_circle["center"]["y"]), 
                pie_circle["radius"], 
                255, 
                -1
            )
            
            # Apply the mask to the HSV image
            masked_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
            
            # Sample points around the circle to detect color changes
            center_x = pie_circle["center"]["x"]
            center_y = pie_circle["center"]["y"] 
            radius = pie_circle["radius"]
            
            # Sample every 5 degrees (72 points)
            samples = 72
            angles = np.linspace(0, 2*np.pi, samples, endpoint=False)
            
            colors = []
            points = []
            
            for angle in angles:
                # Sample at 80% of radius to avoid edge effects
                x = int(center_x + 0.8 * radius * np.cos(angle))
                y = int(center_y + 0.8 * radius * np.sin(angle))
                
                # Ensure point is within image bounds
                x = max(0, min(x, width-1))
                y = max(0, min(y, height-1))
                
                colors.append(image[y, x].tolist())  # BGR color
                points.append((x, y))
            
            # Detect color changes to identify pie segments
            color_changes = []
            prev_color = np.array(colors[0])
            
            for i, color in enumerate(colors[1:], 1):
                color_arr = np.array(color)
                diff = np.sum(np.abs(color_arr - prev_color))
                
                if diff > 80:  # Threshold for color change
                    color_changes.append(i)
                
                prev_color = color_arr
            
            # Create segments
            if len(color_changes) >= 2:  # Need at least 2 changes for segments
                # Add first index if not already present
                if 0 not in color_changes:
                    color_changes.insert(0, 0)
                
                # Process segments
                for i in range(len(color_changes)):
                    start_idx = color_changes[i]
                    end_idx = color_changes[(i+1) % len(color_changes)]
                    
                    # Handle wrap-around
                    if end_idx <= start_idx:
                        end_idx += samples
                    
                    # Get average color for the segment
                    segment_colors = []
                    for j in range(start_idx, min(end_idx, samples)):
                        segment_colors.append(colors[j % samples])
                    
                    if segment_colors:
                        avg_color = np.mean(segment_colors, axis=0).tolist()
                    else:
                        avg_color = [0, 0, 0]
                    
                    # Calculate percentage of the circle
                    percentage = ((end_idx - start_idx) / samples) * 100
                    
                    # Get angles in degrees
                    start_angle = angles[start_idx % samples]
                    end_angle = angles[end_idx % samples] if end_idx < samples else angles[end_idx % samples] + 2*np.pi
                    
                    segments.append({
                        "color": avg_color,
                        "percentage": percentage,
                        "start_angle": np.degrees(start_angle),
                        "end_angle": np.degrees(end_angle % (2*np.pi))
                    })
            
            # If segments were not detected using sampling, try color clustering
            if not segments:
                # Extract colors from the masked image
                pixels = []
                for y in range(height):
                    for x in range(width):
                        if mask[y, x] > 0:
                            pixels.append(image[y, x])
                
                if pixels:
                    try:
                        # Use k-means clustering to group similar colors
                        from sklearn.cluster import KMeans
                        
                        pixels = np.array(pixels)
                        kmeans = KMeans(n_clusters=5, random_state=0).fit(pixels)
                        
                        # Get cluster centers and counts
                        colors = kmeans.cluster_centers_.astype(int).tolist()
                        counts = np.bincount(kmeans.labels_)
                        total = sum(counts)
                        
                        # Create segments from clusters
                        current_angle = 0
                        for i, (count, color) in enumerate(zip(counts, colors)):
                            percentage = (count / total) * 100
                            angle = (percentage / 100) * 360
                            
                            segments.append({
                                "color": color,
                                "percentage": percentage,
                                "start_angle": current_angle,
                                "end_angle": current_angle + angle
                            })
                            
                            current_angle += angle
                    except Exception as e:
                        logger.warning(f"K-means clustering failed: {e}")
                        
                        # Create default segments as fallback
                        segments = [
                            {"color": [255, 0, 0], "percentage": 33.3, "start_angle": 0, "end_angle": 120},
                            {"color": [0, 255, 0], "percentage": 33.3, "start_angle": 120, "end_angle": 240},
                            {"color": [0, 0, 255], "percentage": 33.4, "start_angle": 240, "end_angle": 360}
                        ]
            
            # Create visualization of detected segments
            debug_img = image.copy()
            cv2.circle(
                debug_img,
                (pie_circle["center"]["x"], pie_circle["center"]["y"]),
                pie_circle["radius"],
                (0, 255, 0),
                2
            )
            
            for segment in segments:
                start_angle_rad = np.radians(segment["start_angle"])
                end_angle_rad = np.radians(segment["end_angle"])
                mid_angle_rad = (start_angle_rad + end_angle_rad) / 2
                
                # Draw radius lines
                start_x = int(center_x + radius * np.cos(start_angle_rad))
                start_y = int(center_y + radius * np.sin(start_angle_rad))
                cv2.line(debug_img, (center_x, center_y), (start_x, start_y), (255, 0, 0), 2)
            
            # Find peaks in histogram (dominant colors)
            total_pixels = cv2.countNonZero(mask)
            
            for h in range(h_bins):
                for s in range(s_bins):
                    bin_value = hist[h, s]
                    if bin_value > 0.05:  # Threshold for significant segments
                        segments.append({
                            "hue_bin": h,
                            "saturation_bin": s,
                            "proportion": float(bin_value),
                            "estimated_percentage": float(bin_value * 100)
                        })
        
        return {
            "chart_type": "pie",
            "pie_circle": pie_circle,
            "segments": segments,
            "segment_count": len(segments),
            "image_dimensions": {"width": image.shape[1], "height": image.shape[0]}
        }
    
    def _extract_generic_chart_data(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract basic data from any chart type.
        
        Args:
            image: NumPy array containing the chart image
            
        Returns:
            Dictionary containing extracted chart data
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, self.edge_detection_threshold1, self.edge_detection_threshold2)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size
        significant_contours = [
            contour for contour in contours 
            if cv2.contourArea(contour) > self.min_contour_area
        ]
        
        # Extract basic shapes
        shapes = []
        for contour in significant_contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            
            # Determine shape type
            shape_type = "unknown"
            if len(approx) == 3:
                shape_type = "triangle"
            elif len(approx) == 4:
                shape_type = "rectangle"
            elif len(approx) > 10:
                shape_type = "circle"
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Store shape information
            shapes.append({
                "type": shape_type,
                "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                "area": int(cv2.contourArea(contour)),
                "point_count": len(approx)
            })
        
        return {
            "chart_type": "unknown",
            "detected_shapes": shapes,
            "shape_count": len(shapes),
            "image_dimensions": {"width": image.shape[1], "height": image.shape[0]}
        }

def extract_data_from_chart(chart_image: np.ndarray, chart_type: Optional[str] = None, config=None) -> Dict[str, Any]:
    """
    Extract data from a chart image.
    
    Args:
        chart_image: NumPy array containing the chart image
        chart_type: Optional chart type
        config: Optional configuration manager
        
    Returns:
        Dictionary containing extracted chart data
    """
    extractor = ChartDataExtractor(config)
    return extractor.extract_chart_data(chart_image, chart_type)

def render_chart_data_visualization(chart_image: np.ndarray, chart_data: Dict[str, Any], 
                                  output_path: str) -> bool:
    """
    Render visualization of extracted chart data.
    
    Args:
        chart_image: Original chart image
        chart_data: Extracted chart data
        output_path: Path to save visualization
        
    Returns:
        True if visualization was created, False otherwise
    """
    if not PIL_AVAILABLE or not OPENCV_AVAILABLE:
        logger.warning("Visualization requires PIL and OpenCV")
        return False
    
    # Convert OpenCV image to PIL Image
    image_rgb = cv2.cvtColor(chart_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Create drawing context
    draw = ImageDraw.Draw(pil_image)
    
    # Draw detected elements based on chart type
    chart_type = chart_data.get("chart_type", "unknown")
    
    if chart_type == "bar" and "data_points" in chart_data:
        # Draw bounding boxes around bars
        for bar in chart_data["data_points"]:
            x, y = bar["x"], bar["y"]
            w, h = bar["width"], bar["height"]
            draw.rectangle([x, y, x+w, y+h], outline=(255, 0, 0), width=2)
            
            # Draw value on top of bar
            value_text = str(bar["value"])
            draw.text((x + w//2, y - 10), value_text, fill=(255, 0, 0))
    
    elif chart_type == "line" and "line_segments" in chart_data:
        # Draw line segments
        for segment in chart_data["line_segments"]:
            x1, y1 = segment["start"]["x"], segment["start"]["y"]
            x2, y2 = segment["end"]["x"], segment["end"]["y"]
            draw.line([x1, y1, x2, y2], fill=(0, 255, 0), width=2)
    
    elif chart_type == "pie" and "pie_circle" in chart_data and chart_data["pie_circle"]:
        # Draw pie circle
        circle = chart_data["pie_circle"]
        center_x, center_y = circle["center"]["x"], circle["center"]["y"]
        radius = circle["radius"]
        
        draw.ellipse(
            [center_x-radius, center_y-radius, center_x+radius, center_y+radius],
            outline=(0, 0, 255),
            width=2
        )
        
        # Draw segment information
        if "segments" in chart_data:
            y_pos = 10
            for i, segment in enumerate(chart_data["segments"]):
                percentage = segment.get("estimated_percentage", 0)
                text = f"Segment {i+1}: {percentage:.1f}%"
                draw.text((10, y_pos), text, fill=(255, 0, 0))
                y_pos += 20
    
    else:
        # Generic chart - draw all detected shapes
        if "detected_shapes" in chart_data:
            for shape in chart_data["detected_shapes"]:
                bbox = shape["bbox"]
                x, y = bbox["x"], bbox["y"]
                w, h = bbox["width"], bbox["height"]
                
                # Use different colors for different shapes
                if shape["type"] == "rectangle":
                    color = (255, 0, 0)  # Red
                elif shape["type"] == "circle":
                    color = (0, 255, 0)  # Green
                elif shape["type"] == "triangle":
                    color = (0, 0, 255)  # Blue
                else:
                    color = (255, 255, 0)  # Yellow
                
                draw.rectangle([x, y, x+w, y+h], outline=color, width=2)
                
                # Draw shape type
                draw.text((x, y-15), shape["type"], fill=color)
    
    # Save visualization
    pil_image.save(output_path)
    return True
