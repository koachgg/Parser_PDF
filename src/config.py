"""
Configuration management for PDF parser.
"""
import os
import yaml
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages configuration for PDF parser."""
    
    DEFAULT_CONFIG = {
        "general": {
            "debug": False,
            "output_format": "json",
            "validate_schema": True,
        },
        "text": {
            "extract_headers_footers": True,
            "min_paragraph_length": 5,
        },
        "tables": {
            "mode": "auto",
            "min_confidence": 50,
            "deduplication_threshold": 0.3,
        },
        "charts": {
            "use_ocr": True,
            "min_vector_density": 10,
            "detection_threshold": 0.6,
        },
        "ocr": {
            "enabled": False,
            "dpi": 300,
            "language": "eng",
            "use_for_tables": True,
            "use_for_charts": True,
            "use_for_text": False,
        },
        "visual_debug": {
            "enabled": False,
            "output_dir": "debug_output",
            "output_path": None,  # Will be set at runtime
            "draw_bounding_boxes": True,
            "save_images": True,
            "colors": {
                "paragraph": [0, 0, 255],    # Blue
                "table": [255, 0, 0],        # Red
                "chart": [0, 255, 0],        # Green
                "header": [255, 165, 0],     # Orange
                "footer": [128, 0, 128],     # Purple
                "ocr": [255, 255, 0]         # Yellow
            }
        },
        "content_overlap": {
            "enabled": True,
            "containment_threshold": 0.7,
            "overlap_threshold": 0.3,
            "remove_headers_in_tables": True,
            "remove_footers_in_tables": True,
            "preserve_table_captions": True,
        },
        "performance": {
            "parallel_processing": False,
            "max_workers": 4,
            "batch_size": 5,
        }
    }
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to YAML configuration file
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def load_config(self, config_file: str) -> None:
        """
        Load configuration from YAML file.
        
        Args:
            config_file: Path to YAML configuration file
        """
        try:
            with open(config_file, "r") as f:
                user_config = yaml.safe_load(f)
                
            # Merge user config with default config
            if user_config:
                for section, values in user_config.items():
                    if section in self.config:
                        self.config[section].update(values)
                    else:
                        self.config[section] = values
                        
            logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.warning(f"Failed to load configuration from {config_file}: {str(e)}")
            logger.warning("Using default configuration")
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        try:
            return self.config.get(section, {}).get(key, default)
        except Exception:
            return default
    
    def set(self, section: str, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            value: Configuration value
        """
        if section not in self.config:
            self.config[section] = {}
            
        self.config[section][key] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Configuration section
            
        Returns:
            Configuration section as dictionary
        """
        return self.config.get(section, {})
        
    def get_visual_debug_config(self) -> Dict[str, Any]:
        """
        Get visual debugging configuration with defaults.
        
        Returns:
            Dictionary with visual debugging configuration
        """
        debug_config = self.get_section("visual_debug")
        
        # Convert output_dir and output_path if needed
        output_dir = debug_config.get("output_dir")
        if output_dir and not os.path.isabs(output_dir):
            debug_config["output_dir"] = os.path.abspath(output_dir)
            
        # If output_path not set, use output_dir
        if not debug_config.get("output_path"):
            debug_config["output_path"] = debug_config.get("output_dir", "debug_output")
            
        return debug_config
        
    def get_content_overlap_config(self) -> Dict[str, Any]:
        """
        Get content overlap configuration with defaults.
        
        Returns:
            Dictionary with content overlap configuration
        """
        return self.get_section("content_overlap")
    
    def save_config(self, config_file: str) -> bool:
        """
        Save current configuration to YAML file.
        
        Args:
            config_file: Path to YAML configuration file
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            with open(config_file, "w") as f:
                yaml.dump(self.config, f, default_flow_style=False)
            logger.info(f"Saved configuration to {config_file}")
            return True
        except Exception as e:
            logger.warning(f"Failed to save configuration to {config_file}: {str(e)}")
            return False
