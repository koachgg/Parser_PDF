"""
JSON writer for exporting structured PDF content.
"""
import json
import os
import datetime
import logging
import jsonschema
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class JSONWriter:
    """Write structured PDF content to JSON files."""
    
    def __init__(self, schema_path: Optional[str] = None):
        """
        Initialize JSON writer.
        
        Args:
            schema_path: Path to JSON schema file (optional)
        """
        self.schema = None
        if schema_path and os.path.exists(schema_path):
            try:
                with open(schema_path, "r") as f:
                    self.schema = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load schema: {str(e)}")
    
    def validate_json(self, data: Dict[str, Any]) -> bool:
        """
        Validate JSON against schema.
        
        Args:
            data: JSON data to validate
            
        Returns:
            True if validation passed, False otherwise
        """
        if not self.schema:
            logger.warning("No schema loaded, skipping validation")
            return True
            
        try:
            jsonschema.validate(instance=data, schema=self.schema)
            return True
        except jsonschema.exceptions.ValidationError as e:
            logger.error(f"JSON validation failed: {str(e)}")
            return False
    
    def write(self, output_path: str, pdf_data: Dict[str, Any], validate: bool = True) -> bool:
        """
        Write structured PDF data to JSON file.
        
        Args:
            output_path: Path to write JSON file
            pdf_data: Structured PDF data
            validate: Whether to validate against schema
            
        Returns:
            True if write successful, False otherwise
        """
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Validate if requested
        if validate and not self.validate_json(pdf_data):
            logger.error("JSON validation failed, writing anyway")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(pdf_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to write JSON: {str(e)}")
            return False
