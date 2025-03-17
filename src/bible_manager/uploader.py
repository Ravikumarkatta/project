"""
Bible text upload handling module for Bible-AI.

This module provides functionality for users to upload their own Bible texts
in various formats and integrate them into the Bible-AI system.
"""

import os
import shutil
import uuid
import json
from typing import Dict, List, Optional, Tuple, Union, BinaryIO
from pathlib import Path

# Import logger
try:
    from src.utils.logger import get_logger
except ImportError:
    # Fallback if the import path is different
    try:
        from utils.logger import get_logger
    except ImportError:
        import logging
        # Simple logger fallback if our custom logger isn't available
        get_logger = lambda name: logging.getLogger(name)

# Initialize module logger
logger = get_logger("bible_manager.uploader")


class BibleUploader:
    """
    Handles uploading Bible texts in various formats.
    Supports validation, processing, and integration of user-provided texts.
    """
    
    def __init__(self, upload_dir: Optional[str] = None, 
                 allowed_formats: Optional[List[str]] = None):
        """
        Initialize the Bible uploader with configuration.
        
        Args:
            upload_dir: Directory for temporary upload storage
            allowed_formats: List of allowed file formats
        """
        base_path = Path(os.path.abspath(__file__)).parent.parent.parent
        self.upload_dir = upload_dir or str(base_path / "data" / "uploads")
        
        # Create upload directory if it doesn't exist
        if not os.path.exists(self.upload_dir):
            os.makedirs(self.upload_dir, exist_ok=True)
            
        # Default allowed formats
        self.allowed_formats = allowed_formats or [
            "txt", "json", "xml", "usx", "usfm", "csv", "html"
        ]
        
        logger.info(f"Bible uploader initialized with upload directory: {self.upload_dir}")
        logger.info(f"Allowed formats: {', '.join(self.allowed_formats)}")
        
    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate if a file is in an accepted format and has valid content.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        # Check file existence
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"
        
        # Check file format
        file_ext = os.path.splitext(file_path)[1].lstrip('.').lower()
        if file_ext not in self.allowed_formats:
            return False, f"Unsupported file format: {file_ext}. Allowed formats: {', '.join(self.allowed_formats)}"
        
        # Check file size (limit to 50MB)
        if os.path.getsize(file_path) > 50 * 1024 * 1024:
            return False, "File too large. Maximum size is 50MB."
        
        # Basic content validation based on format
        try:
            if file_ext == "json":
                with open(file_path, 'r', encoding='utf-8') as f:
                    json.load(f)  # Attempt to parse JSON
            elif file_ext in ["xml", "usx", "usfm", "html"]:
                # Basic XML/markup validation
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if "<" not in content or ">" not in content:
                        return False, f"Invalid {file_ext.upper()} format. Missing markup elements."
            
            # For TXT and CSV, we just ensure they're readable
            with open(file_path, 'r', encoding='utf-8') as f:
                f.readline()  # Try to read first line
                
        except UnicodeDecodeError:
            return False, "Invalid file encoding. Please use UTF-8."
        except json.JSONDecodeError:
            return False, "Invalid JSON format."
        except Exception as e:
            return False, f"Error validating file: {str(e)}"
            
        return True, "File is valid."
    
    def upload_file(self, file_obj: BinaryIO, filename: str) -> Tuple[bool, str, Optional[str]]:
        """
        Upload a Bible file to the system.
        
        Args:
            file_obj: File-like object containing the uploaded data
            filename: Original filename
            
        Returns:
            Tuple of (success, message, file_id)
        """
        # Generate unique ID for this upload
        file_id = str(uuid.uuid4())
        
        # Create directory for this upload
        upload_path = os.path.join(self.upload_dir, file_id)
        os.makedirs(upload_path, exist_ok=True)
        
        # Save the file
        file_ext = os.path.splitext(filename)[1].lower()
        if not file_ext or file_ext[1:] not in self.allowed_formats:
            return False, f"Unsupported file format. Allowed formats: {', '.join(self.allowed_formats)}", None
        
        temp_path = os.path.join(upload_path, f"original{file_ext}")
        
        try:
            # Save uploaded file
            with open(temp_path, 'wb') as out_file:
                shutil.copyfileobj(file_obj, out_file)
                
            # Validate saved file
            is_valid, message = self.validate_file(temp_path)
            if not is_valid:
                # Clean up on validation failure
                shutil.rmtree(upload_path)
                return False, message, None
                
            # Log successful upload
            file_size = os.path.getsize(temp_path)
            logger.info(f"Successfully uploaded Bible file: {filename} ({file_size} bytes), ID: {file_id}")
            
            return True, "File uploaded successfully.", file_id
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(upload_path):
                shutil.rmtree(upload_path)
            logger.error(f"Error uploading file {filename}: {str(e)}")
            return False, f"Error uploading file: {str(e)}", None
    
    def get_upload_info(self, file_id: str) -> Dict:
        """
        Get information about an uploaded file.
        
        Args:
            file_id: Unique ID of the uploaded file
            
        Returns:
            Dictionary with upload information
        """
        upload_path = os.path.join(self.upload_dir, file_id)
        
        if not os.path.exists(upload_path):
            return {"exists": False, "message": "Upload not found"}
        
        # Find the original file
        files = [f for f in os.listdir(upload_path) if f.startswith("original")]
        if not files:
            return {"exists": False, "message": "Original file not found"}
        
        original_file = files[0]
        file_path = os.path.join(upload_path, original_file)
        
        # Get file stats
        stats = os.stat(file_path)
        
        return {
            "exists": True,
            "file_id": file_id,
            "filename": original_file,
            "size": stats.st_size,
            "upload_time": stats.st_mtime,
            "format": os.path.splitext(original_file)[1][1:]
        }
    
    def delete_upload(self, file_id: str) -> Tuple[bool, str]:
        """
        Delete an uploaded file.
        
        Args:
            file_id: Unique ID of the uploaded file
            
        Returns:
            Tuple of (success, message)
        """
        upload_path = os.path.join(self.upload_dir, file_id)
        
        if not os.path.exists(upload_path):
            return False, "Upload not found"
        
        try:
            shutil.rmtree(upload_path)
            logger.info(f"Deleted uploaded Bible file with ID: {file_id}")
            return True, "Upload deleted successfully"
        except Exception as e:
            logger.error(f"Error deleting upload {file_id}: {str(e)}")
            return False, f"Error deleting upload: {str(e)}"
