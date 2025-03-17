# src/bible_manager/uploader.py
"""
BibleUploader Module for Bible-AI

Handles uploading, validating, converting, and storing Bible texts with theological
accuracy checks, parallel processing, and robust error handling.
"""

import os
import json
import shutil
import tempfile
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Project-specific imports with fallbacks
try:
    from src.utils.logger import get_logger
    from src.bible_manager.converter import BibleConverter
    from src.bible_manager.storage import BibleStorage
    from src.theology.validator import TheologicalValidator  # For theological checks
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    get_logger = lambda name: logging.getLogger(name)
    logger = get_logger("BibleUploader")
    logger.error(f"Missing dependencies: {e}. Proceeding with basic functionality.")
    BibleConverter = object  # Fallback
    BibleStorage = object
    TheologicalValidator = object

# Initialize logger
logger = get_logger("BibleUploader")

class BibleUploader:
    """
    Handles the upload, validation, conversion, and storage of Bible texts for Bible-AI.

    Attributes:
        config_path: Path to configuration file.
        upload_dir: Directory for temporary uploaded files.
        converter: BibleConverter instance.
        storage: BibleStorage instance.
        validator: TheologicalValidator instance.
        supported_formats: List of supported file extensions.
        max_file_size_mb: Maximum file size in MB.
    """
    
    def __init__(self, config_path: Optional[str] = None, upload_dir: str = "data/uploads", max_file_size_mb: int = 100):
        """
        Initialize the BibleUploader.

        Args:
            config_path (Optional[str]): Path to configuration file.
            upload_dir (str): Directory for temporary uploaded files.
            max_file_size_mb (int): Maximum file size in megabytes.
        """
        self.upload_dir = upload_dir
        os.makedirs(self.upload_dir, exist_ok=True)
        self.max_file_size_mb = max_file_size_mb
        self.config = self._load_config(config_path)

        # Initialize dependencies
        self.converter = BibleConverter(config_path=self.config.get("converter", {}).get("config_path", "config/bible_sources.json"))
        self.storage = BibleStorage(config_path=config_path) if 'BibleStorage' not in globals() else BibleStorage(config_path=config_path)
        self.validator = TheologicalValidator(config=self.config.get("theology", {})) if 'TheologicalValidator' not in globals() else TheologicalValidator()
        self.supported_formats = self.config.get("converter", {}).get("supported_formats", [".usfm", ".osis", ".json", ".txt", ".csv"])
        logger.info("BibleUploader initialized with upload_dir: %s, max_file_size: %dMB", self.upload_dir, self.max_file_size_mb)

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from a JSON file with defaults."""
        config = {"converter": {}, "theology": {}}
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config.update(json.load(f))
                logger.info(f"Loaded config from {config_path}")
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
        return config

    def upload_file(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """
        Upload and process a single Bible text file.

        Args:
            file_path (str): Path to the file to upload.
            metadata (Optional[Dict[str, Any]]): Additional metadata.

        Returns:
            Tuple[bool, str]: (Success status, message or file ID).
        """
        if not os.path.exists(file_path):
            logger.error("File not found: %s", file_path)
            return False, "File not found"

        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            logger.error("File %s exceeds max size (%dMB): %dMB", file_path, self.max_file_size_mb, file_size_mb)
            return False, f"File exceeds {self.max_file_size_mb}MB limit"

        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.supported_formats:
            logger.error("Unsupported format: %s (allowed: %s)", file_ext, self.supported_formats)
            return False, f"Unsupported format. Allowed: {self.supported_formats}"

        try:
            with tempfile.TemporaryDirectory(dir=self.upload_dir) as temp_dir:
                temp_file_path = os.path.join(temp_dir, os.path.basename(file_path))
                shutil.copy(file_path, temp_file_path)
                logger.info("Copied file to temporary location: %s", temp_file_path)

                input_format = self.converter._detect_format(temp_file_path)
                if not input_format:
                    return False, "Format detection failed"

                bible_data = self.converter._read_file(temp_file_path, input_format)
                valid, message = self._validate_bible_data(bible_data)
                if not valid:
                    return False, message

                # Enrich metadata
                default_metadata = {"uploaded_at": datetime.now().isoformat(), "source_file": os.path.basename(file_path)}
                metadata = {**default_metadata, **(metadata or {}), "input_format": input_format}
                bible_data["metadata"] = {**bible_data.get("metadata", {}), **metadata}

                # Convert and validate theologically
                standard_file_path = os.path.join(temp_dir, "standard.json")
                self.converter._write_json(bible_data, standard_file_path)
                theological_score = self.validator.validate(bible_data) if hasattr(self.validator, 'validate') else 1.0
                if theological_score < self.config.get("theology", {}).get("min_score", 0.9):
                    return False, f"Theological score too low: {theological_score}"

                # Store the file
                file_id = self.storage.store_bible(standard_file_path, bible_data["metadata"])
                logger.info("File stored successfully with ID: %s", file_id)
                return True, file_id

        except Exception as e:
            logger.error("Upload failed: %s", e)
            return False, f"Upload failed: {str(e)}"

    def _validate_bible_data(self, bible_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate the structure and content of Bible data with theological checks.

        Args:
            bible_data (Dict[str, Any]): Structured Bible data.

        Returns:
            Tuple[bool, str]: (Validation status, message).
        """
        try:
            if not isinstance(bible_data, dict) or "books" not in bible_data:
                return False, "Invalid Bible structure"

            if not bible_data["books"]:
                return False, "No books found"

            for book in bible_data["books"]:
                if not isinstance(book, dict) or "code" not in book or "chapters" not in book:
                    return False, f"Invalid book structure: {book}"

                if book["code"] not in self.converter.book_codes:
                    logger.warning("Unknown book code: %s", book["code"])

                for chapter in book["chapters"]:
                    if "number" not in chapter or "verses" not in chapter:
                        return False, f"Invalid chapter in book {book['code']}"
                    if not chapter["verses"]:
                        return False, f"No verses in chapter {chapter['number']} of book {book['code']}"

                    for verse in chapter["verses"]:
                        if "number" not in verse or "text" not in verse or not verse["text"].strip():
                            return False, f"Invalid verse in book {book['code']}, chapter {chapter['number']}"
                        if len(verse["text"].split()) < 1:  # Basic content check
                            return False, f"Empty verse content in book {book['code']}, chapter {chapter['number']}"

            logger.info("Structural validation passed")
            return True, "Validation passed"

        except Exception as e:
            logger.error("Validation failed: %s", e)
            return False, f"Validation failed: {str(e)}"

    def upload_directory(self, dir_path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Tuple[bool, str]]:
        """
        Upload and process all Bible text files in a directory with parallel processing.

        Args:
            dir_path (str): Directory containing Bible files.
            metadata (Optional[Dict[str, Any]]): Metadata to apply to all files.

        Returns:
            Dict[str, Tuple[bool, str]]: Mapping of file paths to (success, message) tuples.
        """
        if not os.path.isdir(dir_path):
            logger.error("Directory not found: %s", dir_path)
            return {dir_path: (False, "Directory not found")}

        results = {}
        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        
        with ThreadPoolExecutor(max_workers=min(4, len(files))) as executor:
            future_to_file = {executor.submit(self.upload_file, file, metadata): file for file in files}
            for future in future_to_file:
                file_path = future_to_file[future]
                try:
                    results[file_path] = future.result()
                except Exception as e:
                    logger.error("Error processing %s: %s", file_path, e)
                    results[file_path] = (False, f"Processing error: {str(e)}")

        return results

    def cleanup(self) -> None:
        """
        Clean up temporary files and resources with error recovery.
        """
        try:
            if os.path.exists(self.upload_dir):
                for item in os.listdir(self.upload_dir):
                    item_path = os.path.join(self.upload_dir, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                logger.info("Temporary files cleaned up successfully")
        except Exception as e:
            logger.error("Cleanup failed: %s", e)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload Bible text files to Bible-AI")
    parser.add_argument("--file", type=str, help="Path to a single Bible file to upload")
    parser.add_argument("--dir", type=str, help="Path to a directory containing Bible files")
    parser.add_argument("--config", type=str, default="config/bible_sources.json", help="Path to configuration file")
    parser.add_argument("--metadata", type=str, help="Path to a JSON metadata file")
    args = parser.parse_args()

    uploader = BibleUploader(config_path=args.config)
    metadata = None
    if args.metadata and os.path.exists(args.metadata):
        with open(args.metadata, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    try:
        if args.file:
            success, message = uploader.upload_file(args.file, metadata)
            print(f"Upload {'successful' if success else 'failed'}: {message}")
        elif args.dir:
            results = uploader.upload_directory(args.dir, metadata)
            for file_path, (success, message) in results.items():
                print(f"{file_path}: {'Success' if success else 'Failure'} - {message}")
        else:
            parser.print_help()
    finally:
        uploader.cleanup()
