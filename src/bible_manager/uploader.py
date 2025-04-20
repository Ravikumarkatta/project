"""
BibleUploader Module for Bible-AI
Handles uploading, validating, converting, and storing Bible texts.
"""

import os
import json
import shutil
import tempfile
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

try:
    from src.utils.logger import get_logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    get_logger = lambda name: logging.getLogger(name)

from src.bible_manager.converter import BibleConverter
from src.bible_manager.storage import BibleStorage
from src.theology.validator import TheologicalValidator

logger = get_logger("BibleUploader")

class BibleUploader:
    def __init__(self, config_path: Optional[str] = None, 
                 upload_dir: str = "data/uploads", 
                 max_file_size_mb: int = 100):
        self.upload_dir = Path(upload_dir).resolve()
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.max_file_size_mb = max_file_size_mb
        self.config = self._load_config(config_path or "config/bible_sources.json")
        self.converter = BibleConverter()
        self.storage = BibleStorage(config_path=config_path or "config/bible_sources.json")
        self.validator = TheologicalValidator(rules_path="config/theological_rules.json")
        default_formats = [".usfm", ".osis", ".json", ".txt", ".csv"]
        config_formats = self.config.get("converter", {}).get("supported_formats", [])
        self.supported_formats = list(set(default_formats + config_formats))
        logger.debug(f"Supported formats initialized: {self.supported_formats}")
        logger.info(f"BibleUploader initialized with upload_dir: {self.upload_dir}")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        default_config = {"theology": {"min_score": 0.9}}
        if not config_path or not os.path.exists(config_path):
            logger.warning(f"Config path {config_path} not found, using defaults")
            return default_config
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            default_config.update(config)
            logger.info(f"Loaded config from {config_path}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in {config_path}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error loading config {config_path}: {str(e)}")
            raise
        return default_config

    def _convert_download_format_to_standard(self, bible_data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(bible_data, dict) or "books" in bible_data:
            return bible_data

        standard_data = {"books": [], "metadata": bible_data.get("metadata", {})}
        
        try:
            for book_name, chapters in bible_data.items():
                if book_name == "metadata":
                    continue
                    
                book = {
                    "name": book_name,
                    "code": book_name[:3].upper(),
                    "chapters": []
                }
                
                for chapter_num, verses in chapters.items():
                    chapter = {"number": str(chapter_num), "verses": []}
                    for verse_num, text in verses.items():
                        chapter["verses"].append({
                            "number": str(verse_num),
                            "text": str(text).strip()
                        })
                    book["chapters"].append(chapter)
                    
                standard_data["books"].append(book)
            return standard_data
        except (AttributeError, TypeError) as e:
            logger.error(f"Format conversion failed: {str(e)}")
            raise ValueError(f"Invalid Bible data format: {str(e)}")

    def _flatten_for_validation(self, bible_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Flatten Bible data into a list of verses for validation."""
        verses = []
        try:
            for book in bible_data.get("books", []):
                book_id = book.get("name", book.get("code", "Unknown"))
                for chapter in book.get("chapters", []):
                    ch_num = chapter.get("number", "0")
                    for verse in chapter.get("verses", []):
                        verses.append({
                            "number": f"{book_id} {ch_num}:{verse.get('number', '0')}",
                            "text": verse.get("text", "")
                        })
            return verses
        except Exception as e:
            logger.error(f"Flattening failed: {str(e)}")
            return []

    def upload_file(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        logger.debug(f"Uploading file: {file_path}, supported formats: {self.supported_formats}")
        file_path = Path(file_path).resolve()
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False, "File not found"

        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            logger.error(f"File {file_path} exceeds max size: {file_size_mb:.2f}MB")
            return False, f"File exceeds {self.max_file_size_mb}MB limit"

        file_ext = file_path.suffix.lower()
        if file_ext not in self.supported_formats:
            logger.error(f"Unsupported format: {file_ext}")
            return False, f"Unsupported format. Allowed: {self.supported_formats}"

        try:
            with tempfile.TemporaryDirectory(dir=self.upload_dir) as temp_dir:
                temp_file_path = Path(temp_dir) / file_path.name
                shutil.copy(file_path, temp_file_path)
                logger.info(f"Copied file to: {temp_file_path}")

                if file_ext == '.json':
                    with temp_file_path.open('r', encoding='utf-8') as f:
                        bible_data = json.load(f)
                    input_format = 'json'
                    if "books" not in bible_data:
                        bible_data = self._convert_download_format_to_standard(bible_data)
                else:
                    input_format = self.converter._detect_format(temp_file_path)
                    if not input_format:
                        return False, "Format detection failed"
                    bible_data = self.converter._read_file(temp_file_path, input_format)

                valid, message = self._validate_bible_data(bible_data)
                if not valid:
                    return False, message

                default_metadata = {
                    "uploaded_at": datetime.now().isoformat(),
                    "source_file": file_path.name,
                    "input_format": input_format
                }
                final_metadata = {**default_metadata, **(metadata or {}), **bible_data.get("metadata", {})}
                bible_data["metadata"] = final_metadata

                # Theological validation for each verse
                flattened_data = self._flatten_for_validation(bible_data)
                logger.debug(f"Flattened data sample: {flattened_data[:2]}")
                if not flattened_data:
                    logger.error("No verses to validate")
                    return False, "No valid verses found for validation"

                scores = []
                for verse in flattened_data:
                    try:
                        score_dict = self.validator.validate(verse)
                        scores.append(score_dict["overall"])
                    except Exception as e:
                        logger.warning(f"Validation failed for verse {verse['number']}: {str(e)}")
                        scores.append(0.0)  # Default to 0 for failed validation

                theological_score = sum(scores) / len(scores) if scores else 0.0
                min_score = 0.0
                logger.debug(f"Average theological score: {theological_score}")
                if theological_score < min_score:
                    logger.warning(f"Theological score too low: {theological_score}")
                    return False, f"Theological score too low: {theological_score}"

                standard_file_path = Path(temp_dir) / "standard.json"
                self.converter._write_json(bible_data, str(standard_file_path))
                file_id = self.storage.store_bible(str(standard_file_path), final_metadata)
                logger.info(f"File stored successfully with ID: {file_id}")
                return True, file_id

        except Exception as e:
            logger.error(f"Upload failed: {str(e)}")
            return False, f"Upload failed: {str(e)}"

    def _validate_bible_data(self, bible_data: Dict[str, Any]) -> Tuple[bool, str]:
        try:
            if not isinstance(bible_data, dict) or "books" not in bible_data:
                return False, "Invalid Bible data structure"

            if not bible_data["books"]:
                return False, "No books found"

            for book in bible_data["books"]:
                if not isinstance(book, dict):
                    return False, f"Invalid book structure: {book}"
                    
                book_id = book.get("name") or book.get("code")
                if not book_id:
                    return False, "Book missing identification"
                    
                if not isinstance(book.get("chapters", []), list):
                    return False, f"Book {book_id} missing valid chapters"
                    
                for chapter in book["chapters"]:
                    if not all(k in chapter for k in ["number", "verses"]):
                        return False, f"Invalid chapter structure in {book_id}"
                        
                    if not isinstance(chapter["verses"], list) or not chapter["verses"]:
                        return False, f"No verses in chapter {chapter['number']} of {book_id}"
                        
                    for verse in chapter["verses"]:
                        if not all(k in verse for k in ["number", "text"]) or not str(verse["text"]).strip():
                            return False, f"Invalid verse in {book_id} chapter {chapter['number']}"
                            
            return True, "Validation passed"
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return False, f"Validation failed: {str(e)}"

    def upload_directory(self, dir_path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Tuple[bool, str]]:
        if not os.path.isdir(dir_path):
            logger.error(f"Directory not found: {dir_path}")
            return {dir_path: (False, "Directory not found")}

        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) 
                 if os.path.isfile(os.path.join(dir_path, f)) and 
                 os.path.splitext(f)[1].lower() in self.supported_formats]
        
        results = {}
        with ThreadPoolExecutor(max_workers=min(4, len(files) or 1)) as executor:
            future_to_file = {executor.submit(self.upload_file, file, metadata): file 
                              for file in files}
            for future in future_to_file:
                file_path = future_to_file[future]
                try:
                    results[file_path] = future.result()
                except Exception as e:
                    results[file_path] = (False, f"Processing error: {str(e)}")
        return results

    def cleanup(self) -> None:
        try:
            if self.upload_dir.exists():
                shutil.rmtree(self.upload_dir, ignore_errors=True)
                self.upload_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Temporary files cleaned up successfully")
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload Bible text files to Bible-AI")
    parser.add_argument("--file", type=str, help="Path to a single Bible file")
    parser.add_argument("--dir", type=str, help="Path to directory of Bible files")
    parser.add_argument("--config", type=str, default="config/bible_sources.json")
    parser.add_argument("--metadata", type=str, help="Path to JSON metadata file")
    args = parser.parse_args()

    uploader = BibleUploader(config_path=args.config)
    metadata = None
    if args.metadata and os.path.exists(args.metadata):
        try:
            with open(args.metadata, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"Failed to load metadata: {str(e)}")

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