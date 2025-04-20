"""Bible data validation module."""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass
class ValidationConfig:
    """Configuration for bible data validation."""

    required_books: List[str] = None
    min_verses_per_book: int = 10
    required_fields: List[str] = None

    def __post_init__(self):
        """Set default values if none provided."""
        if self.required_books is None:
            self.required_books = [
                "Genesis",
                "Exodus",
                "Psalms",
                "Isaiah",
                "Matthew",
                "John",
                "Romans",
                "Revelation",
            ]
        if self.required_fields is None:
            self.required_fields = ["book", "chapter", "verse", "text"]


class BibleDataValidator:
    """Validator for biblical data integrity."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize validator with optional config path."""
        self.project_root = Path(__file__).parent.parent.parent
        self.raw_dir = self.project_root / "data/raw/bibles"
        self.processed_dir = self.project_root / "data/processed"

        # Load config
        if config_path:
            with open(config_path) as f:
                config_data = json.load(f)
        else:
            config_data = {}

        self.config = ValidationConfig(**config_data)
        self.logger = logging.getLogger(__name__)

    def check_raw_data(self) -> bool:
        """Verify raw bible data files exist and are readable."""
        try:
            if not self.raw_dir.exists():
                self.logger.error(f"Raw data directory not found: {self.raw_dir}")
                return False

            bible_files = list(self.raw_dir.glob("*.json"))
            if not bible_files:
                bible_files = list(self.raw_dir.glob("*.xml"))

            if not bible_files:
                self.logger.error("No bible data files found in raw directory")
                return False

            # Try to read each file
            for file_path in bible_files:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        if file_path.suffix == ".json":
                            json.load(f)
                except Exception as e:
                    self.logger.error(f"Failed to read {file_path}: {str(e)}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Raw data verification failed: {str(e)}")
            return False

    def check_processed_data(self) -> bool:
        """Verify processed bible data files."""
        try:
            processed_file = self.processed_dir / "bible.db"
            if not processed_file.exists():
                processed_file = self.processed_dir / "bible.json"

            if not processed_file.exists():
                self.logger.error("No processed bible data found")
                return False

            if processed_file.suffix == ".json":
                with open(processed_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Check required books
                found_books = set(data.keys())
                missing_books = set(self.config.required_books) - found_books
                if missing_books:
                    self.logger.error(f"Missing required books: {missing_books}")
                    return False

                # Check verse counts
                for book in self.config.required_books:
                    verse_count = sum(len(chapter) for chapter in data[book].values())
                    if verse_count < self.config.min_verses_per_book:
                        self.logger.error(
                            f"Insufficient verses in {book}: {verse_count}"
                        )
                        return False

            return True

        except Exception as e:
            self.logger.error(f"Processed data verification failed: {str(e)}")
            return False

    def verify_verse_references(self) -> bool:
        """Verify verse reference integrity."""
        try:
            verse_file = self.processed_dir / "verse_references.json"
            if not verse_file.exists():
                # Not critical if missing
                return True

            with open(verse_file, "r", encoding="utf-8") as f:
                refs = json.load(f)

            # Verify reference format
            for ref in refs:
                if not all(field in ref for field in ["book", "chapter", "verse"]):
                    self.logger.error(f"Invalid verse reference format: {ref}")
                    return False

                # Verify chapter and verse are integers
                try:
                    int(ref["chapter"])
                    int(ref["verse"])
                except (ValueError, TypeError):
                    self.logger.error(f"Invalid chapter/verse numbers in {ref}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Verse reference verification failed: {str(e)}")
            return False
