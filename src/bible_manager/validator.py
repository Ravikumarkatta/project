# src/bible_manager/validator.py
"""Bible data validation module."""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

# Corrected import for Optional
from typing import Dict, List, Optional, Set


@dataclass
class ValidationConfig:
    """Configuration for bible data validation."""

    # 4.1: Use Optional for fields initially set to None
    required_books: Optional[List[str]] = None
    min_verses_per_book: int = 10
    # 4.1: Use Optional for fields initially set to None
    required_fields: Optional[List[str]] = None

    # 4.2: Add return type annotation -> None
    def __post_init__(self) -> None:
        """Set default values if none provided."""
        # Note: The logic here correctly handles the initial None values.
        # The Optional type hint makes the initial assignment type-correct.
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
        config_data = {}  # Initialize empty config
        if config_path:
            config_file = Path(config_path)  # Use Path object
            if config_file.exists():
                try:
                    with config_file.open("r", encoding="utf-8") as f:
                        config_data = json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    logging.getLogger(__name__).error(
                        f"Failed to load config from {config_path}: {e}. Using defaults."
                    )
            else:
                logging.getLogger(__name__).warning(
                    f"Config file not found at {config_path}. Using defaults."
                )
        # else: # No config path provided, use defaults implicitly by passing {}

        self.config = ValidationConfig(**config_data)
        self.logger = logging.getLogger(__name__)

    def check_raw_data(self) -> bool:
        """Verify raw bible data files exist and are readable."""
        try:
            if not self.raw_dir.exists():
                self.logger.error(f"Raw data directory not found: {self.raw_dir}")
                return False  # Correctly placed return

            # Prefer JSON, fallback to XML
            bible_files = list(self.raw_dir.glob("*.json"))
            if not bible_files:
                bible_files = list(self.raw_dir.glob("*.xml"))

            if not bible_files:
                self.logger.warning(
                    "No primary bible data files (.json, .xml) found in raw directory. Checking for .txt"
                )
                # Fallback check for .txt if needed, depending on expected raw formats
                bible_files = list(self.raw_dir.glob("*.txt"))
                if not bible_files:
                    self.logger.error(
                        "No suitable bible data files found in raw directory."
                    )
                    return False  # Correctly placed return

            # Try to read each file (basic check)
            readable = True
            for file_path in bible_files:
                try:
                    # Basic read check, specific parsing might happen elsewhere
                    with open(file_path, "r", encoding="utf-8") as f:
                        f.read(10)  # Read a small amount to check readability
                except Exception as e:
                    self.logger.error(f"Failed to read {file_path}: {str(e)}")
                    readable = False
                    # Decide whether to return immediately or check all files
                    # return False # Stricter: fail on first unreadable file

            return readable  # Return overall readability status

        except Exception as e:
            self.logger.error(f"Raw data verification failed unexpectedly: {str(e)}")
            return False  # Correctly placed return

    def check_processed_data(self) -> bool:
        """Verify processed bible data files."""
        try:
            # Determine expected processed file (adjust logic if multiple formats are possible)
            processed_file_json = (
                self.processed_dir / "bible.json"
            )  # Example primary format
            processed_file_db = self.processed_dir / "bible.db"  # Example alternative

            processed_file_to_check = None
            if processed_file_json.exists():
                processed_file_to_check = processed_file_json
            elif processed_file_db.exists():
                # If DB exists, validation might involve querying it, not loading JSON
                self.logger.info(
                    f"Found processed data as DB: {processed_file_db}. Skipping JSON validation logic."
                )
                # Add DB-specific validation here if needed
                return True  # Assuming DB presence is sufficient for this check
            else:
                self.logger.error(
                    f"No processed bible data found at expected locations ({processed_file_json}, {processed_file_db})"
                )
                return False  # Correctly placed return

            # Proceed with JSON validation if bible.json was found
            if processed_file_to_check and processed_file_to_check.suffix == ".json":
                with open(processed_file_to_check, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Ensure config lists are not None before proceeding
                if self.config.required_books is None:
                    self.logger.error(
                        "Validation config error: required_books is None."
                    )
                    return False

                # Check required books (assuming data is dict {book: {chapter: {verse: text}}})
                found_books = set(data.keys())
                missing_books = set(self.config.required_books) - found_books
                if missing_books:
                    self.logger.error(
                        f"Missing required books in processed data: {missing_books}"
                    )
                    return False  # Correctly placed return

                # Check verse counts
                for book in self.config.required_books:
                    if book not in data:
                        # This case should be caught by missing_books check, but added for robustness
                        self.logger.error(
                            f"Required book '{book}' unexpectedly not found in data after initial check."
                        )
                        return False
                    # Calculate verse count for the book
                    verse_count = 0
                    if isinstance(data[book], dict):
                        verse_count = sum(
                            len(verses)
                            for verses in data[book].values()
                            if isinstance(verses, dict)
                        )
                    else:
                        self.logger.warning(
                            f"Unexpected data structure for book '{book}'. Expected dict of chapters."
                        )

                    if verse_count < self.config.min_verses_per_book:
                        self.logger.error(
                            f"Insufficient verses in {book}: found {verse_count}, minimum required {self.config.min_verses_per_book}"
                        )
                        return False  # Correctly placed return

            # If we reached here, all checks passed for the found processed file
            return True

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode processed JSON data: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(
                f"Processed data verification failed unexpectedly: {str(e)}"
            )
            return False  # Correctly placed return

    def verify_verse_references(self) -> bool:
        """Verify verse reference integrity if the reference file exists."""
        try:
            # Optional file: if it doesn't exist, it's not an error for this check
            verse_file = self.processed_dir / "verse_references.json"
            if not verse_file.exists():
                self.logger.info(
                    f"Verse reference file not found at {verse_file}. Skipping verification."
                )
                return True  # Not an error if the file is optional

            with open(verse_file, "r", encoding="utf-8") as f:
                refs = json.load(f)

            if not isinstance(refs, list):
                self.logger.error(
                    f"Verse reference file {verse_file} does not contain a list."
                )
                return False

            # Ensure config list is not None
            if self.config.required_fields is None:
                self.logger.error("Validation config error: required_fields is None.")
                return False

            # Verify reference format for each item in the list
            valid = True
            for i, ref in enumerate(refs):
                if not isinstance(ref, dict):
                    self.logger.error(
                        f"Invalid item type in verse references (index {i}): expected dict, got {type(ref)}"
                    )
                    valid = False
                    continue  # Skip to next item

                # Check for required fields
                missing_fields = [
                    field for field in self.config.required_fields if field not in ref
                ]
                if missing_fields:
                    self.logger.error(
                        f"Invalid verse reference format (index {i}): missing fields {missing_fields} in {ref}"
                    )
                    valid = False
                    continue

                # Verify chapter and verse are integers (or can be cast to int)
                try:
                    int(ref["chapter"])
                    # Verse might be optional depending on structure, handle if needed
                    if "verse" in ref and ref["verse"] is not None:
                        int(ref["verse"])
                except (ValueError, TypeError):
                    self.logger.error(
                        f"Invalid chapter/verse numbers (index {i}): non-integer values in {ref}"
                    )
                    valid = False

            return valid  # Return overall validity status

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode verse reference JSON: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(
                f"Verse reference verification failed unexpectedly: {str(e)}"
            )
            return False  # Correctly placed return
