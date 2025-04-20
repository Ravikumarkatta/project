"""
Bible Converter for Bible-AI.

Converts Bible texts between USFM, JSON, CSV, XML, and TXT formats, with special handling for Gutenberg format.
"""

import csv
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BibleConverter:
    """Converts Bible texts between different formats in Bible-AI."""

    SUPPORTED_FORMATS = {"usfm", "json", "csv", "xml", "txt", "gutenberg"}

    book_name_to_code = {
        "Genesis": "GEN",
        "Exodus": "EXO",
        "Leviticus": "LEV",
        "Numbers": "NUM",
        "Deuteronomy": "DEU",
        "Joshua": "JOS",
        "Judges": "JDG",
        "Ruth": "RUT",
        "1 Samuel": "1SA",
        "2 Samuel": "2SA",
        "1 Kings": "1KI",
        "2 Kings": "2KI",
        "1 Chronicles": "1CH",
        "2 Chronicles": "2CH",
        "Ezra": "EZR",
        "Nehemiah": "NEH",
        "Esther": "EST",
        "Job": "JOB",
        "Psalms": "PSA",
        "Proverbs": "PRO",
        "Ecclesiastes": "ECC",
        "Song of Solomon": "SNG",
        "Isaiah": "ISA",
        "Jeremiah": "JER",
        "Lamentations": "LAM",
        "Ezekiel": "EZK",
        "Daniel": "DAN",
        "Hosea": "HOS",
        "Joel": "JOL",
        "Amos": "AMO",
        "Obadiah": "OBA",
        "Jonah": "JON",
        "Micah": "MIC",
        "Nahum": "NAH",
        "Habakkuk": "HAB",
        "Zephaniah": "ZEP",
        "Haggai": "HAG",
        "Zechariah": "ZEC",
        "Malachi": "MAL",
        "Matthew": "MAT",
        "Mark": "MRK",
        "Luke": "LUK",
        "John": "JHN",
        "Acts": "ACT",
        "Romans": "ROM",
        "1 Corinthians": "1CO",
        "2 Corinthians": "2CO",
        "Galatians": "GAL",
        "Ephesians": "EPH",
        "Philippians": "PHP",
        "Colossians": "COL",
        "1 Thessalonians": "1TH",
        "2 Thessalonians": "2TH",
        "1 Timothy": "1TI",
        "2 Timothy": "2TI",
        "Titus": "TIT",
        "Philemon": "PHM",
        "Hebrews": "HEB",
        "James": "JAS",
        "1 Peter": "1PE",
        "2 Peter": "2PE",
        "1 John": "1JN",
        "2 John": "2JN",
        "3 John": "3JN",
        "Jude": "JUD",
        "Revelation": "REV",
    }

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.logger = logger

    def convert(
        self, input_path: str, input_format: Optional[str] = None
    ) -> Dict[str, Any]:
        input_path = Path(input_path)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return {}
        input_format = input_format or self._detect_format(str(input_path))
        if not input_format:
            logger.error(f"Could not detect format for {input_path}")
            return {}
        if input_format not in self.SUPPORTED_FORMATS:
            logger.error(f"Unsupported input format: {input_format}")
            return {}
        try:
            logger.info(f"Starting conversion of {input_path} as {input_format}")
            bible_data = self._read_file(str(input_path), input_format)
            if not bible_data:
                logger.error(f"No valid data read from {input_path}")
                return {}
            if not self._validate_bible_data(bible_data):
                logger.error(f"Validation failed for data from {input_path}")
                return {}
            logger.info(f"Successfully converted {input_path} to structured data")
            return bible_data
        except Exception as e:
            logger.error(f"Conversion failed for {input_path}: {str(e)}")
            return {}

    def _detect_format(self, file_path: str) -> Optional[str]:
        ext = os.path.splitext(file_path)[1].lower()
        format_map = {
            ".usfm": "usfm",
            ".sfm": "usfm",
            ".json": "json",
            ".csv": "csv",
            ".xml": "xml",
            ".txt": "txt",
        }
        detected = format_map.get(ext)
        if detected:
            return detected
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read(1024)
                if content.startswith("<?xml"):
                    return "xml"
                if content.startswith("{"):
                    return "json"
                if "," in content.splitlines()[0]:
                    return "csv"
                if "\\id" in content:
                    return "usfm"
                if "*** START OF THE PROJECT GUTENBERG" in content:
                    return "gutenberg"
                return "txt"
        except Exception as e:
            logger.warning(
                f"Format detection failed for {file_path}: {str(e)}. Defaulting to 'txt'"
            )
            return "txt"

    def _read_file(self, file_path: str, format_type: str) -> Dict[str, Any]:
        readers = {
            "usfm": self._read_usfm,
            "json": self._read_json,
            "csv": self._read_csv,
            "xml": self._read_xml,
            "txt": self._read_txt,
            "gutenberg": self._read_gutenberg,
        }
        if format_type not in readers:
            raise ValueError(f"Unsupported format: {format_type}")
        return readers[format_type](file_path)

    def _validate_bible_data(self, bible_data: Dict[str, Any]) -> bool:
        if not isinstance(bible_data, dict) or "books" not in bible_data:
            logger.error("Invalid Bible structure: missing 'books' key")
            return False
        if not bible_data["books"]:
            logger.error("No books found in Bible data")
            return False
        for book in bible_data["books"]:
            if (
                "code" not in book
                or book["code"] not in self.book_name_to_code.values()
            ):
                logger.error(
                    f"Invalid or missing book code: {book.get('code', 'missing')}"
                )
                return False
            if not book.get("chapters"):
                logger.error(f"No chapters in book: {book.get('name', 'unknown')}")
                return False
            for chapter in book["chapters"]:
                if "verses" not in chapter or not chapter["verses"]:
                    logger.error(
                        f"No verses in chapter {chapter.get('number', 'unknown')} of book {book.get('name', 'unknown')}"
                    )
                    return False
        logger.info("Bible data validation passed")
        return True

    def _read_json(self, file_path: str) -> Dict[str, Any]:
        """
        Read a JSON Bible file and transform it into the standard structure.

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            Dict[str, Any]: Structured Bible data.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            # If already in standard format, return it
            if isinstance(raw_data, dict) and "books" in raw_data:
                return raw_data

            # Transform nested JSON (book -> chapter -> verse) into standard format
            bible_data = {
                "metadata": {"title": os.path.basename(file_path)},
                "books": [],
            }
            for book_name, chapters in raw_data.items():
                book_code = self.book_name_to_code.get(book_name, "UNKNOWN")
                if book_code == "UNKNOWN":
                    self.logger.warning(f"Unknown book name in JSON: {book_name}")
                    continue

                book = {"name": book_name, "code": book_code, "chapters": []}
                for chapter_num, verses in chapters.items():
                    chapter = {"number": int(chapter_num), "verses": []}
                    for verse_num, verse_text in verses.items():
                        chapter["verses"].append(
                            {"number": verse_num, "text": verse_text}
                        )
                    book["chapters"].append(chapter)
                bible_data["books"].append(book)

            if not bible_data["books"]:
                self.logger.error(f"No valid books parsed from JSON file {file_path}")
                return {}
            return bible_data
        except Exception as e:
            self.logger.error(f"Failed to read JSON file {file_path}: {str(e)}")
            return {}

    def _read_usfm(self, file_path: str) -> Dict[str, Any]:
        bible_data = {"metadata": {"title": os.path.basename(file_path)}, "books": []}
        book_dict = {}
        current_book = None
        current_chapter = None

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("\\id "):
                    book_code = line.split()[1]
                    book_name = next(
                        (
                            name
                            for name, code in self.book_name_to_code.items()
                            if code == book_code
                        ),
                        None,
                    )
                    if book_name:
                        current_book = {
                            "name": book_name,
                            "code": book_code,
                            "chapters": [],
                        }
                        book_dict[book_name] = current_book
                    else:
                        self.logger.warning(f"Unknown book code in USFM: {book_code}")
                elif line.startswith("\\c "):
                    if current_book:
                        chapter_num = int(line.split()[1])
                        current_chapter = {"number": chapter_num, "verses": []}
                        current_book["chapters"].append(current_chapter)
                elif line.startswith("\\v "):
                    if current_chapter:
                        parts = line.split(" ", 2)
                        if len(parts) >= 3:
                            verse_num = parts[1]
                            verse_text = parts[2]
                            current_chapter["verses"].append(
                                {"number": verse_num, "text": verse_text}
                            )
                        else:
                            self.logger.warning(f"Malformed verse line in USFM: {line}")

        bible_data["books"] = list(book_dict.values())
        return bible_data

    def _read_csv(self, file_path: str) -> Dict[str, Any]:
        bible_data = {"metadata": {"title": os.path.basename(file_path)}, "books": []}
        book_dict = {}

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                book_idx = header.index("book")
                chapter_idx = header.index("chapter")
                verse_idx = header.index("verse")
                text_idx = header.index("text")

                for row in reader:
                    book_name = row[book_idx].title()
                    chapter_num = int(row[chapter_idx])
                    verse_num = row[verse_idx]
                    verse_text = row[text_idx]

                    book_code = self.book_name_to_code.get(book_name, "UNKNOWN")
                    if book_code == "UNKNOWN":
                        self.logger.warning(f"Unknown book name in CSV: {book_name}")
                        continue

                    if book_name not in book_dict:
                        book_dict[book_name] = {
                            "name": book_name,
                            "code": book_code,
                            "chapters": {},
                        }

                    if chapter_num not in book_dict[book_name]["chapters"]:
                        book_dict[book_name]["chapters"][chapter_num] = {
                            "number": chapter_num,
                            "verses": [],
                        }

                    book_dict[book_name]["chapters"][chapter_num]["verses"].append(
                        {"number": verse_num, "text": verse_text}
                    )

            for book in book_dict.values():
                book["chapters"] = list(book["chapters"].values())
            bible_data["books"] = list(book_dict.values())
            return bible_data
        except Exception as e:
            self.logger.error(f"Failed to read CSV file {file_path}: {str(e)}")
            return {}

    def _read_xml(self, file_path: str) -> Dict[str, Any]:
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            bible_data = {
                "metadata": {"title": root.get("title", os.path.basename(file_path))},
                "books": [],
            }

            for book_elem in root.findall("book"):
                book_name = book_elem.get("name")
                book_code = self.book_name_to_code.get(book_name, "UNKNOWN")
                if book_code == "UNKNOWN":
                    self.logger.warning(f"Unknown book name in XML: {book_name}")
                    continue

                book = {"name": book_name, "code": book_code, "chapters": []}

                for chapter_elem in book_elem.findall("chapter"):
                    chapter_num = int(chapter_elem.get("number"))
                    chapter = {"number": chapter_num, "verses": []}

                    for verse_elem in chapter_elem.findall("verse"):
                        verse_num = verse_elem.get("number")
                        verse_text = verse_elem.text.strip() if verse_elem.text else ""
                        chapter["verses"].append(
                            {"number": verse_num, "text": verse_text}
                        )

                    book["chapters"].append(chapter)

                bible_data["books"].append(book)

            return bible_data
        except Exception as e:
            self.logger.error(f"Failed to read XML file {file_path}: {str(e)}")
            return {}

    def _read_txt(self, file_path: str) -> Dict[str, Any]:
        bible_data = {"metadata": {"title": os.path.basename(file_path)}, "books": []}
        book_dict = {}

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ref, text = line.split(" ", 1)
                    book_name, chap_verse = ref.rsplit(" ", 1)
                    chapter_num, verse_num = map(int, chap_verse.split(":"))
                    book_name = book_name.title()
                    book_code = self.book_name_to_code.get(book_name, "UNKNOWN")
                    if book_code == "UNKNOWN":
                        self.logger.warning(f"Unknown book name in TXT: {book_name}")
                        continue
                    if book_name not in book_dict:
                        book_dict[book_name] = {
                            "name": book_name,
                            "code": book_code,
                            "chapters": {},
                        }
                    if chapter_num not in book_dict[book_name]["chapters"]:
                        book_dict[book_name]["chapters"][chapter_num] = {
                            "number": chapter_num,
                            "verses": [],
                        }
                    book_dict[book_name]["chapters"][chapter_num]["verses"].append(
                        {"number": str(verse_num), "text": text}
                    )
                except ValueError:
                    self.logger.warning(
                        f"Skipping malformed TXT line in {file_path}: {line}"
                    )

        bible_data["books"] = [book for book in book_dict.values() if book["chapters"]]
        for book in bible_data["books"]:
            book["chapters"] = list(book["chapters"].values())
        return bible_data

    def _read_gutenberg(self, file_path: str) -> Dict[str, Any]:
        bible_data = {"metadata": {"title": "King James Version"}, "books": []}
        book_dict = {}
        in_content = False
        current_book_name = None

        pattern_old = re.compile(
            r"the\s+(first|second|third|fourth|fifth)?\s*book\s+of\s+[\w\s]+?:\s*called\s+(.+)",
            re.IGNORECASE,
        )
        pattern_book = re.compile(r"the\s+book\s+of\s+(.+)", re.IGNORECASE)
        pattern_gospel = re.compile(
            r"the\s+gospel\s+according\s+to\s+(.+)", re.IGNORECASE
        )
        pattern_simple = re.compile(r"^\s*([\w\s]+)$", re.IGNORECASE)

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if "*** START OF THE PROJECT GUTENBERG" in line:
                    in_content = True
                    continue
                if "*** END OF THE PROJECT GUTENBERG" in line:
                    break
                if not in_content:
                    continue

                lower_line = line.lower()
                new_book = None

                if "called" in lower_line:
                    match = pattern_old.search(line)
                    if match and match.group(2):
                        new_book = match.group(2).strip()
                elif "book of" in lower_line:
                    match = pattern_book.search(line)
                    if match:
                        new_book = match.group(1).strip()
                elif "gospel according to" in lower_line:
                    match = pattern_gospel.search(line)
                    if match:
                        new_book = match.group(1).strip()
                elif any(kw in lower_line for kw in ["revelation", "psalms", "acts"]):
                    match = pattern_simple.search(line)
                    if match:
                        candidate = match.group(1).strip()
                        if any(
                            kw in candidate.lower()
                            for kw in ["revelation", "psalms", "acts"]
                        ):
                            new_book = next(
                                (
                                    name
                                    for name in self.book_name_to_code
                                    if name.lower() in candidate.lower()
                                ),
                                None,
                            )

                if not new_book:
                    for book_name in self.book_name_to_code:
                        if book_name.lower() in lower_line and len(line.split()) <= 6:
                            new_book = book_name
                            break

                if new_book:
                    new_book = new_book.lower()
                    for num, word in [
                        ("1", "first"),
                        ("2", "second"),
                        ("3", "third"),
                        ("4", "fourth"),
                        ("5", "fifth"),
                    ]:
                        new_book = new_book.replace(word, num)
                    new_book = (
                        new_book.replace("saint ", "")
                        .replace("the prophet ", "")
                        .strip()
                    )
                    if "song of solomon" in new_book:
                        new_book = "Song of Solomon"
                    current_book_name = new_book.title()
                    book_code = self.book_name_to_code.get(current_book_name, "UNKNOWN")
                    if book_code == "UNKNOWN":
                        self.logger.warning(
                            f"Unknown book name in Gutenberg: {current_book_name}"
                        )
                        current_book_name = None
                        continue
                    if current_book_name not in book_dict:
                        book_dict[current_book_name] = {
                            "name": current_book_name,
                            "code": book_code,
                            "chapters": {},
                        }

                elif current_book_name:
                    line = line.lstrip()
                    if line and line[0].isdigit() and ":" in line[:5]:
                        try:
                            ref, text = line.split(" ", 1)
                            chapter_num, verse_num = map(int, ref.split(":"))
                            if (
                                chapter_num
                                not in book_dict[current_book_name]["chapters"]
                            ):
                                book_dict[current_book_name]["chapters"][
                                    chapter_num
                                ] = {"number": chapter_num, "verses": []}
                            book_dict[current_book_name]["chapters"][chapter_num][
                                "verses"
                            ].append({"number": str(verse_num), "text": text.strip()})
                        except Exception:
                            self.logger.warning(
                                f"Skipping malformed Gutenberg line in {file_path}: {line}"
                            )

            bible_data["books"] = [
                book for book in book_dict.values() if book["chapters"]
            ]
            for book in bible_data["books"]:
                book["chapters"] = list(book["chapters"].values())
            return bible_data

    def _write_json(self, bible_data: Dict[str, Any], output_path: str) -> bool:
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(bible_data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Successfully wrote JSON to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to write JSON file {output_path}: {str(e)}")
            return False
