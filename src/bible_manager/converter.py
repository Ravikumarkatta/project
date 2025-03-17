# src/bible_manager/converter.py
"""
Bible format converter module for Bible-AI.

This module provides utilities for converting between different Bible text formats
such as OSIS, USFM, JSON, plain text, and CSV, with validation and streaming support.
"""

import os
import json
import re
import csv
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Union, Any, Tuple, Iterator
from pathlib import Path
import logging

# Project-specific imports with fallbacks
try:
    from src.utils.logger import get_logger
    from src.theology.validator import TheologicalValidator  # For theological checks
except ImportError:
    import logging
    get_logger = lambda name: logging.getLogger(name)
    TheologicalValidator = None

# Initialize module logger
logger = get_logger("bible_converter")

class BibleConverter:
    """
    Converts Bible texts between different formats.

    Supported formats:
    - USFM (Unified Standard Format Markers)
    - OSIS (Open Scripture Information Standard)
    - JSON
    - Plain text
    - CSV

    Attributes:
        config: Configuration dictionary.
        book_codes: Mapping of Bible book codes to names.
        validator: TheologicalValidator instance for content validation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Bible converter.

        Args:
            config_path: Optional path to converter configuration file.
        """
        self.config = {}
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                logger.info(f"Loaded converter configuration from {config_path}")
            except Exception as e:
                logger.error(f"Failed to load converter configuration: {e}")

        # Default book name mappings
        self.book_codes = {
            # Old Testament
            "GEN": "Genesis", "EXO": "Exodus", "LEV": "Leviticus", "NUM": "Numbers",
            "DEU": "Deuteronomy", "JOS": "Joshua", "JDG": "Judges", "RUT": "Ruth",
            "1SA": "1 Samuel", "2SA": "2 Samuel", "1KI": "1 Kings", "2KI": "2 Kings",
            "1CH": "1 Chronicles", "2CH": "2 Chronicles", "EZR": "Ezra", "NEH": "Nehemiah",
            "EST": "Esther", "JOB": "Job", "PSA": "Psalms", "PRO": "Proverbs",
            "ECC": "Ecclesiastes", "SNG": "Song of Solomon", "ISA": "Isaiah", "JER": "Jeremiah",
            "LAM": "Lamentations", "EZK": "Ezekiel", "DAN": "Daniel", "HOS": "Hosea",
            "JOL": "Joel", "AMO": "Amos", "OBA": "Obadiah", "JON": "Jonah",
            "MIC": "Micah", "NAM": "Nahum", "HAB": "Habakkuk", "ZEP": "Zephaniah",
            "HAG": "Haggai", "ZEC": "Zechariah", "MAL": "Malachi",
            # New Testament
            "MAT": "Matthew", "MRK": "Mark", "LUK": "Luke", "JHN": "John",
            "ACT": "Acts", "ROM": "Romans", "1CO": "1 Corinthians", "2CO": "2 Corinthians",
            "GAL": "Galatians", "EPH": "Ephesians", "PHP": "Philippians", "COL": "Colossians",
            "1TH": "1 Thessalonians", "2TH": "2 Thessalonians", "1TI": "1 Timothy", "2TI": "2 Timothy",
            "TIT": "Titus", "PHM": "Philemon", "HEB": "Hebrews", "JAS": "James",
            "1PE": "1 Peter", "2PE": "2 Peter", "1JN": "1 John", "2JN": "2 John",
            "3JN": "3 John", "JUD": "Jude", "REV": "Revelation"
        }

        # Override with configuration if provided
        if "book_codes" in self.config:
            self.book_codes.update(self.config["book_codes"])

        # Initialize theological validator
        self.validator = TheologicalValidator(self.config.get("theology", {})) if TheologicalValidator else None
        logger.info("BibleConverter initialized with %d book codes", len(self.book_codes))

    def convert_file(self, input_path: str, output_path: str, input_format: Optional[str] = None, output_format: Optional[str] = None) -> bool:
        """
        Convert a Bible file from one format to another.

        Args:
            input_path: Path to the input file.
            output_path: Path to the output file.
            input_format: Format of the input file (auto-detected if None).
            output_format: Format of the output file (auto-detected if None).

        Returns:
            bool: True if conversion was successful, False otherwise.
        """
        # Auto-detect formats if not specified
        if not input_format:
            input_format = self._detect_format(input_path)
            logger.info(f"Detected input format: {input_format}")
        
        if not output_format:
            output_format = self._detect_format(output_path)
            logger.info(f"Detected output format: {output_format}")
        
        # Read the input file
        try:
            bible_data = self._read_file(input_path, input_format)
            valid, message = self._validate_bible_data(bible_data)
            if not valid:
                logger.error("Validation failed: %s", message)
                return False
        except Exception as e:
            logger.error(f"Failed to read input file {input_path}: {e}")
            return False
        
        # Write the output file
        try:
            self._write_file(bible_data, output_path, output_format)
            logger.info(f"Successfully converted {input_path} to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to write output file {output_path}: {e}")
            return False

    def _detect_format(self, file_path: str) -> str:
        """
        Detect the format of a Bible file based on its extension or content.

        Args:
            file_path: Path to the file.

        Returns:
            str: Detected format (usfm, osis, json, txt, csv).
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in ('.usfm', '.sfm'):
            return 'usfm'
        elif ext in ('.xml', '.osis'):
            return 'osis'
        elif ext == '.json':
            return 'json'
        elif ext == '.txt':
            return 'txt'
        elif ext == '.csv':
            return 'csv'
        
        # Content-based detection
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(1000)  # Read first 1000 characters
                if content.startswith('<?xml') and '<osis' in content:
                    return 'osis'
                elif '\\id ' in content or '\\c ' in content:
                    return 'usfm'
                elif content.startswith('{') and '"books":' in content:
                    return 'json'
                elif ',' in content and content.count('\n') > 0:
                    first_line = content.split('\n')[0]
                    if first_line.count(',') >= 2:
                        return 'csv'
                return 'txt'
        except Exception:
            return 'txt'

    def _read_file(self, file_path: str, format_type: str) -> Dict[str, Any]:
        """
        Read a Bible file in the specified format.

        Args:
            file_path: Path to the file.
            format_type: Format of the file.

        Returns:
            Dict: Structured Bible data.
        """
        if format_type == 'usfm':
            return self._read_usfm(file_path)
        elif format_type == 'osis':
            return self._read_osis(file_path)
        elif format_type == 'json':
            return self._read_json(file_path)
        elif format_type == 'csv':
            return self._read_csv(file_path)
        elif format_type == 'txt':
            return self._read_txt(file_path)
        else:
            raise ValueError(f"Unsupported input format: {format_type}")

    def _write_file(self, bible_data: Dict[str, Any], file_path: str, format_type: str) -> None:
        """
        Write Bible data to a file in the specified format.

        Args:
            bible_data: Structured Bible data.
            file_path: Path to the output file.
            format_type: Format of the output file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        if format_type == 'usfm':
            self._write_usfm(bible_data, file_path)
        elif format_type == 'osis':
            self._write_osis(bible_data, file_path)
        elif format_type == 'json':
            self._write_json(bible_data, file_path)
        elif format_type == 'csv':
            self._write_csv(bible_data, file_path)
        elif format_type == 'txt':
            self._write_txt(bible_data, file_path)
        else:
            raise ValueError(f"Unsupported output format: {format_type}")

    def _validate_bible_data(self, bible_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate the structure and theological content of Bible data.

        Args:
            bible_data: Structured Bible data.

        Returns:
            Tuple[bool, str]: (Validation status, message).
        """
        try:
            if not isinstance(bible_data, dict) or "books" not in bible_data:
                return False, "Invalid Bible structure"

            if not bible_data["books"]:
                return False, "No books found"

            for book in bible_data["books"]:
                if "code" not in book or "chapters" not in book:
                    return False, f"Invalid book structure: {book}"
                if book["code"] not in self.book_codes:
                    logger.warning("Unknown book code: %s", book["code"])
                for chapter in book["chapters"]:
                    if "number" not in chapter or "verses" not in chapter:
                        return False, f"Invalid chapter in book {book['code']}"
                    if not chapter["verses"]:
                        return False, f"No verses in chapter {chapter['number']} of book {book['code']}"
                    for verse in chapter["verses"]:
                        if "number" not in verse or "text" not in verse or not verse["text"].strip():
                            return False, f"Invalid verse in book {book['code']}, chapter {chapter['number']}"

            # Theological validation
            if self.validator:
                score = self.validator.validate(bible_data)
                min_score = self.config.get("theology", {}).get("min_score", 0.9)
                if score < min_score:
                    return False, f"Theological score too low: {score} (minimum {min_score})"

            return True, "Validation passed"
        except Exception as e:
            return False, f"Validation failed: {str(e)}"

    def _read_usfm(self, file_path: str) -> Dict[str, Any]:
        """
        Read a USFM Bible file with streaming support.

        Args:
            file_path: Path to the USFM file.

        Returns:
            Dict: Structured Bible data.
        """
        bible_data = {
            "type": "usfm",
            "metadata": {},
            "books": []
        }
        
        current_book = None
        current_chapter = None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Book identification
                if line.startswith('\\id '):
                    book_code = line.split()[1]
                    book_name = self.book_codes.get(book_code, book_code)
                    current_book = {
                        "code": book_code,
                        "name": book_name,
                        "chapters": []
                    }
                    bible_data["books"].append(current_book)
                    continue

                # Metadata (headers)
                if current_book and not current_book["chapters"]:
                    for tag in ["\\h", "\\toc1", "\\toc2", "\\toc3"]:
                        if line.startswith(tag):
                            key = tag.replace("\\", "")
                            value = line[len(tag):].strip()
                            bible_data["metadata"][key] = value

                # Chapter
                if line.startswith('\\c '):
                    chapter_num = int(line.split()[1])
                    current_chapter = {
                        "number": chapter_num,
                        "verses": []
                    }
                    if current_book:
                        current_book["chapters"].append(current_chapter)
                    continue

                # Verse
                if line.startswith('\\v '):
                    parts = line.split(' ', 2)
                    if len(parts) >= 3:
                        verse_num = int(parts[1])
                        verse_text = parts[2]
                        verse_text = re.sub(r'\\[a-z]+\s*', '', verse_text).strip()  # Remove USFM tags
                        verse = {
                            "number": verse_num,
                            "text": verse_text
                        }
                        if current_chapter:
                            current_chapter["verses"].append(verse)

        return bible_data

    def _read_osis(self, file_path: str) -> Dict[str, Any]:
        """
        Read an OSIS Bible file.

        Args:
            file_path: Path to the OSIS file.

        Returns:
            Dict: Structured Bible data.
        """
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        bible_data = {
            "type": "osis",
            "metadata": {},
            "books": []
        }
        
        # Extract metadata
        osisIDWork = root.get('osisIDWork')
        if osisIDWork:
            bible_data["metadata"]["osisIDWork"] = osisIDWork
        
        work_elements = root.findall('.//{http://www.bibletechnologies.net/2003/OSIS/namespace}work')
        for work in work_elements:
            work_id = work.get('osisID')
            if work_id:
                bible_data["metadata"]["work_id"] = work_id
            title = work.find('.//{http://www.bibletechnologies.net/2003/OSIS/namespace}title')
            if title is not None and title.text:
                bible_data["metadata"]["title"] = title.text
        
        # Extract books
        books = root.findall('.//{http://www.bibletechnologies.net/2003/OSIS/namespace}div[@type="book"]')
        for book in books:
            book_osisID = book.get('osisID')
            book_code = book_osisID.split('.')[-1] if book_osisID else "UNK"
            book_name = self.book_codes.get(book_code, book_code)
            
            current_book = {
                "code": book_code,
                "name": book_name,
                "osisID": book_osisID,
                "chapters": []
            }
            
            # Extract chapters
            chapters = book.findall('.//{http://www.bibletechnologies.net/2003/OSIS/namespace}chapter')
            for chapter in chapters:
                chapter_osisID = chapter.get('osisID')
                chapter_num = int(re.search(r'\d+', chapter_osisID.split('.')[-1]).group()) if chapter_osisID else 0
                
                current_chapter = {
                    "number": chapter_num,
                    "verses": []
                }
                
                # Extract verses
                verses = chapter.findall('.//{http://www.bibletechnologies.net/2003/OSIS/namespace}verse')
                for verse in verses:
                    verse_osisID = verse.get('osisID')
                    verse_num = int(re.search(r'\d+', verse_osisID.split('.')[-1]).group()) if verse_osisID else 0
                    verse_text = ''.join(verse.itertext()).strip()
                    
                    current_chapter["verses"].append({
                        "number": verse_num,
                        "text": verse_text
                    })
                
                current_book["chapters"].append(current_chapter)
            
            bible_data["books"].append(current_book)
        
        return bible_data

    def _read_json(self, file_path: str) -> Dict[str, Any]:
        """
        Read a Bible file in JSON format.

        Args:
            file_path: Path to the JSON file.

        Returns:
            Dict: Structured Bible data.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            bible_data = json.load(f)
            bible_data["type"] = "json"
            return bible_data

    def _read_csv(self, file_path: str) -> Dict[str, Any]:
        """
        Read a Bible file in CSV format (book,chapter,verse,text).

        Args:
            file_path: Path to the CSV file.

        Returns:
            Dict: Structured Bible data.
        """
        bible_data = {
            "type": "csv",
            "metadata": {},
            "books": []
        }
        
        book_dict: Dict[str, Any] = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                book_code = row.get("book", "UNK")
                book_name = self.book_codes.get(book_code, book_code)
                
                if book_code not in book_dict:
                    book_dict[book_code] = {
                        "code": book_code,
                        "name": book_name,
                        "chapters": {}
                    }
                
                chapter_num = int(row.get("chapter", 0))
                if chapter_num not in book_dict[book_code]["chapters"]:
                    book_dict[book_code]["chapters"][chapter_num] = {
                        "number": chapter_num,
                        "verses": []
                    }
                
                verse_num = int(row.get("verse", 0))
                verse_text = row.get("text", "")
                book_dict[book_code]["chapters"][chapter_num]["verses"].append({
                    "number": verse_num,
                    "text": verse_text
                })
        
        # Convert chapters to list
        for book in book_dict.values():
            book["chapters"] = list(book["chapters"].values())
            bible_data["books"].append(book)
        
        return bible_data

    def _read_txt(self, file_path: str) -> Dict[str, Any]:
        """
        Read a Bible file in plain text format (assumes verse-per-line with reference).

        Args:
            file_path: Path to the TXT file.

        Returns:
            Dict: Structured Bible data.
        """
        bible_data = {
            "type": "txt",
            "metadata": {},
            "books": []
        }
        
        book_dict: Dict[str, Any] = {}
        ref_pattern = re.compile(r'([A-Za-z\s\d]+)\s(\d+):(\d+)\s(.+)')
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                match = ref_pattern.match(line)
                if not match:
                    continue
                
                book_name, chapter_num, verse_num, text = match.groups()
                book_code = next((code for code, name in self.book_codes.items() if name.lower() == book_name.lower()), "UNK")
                
                if book_code not in book_dict:
                    book_dict[book_code] = {
                        "code": book_code,
                        "name": book_name,
                        "chapters": {}
                    }
                
                chapter_num = int(chapter_num)
                if chapter_num not in book_dict[book_code]["chapters"]:
                    book_dict[book_code]["chapters"][chapter_num] = {
                        "number": chapter_num,
                        "verses": []
                    }
                
                book_dict[book_code]["chapters"][chapter_num]["verses"].append({
                    "number": int(verse_num),
                    "text": text
                })
        
        for book in book_dict.values():
            book["chapters"] = list(book["chapters"].values())
            bible_data["books"].append(book)
        
        return bible_data

    def _write_usfm(self, bible_data: Dict[str, Any], file_path: str) -> None:
        """
        Write Bible data to a USFM file.

        Args:
            bible_data: Structured Bible data.
            file_path: Path to the output file.
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            # Write metadata
            for key, value in bible_data.get("metadata", {}).items():
                f.write(f"\\{key} {value}\n")
            
            for book in bible_data["books"]:
                f.write(f"\\id {book['code']}\n")
                f.write(f"\\h {book['name']}\n")
                for chapter in book["chapters"]:
                    f.write(f"\\c {chapter['number']}\n")
                    for verse in chapter["verses"]:
                        f.write(f"\\v {verse['number']} {verse['text']}\n")

    def _write_osis(self, bible_data: Dict[str, Any], file_path: str) -> None:
        """
        Write Bible data to an OSIS file.

        Args:
            bible_data: Structured Bible data.
            file_path: Path to the output file.
        """
        osis = ET.Element("osis", xmlns="http://www.bibletechnologies.net/2003/OSIS/namespace")
        osisText = ET.SubElement(osis, "osisText", osisIDWork=bible_data["metadata"].get("osisIDWork", "Bible"))
        
        # Add work metadata
        header = ET.SubElement(osisText, "header")
        work = ET.SubElement(header, "work")
        title = ET.SubElement(work, "title")
        title.text = bible_data["metadata"].get("title", "Bible")
        
        # Add books
        for book in bible_data["books"]:
            book_elem = ET.SubElement(osisText, "div", type="book", osisID=f"Bible.{book['code']}")
            for chapter in book["chapters"]:
                chapter_elem = ET.SubElement(book_elem, "chapter", osisID=f"Bible.{book['code']}.{chapter['number']}")
                for verse in chapter["verses"]:
                    verse_elem = ET.SubElement(chapter_elem, "verse", osisID=f"Bible.{book['code']}.{chapter['number']}.{verse['number']}")
                    verse_elem.text = verse['text']
        
        tree = ET.ElementTree(osis)
        tree.write(file_path, encoding='utf-8', xml_declaration=True)

    def _write_json(self, bible_data: Dict[str, Any], file_path: str) -> None:
        """
        Write Bible data to a JSON file.

        Args:
            bible_data: Structured Bible data.
            file_path: Path to the output file.
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(bible_data, f, indent=2, ensure_ascii=False)

    def _write_csv(self, bible_data: Dict[str, Any], file_path: str) -> None:
        """
        Write Bible data to a CSV file.

        Args:
            bible_data: Structured Bible data.
            file_path: Path to the output file.
        """
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["book", "chapter", "verse", "text"])
            for book in bible_data["books"]:
                for chapter in book["chapters"]:
                    for verse in chapter["verses"]:
                        writer.writerow([book["code"], chapter["number"], verse["number"], verse["text"]])

    def _write_txt(self, bible_data: Dict[str, Any], file_path: str) -> None:
        """
        Write Bible data to a plain text file.

        Args:
            bible_data: Structured Bible data.
            file_path: Path to the output file.
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            for book in bible_data["books"]:
                for chapter in book["chapters"]:
                    for verse in chapter["verses"]:
                        f.write(f"{book['name']} {chapter['number']}:{verse['number']} {verse['text']}\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert Bible files between formats for Bible-AI")
    parser.add_argument("--input", type=str, required=True, help="Path to the input Bible file")
    parser.add_argument("--output", type=str, required=True, help="Path to the output Bible file")
    parser.add_argument("--input-format", type=str, help="Input format (usfm, osis, json, txt, csv)")
    parser.add_argument("--output-format", type=str, help="Output format (usfm, osis, json, txt, csv)")
    parser.add_argument("--config", type=str, default="config/bible_sources.json", help="Path to configuration file")
    args = parser.parse_args()

    converter = BibleConverter(config_path=args.config)
    success = converter.convert_file(args.input, args.output, args.input_format, args.output_format)
    print(f"Conversion {'successful' if success else 'failed'}")
