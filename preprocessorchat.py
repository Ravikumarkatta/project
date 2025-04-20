#!/usr/bin/env python3
"""
Robust KJV Bible Text Preprocessor

This script processes raw Gutenberg KJV Bible text files into a standardized format
suitable for analysis, searching, and other applications. It handles:
- Gutenberg header/footer removal
- Standardized book title formatting
- Verse reference normalization
- Chapter handling
- Special formatting for Psalms
- Error detection and reporting for malformed verses
- Support for multiple output formats

Usage:
    python kjv_preprocessor.py --input raw_kjv.txt --output preprocessed_kjv.txt [OPTIONS]

Options:
    --format {text,json,csv}   Output format (default: text)
    --include-apocrypha        Include apocryphal books if present
    --fix-line-breaks          Attempt to fix irregular line breaks
    --validate                 Validate the structure of the output
    --verbose                  Enable verbose logging
"""

import re
import os
import json
import csv
import sys
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Set, Union, Any

# Constants
BIBLE_BOOKS = {
    # Old Testament - 39 books
    "Genesis": {"testament": "old", "abbrev": "Gen"},
    "Exodus": {"testament": "old", "abbrev": "Exod"},
    "Leviticus": {"testament": "old", "abbrev": "Lev"},
    "Numbers": {"testament": "old", "abbrev": "Num"},
    "Deuteronomy": {"testament": "old", "abbrev": "Deut"},
    "Joshua": {"testament": "old", "abbrev": "Josh"},
    "Judges": {"testament": "old", "abbrev": "Judg"},
    "Ruth": {"testament": "old", "abbrev": "Ruth"},
    "1 Samuel": {"testament": "old", "abbrev": "1Sam"},
    "2 Samuel": {"testament": "old", "abbrev": "2Sam"},
    "1 Kings": {"testament": "old", "abbrev": "1Kgs"},
    "2 Kings": {"testament": "old", "abbrev": "2Kgs"},
    "1 Chronicles": {"testament": "old", "abbrev": "1Chr"},
    "2 Chronicles": {"testament": "old", "abbrev": "2Chr"},
    "Ezra": {"testament": "old", "abbrev": "Ezra"},
    "Nehemiah": {"testament": "old", "abbrev": "Neh"},
    "Esther": {"testament": "old", "abbrev": "Esth"},
    "Job": {"testament": "old", "abbrev": "Job"},
    "Psalms": {"testament": "old", "abbrev": "Ps"},
    "Proverbs": {"testament": "old", "abbrev": "Prov"},
    "Ecclesiastes": {"testament": "old", "abbrev": "Eccl"},
    "Song of Solomon": {"testament": "old", "abbrev": "Song"},
    "Isaiah": {"testament": "old", "abbrev": "Isa"},
    "Jeremiah": {"testament": "old", "abbrev": "Jer"},
    "Lamentations": {"testament": "old", "abbrev": "Lam"},
    "Ezekiel": {"testament": "old", "abbrev": "Ezek"},
    "Daniel": {"testament": "old", "abbrev": "Dan"},
    "Hosea": {"testament": "old", "abbrev": "Hos"},
    "Joel": {"testament": "old", "abbrev": "Joel"},
    "Amos": {"testament": "old", "abbrev": "Amos"},
    "Obadiah": {"testament": "old", "abbrev": "Obad"},
    "Jonah": {"testament": "old", "abbrev": "Jonah"},
    "Micah": {"testament": "old", "abbrev": "Mic"},
    "Nahum": {"testament": "old", "abbrev": "Nah"},
    "Habakkuk": {"testament": "old", "abbrev": "Hab"},
    "Zephaniah": {"testament": "old", "abbrev": "Zeph"},
    "Haggai": {"testament": "old", "abbrev": "Hag"},
    "Zechariah": {"testament": "old", "abbrev": "Zech"},
    "Malachi": {"testament": "old", "abbrev": "Mal"},
    
    # New Testament - 27 books
    "Matthew": {"testament": "new", "abbrev": "Matt"},
    "Mark": {"testament": "new", "abbrev": "Mark"},
    "Luke": {"testament": "new", "abbrev": "Luke"},
    "John": {"testament": "new", "abbrev": "John"},
    "Acts": {"testament": "new", "abbrev": "Acts"},
    "Romans": {"testament": "new", "abbrev": "Rom"},
    "1 Corinthians": {"testament": "new", "abbrev": "1Cor"},
    "2 Corinthians": {"testament": "new", "abbrev": "2Cor"},
    "Galatians": {"testament": "new", "abbrev": "Gal"},
    "Ephesians": {"testament": "new", "abbrev": "Eph"},
    "Philippians": {"testament": "new", "abbrev": "Phil"},
    "Colossians": {"testament": "new", "abbrev": "Col"},
    "1 Thessalonians": {"testament": "new", "abbrev": "1Thess"},
    "2 Thessalonians": {"testament": "new", "abbrev": "2Thess"},
    "1 Timothy": {"testament": "new", "abbrev": "1Tim"},
    "2 Timothy": {"testament": "new", "abbrev": "2Tim"},
    "Titus": {"testament": "new", "abbrev": "Titus"},
    "Philemon": {"testament": "new", "abbrev": "Phlm"},
    "Hebrews": {"testament": "new", "abbrev": "Heb"},
    "James": {"testament": "new", "abbrev": "Jas"},
    "1 Peter": {"testament": "new", "abbrev": "1Pet"},
    "2 Peter": {"testament": "new", "abbrev": "2Pet"},
    "1 John": {"testament": "new", "abbrev": "1John"},
    "2 John": {"testament": "new", "abbrev": "2John"},
    "3 John": {"testament": "new", "abbrev": "3John"},
    "Jude": {"testament": "new", "abbrev": "Jude"},
    "Revelation": {"testament": "new", "abbrev": "Rev"},
    
    # Apocrypha - (if included)
    "Tobit": {"testament": "apocrypha", "abbrev": "Tob"},
    "Judith": {"testament": "apocrypha", "abbrev": "Jdt"},
    "Wisdom of Solomon": {"testament": "apocrypha", "abbrev": "Wis"},
    "Sirach": {"testament": "apocrypha", "abbrev": "Sir"},
    "Baruch": {"testament": "apocrypha", "abbrev": "Bar"},
    "Letter of Jeremiah": {"testament": "apocrypha", "abbrev": "LJe"},
    "1 Maccabees": {"testament": "apocrypha", "abbrev": "1Macc"},
    "2 Maccabees": {"testament": "apocrypha", "abbrev": "2Macc"},
    "1 Esdras": {"testament": "apocrypha", "abbrev": "1Esd"},
    "Prayer of Manasseh": {"testament": "apocrypha", "abbrev": "PrMan"},
    "Psalm 151": {"testament": "apocrypha", "abbrev": "Ps151"},
    "3 Maccabees": {"testament": "apocrypha", "abbrev": "3Macc"},
    "2 Esdras": {"testament": "apocrypha", "abbrev": "2Esd"},
    "4 Maccabees": {"testament": "apocrypha", "abbrev": "4Macc"},
}

BOOK_ALIASES = {
    "Psalm": "Psalms",
    "Psalter": "Psalms",
    "Songs": "Song of Solomon",
    "Canticles": "Song of Solomon",
    "Song of Songs": "Song of Solomon",
    "Apocalypse": "Revelation",
    "The Revelation": "Revelation",
    "Saint Matthew": "Matthew",
    "Saint Mark": "Mark",
    "Saint Luke": "Luke",
    "Saint John": "John",
    "The Acts": "Acts",
    "First Samuel": "1 Samuel",
    "Second Samuel": "2 Samuel",
    "First Kings": "1 Kings",
    "Second Kings": "2 Kings",
    "First Chronicles": "1 Chronicles",
    "Second Chronicles": "2 Chronicles",
    "First Corinthians": "1 Corinthians",
    "Second Corinthians": "2 Corinthians",
    "First Thessalonians": "1 Thessalonians",
    "Second Thessalonians": "2 Thessalonians",
    "First Timothy": "1 Timothy",
    "Second Timothy": "2 Timothy",
    "First Peter": "1 Peter",
    "Second Peter": "2 Peter",
    "First John": "1 John",
    "Second John": "2 John",
    "Third John": "3 John",
    "First Book of Moses": "Genesis",
    "Second Book of Moses": "Exodus",
    "Third Book of Moses": "Leviticus",
    "Fourth Book of Moses": "Numbers",
    "Fifth Book of Moses": "Deuteronomy",
}

# Statistics structure
class BibleStatistics:
    def __init__(self):
        self.books_found: Set[str] = set()
        self.chapters_per_book: Dict[str, int] = {}
        self.verses_per_chapter: Dict[str, Dict[int, int]] = {}
        self.total_verses: int = 0
        self.malformed_verses: List[Dict[str, Any]] = []

    def add_book(self, book_name: str) -> None:
        self.books_found.add(book_name)
        if book_name not in self.chapters_per_book:
            self.chapters_per_book[book_name] = 0
            self.verses_per_chapter[book_name] = {}

    def add_chapter(self, book_name: str, chapter_num: int) -> None:
        if book_name in self.chapters_per_book:
            if chapter_num > self.chapters_per_book[book_name]:
                self.chapters_per_book[book_name] = chapter_num
        if book_name not in self.verses_per_chapter:
            self.verses_per_chapter[book_name] = {}
        if chapter_num not in self.verses_per_chapter[book_name]:
            self.verses_per_chapter[book_name][chapter_num] = 0

    def add_verse(self, book_name: str, chapter_num: int, verse_num: int) -> None:
        self.total_verses += 1
        if book_name not in self.verses_per_chapter:
            self.verses_per_chapter[book_name] = {}
        if chapter_num not in self.verses_per_chapter[book_name]:
            self.verses_per_chapter[book_name][chapter_num] = 0
        if verse_num > self.verses_per_chapter[book_name][chapter_num]:
            self.verses_per_chapter[book_name][chapter_num] = verse_num

    def add_malformed_verse(self, line: str, reason: str) -> None:
        self.malformed_verses.append({"line": line, "reason": reason})

    def generate_report(self) -> Dict[str, Any]:
        return {
            "total_books": len(self.books_found),
            "books_found": sorted(list(self.books_found)),
            "total_verses": self.total_verses,
            "chapters_per_book": self.chapters_per_book,
            "malformed_verses_count": len(self.malformed_verses),
            "malformed_verses": self.malformed_verses[:10] if self.malformed_verses else []
        }

class BibleProcessor:
    def __init__(self, include_apocrypha: bool = False, fix_line_breaks: bool = False):
        self.include_apocrypha = include_apocrypha
        self.fix_line_breaks = fix_line_breaks
        self.stats = BibleStatistics()
        self.current_book: Optional[str] = None
        self.current_chapter: Optional[int] = None
        self.parsed_bible: Dict[str, Dict[int, Dict[int, str]]] = {}

    def setup_logging(self, verbose: bool) -> None:
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    def remove_gutenberg_wrappers(self, text: str) -> str:
        start_markers = [
            "*** START OF THE PROJECT GUTENBERG EBOOK",
            "*** START OF THIS PROJECT GUTENBERG EBOOK",
            "***START OF THE PROJECT GUTENBERG EBOOK",
            "*END*THE SMALL PRINT"
        ]
        start_index = -1
        for marker in start_markers:
            pos = text.find(marker)
            if pos != -1:
                if start_index == -1 or pos < start_index:
                    start_index = pos
        if start_index != -1:
            next_line_break = text.find('\n', start_index)
            if next_line_break != -1:
                text = text[next_line_break + 1:]
                logging.debug("Gutenberg start marker found and removed.")
            else:
                text = text[start_index:]
                logging.debug("Gutenberg start marker found but no line break after it.")
        else:
            logging.warning("No Gutenberg start marker found.")

        end_markers = [
            "*** END OF THE PROJECT GUTENBERG",
            "*** END OF THIS PROJECT GUTENBERG",
            "***END OF THE PROJECT GUTENBERG",
            "End of the Project Gutenberg"
        ]
        end_index = -1
        for marker in end_markers:
            pos = text.find(marker)
            if pos != -1:
                if end_index == -1 or pos < end_index:
                    end_index = pos
        if end_index != -1:
            text = text[:end_index]
            logging.debug("Gutenberg end marker found and removed.")
        else:
            logging.warning("No Gutenberg end marker found.")
        return text.strip()

    def normalize_book_name(self, name: str) -> Optional[str]:
        name = name.strip()
        if name in BIBLE_BOOKS:
            return name
        if name in BOOK_ALIASES:
            return BOOK_ALIASES[name]
        numbered_book_pattern = re.compile(r'^(\d+)\s*(\w+)$')
        match = numbered_book_pattern.match(name)
        if match:
            number, book = match.groups()
            number_word = {
                "1": "1", "2": "2", "3": "3",
                "I": "1", "II": "2", "III": "3",
                "First": "1", "Second": "2", "Third": "3"
            }.get(number)
            if number_word and f"{number_word} {book}" in BIBLE_BOOKS:
                return f"{number_word} {book}"
        lower_name = name.lower()
        for book in BIBLE_BOOKS:
            if book.lower().replace(" ", "") in lower_name.replace(" ", ""):
                logging.debug(f"Fuzzy matched '{name}' to '{book}'")
                return book
        logging.warning(f"Could not normalize book name: '{name}'")
        return None

    def identify_verse_reference(self, line: str) -> Optional[Tuple[int, int, str]]:
        patterns = [
            r'^(\d+):(\d+)\s+(.+)$',
            r'^(\d+)\s+(.+)$',
            r'^\[(\d+)\]\s+(.+)$'
        ]
        for pattern in patterns:
            match = re.match(pattern, line.strip())
            if match:
                groups = match.groups()
                if len(groups) == 3:
                    try:
                        chapter = int(groups[0])
                        verse = int(groups[1])
                        return chapter, verse, groups[2]
                    except ValueError:
                        self.stats.add_malformed_verse(line, "Invalid chapter/verse numbers")
                        return None
                elif len(groups) == 2:
                    if self.current_chapter is not None:
                        try:
                            verse = int(groups[0])
                            return self.current_chapter, verse, groups[1]
                        except ValueError:
                            self.stats.add_malformed_verse(line, "Invalid verse number")
                            return None
                    else:
                        self.stats.add_malformed_verse(line, "No current chapter established")
                        return None
        return None

    def identify_book_and_chapter_headers(self, line: str) -> Tuple[Optional[str], Optional[int]]:
        book_marker_match = re.match(r'^BOOK:\s*(.+)$', line.strip())
        if book_marker_match:
            book_name = self.normalize_book_name(book_marker_match.group(1))
            if book_name:
                return book_name, None
        book_title_patterns = [
            r'^(?:The\s+)?(?:Book\s+of\s+)?([A-Za-z\s1-3]+?)(?::\s*Called\s+.+)?$',
            r'^([A-Za-z\s1-3]+)\s+Chapter\s+(\d+)$',
            r'^THE\s+([A-Za-z\s1-3]+)$'
        ]
        for pattern in book_title_patterns:
            match = re.match(pattern, line.strip())
            if match:
                potential_book = match.group(1).strip()
                # Only use fallback if line length is short enough (e.g., under 40 characters)
                if len(line.strip()) < 40:
                    normalized_book = self.normalize_book_name(potential_book)
                    if normalized_book:
                        if len(match.groups()) > 1 and match.group(2):
                            try:
                                chapter = int(match.group(2))
                                return normalized_book, chapter
                            except ValueError:
                                return normalized_book, None
                        return normalized_book, None
        chapter_patterns = [
            r'^Chapter\s+(\d+)$',
            r'^CHAPTER\s+(\d+)$',
            r'^\s*(\d+)\s*$'
        ]
        for pattern in chapter_patterns:
            match = re.match(pattern, line.strip())
            if match:
                try:
                    chapter = int(match.group(1))
                    return None, chapter
                except ValueError:
                    return None, None
        return None, None

    def standardize_book_titles(self, text: str) -> str:
        lines = text.splitlines()
        result_lines = []
        processed_line_count = 0
        # Only use fallback on lines that are short (to avoid modifying verses)
        for i, line in enumerate(lines):
            original_line = line
            line = line.strip()
            if not line:
                result_lines.append("")
                continue
            book_name, chapter_num = self.identify_book_and_chapter_headers(line)
            if book_name:
                if book_name in BIBLE_BOOKS and (self.include_apocrypha or 
                                                 BIBLE_BOOKS[book_name]["testament"] != "apocrypha"):
                    result_lines.append(f"BOOK: {book_name}")
                    processed_line_count += 1
                    logging.debug(f"Standardized book title: {original_line} -> BOOK: {book_name}")
                else:
                    logging.debug(f"Skipping apocryphal or unknown book: {book_name}")
                if chapter_num:
                    result_lines.append(f"CHAPTER: {chapter_num}")
                    processed_line_count += 1
            elif chapter_num:
                result_lines.append(f"CHAPTER: {chapter_num}")
                processed_line_count += 1
                logging.debug(f"Standardized chapter header: {original_line} -> CHAPTER: {chapter_num}")
            else:
                # Fallback: if the line is short (under 40 chars) and contains a known book name
                if len(original_line.strip()) < 40:
                    lower_line = original_line.lower()
                    for book in BIBLE_BOOKS:
                        if book.lower() in lower_line:
                            result_lines.append(f"BOOK: {book}")
                            processed_line_count += 1
                            logging.debug(f"Fallback standardized '{original_line}' to BOOK: {book}")
                            break
                    else:
                        result_lines.append(original_line)
                else:
                    result_lines.append(original_line)
        logging.info(f"Standardized {processed_line_count} book/chapter headers")
        return "\n".join(result_lines)

    def process_verse_lines(self, text: str) -> str:
        lines = text.splitlines()
        result_lines = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                result_lines.append("")
                continue
            if line.startswith("BOOK:"):
                self.current_book = line[5:].strip()
                self.current_chapter = None
                self.stats.add_book(self.current_book)
                if self.current_book not in self.parsed_bible:
                    self.parsed_bible[self.current_book] = {}
                result_lines.append(line)
                continue
            if line.startswith("CHAPTER:"):
                try:
                    chapter_num = int(line[8:].strip())
                    self.current_chapter = chapter_num
                    if self.current_book:
                        self.stats.add_chapter(self.current_book, chapter_num)
                        if self.current_book not in self.parsed_bible:
                            self.parsed_bible[self.current_book] = {}
                        self.parsed_bible[self.current_book][chapter_num] = {}
                    result_lines.append(line)
                    continue
                except ValueError:
                    logging.warning(f"Invalid chapter number in line: {line}")
            verse_info = self.identify_verse_reference(line)
            if verse_info:
                chapter_num, verse_num, verse_text = verse_info
                if self.current_chapter != chapter_num:
                    self.current_chapter = chapter_num
                    if self.current_book:
                        result_lines.append(f"CHAPTER: {chapter_num}")
                        self.stats.add_chapter(self.current_book, chapter_num)
                        if self.current_book not in self.parsed_bible:
                            self.parsed_bible[self.current_book] = {}
                        if chapter_num not in self.parsed_bible[self.current_book]:
                            self.parsed_bible[self.current_book][chapter_num] = {}
                formatted_verse = f"{chapter_num}:{verse_num} {verse_text}"
                result_lines.append(formatted_verse)
                if self.current_book:
                    self.stats.add_verse(self.current_book, chapter_num, verse_num)
                    if self.current_book in self.parsed_bible:
                        if chapter_num not in self.parsed_bible[self.current_book]:
                            self.parsed_bible[self.current_book][chapter_num] = {}
                        self.parsed_bible[self.current_book][chapter_num][verse_num] = verse_text
                continue
            if self.fix_line_breaks and i > 0 and not line.startswith("BOOK:") and not line.startswith("CHAPTER:"):
                prev_line = lines[i-1].strip()
                if (prev_line and not re.match(r'^\d+:\d+', line) and 
                    (re.match(r'^\d+:\d+', prev_line) or (result_lines and re.match(r'^\d+:\d+', result_lines[-1])))):
                    result_lines[-1] += " " + line
                    logging.debug(f"Joined broken verse line: {line}")
                    continue
            result_lines.append(line)
        verse_count = sum(1 for line in result_lines if re.match(r'^\d+:\d+', line.strip()))
        logging.info(f"Processed {verse_count} verses")
        return "\n".join(result_lines)

    def validate_bible_structure(self) -> Dict[str, Any]:
        expected_book_count = len([b for b in BIBLE_BOOKS if self.include_apocrypha or 
                                     BIBLE_BOOKS[b]["testament"] != "apocrypha"])
        report = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "statistics": self.stats.generate_report()
        }
        books_found = len(self.stats.books_found)
        if books_found < expected_book_count:
            missing = [b for b in BIBLE_BOOKS if b not in self.stats.books_found and
                       (self.include_apocrypha or BIBLE_BOOKS[b]["testament"] != "apocrypha")]
            report["warnings"].append(f"Found {books_found} books out of {expected_book_count} expected")
            report["warnings"].append(f"Missing books: {', '.join(missing[:5])}")
            if len(missing) > 5:
                report["warnings"].append(f"... and {len(missing) - 5} more")
        if self.stats.total_verses < 30000:
            report["warnings"].append(f"Low verse count: {self.stats.total_verses} (expected ~31,000)")
            report["valid"] = False
        if self.stats.malformed_verses:
            report["warnings"].append(f"Found {len(self.stats.malformed_verses)} malformed verses")
        return report

    def write_output(self, text: str, output_file: str, output_format: str) -> None:
        if output_format == "text":
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
        elif output_format == "json":
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.parsed_bible, f, ensure_ascii=False, indent=2)
        elif output_format == "csv":
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Book", "Chapter", "Verse", "Text"])
                for book in sorted(self.parsed_bible.keys()):
                    for chapter in sorted(self.parsed_bible[book].keys()):
                        for verse in sorted(self.parsed_bible[book][chapter].keys()):
                            writer.writerow([book, chapter, verse, self.parsed_bible[book][chapter][verse]])
        else:
            logging.error(f"Unsupported output format: {output_format}")
            sys.exit(1)
        logging.info(f"Output written to {output_file} in {output_format} format.")

    def process_bible_text(self, input_file: str, output_file: str, output_format: str = "text",
                           validate: bool = False) -> Dict[str, Any]:
        logging.info(f"Reading input file: {input_file}")
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            logging.warning("UTF-8 decoding failed, trying latin-1 encoding")
            with open(input_file, 'r', encoding='latin-1') as f:
                text = f.read()
        text = self.remove_gutenberg_wrappers(text)
        text = self.standardize_book_titles(text)
        text = self.process_verse_lines(text)
        self.write_output(text, output_file, output_format)
        validation_report = None
        if validate:
            validation_report = self.validate_bible_structure()
            log_level = logging.WARNING if validation_report["warnings"] else logging.INFO
            logging.log(log_level, f"Validation complete: {len(validation_report['warnings'])} warnings")
        return {
            "input_file": input_file,
            "output_file": output_file,
            "format": output_format,
            "verses_processed": self.stats.total_verses,
            "books_processed": len(self.stats.books_found),
            "validation": validation_report
        }

    def format_parsed_bible_as_text(self) -> str:
        lines = []
        for book in sorted(self.parsed_bible.keys()):
            lines.append(f"BOOK: {book}")
            for chapter in sorted(self.parsed_bible[book].keys()):
                lines.append(f"CHAPTER: {chapter}")
                for verse in sorted(self.parsed_bible[book][chapter].keys()):
                    lines.append(f"{chapter}:{verse} {self.parsed_bible[book][chapter][verse]}")
        return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(
        description="Robust KJV Bible Text Preprocessor"
    )
    parser.add_argument("--input", required=True, help="Path to the raw Gutenberg KJV text file.")
    parser.add_argument("--output", required=True, help="Path to save the preprocessed output.")
    parser.add_argument("--format", choices=["text", "json", "csv"], default="text",
                        help="Output format (default: text)")
    parser.add_argument("--include-apocrypha", action="store_true", help="Include apocryphal books if present")
    parser.add_argument("--fix-line-breaks", action="store_true", help="Attempt to fix irregular line breaks")
    parser.add_argument("--validate", action="store_true", help="Validate the structure of the output")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    processor = BibleProcessor(include_apocrypha=args.include_apocrypha, fix_line_breaks=args.fix_line_breaks)
    processor.setup_logging(args.verbose)
    report = processor.process_bible_text(args.input, args.output, output_format=args.format, validate=args.validate)
    
    logging.info("Processing complete. Report:")
    logging.info(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
