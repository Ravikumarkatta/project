#!/usr/bin/env python3
"""
KJV Bible Preprocessor for OpenBible Format
This script processes the KJV Bible text file from OpenBible (e.g., https://openbible.com/textfiles/kjv.txt)
into a standardized format with explicit BOOK: and CHAPTER: headers, and verses formatted as:
    chapter:verse verse_text
Usage:
    python kjv_preprocessor.py --input kjv.txt --output preprocessed_kjv.txt [OPTIONS]
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
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict, OrderedDict

# Define canonical book order and categorization
BIBLE_STRUCTURE = {
    "Old Testament": [
        "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
        "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel", "1 Kings", "2 Kings",
        "1 Chronicles", "2 Chronicles", "Ezra", "Nehemiah", "Esther",
        "Job", "Psalms", "Proverbs", "Ecclesiastes", "Song of Solomon",
        "Isaiah", "Jeremiah", "Lamentations", "Ezekiel", "Daniel",
        "Hosea", "Joel", "Amos", "Obadiah", "Jonah", "Micah", "Nahum",
        "Habakkuk", "Zephaniah", "Haggai", "Zechariah", "Malachi"
    ],
    "New Testament": [
        "Matthew", "Mark", "Luke", "John", "Acts",
        "Romans", "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians",
        "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians",
        "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews",
        "James", "1 Peter", "2 Peter", "1 John", "2 John", "3 John", "Jude", "Revelation"
    ],
    "Apocrypha": [
        "Tobit", "Judith", "Wisdom", "Sirach", "Baruch", "1 Maccabees", "2 Maccabees",
        "1 Esdras", "2 Esdras", "Prayer of Manasseh", "Additions to Esther",
        "Additions to Daniel", "Letter of Jeremiah", "Prayer of Azariah", "Susanna", "Bel and the Dragon"
    ]
}

# Flatten the book list for easier lookup
ALL_CANONICAL_BOOKS = []
for section, books in BIBLE_STRUCTURE.items():
    ALL_CANONICAL_BOOKS.extend(books)

# Common book name variations
BOOK_NAME_VARIANTS = {
    "1 Samuel": ["1 Samuel", "1 Sam", "1Samuel", "1Sam", "I Samuel", "I Sam"],
    "2 Samuel": ["2 Samuel", "2 Sam", "2Samuel", "2Sam", "II Samuel", "II Sam"],
    # Add more variants as needed for other books
    "Song of Solomon": ["Song of Solomon", "Song of Songs", "Canticles", "SongOfSolomon"],
    "Psalms": ["Psalms", "Psalm", "Ps", "Psa"],
    "Revelation": ["Revelation", "Revelations", "Rev", "Apocalypse"],
}

# Create reverse lookup from variants to canonical names
BOOK_NAME_MAPPING = {}
for canonical, variants in BOOK_NAME_VARIANTS.items():
    for variant in variants:
        BOOK_NAME_MAPPING[variant.lower()] = canonical

def setup_logging(verbose: bool) -> None:
    """Configure the logging system based on verbosity level."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def is_apocryphal(book_name: str) -> bool:
    """Check if a book is part of the Apocrypha."""
    return book_name in BIBLE_STRUCTURE.get("Apocrypha", [])

def get_canonical_book_name(book_name: str) -> str:
    """
    Convert possible book name variants to canonical form.
    Returns the input if no canonical form is found.
    """
    return BOOK_NAME_MAPPING.get(book_name.lower(), book_name)

def fix_line_breaks(lines: List[str]) -> List[str]:
    """
    Attempt to fix irregular line breaks by joining lines that appear to be
    continuations of verses rather than new verses.
    """
    verse_pattern = re.compile(
        r'^(?P<book>(?:\d+\s+)?[A-Za-z]+(?:\s+[A-Za-z]+)*)\s+(?P<chapter>\d+):(?P<verse>\d+)\s+(?P<text>.+)$'
    )
    
    fixed_lines = []
    current_verse = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        match = verse_pattern.match(line)
        if match:
            # This is a new verse
            if current_verse:
                fixed_lines.append(current_verse)
            current_verse = line
        elif current_verse:
            # This appears to be a continuation of the previous verse
            current_verse += " " + line
        else:
            # Not a verse and no current verse to append to
            fixed_lines.append(line)
    
    # Don't forget the last verse
    if current_verse:
        fixed_lines.append(current_verse)
        
    return fixed_lines

def parse_kjv(lines: List[str], include_apocrypha: bool = False, fix_breaks: bool = False) -> Dict[str, Dict[int, Dict[int, str]]]:
    """
    Parse KJV text into a structured dictionary.
    Returns:
    {
        "Genesis": {
            1: {  # Chapter 1
                1: "In the beginning...",  # Verse 1
                2: "And the earth was without form...",  # Verse 2
                ...
            },
            ...
        },
        ...
    }
    """
    if fix_breaks:
        lines = fix_line_breaks(lines)
    
    verse_pattern = re.compile(
        r'^(?P<book>(?:\d+\s+)?[A-Za-z]+(?:\s+[A-Za-z]+)*)\s+(?P<chapter>\d+):(?P<verse>\d+)\s+(?P<text>.+)$'
    )
    
    bible_data = defaultdict(lambda: defaultdict(dict))
    skipped_books = set()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        match = verse_pattern.match(line)
        if match:
            book = match.group('book').strip()
            canonical_book = get_canonical_book_name(book)
            
            if not include_apocrypha and is_apocryphal(canonical_book):
                if canonical_book not in skipped_books:
                    logging.info(f"Skipping apocryphal book: {canonical_book}")
                    skipped_books.add(canonical_book)
                continue
                
            chapter = int(match.group('chapter'))
            verse = int(match.group('verse'))
            text = match.group('text').strip()
            
            bible_data[canonical_book][chapter][verse] = text
        else:
            logging.warning(f"Skipping malformed line: {line}")
    
    return bible_data

def validate_bible_structure(bible_data: Dict[str, Dict[int, Dict[int, str]]]) -> Tuple[bool, List[str]]:
    """
    Validate the structure of the parsed Bible data.
    Checks for:
    - Missing books
    - Missing chapters
    - Missing verses
    - Discontinuities in chapter or verse numbering
    """
    issues = []
    is_valid = True
    
    # Check for missing canonical books
    found_books = set(bible_data.keys())
    missing_books = set(ALL_CANONICAL_BOOKS) - found_books
    unknown_books = found_books - set(ALL_CANONICAL_BOOKS)
    
    if missing_books:
        issues.append(f"Missing canonical books: {', '.join(sorted(missing_books))}")
        is_valid = False
    
    if unknown_books:
        issues.append(f"Found non-canonical books: {', '.join(sorted(unknown_books))}")
        # Not marking as invalid as these could be valid variants or apocrypha
    
    # Check chapters and verses for each book
    for book, chapters in bible_data.items():
        chapter_nums = sorted(chapters.keys())
        
        # Check for discontinuities in chapter numbering
        if chapter_nums and chapter_nums[0] != 1:
            issues.append(f"{book}: First chapter is {chapter_nums[0]}, expected 1")
            is_valid = False
            
        for i in range(len(chapter_nums) - 1):
            if chapter_nums[i+1] != chapter_nums[i] + 1:
                issues.append(f"{book}: Missing chapters between {chapter_nums[i]} and {chapter_nums[i+1]}")
                is_valid = False
        
        # Check each chapter for verse continuity
        for chapter_num, verses in chapters.items():
            verse_nums = sorted(verses.keys())
            
            if verse_nums and verse_nums[0] != 1:
                issues.append(f"{book} {chapter_num}: First verse is {verse_nums[0]}, expected 1")
                is_valid = False
                
            for i in range(len(verse_nums) - 1):
                if verse_nums[i+1] != verse_nums[i] + 1:
                    issues.append(f"{book} {chapter_num}: Missing verses between {verse_nums[i]} and {verse_nums[i+1]}")
                    is_valid = False
    
    return is_valid, issues

def write_text_output(bible_data: Dict[str, Dict[int, Dict[int, str]]], output_file: str) -> None:
    """Write Bible data to a text file with BOOK: and CHAPTER: headers."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for book in sorted(bible_data.keys(), key=lambda x: ALL_CANONICAL_BOOKS.index(x) if x in ALL_CANONICAL_BOOKS else 999):
            f.write(f"BOOK: {book}\n")
            
            for chapter in sorted(bible_data[book].keys()):
                f.write(f"CHAPTER: {chapter}\n")
                
                for verse in sorted(bible_data[book][chapter].keys()):
                    f.write(f"{chapter}:{verse} {bible_data[book][chapter][verse]}\n")

def write_json_output(bible_data: Dict[str, Dict[int, Dict[int, str]]], output_file: str) -> None:
    """Write Bible data to a JSON file."""
    # Convert defaultdict to regular dict for JSON serialization
    regular_dict = {}
    for book, chapters in bible_data.items():
        regular_dict[book] = {}
        for chapter, verses in chapters.items():
            regular_dict[book][str(chapter)] = {}
            for verse, text in verses.items():
                regular_dict[book][str(chapter)][str(verse)] = text
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(regular_dict, f, indent=2, ensure_ascii=False)

def write_csv_output(bible_data: Dict[str, Dict[int, Dict[int, str]]], output_file: str) -> None:
    """Write Bible data to a CSV file."""
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Book', 'Chapter', 'Verse', 'Text'])
        
        for book in sorted(bible_data.keys(), key=lambda x: ALL_CANONICAL_BOOKS.index(x) if x in ALL_CANONICAL_BOOKS else 999):
            for chapter in sorted(bible_data[book].keys()):
                for verse in sorted(bible_data[book][chapter].keys()):
                    writer.writerow([book, chapter, verse, bible_data[book][chapter][verse]])

def preprocess_kjv(input_file: str, output_file: str, output_format: str = 'text', 
                  include_apocrypha: bool = False, fix_line_breaks: bool = False,
                  validate: bool = False) -> None:
    """
    Process the KJV text file from OpenBible into a structured format.
    
    Args:
        input_file: Path to the input KJV text file
        output_file: Path to save the preprocessed output
        output_format: Output format ('text', 'json', or 'csv')
        include_apocrypha: Whether to include apocryphal books
        fix_line_breaks: Attempt to fix irregular line breaks
        validate: Validate the structure of the output
    """
    logging.info(f"Processing {input_file} to {output_file} in {output_format} format")
    logging.info(f"Include apocrypha: {include_apocrypha}")
    logging.info(f"Fix line breaks: {fix_line_breaks}")
    logging.info(f"Validate output: {validate}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    logging.info(f"Read {len(lines)} lines from {input_file}")
    
    bible_data = parse_kjv(lines, include_apocrypha, fix_line_breaks)
    
    logging.info(f"Parsed {len(bible_data)} books")
    
    if validate:
        is_valid, issues = validate_bible_structure(bible_data)
        if not is_valid:
            logging.warning("Bible structure validation failed:")
            for issue in issues:
                logging.warning(f"  - {issue}")
        else:
            logging.info("Bible structure validation successful")
    
    # Write output in the requested format
    if output_format == 'json':
        write_json_output(bible_data, output_file)
    elif output_format == 'csv':
        write_csv_output(bible_data, output_file)
    else:  # default to text
        write_text_output(bible_data, output_file)
    
    logging.info(f"Successfully wrote output to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess KJV Bible text from OpenBible format into structured output."
    )
    parser.add_argument("--input", required=True, help="Path to the input KJV text file.")
    parser.add_argument("--output", required=True, help="Path to save the preprocessed output.")
    parser.add_argument("--format", choices=['text', 'json', 'csv'], default='text',
                        help="Output format (default: text)")
    parser.add_argument("--include-apocrypha", action="store_true", 
                        help="Include apocryphal books if present")
    parser.add_argument("--fix-line-breaks", action="store_true",
                        help="Attempt to fix irregular line breaks")
    parser.add_argument("--validate", action="store_true",
                        help="Validate the structure of the output")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    preprocess_kjv(
        args.input, 
        args.output,
        args.format,
        args.include_apocrypha,
        args.fix_line_breaks,
        args.validate
    )

if __name__ == "__main__":
    main()