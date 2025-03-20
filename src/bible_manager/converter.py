# src/bible_manager/converter.py
"""
Bible Converter for Bible-AI.

Converts Bible texts between USFM, JSON, CSV, XML, and TXT formats.
"""

import os
import json
import csv
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class BibleConverter:
    """Converts Bible texts between different formats in Bible-AI."""
    
    SUPPORTED_FORMATS = {'usfm', 'json', 'csv', 'xml', 'txt'}
    
    def __init__(self, config_path=None):
        """Initialize the converter with supported formats."""
        self.config_path = config_path  # Store the configuration path
        self.logger = logger  # Ensure 'logger' is defined elsewhere


    def convert(self, input_path: str, output_path: str, 
                input_format: Optional[str] = None, 
                output_format: Optional[str] = None) -> bool:
        """
        Convert a Bible file from one format to another.
        
        Args:
            input_path: Path to the input file.
            output_path: Path to the output file.
            input_format: Input format (auto-detected if None).
            output_format: Output format (auto-detected if None).
        
        Returns:
            bool: True if conversion succeeds, False otherwise.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return False

        # Detect formats if not provided
        input_format = input_format or self._detect_format(str(input_path))
        output_format = output_format or self._detect_format(str(output_path))
        logger.info(f"Detected formats: input={input_format}, output={output_format}")

        if input_format not in self.SUPPORTED_FORMATS or output_format not in self.SUPPORTED_FORMATS:
            logger.error(f"Unsupported format: input={input_format}, output={output_format}")
            return False

        try:
            bible_data = self._read_file(str(input_path), input_format)
            if not bible_data:
                logger.error(f"No valid data read from {input_path}")
                return False
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_file(bible_data, str(output_path), output_format)
            logger.info(f"Converted {input_path} to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Conversion failed: {str(e)}")
            return False

    def _detect_format(self, file_path: str) -> str:
        """Detect file format based on extension or content."""
        ext = os.path.splitext(file_path)[1].lower()
        format_map = {
            '.usfm': 'usfm', '.sfm': 'usfm',
            '.json': 'json',
            '.csv': 'csv',
            '.xml': 'xml',
            '.txt': 'txt'
        }
        if ext in format_map:
            return format_map[ext]

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(1024)
                if content.startswith('<?xml'): return 'xml'
                if content.startswith('{'): return 'json'
                if ',' in content.splitlines()[0]: return 'csv'
                if '\\id' in content: return 'usfm'
                return 'txt'
        except Exception as e:
            logger.warning(f"Format detection failed for {file_path}: {str(e)}. Defaulting to 'txt'")
            return 'txt'

    def _read_file(self, file_path: str, format_type: str) -> Dict[str, Any]:
        """Read a Bible file based on its format."""
        readers = {
            'usfm': self._read_usfm,
            'json': self._read_json,
            'csv': self._read_csv,
            'xml': self._read_xml,
            'txt': self._read_txt
        }
        if format_type not in readers:
            raise ValueError(f"Unsupported format: {format_type}")
        return readers[format_type](file_path)

    def _write_file(self, bible_data: Dict[str, Any], file_path: str, format_type: str) -> None:
        """Write Bible data to a file in the specified format."""
        writers = {
            'usfm': self._write_usfm,
            'json': self._write_json,
            'csv': self._write_csv,
            'xml': self._write_xml,
            'txt': self._write_txt
        }
        if format_type not in writers:
            raise ValueError(f"Unsupported format: {format_type}")
        writers[format_type](bible_data, file_path)

    # Format-specific methods
    def _read_usfm(self, file_path: str) -> Dict[str, Any]:
        """Read USFM file into structured data."""
        bible_data = {'metadata': {}, 'books': []}
        current_book = None
        current_chapter = None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=2)
                marker = parts[0]
                if marker == '\\id':
                    bible_data['metadata']['id'] = parts[1] if len(parts) > 1 else 'UNK'
                elif marker == '\\h':
                    bible_data['metadata']['title'] = parts[1] if len(parts) > 1 else 'Bible'
                elif marker == '\\c':
                    if current_chapter and current_book:
                        current_book['chapters'].append(current_chapter)
                    current_chapter = {'number': int(parts[1]), 'verses': []} if len(parts) > 1 else None
                elif marker == '\\v' and current_chapter:
                    verse_num = parts[1] if len(parts) > 1 else '0'
                    verse_text = parts[2] if len(parts) > 2 else ''
                    current_chapter['verses'].append({'number': verse_num, 'text': verse_text})
                elif marker == '\\b':
                    if current_book:
                        bible_data['books'].append(current_book)
                    current_book = {'name': parts[1] if len(parts) > 1 else 'Unknown', 'chapters': []}
            if current_chapter and current_book:
                current_book['chapters'].append(current_chapter)
            if current_book:
                bible_data['books'].append(current_book)
        return bible_data if bible_data['books'] else {}

    def _write_usfm(self, bible_data: Dict[str, Any], file_path: str) -> None:
        """Write data to USFM file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"\\id {bible_data['metadata'].get('id', 'UNK')}\n")
            f.write(f"\\h {bible_data['metadata'].get('title', 'Bible')}\n")
            for book in bible_data.get('books', []):
                f.write(f"\\b {book['name']}\n")
                for chapter in book.get('chapters', []):
                    f.write(f"\\c {chapter['number']}\n")
                    for verse in chapter.get('verses', []):
                        f.write(f"\\v {verse['number']} {verse['text']}\n")

    def _read_json(self, file_path: str) -> Dict[str, Any]:
        """Read JSON file into structured data."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) and 'books' in data else {}

    def _write_json(self, bible_data: Dict[str, Any], file_path: str) -> None:
        """Write data to JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(bible_data, f, ensure_ascii=False, indent=2)

    def _read_csv(self, file_path: str) -> Dict[str, Any]:
        """Read CSV file (book,chapter,verse,text) into structured data."""
        bible_data = {'metadata': {'title': os.path.basename(file_path)}, 'books': []}
        book_dict = {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    book_name = row['book']
                    chapter_num = int(row['chapter'])
                    if book_name not in book_dict:
                        book_dict[book_name] = {'name': book_name, 'chapters': {}}
                    if chapter_num not in book_dict[book_name]['chapters']:
                        book_dict[book_name]['chapters'][chapter_num] = {'number': chapter_num, 'verses': []}
                    book_dict[book_name]['chapters'][chapter_num]['verses'].append({
                        'number': row['verse'],
                        'text': row['text']
                    })
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping invalid CSV row in {file_path}: {row}")
                    bible_data['books'] = [book for book in book_dict.values() if list(book['chapters'].values())]
                    for book in bible_data['books']:
                     book['chapters'] = list(book['chapters'].values())
                     return bible_data

    def _write_csv(self, bible_data: Dict[str, Any], file_path: str) -> None:
        """Write data to CSV file."""
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['book', 'chapter', 'verse', 'text'])
            for book in bible_data.get('books', []):
                for chapter in book.get('chapters', []):
                    for verse in chapter.get('verses', []):
                        writer.writerow([book['name'], chapter['number'], verse['number'], verse['text']])

    def _read_xml(self, file_path: str) -> Dict[str, Any]:
        """Read XML file into structured data."""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            bible_data = {'metadata': {'title': root.get('title', 'Bible')}, 'books': []}
            for book_elem in root.findall('book'):
                book = {'name': book_elem.get('name', 'Unknown'), 'chapters': []}
                for chapter_elem in book_elem.findall('chapter'):
                    chapter = {'number': int(chapter_elem.get('number', '0')), 'verses': []}
                    for verse_elem in chapter_elem.findall('verse'):
                        chapter['verses'].append({
                            'number': verse_elem.get('number', '0'),
                            'text': verse_elem.text.strip() if verse_elem.text else ''
                        })
                    book['chapters'].append(chapter)
                bible_data['books'].append(book)
            return bible_data if bible_data['books'] else {}
        except ET.ParseError as e:
            logger.error(f"Invalid XML in {file_path}: {str(e)}")
            return {}

    def _write_xml(self, bible_data: Dict[str, Any], file_path: str) -> None:
        """Write data to XML file."""
        root = ET.Element('bible', title=bible_data['metadata'].get('title', 'Bible'))
        for book in bible_data.get('books', []):
            book_elem = ET.SubElement(root, 'book', name=book['name'])
            for chapter in book.get('chapters', []):
                chapter_elem = ET.SubElement(book_elem, 'chapter', number=str(chapter['number']))
                for verse in chapter.get('verses', []):
                    verse_elem = ET.SubElement(chapter_elem, 'verse', number=str(verse['number']))
                    verse_elem.text = verse['text']
        ET.ElementTree(root).write(file_path, encoding='utf-8', xml_declaration=True)

    def _read_txt(self, file_path: str) -> Dict[str, Any]:
        """Read TXT file (Book Chapter:Verse Text) into structured data."""
        bible_data = {'metadata': {'title': os.path.basename(file_path)}, 'books': []}
        book_dict = {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ref, text = line.split(' ', 1)
                    book_name, chap_verse = ref.rsplit(' ', 1)
                    chapter_num, verse_num = map(int, chap_verse.split(':'))
                    if book_name not in book_dict:
                        book_dict[book_name] = {'name': book_name, 'chapters': {}}
                    if chapter_num not in book_dict[book_name]['chapters']:
                        book_dict[book_name]['chapters'][chapter_num] = {'number': chapter_num, 'verses': []}
                    book_dict[book_name]['chapters'][chapter_num]['verses'].append({
                        'number': str(verse_num),
                        'text': text
                    })
                except ValueError:
                    logger.warning(f"Skipping malformed TXT line in {file_path}: {line}")
                    bible_data['books'] = [book for book in book_dict.values() if list(book['chapters'].values())]
                    for book in bible_data['books']:
                     book['chapters'] = list(book['chapters'].values())
                    return bible_data

    def _write_txt(self, bible_data: Dict[str, Any], file_path: str) -> None:
        """Write data to TXT file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            for book in bible_data.get('books', []):
                for chapter in book.get('chapters', []):
                    for verse in chapter.get('verses', []):
                        f.write(f"{book['name']} {chapter['number']}:{verse['number']} {verse['text']}\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    converter = BibleConverter()
    result = converter.convert("input.usfm", "output.json")
    print(f"Conversion result: {result}")