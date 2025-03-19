import os
import json
import csv
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class BibleConverter:
    """A module for converting Bible texts between different formats in Bible-AI."""
    
    SUPPORTED_FORMATS = {'usfm', 'json', 'csv', 'xml', 'txt'}
    
    def __init__(self):
        """Initialize the converter with supported formats."""
        self.logger = logger

    def convert(self, input_path: str, output_path: str, input_format: Optional[str] = None, 
                output_format: Optional[str] = None) -> bool:
        """
        Convert a Bible file from one format to another.
        
        Args:
            input_path: Path to the input file.
            output_path: Path to the output file.
            input_format: Format of the input file (auto-detected if None).
            output_format: Format of the output file (auto-detected if None).
        
        Returns:
            bool: True if conversion succeeds, False otherwise.
        """
        # Ensure input file exists
        if not os.path.exists(input_path):
            self.logger.error(f"Input file not found: {input_path}")
            return False

        # Auto-detect formats if not specified
        if not input_format:
            input_format = self._detect_format(input_path)
            self.logger.info(f"Detected input format: {input_format}")
        if not output_format:
            output_format = self._detect_format(output_path)
            self.logger.info(f"Detected output format: {output_format}")

        # Validate formats
        if input_format not in self.SUPPORTED_FORMATS or output_format not in self.SUPPORTED_FORMATS:
            self.logger.error(f"Unsupported format: input={input_format}, output={output_format}")
            return False

        # Read input file
        try:
            bible_data = self._read_file(input_path, input_format)
            if not bible_data or not self._validate_bible_data(bible_data):
                self.logger.error(f"Invalid or empty data from {input_path}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to read {input_path}: {str(e)}")
            return False

        # Write output file
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure directory exists
            self._write_file(bible_data, output_path, output_format)
            self.logger.info(f"Successfully converted {input_path} to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to write {output_path}: {str(e)}")
            return False

    def _detect_format(self, file_path: str) -> str:
        """
        Detect the format of a Bible file based on extension or content.
        
        Args:
            file_path: Path to the file.
        
        Returns:
            str: Detected format ('usfm', 'json', 'csv', 'xml', 'txt').
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext in {'.usfm', '.sfm'}:
            return 'usfm'
        elif ext == '.json':
            return 'json'
        elif ext == '.csv':
            return 'csv'
        elif ext == '.xml':
            return 'xml'
        elif ext == '.txt':
            return 'txt'

        # Fallback to content-based detection
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(1024)  # Read first 1KB for efficiency
                if content.startswith('<?xml'):
                    return 'xml'
                elif content.startswith('{'):
                    return 'json'
                elif ',' in content.splitlines()[0]:
                    return 'csv'
                elif '\\id' in content:  # USFM marker
                    return 'usfm'
                return 'txt'  # Default
        except Exception as e:
            self.logger.warning(f"Content detection failed for {file_path}: {str(e)}. Defaulting to 'txt'")
            return 'txt'

    def _validate_bible_data(self, bible_data: Dict[str, Any]) -> bool:
        """
        Validate the structure of Bible data.
        
        Args:
            bible_data: Structured Bible data.
        
        Returns:
            bool: True if valid, False otherwise.
        """
        required_keys = {'metadata', 'books'}
        if not isinstance(bible_data, dict) or not all(key in bible_data for key in required_keys):
            return False
        if not bible_data['books'] or not isinstance(bible_data['books'], list):
            return False
        return True

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
        elif format_type == 'json':
            return self._read_json(file_path)
        elif format_type == 'csv':
            return self._read_csv(file_path)
        elif format_type == 'xml':
            return self._read_xml(file_path)
        elif format_type == 'txt':
            return self._read_txt(file_path)
        raise ValueError(f"Unsupported format: {format_type}")

    def _write_file(self, bible_data: Dict[str, Any], file_path: str, format_type: str) -> None:
        """
        Write Bible data to a file in the specified format.
        
        Args:
            bible_data: Structured Bible data.
            file_path: Path to the output file.
            format_type: Format of the output file.
        """
        if format_type == 'usfm':
            self._write_usfm(bible_data, file_path)
        elif format_type == 'json':
            self._write_json(bible_data, file_path)
        elif format_type == 'csv':
            self._write_csv(bible_data, file_path)
        elif format_type == 'xml':
            self._write_xml(bible_data, file_path)
        elif format_type == 'txt':
            self._write_txt(bible_data, file_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    # Format-specific read/write methods
    def _read_usfm(self, file_path: str) -> Dict[str, Any]:
        """Read a USFM file into structured data."""
        bible_data = {'metadata': {}, 'books': []}
        current_book = None
        current_chapter = None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('\\id'):
                    bible_data['metadata']['id'] = line.split()[1]
                elif line.startswith('\\h'):
                    bible_data['metadata']['title'] = line[3:].strip()
                elif line.startswith('\\c'):
                    if current_chapter:
                        current_book['chapters'].append(current_chapter)
                    current_chapter = {'number': int(line.split()[1]), 'verses': []}
                elif line.startswith('\\v'):
                    parts = line.split(maxsplit=2)
                    verse_num = parts[1]
                    verse_text = parts[2] if len(parts) > 2 else ""
                    current_chapter['verses'].append({'number': verse_num, 'text': verse_text})
                elif line.startswith('\\b'):
                    if current_book:
                        bible_data['books'].append(current_book)
                    current_book = {'name': line[3:].strip(), 'chapters': []}
            if current_chapter and current_book:
                current_book['chapters'].append(current_chapter)
            if current_book:
                bible_data['books'].append(current_book)
        return bible_data

    def _write_usfm(self, bible_data: Dict[str, Any], file_path: str) -> None:
        """Write structured data to a USFM file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"\\id {bible_data['metadata'].get('id', 'UNK')}\n")
            f.write(f"\\h {bible_data['metadata'].get('title', 'Bible')}\n")
            for book in bible_data['books']:
                f.write(f"\\b {book['name']}\n")
                for chapter in book['chapters']:
                    f.write(f"\\c {chapter['number']}\n")
                    for verse in chapter['verses']:
                        f.write(f"\\v {verse['number']} {verse['text']}\n")

    def _read_json(self, file_path: str) -> Dict[str, Any]:
        """Read a JSON file into structured data."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _write_json(self, bible_data: Dict[str, Any], file_path: str) -> None:
        """Write structured data to a JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(bible_data, f, ensure_ascii=False, indent=2)

    def _read_csv(self, file_path: str) -> Dict[str, Any]:
        """Read a CSV file into structured data (assumes book,chapter,verse,text columns)."""
        bible_data = {'metadata': {'title': os.path.basename(file_path)}, 'books': []}
        book_dict = {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                book_name = row['book']
                if book_name not in book_dict:
                    book_dict[book_name] = {'name': book_name, 'chapters': {}}
                chapter_num = int(row['chapter'])
                if chapter_num not in book_dict[book_name]['chapters']:
                    book_dict[book_name]['chapters'][chapter_num] = {'number': chapter_num, 'verses': []}
                book_dict[book_name]['chapters'][chapter_num]['verses'].append({
                    'number': row['verse'],
                    'text': row['text']
                })
        
        for book_name, book in book_dict.items():
            book['chapters'] = list(book['chapters'].values())
            bible_data['books'].append(book)
        return bible_data

    def _write_csv(self, bible_data: Dict[str, Any], file_path: str) -> None:
        """Write structured data to a CSV file."""
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['book', 'chapter', 'verse', 'text'])
            for book in bible_data['books']:
                for chapter in book['chapters']:
                    for verse in chapter['verses']:
                        writer.writerow([book['name'], chapter['number'], verse['number'], verse['text']])

    def _read_xml(self, file_path: str) -> Dict[str, Any]:
        """Read an XML file into structured data."""
        tree = ET.parse(file_path)
        root = tree.getroot()
        bible_data = {'metadata': {'title': root.get('title', 'Bible')}, 'books': []}
        
        for book_elem in root.findall('book'):
            book = {'name': book_elem.get('name'), 'chapters': []}
            for chapter_elem in book_elem.findall('chapter'):
                chapter = {'number': int(chapter_elem.get('number')), 'verses': []}
                for verse_elem in chapter_elem.findall('verse'):
                    chapter['verses'].append({
                        'number': verse_elem.get('number'),
                        'text': verse_elem.text.strip() if verse_elem.text else ''
                    })
                book['chapters'].append(chapter)
            bible_data['books'].append(book)
        return bible_data

    def _write_xml(self, bible_data: Dict[str, Any], file_path: str) -> None:
        """Write structured data to an XML file."""
        root = ET.Element('bible', title=bible_data['metadata'].get('title', 'Bible'))
        for book in bible_data['books']:
            book_elem = ET.SubElement(root, 'book', name=book['name'])
            for chapter in book['chapters']:
                chapter_elem = ET.SubElement(book_elem, 'chapter', number=str(chapter['number']))
                for verse in chapter['verses']:
                    verse_elem = ET.SubElement(chapter_elem, 'verse', number=str(verse['number']))
                    verse_elem.text = verse['text']
        tree = ET.ElementTree(root)
        tree.write(file_path, encoding='utf-8', xml_declaration=True)

    def _read_txt(self, file_path: str) -> Dict[str, Any]:
        """Read a plain text file (assumes verse-per-line with reference)."""
        bible_data = {'metadata': {'title': os.path.basename(file_path)}, 'books': []}
        book_dict = {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Assume format: "Book Chapter:Verse Text"
                try:
                    ref, text = line.split(' ', 1)
                    book_name, chap_verse = ref.rsplit(' ', 1)
                    chapter_num, verse_num = chap_verse.split(':')
                    chapter_num = int(chapter_num)
                    
                    if book_name not in book_dict:
                        book_dict[book_name] = {'name': book_name, 'chapters': {}}
                    if chapter_num not in book_dict[book_name]['chapters']:
                        book_dict[book_name]['chapters'][chapter_num] = {'number': chapter_num, 'verses': []}
                    book_dict[book_name]['chapters'][chapter_num]['verses'].append({
                        'number': verse_num,
                        'text': text
                    })
                except ValueError:
                    self.logger.warning(f"Skipping malformed line in {file_path}: {line}")
                    continue
        
        for book_name, book in book_dict.items():
            book['chapters'] = list(book['chapters'].values())
            bible_data['books'].append(book)
        return bible_data

    def _write_txt(self, bible_data: Dict[str, Any], file_path: str) -> None:
        """Write structured data to a plain text file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            for book in bible_data['books']:
                for chapter in book['chapters']:
                    for verse in chapter['verses']:
                        f.write(f"{book['name']} {chapter['number']}:{verse['number']} {verse['text']}\n")

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    converter = BibleConverter()
    result = converter.convert("input.usfm", "output.json")
    print(f"Conversion result: {result}")
