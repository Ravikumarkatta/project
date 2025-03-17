# src/data/preprocessing.py
import re
import unicodedata
import json
import os
from typing import Dict, List, Any, Tuple
import logging
from bs4 import BeautifulSoup
import pandas as pd

logger = logging.getLogger(__name__)

class BiblicalTextPreprocessor:
    """Preprocess biblical texts and commentaries for model training."""
    
    def __init__(self, config_path: str):
        """Initialize with configuration.
        
        Args:
            config_path: Path to preprocessing configuration
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.raw_dir = self.config['raw_data_dir']
        self.processed_dir = self.config['processed_data_dir']
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Compile common cleanup patterns
        self.cleanup_patterns = [
            (re.compile(r'\s+'), ' '),                       # Standardize whitespace
            (re.compile(r'["“”]'), '"'),                     # Standardize quotes
            (re.compile(r"[‘’']"), "'"),                     # Standardize apostrophes
            (re.compile(r'†|‡|\*|\#|¶'), ''),                # Remove footnote markers
            (re.compile(r'\[.*?\]'), ''),                    # Remove square bracket content
        ]
        
        
        # Regex for verse detection
        self.verse_pattern = re.compile(r'(\d+)[:\.](\d+)')
    
    def normalize_text(self, text: str) -> str:
        """Basic text normalization."""
        # Normalize unicode forms
        text = unicodedata.normalize('NFKC', text)
        
        # Apply cleanup patterns
        for pattern, replacement in self.cleanup_patterns:
            text = pattern.sub(replacement, text)
        
        return text.strip()
    
    def clean_bible_text(self, text: str) -> str:
        """Clean Bible text with special handling for verse structure."""
        text = self.normalize_text(text)
        
        # Special handling for Bible text
        # Replace verse numbers with standardized format
        text = re.sub(r'(\d+)[:\.](\d+)', r'[\1:\2] ', text)
        
        # Remove excess spaces around punctuation
        text = re.sub(r'\s+([,.;:?!])', r'\1', text)
        
        return text
    
    def clean_commentary_text(self, text: str) -> str:
        """Clean commentary text preserving theological terminology."""
        text = self.normalize_text(text)
        
        # Preserve special theological terms that might get normalized
        theological_terms = {
            'YHWH': 'YHWH',
            'JHVH': 'JHVH',
            'LORD': 'LORD',
            'Son of Man': 'Son of Man'
        }
        
        for term, replacement in theological_terms.items():
            text = re.sub(fr'\b{term}\b', replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def process_bible_file(self, file_path: str, translation: str) -> Dict[str, Dict[int, Dict[int, str]]]:
        """Process a single Bible file into structured format.
        
        Args:
            file_path: Path to Bible file
            translation: Bible translation identifier (e.g., 'KJV', 'NIV')
            
        Returns:
            Dict with structure: {book: {chapter: {verse: text}}}
        """
        bible_data = {}
        current_book = None
        current_chapter = None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Process based on file format
        file_format = os.path.splitext(file_path)[1].lower()
        
        if file_format == '.xml':
            bible_data = self._process_xml_bible(content, translation)
        elif file_format == '.json':
            bible_data = self._process_json_bible(content, translation)
        elif file_format == '.txt':
            bible_data = self._process_txt_bible(content, translation)
        else:
            logger.warning(f"Unsupported Bible file format: {file_format}")
        
        return bible_data
    
    def _process_xml_bible(self, content: str, translation: str) -> Dict[str, Dict[int, Dict[int, str]]]:
        """Process XML formatted Bible."""
        bible_data = {}
        soup = BeautifulSoup(content, 'xml')
        
        for book_elem in soup.find_all('book'):
            book_name = book_elem.get('name')
            book_data = {}
            
            for chapter_elem in book_elem.find_all('chapter'):
                chapter_num = int(chapter_elem.get('number'))
                chapter_data = {}
                
                for verse_elem in chapter_elem.find_all('verse'):
                    verse_num = int(verse_elem.get('number'))
                    verse_text = self.clean_bible_text(verse_elem.text)
                    chapter_data[verse_num] = verse_text
                
                book_data[chapter_num] = chapter_data
            
            bible_data[book_name] = book_data
        
        return bible_data
    
    def _process_json_bible(self, content: str, translation: str) -> Dict[str, Dict[int, Dict[int, str]]]:
        """Process JSON formatted Bible."""
        try:
            data = json.loads(content)
            bible_data = {}
            
            # Handle different JSON structures
            if 'books' in data:
                # Format: {"books": [{book data}]}
                for book in data['books']:
                    book_name = book['name']
                    book_data = {}
                    
                    for chapter in book['chapters']:
                        chapter_num = int(chapter['number'])
                        chapter_data = {}
                        
                        for verse in chapter['verses']:
                            verse_num = int(verse['number'])
                            verse_text = self.clean_bible_text(verse['text'])
                            chapter_data[verse_num] = verse_text
                        
                        book_data[chapter_num] = chapter_data
                    
                    bible_data[book_name] = book_data
            else:
                # Assume format: {book: {chapter: {verse: text}}}
                for book_name, chapters in data.items():
                    book_data = {}
                    
                    for chapter_num, verses in chapters.items():
                        chapter_num = int(chapter_num)
                        chapter_data = {}
                        
                        for verse_num, verse_text in verses.items():
                            verse_num = int(verse_num)
                            verse_text = self.clean_bible_text(verse_text)
                            chapter_data[verse_num] = verse_text
                        
                        book_data[chapter_num] = chapter_data
                    
                    bible_data[book_name] = book_data
            
            return bible_data
        
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON Bible file for {translation}")
            return {}
    
    def _process_txt_bible(self, content: str, translation: str) -> Dict[str, Dict[int, Dict[int, str]]]:
        """Process plain text formatted Bible."""
        bible_data = {}
        current_book = None
        current_chapter = None
        
        # Common patterns for book chapter:verse format
        book_chapter_verse_pattern = re.compile(
            r'([1-3]?\s*[A-Za-z]+)\s+(\d+):(\d+)\s+(.*?)(?=(?:[1-3]?\s*[A-Za-z]+\s+\d+:\d+)|$)',
            re.DOTALL
        )
        
        for match in book_chapter_verse_pattern.finditer(content):
            book = match.group(1).strip()
            chapter = int(match.group(2))
            verse = int(match.group(3))
            text = self.clean_bible_text(match.group(4))
            
            if book not in bible_data:
                bible_data[book] = {}
            
            if chapter not in bible_data[book]:
                bible_data[book][chapter] = {}
            
            bible_data[book][chapter][verse] = text
        
        return bible_data
    
    def process_commentary_file(self, file_path: str, source: str) -> List[Dict[str, Any]]:
        """Process a commentary file into structured format.
        
        Args:
            file_path: Path to commentary file
            source: Commentary source identifier
            
        Returns:
            List of commentary entries with metadata
        """
        entries = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Process based on file format
        file_format = os.path.splitext(file_path)[1].lower()
        
        if file_format == '.xml':
            entries = self._process_xml_commentary(content, source)
        elif file_format == '.json':
            entries = self._process_json_commentary(content, source)
        elif file_format == '.txt':
            entries = self._process_txt_commentary(content, source)
        elif file_format == '.csv':
            entries = self._process_csv_commentary(file_path, source)
        else:
            logger.warning(f"Unsupported commentary file format: {file_format}")
        
        return entries
    
    def _process_xml_commentary(self, content: str, source: str) -> List[Dict[str, Any]]:
        """Process XML formatted commentary."""
        entries = []
        soup = BeautifulSoup(content, 'xml')
        
        for entry_elem in soup.find_all('entry'):
            entry = {
                'source': source,
                'content': self.clean_commentary_text(entry_elem.find('content').text),
                'tradition': entry_elem.get('tradition', 'unknown')
            }
            
            # Extract reference information
            ref_elem = entry_elem.find('reference')
            if ref_elem:
                entry['book'] = ref_elem.get('book')
                entry['chapter'] = int(ref_elem.get('chapter')) if ref_elem.get('chapter') else None
                entry['verse_start'] = int(ref_elem.get('verse_start')) if ref_elem.get('verse_start') else None
                entry['verse_end'] = int(ref_elem.get('verse_end')) if ref_elem.get('verse_end') else entry['verse_start']
            
            # Extract author information
            author_elem = entry_elem.find('author')
            if author_elem:
                entry['author'] = author_elem.text
                entry['year'] = int(author_elem.get('year')) if author_elem.get('year') else None
            
            entries.append(entry)
        
        return entries
    
    def _process_json_commentary(self, content: str, source: str) -> List[Dict[str, Any]]:
        """Process JSON formatted commentary."""
        try:
            data = json.loads(content)
            entries = []
            
            if isinstance(data, list):
                for item in data:
                    item['source'] = source
                    if 'content' in item:
                        item['content'] = self.clean_commentary_text(item['content'])
                    entries.append(item)
            elif isinstance(data, dict) and 'entries' in data:
                for item in data['entries']:
                    item['source'] = source
                    if 'content' in item:
                        item['content'] = self.clean_commentary_text(item['content'])
                    entries.append(item)
            
            return entries
        
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON commentary file for {source}")
            return []
    
    def _process_txt_commentary(self, content: str, source: str) -> List[Dict[str, Any]]:
        """Process plain text formatted commentary."""
        entries = []
        
        # Try to split by common delimiter patterns
        sections = re.split(r'\n\s*\n|\r\n\s*\r\n', content)
        
        for section in sections:
            if not section.strip():
                continue
            
            # Try to extract reference and content
            ref_match = re.search(r'([1-3]?\s*[A-Za-z]+\s+\d+:\d+(?:-\d+)?)', section)
            
            if ref_match:
                reference = ref_match.group(1)
                remaining_text = section.replace(reference, '', 1).strip()
                
                # Parse reference
                book_chapter_verse = re.match(r'([1-3]?\s*[A-Za-z]+)\s+(\d+):(\d+)(?:-(\d+))?', reference)
                
                if book_chapter_verse:
                    entry = {
                        'source': source,
                        'content': self.clean_commentary_text(remaining_text),
                        'book': book_chapter_verse.group(1),
                        'chapter': int(book_chapter_verse.group(2)),
                        'verse_start': int(book_chapter_verse.group(3)),
                        'verse_end': int(book_chapter_verse.group(4)) if book_chapter_verse.group(4) else int(book_chapter_verse.group(3))
                    }
                    entries.append(entry)
                else:
                    # If we can't parse the reference, just add the content
                    entries.append({
                        'source': source,
                        'content': self.clean_commentary_text(section),
                        'reference': reference if ref_match else None
                    })
            else:
                # No reference found, treat whole section as content
                entries.append({
                    'source': source,
                    'content': self.clean_commentary_text(section)
                })
        
        return entries
    
    def _process_csv_commentary(self, file_path: str, source: str) -> List[Dict[str, Any]]:
        """Process CSV formatted commentary."""
        try:
            df = pd.read_csv(file_path)
            entries = []
            
            for _, row in df.iterrows():
                entry = row.to_dict()
                entry['source'] = source
                
                if 'content' in entry:
                    entry['content'] = self.clean_commentary_text(entry['content'])
                
                entries.append(entry)
            
            return entries
        
        except Exception as e:
            logger.error(f"Error processing CSV commentary file {file_path}: {e}")
            return []
    
    def save_processed_bible(self, bible_data: Dict[str, Dict[int, Dict[int, str]]], translation: str):
        """Save processed Bible data."""
        output_path = os.path.join(self.processed_dir, f"bible_{translation.lower()}.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(bible_data, f, indent=2)
        
        logger.info(f"Saved processed Bible data for {translation} to {output_path}")
    
    def save_processed_commentaries(self, entries: List[Dict[str, Any]], source: str):
        """Save processed commentary data."""
        output_path = os.path.join(self.processed_dir, f"commentary_{source.lower().replace(' ', '_')}.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(entries, f, indent=2)
        
        logger.info(f"Saved {len(entries)} processed commentary entries for {source} to {output_path}")
    
    def create_verse_aligned_dataset(self, bibles: Dict[str, Dict[str, Dict[int, Dict[int, str]]]], 
                                     commentaries: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
        """Create a verse-aligned dataset combining Bibles and commentaries.
        
        Args:
            bibles: Dictionary of Bible translations
            commentaries: Dictionary of commentary sources
            
        Returns:
            DataFrame with aligned verses and commentaries
        """
        # Start with a list to collect all verse references
        all_refs = []
        
        # Collect all unique verse references from all Bibles
        for translation, bible_data in bibles.items():
            for book, chapters in bible_data.items():
                for chapter, verses in chapters.items():
                    for verse, text in verses.items():
                        ref = {
                            'book': book,
                            'chapter': chapter,
                            'verse': verse,
                            'reference': f"{book} {chapter}:{verse}"
                        }
                        if ref not in all_refs:
                            all_refs.append(ref)
        
        # Create DataFrame with all references
        df = pd.DataFrame(all_refs)
        
        # Add translations
        for translation, bible_data in bibles.items():
            df[f"text_{translation}"] = df.apply(
                lambda row: bible_data.get(row['book'], {}).get(row['chapter'], {}).get(row['verse'], ""), 
                axis=1
            )
        
        # Add commentaries
        for source, entries in commentaries.items():
            # Create a mapping of references to commentaries
            commentary_map = {}
            
            for entry in entries:
                if 'book' in entry and 'chapter' in entry and 'verse_start' in entry:
                    book = entry['book']
                    chapter = entry['chapter']
                    verse_start = entry['verse_start']
                    verse_end = entry.get('verse_end', verse_start)
                    
                    for verse in range(verse_start, verse_end + 1):
                        key = (book, chapter, verse)
                        if key not in commentary_map:
                            commentary_map[key] = []
                        commentary_map[key].append(entry['content'])
            
            # Add to DataFrame
            df[f"commentary_{source}"] = df.apply(
                lambda row: "; ".join(commentary_map.get((row['book'], row['chapter'], row['verse']), [])),
                axis=1
            )
        
        # Save the aligned dataset
        output_path = os.path.join(self.processed_dir, "verse_aligned_dataset.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Created verse-aligned dataset with {len(df)} rows at {output_path}")
        
        return df

# ===================== Added Code: Training Data Pipeline =====================
# The following code integrates tokenizer-based data preparation for training.
# It provides a dataset class for instruction fine-tuning and a helper to create DataLoaders.

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from typing import Dict, List, Tuple

class BibleInstructionDataset(Dataset):
    """Dataset for instruction fine-tuning with biblical data."""
    
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        """
        Initialize dataset from instruction data.
        
        Args:
            data_path: Path to instruction JSON file.
            tokenizer: HuggingFace tokenizer to use.
            max_length: Maximum sequence length.
        """
        self.data = self._load_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info(f"Loaded {len(self.data)} instruction examples")
        
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load instruction data from JSON file."""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get tokenized instruction example."""
        item = self.data[idx]
        
        # Format as instruction prompt
        instruction = item['instruction']
        input_text = item['input']
        output = item['output']
        
        # Format prompt according to instruction tuning format
        prompt = f"Instruction: {instruction}\n\nInput: {input_text}\n\nOutput: "
        
        # Tokenize prompt
        prompt_tokenized = self.tokenizer(
            prompt, 
            max_length=self.max_length // 2,  # Reserve half length for output
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Tokenize output (labels)
        output_tokenized = self.tokenizer(
            output,
            max_length=self.max_length // 2,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Combine input_ids: prompt followed by output
        input_ids = torch.cat([
            prompt_tokenized['input_ids'].squeeze(),
            output_tokenized['input_ids'].squeeze()
        ])[:self.max_length]
        
        # Create attention mask (1 for prompt and output tokens, 0 for padding)
        attention_mask = torch.cat([
            prompt_tokenized['attention_mask'].squeeze(),
            output_tokenized['attention_mask'].squeeze()
        ])[:self.max_length]
        
        # Create labels tensor: -100 for prompt tokens (ignored in loss), actual ids for output
        labels = torch.cat([
            torch.full_like(prompt_tokenized['input_ids'].squeeze(), -100),
            output_tokenized['input_ids'].squeeze()
        ])[:self.max_length]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def create_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    max_length: int = 512
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders.
    
    Args:
        train_path: Path to training data JSON file.
        val_path: Path to validation data JSON file.
        tokenizer: HuggingFace tokenizer.
        batch_size: Batch size for training.
        max_length: Maximum sequence length.
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create datasets
    train_dataset = BibleInstructionDataset(train_path, tokenizer, max_length)
    val_dataset = BibleInstructionDataset(val_path, tokenizer, max_length)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader

class BiblicalDataset(Dataset):
    """Custom Dataset for biblical data."""
    
    def __init__(self, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor):
        """
        Initialize the dataset with input_ids, labels, and attention_mask.
        
        Args:
            input_ids: Tensor of input token IDs.
            labels: Tensor of label token IDs.
            attention_mask: Tensor of attention masks.
        """
        self.input_ids = input_ids
        self.labels = labels
        self.attention_mask = attention_mask
    
    def __len__(self) -> int:
        return len(self.input_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': self.input_ids[idx],
            'labels': self.labels[idx],
            'attention_mask': self.attention_mask[idx]
        }

def load_datasets(data_path: str) -> Tuple[BiblicalDataset, BiblicalDataset]:
    """
    Load processed datasets and return as BiblicalDataset instances.
    
    Args:
        data_path: Path to the directory containing processed data files.
        
    Returns:
        Tuple (train_dataset, val_dataset) as BiblicalDataset instances.
    """
    data_dir = os.path.abspath(data_path)
    train_file = os.path.join(data_dir, 'train.pt')
    val_file = os.path.join(data_dir, 'val.pt')

    if not os.path.exists(train_file) or not os.path.exists(val_file):
        raise FileNotFoundError(f"Processed data files not found in {data_dir}")

    train_data = torch.load(train_file)
    val_data = torch.load(val_file)

    train_dataset = BiblicalDataset(
        input_ids=train_data['input_ids'],
        labels=train_data['labels'],
        attention_mask=train_data['attention_mask']
    )
    val_dataset = BiblicalDataset(
        input_ids=val_data['input_ids'],
        labels=val_data['labels'],
        attention_mask=val_data['attention_mask']
    )

    return train_dataset, val_dataset

# ===================== End of Added Code =====================

# Example usage
if __name__ == '__main__':
    preprocessor = BiblicalTextPreprocessor('config/data_config.json')
    
    # Process Bibles
    bibles = {}
    bibles_dir = os.path.join(preprocessor.raw_dir, 'bibles')
    for bible_file in os.listdir(bibles_dir):
        if bible_file.endswith(('.xml', '.json', '.txt')):
            translation = os.path.splitext(bible_file)[0].upper()
            file_path = os.path.join(bibles_dir, bible_file)
            bible_data = preprocessor.process_bible_file(file_path, translation)
            bibles[translation] = bible_data
            preprocessor.save_processed_bible(bible_data, translation)
    
    # Process commentaries
    commentaries = {}
    commentaries_dir = os.path.join(preprocessor.raw_dir, 'commentaries')
    for commentary_file in os.listdir(commentaries_dir):
        if commentary_file.endswith(('.xml', '.json', '.txt', '.csv')):
            source = os.path.splitext(commentary_file)[0]
            file_path = os.path.join(commentaries_dir, commentary_file)
            entries = preprocessor.process_commentary_file(file_path, source)
            commentaries[source] = entries
            preprocessor.save_processed_commentaries(entries, source)
    
    # Create verse-aligned dataset
    preprocessor.create_verse_aligned_dataset(bibles, commentaries)
