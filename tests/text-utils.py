"""
Utility functions for text processing in biblical data preparation.
"""
import re
import unicodedata
from typing import List, Union, Optional

def clean_text(text: str) -> str:
    """
    Basic text cleaning and normalization.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned and normalized text
    """
    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)
    
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Standardize quotes and apostrophes
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    
    # Remove footnote markers and other special characters
    text = re.sub(r"†|‡|\*|\#|¶", "", text)
    
    # Remove square bracket content (often annotations)
    text = re.sub(r"\[.*?\]", "", text)
    
    return text

def normalize_verses(text: str) -> str:
    """
    Normalize verse references in text.
    
    Args:
        text: Text containing verse references
        
    Returns:
        Text with standardized verse references
    """
    # First clean the text
    text = clean_text(text)
    
    # Standardize verse references
    text = re.sub(r"(\d+)[:\.](\d+)", r"[\1:\2] ", text)
    
    # Remove excess spaces around punctuation
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)
    
    return text

def tokenize_text(text: str, tokenizer=None) -> Union[List[str], List[int]]:
    """
    Tokenize text using the provided tokenizer or simple whitespace splitting.
    
    Args:
        text: Text to tokenize
        tokenizer: Optional tokenizer function or object with a __call__ method
        
    Returns:
        List of tokens (strings or token IDs depending on tokenizer)
    """
    if tokenizer:
        return tokenizer(text)
    else:
        # Simple whitespace tokenization as fallback
        return text.split()

def extract_verse_refs(text: str) -> List[str]:
    """
    Extract biblical verse references from text.
    
    Args:
        text: Text containing verse references
        
    Returns:
        List of extracted verse references
    """
    # Pattern to match common verse reference formats
    # e.g., "Genesis 1:1", "Gen 1:1", "Gen. 1:1-3", "1 Cor 13:4-7"
    pattern = r'(?:\d\s)?[A-Za-z]+\.?\s\d+:\d+(?:-\d+)?'
    
    # Find all matches
    matches = re.findall(pattern, text)
    return matches

def standardize_book_names(book_name: str) -> str:
    """
    Standardize biblical book names to full names.
    
    Args:
        book_name: Short or abbreviated book name
        
    Returns:
        Standardized full book name
    """
    book_name_mapping = {
        "Gen": "Genesis",
        "Exo": "Exodus", 
        "Ex": "Exodus",
        "Lev": "Leviticus",
        "Num": "Numbers",
        "Deut": "Deuteronomy", 
        "Deu": "Deuteronomy",
        "Josh": "Joshua",
        "Jos": "Joshua",
        "Judg": "Judges", 
        "Jdg": "Judges",
        "Rut": "Ruth",
        "1Sa": "1 Samuel", 
        "1 Sa": "1 Samuel",
        "2Sa": "2 Samuel", 
        "2 Sa": "2 Samuel",
        "1Ki": "1 Kings", 
        "1 Ki": "1 Kings",
        "2Ki": "2 Kings", 
        "2 Ki": "2 Kings",
        "1Ch": "1 Chronicles", 
        "1 Ch": "1 Chronicles",
        "2Ch": "2 Chronicles", 
        "2 Ch": "2 Chronicles",
        "Ezr": "Ezra",
        "Neh": "Nehemiah",
        "Est": "Esther",
        "Job": "Job",
        "Ps": "Psalms", 
        "Psa": "Psalms",
        "Prov": "Proverbs", 
        "Pro": "Proverbs",
        "Eccl": "Ecclesiastes", 
        "Ecc": "Ecclesiastes",
        "Song": "Song of Solomon", 
        "Son": "Song of Solomon",
        "Isa": "Isaiah",
        "Jer": "Jeremiah",
        "Lam": "Lamentations",
        "Ezek": "Ezekiel", 
        "Eze": "Ezekiel",
        "Dan": "Daniel",
        "Hos": "Hosea",
        "Joel": "Joel", 
        "Joe": "Joel",
        "Amos": "Amos", 
        "Amo": "Amos",
        "Obad": "Obadiah", 
        "Oba": "Obadiah",
        "Jonah": "Jonah", 
        "Jon": "Jonah",
        "Mic": "Micah",
        "Nah": "Nahum",
        "Hab": "Habakkuk",
        "Zeph": "Zephaniah", 
        "Zep": "Zephaniah",
        "Hag": "Haggai",
        "Zech": "Zechariah", 
        "Zec": "Zechariah",
        "Mal": "Malachi",
        "Matt": "Matthew", 
        "Mat": "Matthew",
        "Mk": "Mark", 
        "Mar": "Mark",
        "Lk": "Luke", 
        "Luk": "Luke",
        "Jn": "John", 
        "Joh": "John",
        "Acts": "Acts", 
        "Act": "Acts",
        "Rom": "Romans",
        "1Cor": "1 Corinthians", 
        "1 Cor": "1 Corinthians", 
        "1Co": "1 Corinthians",
        "2Cor": "2 Corinthians", 
        "2 Cor": "2 Corinthians", 
        "2Co": "2 Corinthians",
        "Gal": "Galatians",
        "Eph": "Ephesians",
        "Phil": "Philippians", 
        "Php": "Philippians",
        "Col": "Colossians",
        "1Thess": "1 Thessalonians", 
        "1 Thess": "1 Thessalonians", 
        "1Th": "1 Thessalonians",
        "2Thess": "2 Thessalonians", 
        "2 Thess": "2 Thessalonians", 
        "2Th": "2 Thessalonians",
        "1Tim": "1 Timothy", 
        "1 Tim": "1 Timothy", 
        "1Ti": "1 Timothy",
        "2Tim": "2 Timothy", 
        "2 Tim": "2 Timothy", 
        "2Ti": "2 Timothy",
        "Tit": "Titus",
        "Phlm": "Philemon", 
        "Phm": "Philemon",
        "Heb": "Hebrews",
        "Jas": "James", 
        "Jam": "James",
        "1Pet": "1 Peter", 
        "1 Pet": "1 Peter", 
        "1Pe": "1 Peter",
        "2Pet": "2 Peter", 
        "2 Pet": "2 Peter", 
        "2Pe": "2 Peter",
        "1Jn": "1 John", 
        "1 Jn": "1 John", 
        "1Jo": "1 John",
        "2Jn": "2 John", 
        "2 Jn": "2 John", 
        "2Jo": "2 John",
        "3Jn": "3 John", 
        "3 Jn": "3 John", 
        "3Jo": "3 John",
        "Jude": "Jude", 
        "Jud": "Jude",
        "Rev": "Revelation"
    }
    
    return book_name_mapping.get(book_name, book_name)
