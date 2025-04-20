import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from typing import List, Dict, Tuple, Optional, Union
from src.bible_manager.storage import BibleStorage
import logging

# Configure logging only if not already configured
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VerseDetector(nn.Module):
    """
    A module for detecting verse patterns and resolving Bible references in text.
    
    Combines structural pattern detection (e.g., poetry, scripture) with reference
    resolution (e.g., "John 3:16") using stored Bible data. Designed for integration
    with the Bible-AI transformer model.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_verse_types: int = 5,
        dropout_rate: float = 0.1,
        bible_id: str = None,
        storage_dir: str = "data/bible_storage"
    ):
        """
        Initialize the verse detector module.

        Args:
            hidden_dim (int): Dimension of hidden representations from the transformer.
            num_verse_types (int): Number of verse types to detect (e.g., prose, poetry).
            dropout_rate (float): Dropout probability for regularization.
            bible_id (str): ID of the stored Bible data to use for reference resolution.
            storage_dir (str): Directory where Bible data is stored.
        """
        super(VerseDetector, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_verse_types = num_verse_types
        self.storage_dir = storage_dir
        
        # Load Bible data for reference resolution
        self.storage = BibleStorage(storage_dir=storage_dir)
        self.bible_data = self.storage.retrieve_bible(bible_id) if bible_id else {}
        if not self.bible_data:
            logger.warning("No Bible data loaded; reference resolution will be limited")
        
        # Book aliases for reference resolution
        self.book_aliases = {
            "gen": "Genesis", "exo": "Exodus", "lev": "Leviticus", "num": "Numbers",
            "deut": "Deuteronomy", "josh": "Joshua", "judg": "Judges", "ruth": "Ruth",
            "1sam": "1 Samuel", "2sam": "2 Samuel", "1kgs": "1 Kings", "2kgs": "2 Kings",
            "1chr": "1 Chronicles", "2chr": "2 Chronicles", "ezra": "Ezra", "neh": "Nehemiah",
            "esth": "Esther", "job": "Job", "ps": "Psalms", "prov": "Proverbs",
            "eccl": "Ecclesiastes", "song": "Song of Solomon", "isa": "Isaiah",
            "jer": "Jeremiah", "lam": "Lamentations", "ezek": "Ezekiel", "dan": "Daniel",
            "hos": "Hosea", "joel": "Joel", "amos": "Amos", "obad": "Obadiah",
            "jonah": "Jonah", "mic": "Micah", "nah": "Nahum", "hab": "Habakkuk",
            "zeph": "Zephaniah", "hag": "Haggai", "zech": "Zechariah", "mal": "Malachi",
            "matt": "Matthew", "mark": "Mark", "luke": "Luke", "john": "John",
            "acts": "Acts", "rom": "Romans", "1cor": "1 Corinthians", "2cor": "2 Corinthians",
            "gal": "Galatians", "eph": "Ephesians", "phil": "Philippians", "col": "Colossians",
            "1thess": "1 Thessalonians", "2thess": "2 Thessalonians", "1tim": "1 Timothy",
            "2tim": "2 Timothy", "titus": "Titus", "philem": "Philemon", "heb": "Hebrews",
            "jas": "James", "1pet": "1 Peter", "2pet": "2 Peter", "1john": "1 John",
            "2john": "2 John", "3john": "3 John", "jude": "Jude", "rev": "Revelation"
        }
        
        # Layers for verse pattern detection (neural network component)
        self.verse_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout_rate
        )
        self.verse_ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.verse_classifier = nn.Linear(hidden_dim, num_verse_types)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for verse pattern detection (neural network component).

        Args:
            hidden_states (torch.Tensor): Hidden representations [batch_size, seq_len, hidden_dim].
            attention_mask (torch.Tensor, optional): Mask for padding tokens [batch_size, seq_len].

        Returns:
            Dict[str, torch.Tensor]: 
                - 'verse_logits': Logits for verse type classification [batch_size, seq_len, num_verse_types].
                - 'verse_features': Enhanced features [batch_size, seq_len, hidden_dim].
        """
        key_padding_mask = (1 - attention_mask).bool() if attention_mask is not None else None
        
        # Self-attention with sparse mask
        hidden_states_t = hidden_states.transpose(0, 1)  # [seq_len, batch_size, hidden_dim]
        seq_len = hidden_states_t.size(0)
        sparse_mask = torch.ones((seq_len, seq_len), device=hidden_states.device, dtype=torch.bool)
        for i in range(seq_len):
            start = max(0, i - 5)
            end = min(seq_len, i + 6)
            sparse_mask[i, start:end] = 0
        
        attn_output, _ = self.verse_attention(
            query=hidden_states_t,
            key=hidden_states_t,
            value=hidden_states_t,
            key_padding_mask=key_padding_mask,
            attn_mask=sparse_mask
        )
        attn_output = attn_output.transpose(0, 1)  # [batch_size, seq_len, hidden_dim]
        
        # Residual and normalization
        hidden_states = self.norm1(hidden_states + self.dropout(attn_output))
        ff_output = self.verse_ff(hidden_states)
        verse_features = self.norm2(hidden_states + self.dropout(ff_output))
        
        # Classify verse types
        verse_logits = self.verse_classifier(verse_features)
        
        return {
            'verse_logits': verse_logits,
            'verse_features': verse_features
        }
    
    def detect_verse_patterns(
        self,
        text: str,
        tokenizer=None
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Detect verse patterns in raw text using rule-based methods.

        Args:
            text (str): Raw input text to analyze.
            tokenizer: Optional tokenizer (unused here but kept for future compatibility).

        Returns:
            List[Dict[str, Union[str, float]]]: Detected verse segments with types and confidence scores.
        """
        lines = text.strip().split('\n')
        results = []
        
        verse_patterns = {
            'numbered_verse': r'^\d+[:\.]\d+\s+',  # e.g., "3:16 For God so loved..."
            'poetry_stanza': r'^(\s{2,}|\t+)',     # Indented lines for poetry
            'list_item': r'^\s*[\*\-\•]\s+',       # Bullet points
            'quotation': r'^[\"\']',               # Starting with quotes
        }
        
        for line in lines:
            verse_type = 'prose'
            confidence = 0.5
            
            for pattern_name, pattern in verse_patterns.items():
                if re.match(pattern, line):
                    verse_type = pattern_name
                    confidence = 0.8
                    break
            
            results.append({
                'text': line,
                'verse_type': verse_type,
                'confidence': confidence
            })
        
        return results
    
    def resolve_reference(self, ref: str) -> Optional[Dict[str, str]]:
        """
        Resolve a Bible reference to its text using stored data.

        Args:
            ref (str): Reference like "John 3:16" or "Gen 1:1-2".

        Returns:
            Optional[Dict[str, str]]: Resolved reference with book, chapter, verse, and text,
                                      or None if invalid.
        """
        if not self.bible_data:
            logger.error("No Bible data loaded for reference resolution")
            return None
        
        # Handle range (e.g., "Gen 1:1-3") or single verse
        pattern = r"(\w+)\s*(\d+):(\d+)(?:-(\d+))?"
        match = re.match(pattern, ref.strip())
        if not match:
            logger.warning(f"Invalid reference format: {ref}")
            return None
        
        book_abbr, chapter, start_verse, end_verse = match.groups()
        book = self.book_aliases.get(book_abbr.lower(), book_abbr.capitalize())
        end_verse = end_verse or start_verse
        
        try:
            verses = {}
            for verse_num in range(int(start_verse), int(end_verse) + 1):
                verse_text = self.bible_data[book][chapter][str(verse_num)]
                verses[str(verse_num)] = verse_text
            full_text = " ".join(verses.values())
            return {
                "book": book,
                "chapter": chapter,
                "verses": verses,
                "text": full_text
            }
        except KeyError as e:
            logger.warning(f"Reference not found: {book} {chapter}:{start_verse}-{end_verse} ({str(e)})")
            return None

if __name__ == "__main__":
    detector = VerseDetector(
        hidden_dim=768,
        bible_id="2d4db0be-da7c-4cf2-aa0a-65b8ea930c20"
    )
    
    ref = detector.resolve_reference("John 3:16")
    print("Reference Resolution:", ref)
    
    sample_text = """
    1:1 In the beginning God created the heaven and the earth.
      The Lord is my shepherd; I shall not want.
    * Blessed are the meek
    "For God so loved the world"
    This is normal prose text.
    """
    patterns = detector.detect_verse_patterns(sample_text)
    print("\nVerse Patterns:")
    for pattern in patterns:
        print(pattern)