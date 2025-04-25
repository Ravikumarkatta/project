"""
Biblical Text Tokenizer
- Verse reference preservation (Rom 3:16-18)
- Theological term handling (YHWH[_FULL] → [LORD])
- Configurable memory mapping
- Safe loading with binary format support
- PyTorch Dataloader integration
"""

import re
import os
import logging
from io import BytesIO
from typing import Dict, Optional, List, Union, Any, Tuple
import torch
from torch.serialization import load, SourceChangeWarning
from transformers import AutoTokenizer, PreTrainedTokenizer, logging as hf_logging
from pydantic import BaseModel
import safetensors.torch as safe

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
hf_logging.set_verbosity_warning()

# Terminal colors
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Book abbreviation mapping
BOOK_ABBS: Dict[str, str] = {
    # Old Testament
    "Gen": "Genesis", "Ex": "Exodus", "Lev": "Leviticus", 
    "Num": "Numbers", "Deut": "Deuteronomy", "Josh": "Joshua",
    "Judg": "Judges", "Ruth": "Ruth", "1Sm": "1 Samuel",
    "2Sm": "2 Samuel", "1Kgs": "1 Kings", "2Kgs": "2 Kings",
    "1Chr": "1 Chronicles", "2Chr": "2 Chronicles", "Ezra": "Ezra",
    "Neh": "Nehemiah", "Esth": "Esther", "Job": "Job",
    "Ps": "Psalms", "Pr": "Proverbs", "Eccl": "Ecclesiastes",
    "Song": "Song of Solomon", "Isa": "Isaiah", "Jer": "Jeremiah",
    "Lam": "Lamentations", "Ezek": "Ezekiel", "Dan": "Daniel",
    "Hos": "Hosea", "Joel": "Joel", "Amos": "Amos", "Obad": "Obadiah",
    "Jonah": "Jonah", "Mic": "Micah", "Nah": "Nahum", "Hab": "Habakkuk",
    "Zeph": "Zephaniah", "Hag": "Haggai", "Zech": "Zechariah",
    "Mal": "Malachi", "Matt": "Matthew", "Mark": "Mark", 
    "Lk": "Luke", "Jn": "John", "Acts": "Acts", "Rom": "Romans",
    "1Cor": "1 Corinthians", "2Cor": "2 Corinthians", "Gal": "Galatians",
    "Eph": "Ephesians", "Phil": "Philippians", "Col": "Colossians",
    "1Thess": "1 Thessalonians", "2Thess": "2 Thessalonians",
    "1Tim": "1 Timothy", "2Tim": "2 Timothy", "Titus": "Titus",
    "Phlm": "Philemon", "Heb": "Hebrews", "Jas": "James",
    "1Pet": "1 Peter", "2Pet": "2 Peter", "1Jn": "1 John",
    "2Jn": "2 John", "3Jn": "3 John", "Jude": "Jude", 
    "Rev": "Revelation"
}

class TokenizerConfig(BaseModel):
    """Configuration schema for tokenizer validation"""
    
    max_tokens: int = 512
    safe_load: bool = True
    device: str = "cuda"
    book_abbreviations: Dict[str, str] = BOOK_ABBS
    special_terms: List[str] = ["YHWH", "Trinity", "Ark", "Law"]
    verse_pattern: str = r"\b([Jj]n|[Mm]att|[Rr]om|[Rr]ev?[[:alnum:]])[:,.]\s*\d+[:.]\d+"
    output_type: str = "pt"
    encoding: str = "utf-8"

class BibleTokenizer:
    """
    Tokenizer specialized for biblical texts with:
    - Verse reference preservation (e.g., "Rev 1:7" → "__VREF_001__")
    - Theological term standardization (e.g., "YHWH" → "[LORD]")
    - Book abbreviation expansion (e.g., "1Chr" → "1 Chronicles")
    - Memory-efficient loading for large checkpoints
    - Integrated error recovery and validation

    Attributes:
        base_tokenizer: Underlying HuggingFace tokenizer
        config: Tokenizer configuration
        verse_pattern: Compiled regex pattern for verse references
        term_replacer: Dictionary for special term replacements
    """
    
    def __init__(
        self,
        base_model: str = "sentence-transformers/LaBSE",
        config_path: Optional[str] = None,
        disable_warnings: bool = False
    ):
        """
        Initialize the Bible tokenizer
        
        Args:
            base_model: HuggingFace model name to use as base
            config_path: Optional custom configuration path
            disable_warnings: Suppress warning messages
        """
        self._setup_logging(disable_warnings)
        self.config = self._load_config(config_path)
        self.term_replacer = {
            k: f"[{k.upper()}]" for k in self.config.special_terms
        }
        
        # Initialize base tokenizer
        self.base_tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            use_fast=True,
            trust_remote_code=True
        )
        
        # Compile regex patterns
        self.verse_pattern = re.compile(self.config.verse_pattern)
        self.book_pattern = re.compile(
            r"\b(" + "|".join(map(re.escape, self.config.book_abbreviations.keys())) + r")\b",
            flags=re.IGNORECASE
        )
        
        logger.info(f"Initialized BibleTokenizer using {base_model}")

    def _setup_logging(self, disable_warnings: bool):
        """ Configure logging levels and warnings """
        if disable_warnings:
            logger.setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.ERROR)
            
        format_str = "%(asctime)s - [%(levelname)s] %(name)s: %(message)s"
        logging.basicConfig(format=format_str, level=logging.INFO)

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load validated configuration from file or defaults"""
        config = TokenizerConfig()
        
        if config_path:
            try:
                with open(config_path) as f:
                    data = json.load(f)
                merged = config.model_copy(update=data)
                return merged
            except Exception as e:
                logger.error(f"Config loading failed: {str(e)}")
                raise ConfigurationError("Invalid configuration file") from e
        else:
            logger.warning("Using default configuration - consider providing config file")
            return config

    def normalize_text(self, text: str) -> str:
        """
        Process text to handle:
        1. Verse reference standardization
        2. Book abbreviation expansion
        3. Special term formatting
        
        Args:
            text: Raw input text
            
        Returns:
            Processed text with standardized elements
        """
        try:
            steps = 0
            original_len = len(text)
            
            # Step 1: Normalize verse references
            verse_replaced, count = self._replace_verses(text)
            steps += count
            
            # Step 2: Expand book abbreviations
            expanded = self._expand_abbreviations(verse_replaced)
            steps += 1
            
            # Step 3: Replace special theological terms
            final_text, term_count = self._replace_special_terms(expanded)
            steps += term_count
            
            logger.debug(
                f"Processed text in {steps} steps: {len(final_text)/original_len:.2%} length preserved"
            )
            return final_text
            
        except Exception as e:
            logger.error(f"Text normalization failed: {str(e)}")
            raise RuntimeError("Critical normalization error") from e

    def _replace_verses(self, text: str) -> Tuple[str, int]:
        """Convert verse references to standardized format"""
        replacement_map = {}
        count = 0
        
        # Process verse patterns
        for match in self.verse_pattern.finditer(text):
            full_ref = match.group()
            cleaned_ref = re.sub(r"[^\w:]", "", full_ref)
            replacement = f"__VREF_{count:03d}__"
            text = text.replace(full_ref, replacement)
            replacement_map[replacement] = cleaned_ref
            count += 1

        # Process book abbreviations
        for abbrev, full_name in self.config.book_abbreviations.items():
            pattern = re.compile(r"\b" + re.escape(abbrev) + r"\b", re.IGNORECASE)
            if pattern.search(text):
                text = pattern.sub(full_name, text)
                count += 1

        logger.info(f"Replaced {count} verse references")
        return text, count

    def _expand_abbreviations(self, text: str) -> str:
        """Expand all book abbreviations to full names"""
        for abbrev, full in self.config.book_abbreviations.items():
            if abbrev in ("Matt", "Matt"):
                text = re.sub(
                    r"\b(M[m]a?t?t?r?[s]?)\b", 
                    lambda m: self.config.book_abbreviations.get(m.group(1), m.group(1)),
                    text,
                    flags=re.IGNORECASE
                )
            else:
                text = re.sub(
                    r"\b" + re.escape(abbrev) + r"\b",
                    self.config.book_abbreviations[abbrev],
                    text,
                    flags=re.IGNORECASE
                )
        return text

    def _replace_special_terms(self, text: str) -> Tuple[str, int]:
        """Replace theological terms with standardized tokens"""
        count = 0
        for term in self.config.special_terms:
            term = term.lower()
            pattern = re.compile(
                rf"(?i)\b({term})\b",
                flags=re.IGNORECASE | re.MULTILINE
            )
            if matches := pattern.finditer(text):
                for match in matches:
                    term_start, term_end = match.span()
                    preceding_space = text[max(0, term_start-1):term_start].isspace()
                    new_term = f"[{term.upper()}]"
                    if preceding_space:
                        text = text[:term_start] + new_term + text[term_end:]
                    else:
                        text = text[:term_start] + " " + new_term + text[term_end:]
                    count += 1
        logger.debug(f"Replaced {count} special terms")
        return text, count

    def tokenize(self, text: str, return_tensors: str = "pt") -> Dict[str, Any]:
        """
        Main tokenization interface
        
        Args:
            text: Input text to tokenize
            return_tensors: Return type ("pt", "np", or "jax")
            
        Returns:
            Dictionary containing:
            - input_ids: Tokenized sequence ids
            - attention_mask: Mask for valid tokens
            - special_tokens: Dictionary of replaced special terms
        """
        try:
            # Normalization pipeline
            processed = self.normalize_text(text)
            
            # Base tokenization
            encoding = self.base_tokenizer(
                processed,
                max_length=self.config.max_tokens,
                padding="longest",
                truncation=True,
                return_tensors=return_tensors
            )
            
            return {
                **encoding,
                "special_tokens": self._collect_special_tokens(processed)
            }
            
        except Exception as e:
            logger.error(f"Tokenization failed: {str(e)}")
            raise RuntimeError("Fatal tokenization error") from e

    def load_state_dict(self, checkpoint_path: str, **kwargs) -> Dict:
        """
        Safe state dictionary loading with recovery options
        
        Args:
            checkpoint_path: Path to model checkpoint
            kwargs: Additional arguments for torch.load
            
        Returns:
            Dictionary of loaded tensors
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: For non-recoverable errors
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint file not found: {checkpoint_path}\n"
                f"Ensure you've run:\n"
                f"python -m torch.serialization.save_module ...  # or appropriate saving method",
                bcolors.FAIL
            )
        
        try:
            return torch.load(
                checkpoint_path,
                map_location=kwargs.get("map_location", "cpu"),
                weights_only=kwargs.get("weights_only", True),
                pickle_module=kwargs.get("pickle_module", pickle),
            )
            
        except Exception as e:
            error_msg = str(e)
            logger.critical(f"Checkpoint load failed: {error_msg}")

            # Attempt recovery routes:
            recovery_steps = [
                self._try_mmap_load(checkpoint_path, **kwargs),
                self._try_safetensors_load(checkpoint_path, **kwargs),
                self._try_legacy_load(checkpoint_path, **kwargs),
            ]
            
            for i, (result, error) in enumerate(zip(recovery_steps, self._get_recovery_errors())):
                if result is not None:
                    logger.warning(f"Recovery step {i+1} succeeded with warning: {error}")
                    return result
                logger.debug(f"Recovery step {i+1} failed: {error}")

            raise RuntimeError(
                f"Failed all recovery attempts for: {checkpoint_path}\n"
                f"Error details: '{error_msg}'"
            ) from e

    def _try_mmap_load(self, path: str, **kwargs) -> Optional[Dict]:
        """Attempt memory-mapped loading"""
        try:
            from torch._utils import _getillow_version
            if _getpillow_version() < (8,):
                raise RuntimeError(f"Pillow >=8.0 required, version {_getpillow_version()} found")
            
            with open(path, "rb") as f:
                return torch.load(f, mmap=True, **kwargs)
                
        except Exception as e:
            logger.warning(
                "Memory-mapped loading failed: %s\nFallback required", 
                str(e)
            )
            return None

    def _try_safetensors_load(self, path: str, **kwargs) -> Optional[Dict]:
        """Attempt loading from safetensors format"""
        try:
            return safe.load_file(path, **kwargs)
        except Exception as e:
            logger.debug(f"Safetensors load failed: {str(e)}")
            return None

    def _try_legacy_load(self, path: str, **kwargs) -> Optional[Dict]:
        """Legacy pickle format fallback"""
        try:
            from torch.serialization import pickle_load
            with open(path, "rb") as f:
                return pickle_load(f, **kwargs)
        except Exception as e:
            logger.error(f"Legacy loading failed: {str(e)}")
            return None

    def _get_recovery_errors(self) -> List[str]:
        """Predefined recovery error messages"""
        return [
            "MMAP failed: File might be corrupted or incompatible",
            "Safetensors load failed: Format mismatch or version issue",
            "Legacy load failed: Unsupported pickle protocol or data corruption"
        ]


# Default instance
default_tokenizer = BibleTokenizer()

def normalize_biblical_text(text: str) -> str:
    """Wrapper function for external API"""
    return default_tokenizer.normalize_text(text)

def tokenize_biblical_text(
    text: str,
    return_tensors: str = "pt",
) -> Dict[str, Union[List[int], torch.Tensor]]:
    """Unified API for external use"""
    return default_tokenizer.tokenize(text, return_tensors=return_tensors)
                
