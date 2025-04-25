import re
import os
import logging
from io import BytesIO
from typing import Dict, Optional, List, Union, Any, Tuple
import torch
from torch.serialization import load, SourceChangeWarning
from transformers import AutoTokenizer, PreTrainedTokenizer, logging as hf_logging
from pydantic import BaseModel, ValidationError # Import ValidationError
import safetensors.torch as safe
import json
import pickle
import sys # Import sys for Python version check
import warnings # Import warnings to catch SourceChangeWarning

# --- Configuration and Setup ---

# Configure logging for the module
logging.basicConfig(
    level=logging.INFO, # Set default logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__) # Logger specific to this module
hf_logging.set_verbosity_warning() # Keep HuggingFace warnings at warning level

# Define custom exception for configuration errors
class ConfigurationError(Exception):
    """Custom exception raised for errors in tokenizer configuration."""
    pass

# Define the configuration schema using Pydantic for validation
class TokenizerConfig(BaseModel):
    """
    Configuration schema for the BibleTokenizer.

    Attributes:
        max_tokens (int): Maximum sequence length for tokenization.
        safe_load (bool): Whether to attempt safe loading for checkpoints (currently not fully utilized in load_state_dict).
        device (str): Default device for PyTorch tensors ('cuda' or 'cpu').
        book_abbreviations (Dict[str, str]): Mapping of book abbreviations to full names.
        special_terms (List[str]): List of theological terms to standardize.
        verse_pattern (str): Regex pattern for identifying verse references.
        output_type (str): Desired output tensor type ('pt', 'np', 'jax').
        encoding (str): File encoding for reading configuration files.
    """
    max_tokens: int = 512
    safe_load: bool = True # Note: This attribute is currently not actively used in load_state_dict logic but kept for config schema.
    device: str = "cuda" if torch.cuda.is_available() else "cpu" # Default to cuda if available
    # Default book abbreviations (can be overridden by config file)
    book_abbreviations: Dict[str, str] = {
        "Gen": "Genesis", "Ex": "Exodus", "Lev": "Leviticus", "Num": "Numbers",
        "Deut": "Deuteronomy", "Josh": "Joshua", "Judg": "Judges", "Ruth": "Ruth",
        "1Sm": "1 Samuel", "2Sm": "2 Samuel", "1Kgs": "1 Kings", "2Kgs": "2 Kings",
        "1Chr": "1 Chronicles", "2Chr": "2 Chronicles", "Ezra": "Ezra", "Neh": "Nehemiah",
        "Esth": "Esther", "Job": "Job", "Ps": "Psalms", "Pr": "Proverbs",
        "Eccl": "Ecclesiastes", "Song": "Song of Solomon", "Isa": "Isaiah", "Jer": "Jeremiah",
        "Lam": "Lamentations", "Ezek": "Ezekiel", "Dan": "Daniel", "Hos": "Hosea",
        "Joel": "Joel", "Amos": "Amos", "Obad": "Obadiah", "Jonah": "Jonah",
        "Mic": "Micah", "Nah": "Nahum", "Hab": "Habakkuk", "Zeph": "Zephaniah",
        "Hag": "Haggai", "Zech": "Zechariah", "Mal": "Malachi",
        "Matt": "Matthew", "Mark": "Mark", "Lk": "Luke", "Jn": "John", "Acts": "Acts",
        "Rom": "Romans", "1Cor": "1 Corinthians", "2Cor": "2 Corinthians", "Gal": "Galatians",
        "Eph": "Ephesians", "Phil": "Philippians", "Col": "Colossians",
        "1Thess": "1 Thessalonians", "2Thess": "2 Thessalonians", "1Tim": "1 Timothy",
        "2Tim": "2 Timothy", "Titus": "Titus", "Phlm": "Philemon", "Heb": "Hebrews",
        "Jas": "James", "1Pet": "1 Peter", "2Pet": "2 Peter", "1Jn": "1 John",
        "2Jn": "2 John", "3Jn": "3 John", "Jude": "Jude", "Rev": "Revelation"
    }
    # Regex pattern to capture common verse reference formats (e.g., Gen 1:1, Jn 3:16-18, Rev 21.1)
    # It dynamically includes all defined book abbreviations.
    verse_pattern: str = "" # Will be set in __init__ based on book_abbreviations
    special_terms: List[str] = ["YHWH", "Trinity", "Ark", "Law", "Covenant", "Grace"] # Added more default terms
    output_type: str = "pt" # Default to PyTorch tensors
    encoding: str = "utf-8" # Default file encoding

    # Pydantic model validation - ensures verse_pattern is generated after book_abbreviations are loaded
    def model_post_init(self, __context: Any) -> None:
        """Post-initialization hook to generate the verse pattern."""
        # Generate the verse pattern dynamically based on the loaded abbreviations
        book_abbs_escaped = "|".join(re.escape(k) for k in self.book_abbreviations.keys())
        # Pattern: word boundary, followed by a book abbreviation (case-insensitive),
        # optional space, chapter number, colon/period, verse number,
        # optional range (- followed by number), optional sub-verse (: followed by number), word boundary.
        self.verse_pattern = r"\b(?:{})\s*\d+[:.]\d+(?:-\d+)?(?:[:.]\d+)?\b".format(book_abbs_escaped)
        logger.debug(f"Generated verse pattern: {self.verse_pattern}")

# --- Main Tokenizer Class ---

class BibleTokenizer:
    """
    A specialized tokenizer for biblical texts designed to handle:
    - Preservation and standardization of verse references (e.g., "Rev 1:7" -> "__VREF_001__").
    - Standardization of theological terms (e.g., "YHWH" -> "[LORD]").
    - Expansion of book abbreviations (e.g., "1Chr" -> "1 Chronicles").
    - Configuration loading and validation.
    - Safe and memory-efficient loading of model checkpoints.
    - Integration with HuggingFace tokenizers and PyTorch.

    Attributes:
        config (TokenizerConfig): The validated configuration object.
        base_tokenizer (PreTrainedTokenizer): The underlying HuggingFace tokenizer.
        verse_pattern_compiled (re.Pattern): Compiled regex pattern for verse references.
        book_pattern_compiled (re.Pattern): Compiled regex pattern for book abbreviations.
        term_replacer (Dict[str, str]): Mapping of lowercase special terms to their standardized tokens.
        _verse_map (Dict[str, str]): Internal mapping of verse reference placeholders to original references for the last processed text.
        _term_map (Dict[str, str]): Internal mapping of term tokens to original terms for the last processed text (needs refinement).
    """

    def __init__(
        self,
        base_model: str = "sentence-transformers/LaBSE", # Default base model
        config_path: Optional[str] = None,
        disable_warnings: bool = False
    ):
        """
        Initializes the BibleTokenizer.

        Args:
            base_model (str): The name or path of the HuggingFace model to use as the base tokenizer.
            config_path (Optional[str]): Path to a JSON configuration file. If None, uses default settings.
            disable_warnings (bool): If True, suppresses warnings from this tokenizer and HuggingFace.
        """
        self._setup_logging(disable_warnings)
        self.config = self._load_config(config_path)

        # Compile regex patterns based on the loaded configuration
        self.verse_pattern_compiled = re.compile(self.config.verse_pattern, flags=re.IGNORECASE)
        # Pattern for matching book abbreviations (case-insensitive, whole word)
        book_abbs_escaped = "|".join(re.escape(k) for k in self.config.book_abbreviations.keys())
        self.book_pattern_compiled = re.compile(r"\b(" + book_abbs_escaped + r")\b", flags=re.IGNORECASE)

        # Create the term replacer dictionary
        self.term_replacer = {
            term.lower(): f"[{term.upper()}]" for term in self.config.special_terms
        }

        # Initialize the base HuggingFace tokenizer
        try:
            self.base_tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                use_fast=True, # Prefer fast tokenizers if available
                trust_remote_code=True # Allow loading models with custom code (use with caution)
            )
            logger.info(f"Initialized base tokenizer: {base_model}")
        except Exception as e:
            logger.critical(f"Failed to load base tokenizer '{base_model}': {e}")
            raise RuntimeError(f"Could not load base tokenizer: {base_model}") from e

        # Internal maps to store original text for placeholders/tokens for the *last* processed text
        self._verse_map: Dict[str, str] = {}
        self._term_map: Dict[str, str] = {} # Note: Capturing original terms needs refinement in replacement logic

        logger.info("BibleTokenizer initialized successfully.")

    def _setup_logging(self, disable_warnings: bool):
        """Configures logging levels based on the disable_warnings flag."""
        if disable_warnings:
            logger.setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.ERROR)
            # Optionally suppress other libraries if they are too noisy
            # logging.getLogger("some_other_library").setLevel(logging.ERROR)
        else:
            logger.setLevel(logging.INFO)
            logging.getLogger("transformers").setLevel(logging.WARNING)


    def _load_config(self, config_path: Optional[str]) -> TokenizerConfig:
        """
        Loads and validates the configuration from a JSON file or returns the default config.

        Args:
            config_path (Optional[str]): Path to the JSON configuration file.

        Returns:
            TokenizerConfig: The validated configuration object.

        Raises:
            ConfigurationError: If the config file is not found, invalid JSON, or validation fails.
        """
        if config_path:
            logger.info(f"Attempting to load configuration from {config_path}")
            try:
                with open(config_path, 'r', encoding=self.config.encoding) as f:
                    data = json.load(f)
                # Validate the loaded data using the Pydantic model
                loaded_config = TokenizerConfig(**data)
                logger.info("Configuration loaded and validated successfully.")
                # Merge loaded config with defaults, prioritizing loaded values
                # This ensures any missing keys in the file use the default values
                default_config = TokenizerConfig() # Get default instance to merge from
                merged_config_data = default_config.model_dump() # Start with defaults
                merged_config_data.update(loaded_config.model_dump()) # Update with loaded values
                merged_config = TokenizerConfig(**merged_config_data) # Create new config object from merged data
                return merged_config
            except FileNotFoundError:
                logger.error(f"Configuration file not found at {config_path}")
                raise ConfigurationError(f"Configuration file not found: {config_path}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON format in configuration file {config_path}: {e}")
                raise ConfigurationError(f"Invalid JSON format in configuration file: {e}") from e
            except ValidationError as e:
                logger.error(f"Configuration validation failed for {config_path}: {e.errors()}")
                raise ConfigurationError(f"Configuration validation failed: {e.errors()}") from e
            except Exception as e:
                logger.error(f"An unexpected error occurred while loading config {config_path}: {e}")
                raise ConfigurationError(f"Unexpected error loading configuration: {e}") from e
        else:
            logger.warning("No configuration path provided. Using default configuration.")
            return TokenizerConfig() # Return default config object

    def normalize_text(self, text: str) -> str:
        """
        Applies a series of transformations to the input text:
        1. Standardizes verse references.
        2. Expands book abbreviations.
        3. Replaces special theological terms with standardized tokens.
        Populates _verse_map and _term_map during this process.

        Args:
            text (str): The raw input text string.

        Returns:
            str: The processed text with standardized elements.

        Raises:
            RuntimeError: If a critical error occurs during normalization.
        """
        if not isinstance(text, str):
            logger.error(f"Input to normalize_text must be a string, but received {type(text)}")
            raise TypeError("Input text must be a string.")

        # Reset maps for the new text
        self._verse_map = {}
        self._term_map = {}

        try:
            logger.debug("Starting text normalization.")
            original_len = len(text)

            # Step 1: Standardize verse references
            # We replace verses first to prevent abbreviation expansion from affecting verse patterns.
            text_after_verses, verse_count = self._replace_verses(text)
            logger.debug(f"Step 1 (Verse Replacement) complete. Replaced {verse_count} references.")

            # Step 2: Expand book abbreviations
            # Applied to the text after verse replacement.
            text_after_abbs = self._expand_abbreviations(text_after_verses)
            logger.debug("Step 2 (Abbreviation Expansion) complete.")

            # Step 3: Replace special theological terms
            # Applied to the text after abbreviation expansion.
            final_text, term_count = self._replace_special_terms(text_after_abbs)
            logger.debug(f"Step 3 (Term Replacement) complete. Replaced {term_count} terms.")

            logger.info(f"Text normalization finished. Original length: {original_len}, Final length: {len(final_text)}")
            return final_text

        except Exception as e:
            logger.critical(f"Text normalization failed: {e}", exc_info=True) # Log traceback
            raise RuntimeError("Critical text normalization error.") from e

    def _replace_verses(self, text: str) -> Tuple[str, int]:
        """
        Finds and replaces verse references with standardized placeholders (e.g., __VREF_001__).
        Stores a mapping from placeholder to original reference in _verse_map.

        Args:
            text (str): The input text.

        Returns:
            Tuple[str, int]: The text with verses replaced and the count of replacements.
        """
        count = 0
        processed_text = text

        # Find all matches first to avoid issues with replacement changing indices
        matches = list(self.verse_pattern_compiled.finditer(text))

        # Sort matches by start position descending to replace from end to beginning
        # This prevents earlier replacements from affecting the indices of later matches.
        sorted_matches = sorted(matches, key=lambda m: m.start(), reverse=True)

        for i, match in enumerate(sorted_matches):
            full_ref = match.group()
            # Create a unique placeholder based on the order of appearance in the original text (after sorting)
            # Using original match order might be more intuitive, but replacing from end requires sorting.
            # Let's use the index from the sorted list for the placeholder.
            # Placeholder index reflects the order in the *sorted* list of matches.
            placeholder = f"__VREF_{len(sorted_matches) - 1 - i:03d}__"

            # Store the mapping from the placeholder to the original reference
            self._verse_map[placeholder] = full_ref

            # Replace the original reference with the placeholder in the text
            start, end = match.span()
            processed_text = processed_text[:start] + placeholder + processed_text[end:]
            count += 1

        logger.debug(f"Replaced {count} verse references with placeholders.")
        return processed_text, count

    def _expand_abbreviations(self, text: str) -> str:
        """
        Expands book abbreviations to their full names (e.g., "Gen" -> "Genesis").
        Uses the compiled book pattern.

        Args:
            text (str): The input text (potentially with verse placeholders).

        Returns:
            str: The text with abbreviations expanded.
        """
        processed_text = text
        # Find all matches first
        matches = list(self.book_pattern_compiled.finditer(text))

        # Sort matches by start position descending to replace from end to beginning
        sorted_matches = sorted(matches, key=lambda m: m.start(), reverse=True)

        for match in sorted_matches:
            abbrev = match.group(1) # Capture the abbreviation
            # Get the full name, using the original abbreviation if not found (shouldn't happen with regex)
            # Capitalize the matched abbreviation to match the dictionary keys
            full_name = self.config.book_abbreviations.get(abbrev.capitalize(), abbrev)

            start, end = match.span()
            # Replace the abbreviation with the full name
            processed_text = processed_text[:start] + full_name + processed_text[end:]

        logger.debug("Expanded book abbreviations.")
        return processed_text

    def _replace_special_terms(self, text: str) -> Tuple[str, int]:
        """
        Replaces defined special theological terms with standardized tokens (e.g., "YHWH" -> "[LORD]").
        Uses the term_replacer mapping. Stores mapping in _term_map (Note: Refinement needed for uniqueness).

        Args:
            text (str): The input text (potentially with verse placeholders and expanded abbreviations).

        Returns:
            Tuple[str, int]: The text with special terms replaced and the count of replacements.
        """
        count = 0
        processed_text = text

        # Sort terms by length descending to handle cases where one term is a substring of another
        sorted_terms = sorted(self.term_replacer.keys(), key=len, reverse=True)

        for term_lower in sorted_terms:
            new_token = self.term_replacer[term_lower]
            # Create a regex pattern for the specific term with word boundaries and case-insensitivity
            pattern = re.compile(rf"\b{re.escape(term_lower)}\b", flags=re.IGNORECASE)

            # Find all occurrences of this term in the current state of the text
            matches = list(pattern.finditer(processed_text))

            # Replace from end to start to avoid index issues
            for match in reversed(matches):
                start, end = match.span()
                original_term_match = match.group() # Capture the original matched text (e.g., "YHWH", "yHwH")

                # Note: Storing the original term mapping here.
                # If multiple original terms map to the same token (e.g., "Lord" and "LORD" both map to "[LORD]"),
                # this mapping will only store the last one encountered for that token.
                # A more robust approach might involve unique tokens or a list of original terms per token.
                # For now, we'll just store the mapping, acknowledging this limitation.
                self._term_map[new_token] = original_term_match # Store the mapping

                # Check if there's a space before the replacement to avoid creating double spaces
                preceding_char = processed_text[start - 1] if start > 0 else ''
                if preceding_char.isspace():
                     processed_text = processed_text[:start] + new_token + processed_text[end:]
                else:
                    # Add a space before the token if there isn't one, to ensure separation
                    processed_text = processed_text[:start] + " " + new_token + processed_text[end:]
                count += 1

        logger.debug(f"Replaced {count} special terms with tokens.")
        return processed_text, count

    def get_special_token_map(self) -> Dict[str, str]:
        """
        Returns the mapping of special tokens (verse placeholders, term tokens)
        to their original text segments from the *last* processed text.

        Note: This map is popu
