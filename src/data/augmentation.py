# bible/src/data/augmentation.py
"""
Bible data augmentation module for Bible-AI.

This module provides techniques to augment Bible text data (e.g., USFM, OSIS, JSON)
while preserving theological accuracy. Augmentations include synonym replacement,
sentence shuffling, and minor noise addition.

Dependencies:
- nltk (for synonym replacement)
- random (for shuffling and noise)
"""

import os
import json
import random
import logging
from typing import Dict, List, Any, Optional
import nltk
from nltk.corpus import wordnet

# Project-specific imports with fallbacks
try:
    from src.utils.logger import get_logger
    from src.bible_manager.converter import BibleConverter
except ImportError:
    import logging
    get_logger = lambda name: logging.getLogger(name)
    BibleConverter = None

# Download required NLTK data (run once)
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception as e:
    logging.warning("Failed to download NLTK data: %s", e)

# Initialize logger
logger = get_logger("bible_augmentation")

class BibleAugmenter:
    """
    Augments Bible text data while preserving theological integrity.

    Attributes:
        config: Configuration dictionary.
        converter: BibleConverter instance for format handling.
        safe_synonyms: Dictionary of safe synonym mappings for theological terms.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the BibleAugmenter.

        Args:
            config_path: Optional path to configuration file.
        """
        self.config = {}
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                logger.info(f"Loaded augmentation configuration from {config_path}")
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")

        self.converter = BibleConverter(config_path=config_path) if BibleConverter else None
        self.safe_synonyms = self._load_safe_synonyms()
        logger.info("BibleAugmenter initialized")

    def _load_safe_synonyms(self) -> Dict[str, List[str]]:
        """
        Load a dictionary of safe synonym mappings for theological terms.

        Returns:
            Dict: Mapping of words to safe synonym lists.
        """
        # Default safe synonyms (expandable via config)
        default_synonyms = {
            "lord": ["God", "Yahweh", "Jehovah"],
            "king": ["ruler", "sovereign"],
            "holy": ["sacred", "divine"],
            "sin": ["transgression", "wrongdoing"]
        }
        return {**default_synonyms, **self.config.get("safe_synonyms", {})}

    def _get_synonyms(self, word: str) -> List[str]:
        """
        Get safe synonyms for a word, prioritizing theological accuracy.

        Args:
            word: Word to find synonyms for.

        Returns:
            List: List of safe synonyms.
        """
        word = word.lower()
        synonyms = self.safe_synonyms.get(word, [])
        if not synonyms:
            syn_sets = wordnet.synsets(word)
            synonyms = [lemma.name() for syn in syn_sets for lemma in syn.synonyms() if lemma.name().lower() != word]
            synonyms = [syn for syn in synonyms if syn in self.safe_synonyms.get(word, []) or len(syn.split()) == 1]
        return synonyms[:3]  # Limit to 3 to avoid excessive variation

    def _augment_text(self, text: str, intensity: float = 0.2) -> str:
        """
        Augment a single text string with synonym replacement and noise.

        Args:
            text: Input text to augment.
            intensity: Probability of applying augmentation (0 to 1).

        Returns:
            str: Augmented text.
        """
        words = text.split()
        augmented_words = []
        for word in words:
            if random.random() < intensity and word.lower() not in ["god", "jesus", "christ"]:  # Protect key terms
                synonyms = self._get_synonyms(word)
                if synonyms and random.random() < 0.5:  # 50% chance to replace
                    augmented_words.append(random.choice(synonyms))
                else:
                    augmented_words.append(word)
            else:
                augmented_words.append(word)

        # Add minor noise (e.g., extra space or punctuation)
        if random.random() < intensity / 2:
            pos = random.randint(0, len(augmented_words) - 1)
            augmented_words.insert(pos, random.choice(["", ".", ","]))

        return " ".join(augmented_words).strip()

    def _shuffle_sentences(self, text: str, intensity: float = 0.2) -> str:
        """
        Shuffle sentences in the text with controlled intensity.

        Args:
            text: Input text to shuffle.
            intensity: Probability of shuffling (0 to 1).

        Returns:
            str: Shuffled text.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if len(sentences) > 1 and random.random() < intensity:
            shuffle_count = max(1, int(len(sentences) * intensity))
            for _ in range(shuffle_count):
                i, j = random.sample(range(len(sentences)), 2)
                sentences[i], sentences[j] = sentences[j], sentences[i]
        return " ".join(sentences)

    def augment_bible_data(self, bible_data: Dict[str, Any], intensity: float = 0.2, max_augmentations: int = 3) -> List[Dict[str, Any]]:
        """
        Augment Bible data with multiple variations.

        Args:
            bible_data: Input Bible data in standardized format.
            intensity: Probability of applying augmentation per text segment.
            max_augmentations: Maximum number of augmented versions to generate.

        Returns:
            List: List of augmented Bible data dictionaries.
        """
        if not bible_data.get("books"):
            logger.warning("No books found in bible_data")
            return [bible_data]

        augmented_data = [bible_data.copy()]
        for _ in range(min(max_augmentations, len(bible_data["books"]))):
            new_data = bible_data.copy()
            for book in new_data["books"]:
                for chapter in book["chapters"]:
                    for verse in chapter["verses"]:
                        verse["text"] = self._augment_text(verse["text"], intensity)
                        verse["text"] = self._shuffle_sentences(verse["text"], intensity)
                        # Ensure text is not empty
                        if not verse["text"].strip():
                            verse["text"] = bible_data["books"][book_idx]["chapters"][chapter_idx]["verses"][verse_idx]["text"]
            augmented_data.append(new_data)

        logger.info("Generated %d augmented versions", len(augmented_data) - 1)
        return augmented_data

    def save_augmentations(self, augmented_data: List[Dict[str, Any]], base_path: str) -> List[str]:
        """
        Save augmented Bible data to files.

        Args:
            augmented_data: List of augmented Bible data dictionaries.
            base_path: Base directory to save augmented files.

        Returns:
            List: List of file paths where augmentations were saved.
        """
        paths = []
        os.makedirs(base_path, exist_ok=True)
        for i, data in enumerate(augmented_data):
            file_path = os.path.join(base_path, f"augmented_bible_{i}.json")
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                paths.append(file_path)
                logger.info("Saved augmented data to %s", file_path)
            except Exception as e:
                logger.error("Failed to save augmented data to %s: %s", file_path, e)
        return paths

if __name__ == "__main__":
    import argparse
    import re

    parser = argparse.ArgumentParser(description="Augment Bible text data for Bible-AI")
    parser.add_argument("--input", type=str, required=True, help="Path to input Bible JSON file")
    parser.add_argument("--output", type=str, required=True, help="Base directory to save augmented files")
    parser.add_argument("--intensity", type=float, default=0.2, help="Augmentation intensity (0 to 1)")
    parser.add_argument("--max-augmentations", type=int, default=3, help="Maximum number of augmentations")
    parser.add_argument("--config", type=str, default="config/bible_sources.json", help="Path to configuration file")
    args = parser.parse_args()

    # Load input Bible data
    with open(args.input, 'r', encoding='utf-8') as f:
        bible_data = json.load(f)

    # Initialize augmenter
    augmenter = BibleAugmenter(config_path=args.config)

    # Augment data
    augmented_data = augmenter.augment_bible_data(bible_data, args.intensity, args.max_augmentations)

    # Save augmentations
    file_paths = augmenter.save_augmentations(augmented_data, args.output)
    print(f"Augmented files saved at: {file_paths}")
