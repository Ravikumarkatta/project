"""
Ultimate Text & Biblical Data Augmentation Module for Bible-AI

This module merges the best features from previous implementations:
- Generic augmentation: Synonym replacement, random deletion, random swap, random insertion
- Biblical augmentation: Verse shuffling, translation swap, context expansion, theological validation
- Optimized for Bible-AI integration with converter, storage, and validator modules

Dependencies:
- nltk (for synonym replacement, POS tagging, tokenization)
- concurrent.futures (for parallel processing)
- jsonschema (optional, for config validation)
- Custom modules: src.utils.logger, src.bible_manager.converter, src.bible_manager.storage,
  src.theology.validator
"""

import json
import logging
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Set, Any

# NLTK setup - ensure nltk and types-nltk are installed
import nltk
from nltk.corpus import wordnet
from nltk.tag import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)

# Optional: jsonschema for config validation
try:
    from jsonschema import validate
except ImportError:
    validate = lambda instance, schema: None  # Fallback if jsonschema is missing

# Project-specific imports with mock fallbacks
try:
    from src.bible_manager.converter import BibleConverter
    from src.bible_manager.storage import BibleStorage
    from src.theology.validator import TheologicalValidator
    from src.utils.logger import get_logger
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    get_logger = lambda name: logging.getLogger(name)
    logger = get_logger("UltimateAugmenter")
    logger.warning("Missing dependencies: %s. Some features may be limited.", e)
    class MockBibleConverter:
        def __init__(self, config_path: Optional[str] = None):
            pass

    class MockBibleStorage:
        def __init__(self, config_path: Optional[str] = None):
            pass
        def save(self, data: Dict[str, Any], path: str) -> None:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    class MockTheologicalValidator:
        def __init__(self, config: str):
            pass
        def validate(self, text: Any) -> float:
            logger.warning("Using mock validator; assuming valid.")
            return 1.0

    BibleConverter = MockBibleConverter
    BibleStorage = MockBibleStorage
    TheologicalValidator = MockTheologicalValidator

logger = get_logger("UltimateAugmenter")

# Config schema for validation
CONFIG_SCHEMA = {
    "type": "object",
    "required": ["translation_paths", "theological_terms"],
    "properties": {
        "translation_paths": {"type": "object"},
        "theological_terms": {"type": "array", "items": {"type": "string"}},
        "prob_synonym_replacement": {"type": "number"},
        "max_synonym_replacements": {"type": "integer"},
        "prob_deletion": {"type": "number"},
        "prob_swap": {"type": "number"},
        "prob_insertion": {"type": "number"},
        "prob_verse_shuffle": {"type": "number"},
        "prob_translation_swap": {"type": "number"},
        "min_context_verses": {"type": "integer"},
        "max_context_verses": {"type": "integer"},
        "theology": {"type": "object"},
    },
}

# -------------------------------
# Generic Text Augmentation Class
# -------------------------------
class GenericAugmenter:
    """Provides generic text augmentation methods with theological safeguards."""

    def __init__(self, config: Dict[str, Any]):
        self.prob_synonym_replacement = config.get("prob_synonym_replacement", 0.1)
        self.max_synonym_replacements = config.get("max_synonym_replacements", 3)
        self.prob_deletion = config.get("prob_deletion", 0.05)
        self.prob_swap = config.get("prob_swap", 0.05)
        self.prob_insertion = config.get("prob_insertion", 0.05)
        self.theological_terms = set(config.get("theological_terms", []))

    def _get_synonyms(self, word: str, pos: str) -> List[str]:
        """Get synonyms, protecting theological terms."""
        if word.lower() in self.theological_terms:
            return [word]
        wordnet_pos = self._get_wordnet_pos(pos)
        if not wordnet_pos:
            return [word]
        return [
            lemma.name().replace("_", " ")
            for syn in wordnet.synsets(word, pos=wordnet_pos)
            for lemma in syn.lemmas()
            if lemma.name().lower() != word
        ][:3] or [word]

    def _get_wordnet_pos(self, treebank_tag: str) -> Optional[str]:
        """Convert Penn Treebank tags to WordNet tags."""
        return {
            "J": wordnet.ADJ,
            "V": wordnet.VERB,
            "N": wordnet.NOUN,
            "R": wordnet.ADV,
        }.get(treebank_tag[0])

    def apply_synonym_replacement(self, text: str) -> str:
        """Replace words with synonyms, avoiding theological terms."""
        words = word_tokenize(text)
        tagged_words = pos_tag(words)
        indices = random.sample(
            range(len(words)), min(self.max_synonym_replacements, len(words))
        )
        for idx in indices:
            word, pos = tagged_words[idx]
            if (
                not word.isalnum()
                or len(word) <= 3
                or word.lower() in self.theological_terms
            ):
                continue
            synonyms = self._get_synonyms(word, pos)
            if len(synonyms) > 1:
                words[idx] = random.choice(synonyms[1:])
        return " ".join(words)

    def random_deletion(self, text: str) -> str:
        """Randomly delete words, preserving theological terms."""
        words = word_tokenize(text)
        if len(words) <= 1:
            return text
        return " ".join(
            [
                w
                for w in words
                if random.random() > self.prob_deletion
                or w.lower() in self.theological_terms
            ]
        )

    def random_swap(self, text: str) -> str:
        """Randomly swap two words, minimizing theological term disruption."""
        words = word_tokenize(text)
        if len(words) < 2:
            return text
        for _ in range(1):
            idx1, idx2 = random.sample(range(len(words)), 2)
            if (
                words[idx1].lower() not in self.theological_terms
                and words[idx2].lower() not in self.theological_terms
            ):
                words[idx1], words[idx2] = words[idx2], words[idx1]
        return " ".join(words)

    def random_insertion(self, text: str) -> str:
        """Randomly insert synonyms, avoiding theological term context."""
        words = word_tokenize(text)
        if not words:
            return text
        idx = random.randint(0, len(words) - 1)
        word, pos = pos_tag([words[idx]])[0]
        if word.lower() in self.theological_terms:
            return text
        synonyms = self._get_synonyms(word, pos)
        if synonyms and len(synonyms) > 1:
            words.insert(idx, random.choice(synonyms[1:]))
        return " ".join(words)

    def augment_text(self, text: str, intensity: float = 0.2, **kwargs: Any) -> str:
        """Apply basic augmentation techniques based on intensity."""
        if not text:
            return text
        augmented = text
        if random.random() < self.prob_synonym_replacement * intensity:
            augmented = self.apply_synonym_replacement(augmented)
        if random.random() < self.prob_deletion * intensity:
            augmented = self.random_deletion(augmented)
        if random.random() < self.prob_swap * intensity:
            augmented = self.random_swap(augmented)
        if random.random() < self.prob_insertion * intensity:
            augmented = self.random_insertion(augmented)
        return augmented

    def augment_bible_data(
        self, data: Dict[str, Any], intensity: float = 0.2, max_augmentations: int = 3
    ) -> List[Dict[str, Any]]:
        """Augment Bible data structure with multiple variations."""
        if not data.get("books"):
            logger.warning("No books found in data")
            return [data]
        augmented_data = [data.copy()]
        for _ in range(max_augmentations):
            new_data = json.loads(json.dumps(data))  # Deep copy
            for book in new_data.get("books", []):
                for chapter in book.get("chapters", []):
                    for verse in chapter.get("verses", []):
                        if "text" in verse:
                            verse["text"] = self.augment_text(verse["text"], intensity)
            augmented_data.append(new_data)
        return augmented_data

# -------------------------------
# Biblical Augmentation Class
# -------------------------------
class BiblicalAugmenter(GenericAugmenter):
    """Provides Bible-specific augmentation with validation and integration."""

    def __init__(self, config_path: Optional[str] = "config/bible_sources.json"):
        config: Dict[str, Any] = {}
        if config_path is not None:
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                validate(config, CONFIG_SCHEMA)
            except Exception as e:
                logger.error("Failed to load or validate config %s: %s", config_path, e)
                config = {}
        super().__init__(config)
        self.prob_verse_shuffle = config.get("prob_verse_shuffle", 0.3)
        self.prob_translation_swap = config.get("prob_translation_swap", 0.4)
        self.min_context_verses = config.get("min_context_verses", 1)
        self.max_context_verses = config.get("max_context_verses", 5)
        self.bible_translations = self._load_bible_translations(config)
        self.converter = BibleConverter(config_path=config_path)
        self.storage = BibleStorage(config_path=config_path)
        self.validator = TheologicalValidator(str(config.get("theology", {})))
        logger.info(
            "BiblicalAugmenter initialized with %d theological terms and %d translations",
            len(self.theological_terms),
            len(self.bible_translations),
        )

    def _load_bible_translations(self, config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Load Bible translations from config paths."""
        translations: Dict[str, Dict[str, Any]] = config.get("translation_paths", {})
        for code, info in translations.items():
            path = info.get("path", "")
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        info["data"] = json.load(f)
                except Exception as e:
                    logger.error("Failed to load translation %s from %s: %s", code tos, path, e)
        return translations

    def _apply_verse_shuffle(self, text: str) -> str:
        """Shuffle verses or sentences based on probability."""
        verses: List[str] = re.findall(r"(\d+:\d+[\-\d+]*\s+[^.!?\n]+[.!?])", text)
        if len(verses) <= 1:
            sentences: List[str] = sent_tokenize(text)
            if len(sentences) > 1 and random.random() < self.prob_verse_shuffle:
                random.shuffle(sentences)
                return " ".join(sentences)
            return text
        if random.random() < self.prob_verse_shuffle:
            random.shuffle(verses)
        return " ".join(verses)

    def _apply_translation_swap(self, text: str, ref: Optional[str] = None) -> str:
        """Swap text with a verse from a different translation."""
        if not ref or random.random() >= self.prob_translation_swap:
            return text
        try:
            book, chapter_verse = ref.split(" ", 1)
            chapter, verse = chapter_verse.split(":")
            start_verse = int(verse.split("-")[0])
        except (ValueError, IndexError) as e:
            logger.error("Invalid reference %s: %s", ref, e)
            return text

        for code, info in self.bible_translations.items():
            if "data" not in info:
                logger.warning("No data for translation %s", code)
                continue
            for book_data in info["data"].get("books", []):
                if book_data.get("code", "").lower() == book.lower():
                    for chapter_data in book_data.get("chapters", []):
                        if int(chapter_data.get("number", 0)) == int(chapter):
                            for verse_data in chapter_data.get("verses", []):
                                if int(verse_data.get("number", 0)) == start_verse:
                                    return verse_data.get("text", text)
        logger.debug("No matching verse found for %s", ref)
        return text

    def expand_context(self, ref: str, text: str) -> str:
        """Expand verse context with simulated or real surrounding verses."""
        try:
            book, chapter_verse = ref.split(" ", 1)
            chapter, verse = chapter_verse.split(":")
            start_verse = int(verse.split("-")[0])
            verses_to_add = random.randint(
                self.min_context_verses, self.max_context_verses
            )
            context_before = f"[Context from {book} {chapter}:{max(1, start_verse-verses_to_add)} to {start_verse-1}] "
            context_after = (
                f" [Context to {book} {chapter}:{start_verse+verses_to_add}]"
            )
            return context_before + text + context_after
        except (ValueError, IndexError) as e:
            logger.error("Failed to expand context for %s: %s", ref, e)
            return text

    def augment_text(
        self, text: str, ref: Optional[str] = None, intensity: float = 0.2
    ) -> str:
        """Apply augmentation techniques based on intensity."""
        if not text:
            return text
        augmented = text
        if random.random() < self.prob_synonym_replacement * intensity:
            augmented = self.apply_synonym_replacement(augmented)
        if random.random() < self.prob_deletion * intensity:
            augmented = self.random_deletion(augmented)
        if random.random() < self.prob_swap * intensity:
            augmented = self.random_swap(augmented)
        if random.random() < self.prob_insertion * intensity:
            augmented = self.random_insertion(augmented)
        if random.random() < self.prob_verse_shuffle * intensity:
            augmented = self._apply_verse_shuffle(augmented)
        if random.random() < self.prob_translation_swap * intensity and ref:
            augmented = self._apply_translation_swap(augmented, ref)
        if random.random() < 0.3 * intensity and ref:
            augmented = self.expand_context(ref, augmented)
        if self.validator:
            score = self.validator.validate(augmented)
            if score < 0.9:
                logger.warning("Theological validation failed for %s (score: %.2f)", ref, score)
                return text
        return augmented

    def augment_bible_data(
        self,
        bible_data: Dict[str, Any],
        intensity: float = 0.2,
        max_augmentations: int = 3,
    ) -> List[Dict[str, Any]]:
        """Augment Bible data structure with multiple variations."""
        if not bible_data.get("books"):
            logger.warning("No books found in bible_data")
            return [bible_data]
        augmented_data = [bible_data.copy()]
        for i in range(max_augmentations):
            new_data = json.loads(json.dumps(bible_data))  # Deep copy
            overall_score = 1.0
            for book in new_data.get("books", []):
                for chapter in book.get("chapters", []):
                    for verse in chapter.get("verses", []):
                        book_code = book.get("code", book.get("name", "UnknownBook"))
                        chapter_num = chapter.get("number", "UnknownChapter")
                        verse_num = verse.get("number", "UnknownVerse")
                        ref = f"{book_code} {chapter_num}:{verse_num}"
                        original_text = verse.get("text", "")
                        if not original_text:
                            continue
                        verse["text"] = self.augment_text(original_text, ref, intensity)
                        if self.validator:
                            score = self.validator.validate(verse["text"])
                            if score < 0.9:
                                verse["text"] = original_text
                                logger.debug(
                                    "Reverted augmentation for %s due to low score (%.2f)",
                                    ref,
                                    score,
                                )
            if self.validator:
                try:
                    representative_text = " ".join(
                        verse.get("text", "")
                        for book in new_data.get("books", [])
                        for chapter in book.get("chapters", [])
                        for verse in chapter.get("verses", [])
                    )
                    overall_score = self.validator.validate(representative_text)
                except Exception as e:
                    logger.error("Error during overall validation: %s", e)
                    overall_score = 0.0
                if overall_score < 0.9:
                    logger.warning(
                        "Skipping augmented version %d due to low overall score (%.2f)",
                        i + 1,
                        overall_score,
                    )
                    continue
            augmented_data.append(new_data)
            logger.info(
                "Generated augmented version %d with score %.2f",
                len(augmented_data) - 1,
                overall_score,
            )
        return augmented_data

    def augment_batch(
        self,
        texts: List[str],
        refs: Optional[List[str]] = None,
        intensity: float = 0.2,
    ) -> List[str]:
        """Augment a batch of texts in parallel."""
        if not texts:
            return []
        if refs and len(refs) != len(texts):
            logger.warning("Length of refs does not match length of texts. Ignoring refs.")
            refs = None
        with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 1, len(texts))) as executor:
            future_to_text = {
                executor.submit(
                    self.augment_text,
                    text,
                    refs[i] if refs else None,
                    intensity,
                ): (text, refs[i] if refs else None)
                for i, text in enumerate(texts)
            }
            results = []
            for future in future_to_text:
                original_text, original_ref = future_to_text[future]
                try:
                    augmented_text = future.result()
                    results.append(augmented_text)
                except Exception as e:
                    logger.error("Failed to augment text (ref: %s): %s", original_ref, e)
                    results.append(original_text)
        return results

    def save_augmentations(
        self, augmented_data: List[Dict[str, Any]], base_path: str
    ) -> List[str]:
        """Save augmented data using storage or filesystem."""
        paths = []
        output_dir = (
            os.path.dirname(base_path) if "." in os.path.basename(base_path)
            else base_path
        )
        os.makedirs(output_dir, exist_ok=True)
        for i, data in enumerate(augmented_data):
            file_path = os.path.join(output_dir, f"augmented_bible_{i}.json
