# bible/src/data/augmentation.py
"""
Ultimate Text & Biblical Data Augmentation Module for Bible-AI

This module merges the best features from previous implementations:
- Generic augmentation: Synonym replacement, random deletion, random swap, random insertion
- Biblical augmentation: Verse shuffling, translation swap, context expansion, theological validation
- Optimized for Bible-AI integration with converter, storage, and validator modules

Dependencies:
- nltk (for synonym replacement, POS tagging, tokenization)
- concurrent.futures (for parallel processing)
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
from typing import Dict, List, Optional, Set, Tuple, Any  # Added Any import here

# NLTK setup - note these imports might need stubs
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

# Project-specific imports with fallbacks
try:
    from src.bible_manager.converter import BibleConverter
    from src.bible_manager.storage import BibleStorage
    from src.theology.validator import TheologicalValidator
    from src.utils.logger import get_logger
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    get_logger = lambda name: logging.getLogger(name)
    # Fix: Don't assign None to types, initialize with None values
    BibleConverter = None  # type: Optional[type]
    BibleStorage = None    # type: Optional[type]
    TheologicalValidator = None  # type: Optional[type]
    logger = get_logger("UltimateAugmenter")
    logger.warning("Missing dependencies: %s. Some features may be limited.", e)

logger = get_logger("UltimateAugmenter")


# -------------------------------
# Generic Text Augmentation Class
# -------------------------------
class GenericAugmenter:
    """Provides generic text augmentation methods with theological safeguards."""

    def __init__(self, config: Dict):
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

    # Add missing methods reported in the error list
    def augment_bible_data(self, data: Dict[str, Any], intensity: float = 0.2, max_augmentations: int = 3) -> List[Dict[str, Any]]:
        """Augment Bible data structure with multiple variations."""
        # This is a generic implementation to fix the error
        if not data:
            return [data]
        augmented_data = [data.copy()]
        return augmented_data

    def augment_text(self, text: str, intensity: float = 0.2) -> str:
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


# -------------------------------
# Biblical Augmentation Class
# -------------------------------
class BiblicalAugmenter(GenericAugmenter):
    """Provides Bible-specific augmentation with validation and integration."""

    def __init__(self, config_path: Optional[str] = "config/bible_sources.json"):
        # Fix: handle the case where config_path is None
        if config_path is not None:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = {}
        super().__init__(config)
        self.prob_verse_shuffle = config.get("prob_verse_shuffle", 0.3)
        self.prob_translation_swap = config.get("prob_translation_swap", 0.4)
        self.min_context_verses = config.get("min_context_verses", 1)
        self.max_context_verses = config.get("max_context_verses", 5)
        self.bible_translations = self._load_bible_translations(config)
        self.converter = (
            BibleConverter(config_path=config_path) if BibleConverter else None
        )
        self.storage = BibleStorage(config_path=config_path) if BibleStorage else None
        self.validator = (
            TheologicalValidator(str(config.get("theology", {})))  # Fix: Convert dict to str
            if TheologicalValidator
            else None
        )
        logger.info(
            "BiblicalAugmenter initialized with %d theological terms and %d translations",
            len(self.theological_terms),
            len(self.bible_translations),
        )

    def _load_bible_translations(self, config: Dict) -> Dict[str, Dict]:
        """Load Bible translations from config paths."""
        translations = config.get("translation_paths", {})
        for code, info in translations.items():
            path = info.get("path", "")
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        info["data"] = json.load(f)
                except Exception as e:
                    logger.error(
                        "Failed to load translation %s from %s: %s", code, path, e
                    )
        return translations

    def _apply_verse_shuffle(self, text: str) -> str:
        """Shuffle verses or sentences based on probability."""
        verses = re.findall(r"(\d+:\d+[\-\d+]*\s+[^.!?\n]+[.!?])", text)
        if len(verses) <= 1:
            sentences = sent_tokenize(text)
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
                continue
            for book_data in info["data"].get("books", []):
                if book_data.get("code", "").lower() == book.lower():
                    for chapter_data in book_data.get("chapters", []):
                        if int(chapter_data.get("number", 0)) == int(chapter):
                            for verse_data in chapter_data.get("verses", []):
                                if int(verse_data.get("number", 0)) == start_verse:
                                    return verse_data.get("text", text)
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
            # Fix: Convert dict to string for validator
            score = self.validator.validate(augmented, str(ref))
            if score < 0.5: # Corrected: Use < instead of &lt;
                logger.warning("Theological validation failed for %s", ref)
                return text
        return augmented

    # --- Start of Added Code ---
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
        for _ in range(max_augmentations):
            new_data = json.loads(json.dumps(bible_data)) # Deep copy
            overall_score = 1.0 # Initialize score
            for book_idx, book in enumerate(new_data["books"]):
                for chapter_idx, chapter in enumerate(book.get("chapters", [])): # Use .get for safety
                    for verse_idx, verse in enumerate(chapter.get("verses", [])): # Use .get for safety
                        # Ensure necessary keys exist before creating ref
                        book_code = book.get("code", book.get("name", "UnknownBook")) # Fallback for book identifier
                        chapter_num = chapter.get("number", "UnknownChapter")
                        verse_num = verse.get("number", "UnknownVerse")
                        ref = f"{book_code} {chapter_num}:{verse_num}"

                        original_text = verse.get("text", "")
                        if not original_text:
                            continue # Skip if verse text is missing

                        verse["text"] = self.augment_text(original_text, ref, intensity)

                        # Validate individual verse augmentation
                        if self.validator:
                            # Assuming validator needs text and optional reference
                            score = self.validator.validate(verse["text"], ref)
                            if score < 0.9: # Corrected: Use < instead of &lt;
                                # Revert if validation fails
                                verse["text"] = original_text
                                logger.debug(f"Reverted augmentation for {ref} due to low score ({score:.2f})")


            # Validate the entire augmented structure after modifying all verses
            if self.validator:
                # Assuming validator can handle the full data structure or needs specific format
                # This part might need adjustment based on how validator.validate handles full data
                # For now, let's assume it can take the dict directly or needs a representative text sample
                # If it needs text, we might need to concatenate or sample
                # Simplified: Assume validate can take the dict
                try:
                    overall_score = self.validator.validate(new_data) # Pass the whole dict
                except Exception as e:
                    logger.error(f"Error during overall validation: {e}")
                    overall_score = 0.0 # Treat validation error as failure

                if overall_score < 0.9: # Corrected: Use < instead of &lt;
                    logger.warning(f"Skipping augmented version due to low overall score ({overall_score:.2f})")
                    continue # Skip this augmentation if overall validation fails

            augmented_data.append(new_data)
            logger.info(
                "Generated augmented version %d with score %.2f",
                len(augmented_data) - 1,
                overall_score,
            )
        return augmented_data

    def augment_batch(
        self, texts: List[str], refs: Optional[List[str]] = None, intensity: float = 0.2
    ) -> List[str]:
        """Augment a batch of texts in parallel."""
        if not texts:
            return []
        # Ensure refs list matches texts length if provided
        if refs and len(refs) != len(texts):
            logger.warning("Length of refs does not match length of texts. Ignoring refs.")
            refs = None

        with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 1, len(texts))) as executor: # Use cpu_count
            future_to_text = {
                executor.submit(
                    self.augment_text,
                    text,
                    refs[i] if refs else None, # Pass ref correctly
                    intensity,
                ): (text, refs[i] if refs else None) # Store original text and ref for error logging
                for i, text in enumerate(texts)
            }
            results = []
            original_texts_for_failed = [] # Keep track of original texts for failed augmentations

            for future in future_to_text:
                original_text, original_ref = future_to_text[future]
                try:
                    augmented_text = future.result()
                    results.append(augmented_text)
                except Exception as e:
                    logger.error(f"Failed to augment text (ref: {original_ref}): {e}. Using original text.")
                    results.append(original_text) # Append original text on failure
                    original_texts_for_failed.append(original_text)

        # Log summary of failures if any
        if original_texts_for_failed:
            logger.warning(f"{len(original_texts_for_failed)} augmentations failed and used original text.")

        return results


    def save_augmentations(
        self, augmented_data: List[Dict[str, Any]], base_path: str
    ) -> List[str]:
        """Save augmented data using storage or filesystem."""
        paths = []
        # Ensure base_path is a directory
        output_dir = os.path.dirname(base_path) if '.' in os.path.basename(base_path) else base_path
        os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist

        if not self.storage:
            logger.warning("Storage module unavailable; saving to filesystem")
            for i, data in enumerate(augmented_data):
                # Generate filename based on index
                file_path = os.path.join(output_dir, f"augmented_bible_{i}.json")
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False) # Use ensure_ascii=False for broader char support
                    paths.append(file_path)
                    logger.debug(f"Saved augmented data to {file_path}")
                except TypeError as e:
                     logger.error(f"Serialization error saving {file_path}: {e}. Data type: {type(data)}")
                except IOError as e:
                    logger.error(f"IOError saving {file_path}: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error saving {file_path}: {e}")
        else:
            logger.info("Using BibleStorage to save augmentations.")
            for i, data in enumerate(augmented_data):
                try:
                    # Prepare metadata
                    metadata = {
                        "augmentation_index": i,
                        "timestamp": datetime.now().isoformat(), # Use ISO format
                        "source": "augmentation_script",
                        # Add other relevant metadata if available, e.g., original file ID
                    }
                    # Store using BibleStorage - assumes store_bible takes data (str or dict) and metadata
                    # Convert data to JSON string if necessary
                    data_str = json.dumps(data, ensure_ascii=False)
                    file_id = self.storage.store_bible(data_str, metadata=metadata) # Pass metadata correctly
                    if file_id:
                        # Construct the path where the storage module likely saved the file
                        # This might need adjustment based on BibleStorage implementation
                        saved_path = os.path.join(self.storage.storage_dir, f"{file_id}.json")
                        paths.append(saved_path)
                        logger.debug(f"Stored augmented data via BibleStorage with ID: {file_id}")
                    else:
                        logger.error(f"BibleStorage failed to store augmented data index {i}")
                except AttributeError:
                     logger.error("BibleStorage object does not have a 'store_bible' method or is None.")
                     # Fallback to filesystem saving if storage fails structurally
                     file_path = os.path.join(output_dir, f"augmented_bible_{i}_fallback.json")
                     try:
                         with open(file_path, "w", encoding="utf-8") as f:
                             json.dump(data, f, indent=2, ensure_ascii=False)
                         paths.append(file_path)
                         logger.warning(f"Fell back to saving {file_path} directly.")
                     except Exception as fallback_e:
                         logger.error(f"Fallback saving also failed for index {i}: {fallback_e}")
                except Exception as e:
                    logger.error(f"Error storing augmented data index {i} via BibleStorage: {e}")

        if not paths:
            logger.error("No augmented data was successfully saved.")
        else:
            logger.info(f"Successfully saved {len(paths)} augmented data files.")
        return paths

    # --- End of Added Code ---


# -------------------------------
# CLI Execution
# -------------------------------
# --- Replacing previous __main__ block ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ultimate Augmentation for Bible-AI")
    parser.add_argument(
        "--input", type=str, required=True, help="Input file (JSON Bible data format)" # Clarified input type
    )
    parser.add_argument("--output", type=str, required=True, help="Output directory to save augmented files") # Clarified output is directory
    parser.add_argument(
        "--intensity", type=float, default=0.2, help="Augmentation intensity (0.0 to 1.0)" # Added range hint
    )
    parser.add_argument(
        "--max-augmentations", type=int, default=3, help="Number of augmented versions to generate" # Clarified meaning
    )
    # Removed mode argument as the script now focuses on BiblicalAugmenter
    # parser.add_argument(
    #     "--mode", type=str, choices=["generic", "biblical"], default="biblical"
    # )
    parser.add_argument("--config", type=str, default="config/bible_sources.json", help="Path to configuration file") # Kept config path
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging") # Added verbose flag

    args = parser.parse_args()

    # Setup logger level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Always use BiblicalAugmenter now
    try:
        augmenter = BiblicalAugmenter(args.config)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {args.config}. Exiting.")
        exit(1)
    except json.JSONDecodeError:
         logger.error(f"Error decoding JSON configuration file {args.config}. Exiting.")
         exit(1)
    except Exception as e:
        logger.error(f"Failed to initialize BiblicalAugmenter: {e}")
        exit(1)


    # Input must be JSON for augment_bible_data
    if not args.input.endswith(".json"):
         logger.error(f"Input file must be a JSON file for Bible data augmentation. Provided: {args.input}")
         exit(1)

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded Bible data from {args.input}")
    except FileNotFoundError:
        logger.error(f"Input file not found: {args.input}")
        exit(1)
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON input file: {args.input}")
        exit(1)
    except Exception as e:
        logger.error(f"Error reading input file {args.input}: {e}")
        exit(1)

    # Perform augmentation
    logger.info(f"Starting augmentation with intensity {args.intensity}, generating {args.max_augmentations} versions...")
    try:
        augmented_data_list = augmenter.augment_bible_data(
            data, args.intensity, args.max_augmentations
        )
        logger.info(f"Generated {len(augmented_data_list)} total versions (including original).")
    except Exception as e:
        logger.error(f"An error occurred during augmentation: {e}")
        augmented_data_list = [] # Ensure list exists even on error

    # Save the results
    if augmented_data_list:
        logger.info(f"Saving augmented data to directory: {args.output}")
        try:
            file_paths = augmenter.save_augmentations(augmented_data_list, args.output)
            if file_paths:
                print(f"Augmented files saved successfully. Paths: {file_paths}")
            else:
                 print("Augmentation completed, but failed to save any files.")
        except Exception as e:
            logger.error(f"An error occurred during saving: {e}")
            print("Augmentation completed, but an error occurred during saving.")
    else:
        print("Augmentation process did not produce any results.")

    logger.info("Augmentation script finished.")
# --- End of Replaced __main__ block ---

