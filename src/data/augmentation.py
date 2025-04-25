"""
Data augmentation module for Bible-AI.

This module implements various data augmentation techniques specifically
designed for biblical text data to improve model training and performance.
"""

import json
import logging
import os
import random
import re
from typing import Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("augmentation")

# Constants
CONFIG_DIR = "config"
AUGMENTATION_CONFIG_FILE = os.path.join(CONFIG_DIR, "augmentation_config.json")


class BiblicalDataAugmenter:
    """
    Class for augmenting biblical text data to improve model training.
    """

    def __init__(self, config_path: str = AUGMENTATION_CONFIG_FILE):
        """
        Initialize the augmenter with configuration.

        Args:
            config_path: Path to the augmentation configuration file
        """
        self.config = self._load_config(config_path)
        self.synonyms = self._load_synonyms()
        self.paraphrases = self._load_paraphrases()
        logger.info("Biblical Data Augmenter initialized")

    def _load_config(self, config_path: str) -> Dict:
        """
        Load augmentation configuration from file.

        Args:
            config_path: Path to the configuration file

        Returns:
            Dictionary containing augmentation configuration
        """
        default_config = {
            "synonym_replacement_prob": 0.3,
            "paraphrase_prob": 0.2,
            "verse_reference_variation_prob": 0.4,
            "theological_concept_insertion_prob": 0.15,
            "max_augmentations_per_sample": 3,
            "resources_path": "data/augmentation",
        }

        if not os.path.exists(config_path):
            logger.warning(
                f"Config file not found at {config_path}. Using default configuration."
            )
            return default_config

        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            logger.info(f"Loaded augmentation config from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return default_config

    def _load_synonyms(self) -> Dict[str, List[str]]:
        """
        Load biblical word synonyms for replacement.

        Returns:
            Dictionary mapping words to lists of synonyms
        """
        resources_path = self.config.get("resources_path", "data/augmentation")
        synonyms_path = os.path.join(resources_path, "biblical_synonyms.json")

        if not os.path.exists(synonyms_path):
            logger.warning(
                f"Synonyms file not found at {synonyms_path}. Synonym replacement will be limited."
            )
            return {}

        try:
            with open(synonyms_path, "r") as f:
                synonyms = json.load(f)
            logger.info(f"Loaded {len(synonyms)} synonym sets from {synonyms_path}")
            return synonyms
        except Exception as e:
            logger.error(f"Failed to load synonyms from {synonyms_path}: {e}")
            return {}

    def _load_paraphrases(self) -> Dict[str, List[str]]:
        """
        Load biblical phrase paraphrases.

        Returns:
            Dictionary mapping phrases to lists of paraphrases
        """
        resources_path = self.config.get("resources_path", "data/augmentation")
        paraphrases_path = os.path.join(resources_path, "biblical_paraphrases.json")

        if not os.path.exists(paraphrases_path):
            logger.warning(
                f"Paraphrases file not found at {paraphrases_path}. Paraphrase augmentation will be limited."
            )
            return {}

        try:
            with open(paraphrases_path, "r") as f:
                paraphrases = json.load(f)
            logger.info(
                f"Loaded {len(paraphrases)} paraphrase sets from {paraphrases_path}"
            )
            return paraphrases
        except Exception as e:
            logger.error(f"Failed to load paraphrases from {paraphrases_path}: {e}")
            return {}

    def augment_text(self, text: str, techniques: List[str] = None) -> List[str]:
        """
        Augment a single text sample using specified techniques.

        Args:
            text: The text to augment
            techniques: List of augmentation techniques to apply.
                        If None, uses all available techniques.

        Returns:
            List of augmented text samples
        """
        available_techniques = {
            "synonym_replacement": self._synonym_replacement,
            "paraphrase": self._paraphrase,
            "verse_reference_variation": self._verse_reference_variation,
            "theological_concept_insertion": self._theological_concept_insertion,
        }

        if techniques is None:
            techniques = list(available_techniques.keys())

        augmented_texts = []
        for technique in techniques:
            if technique in available_techniques:
                try:
                    augmented = available_techniques[technique](text)
                    if augmented and augmented != text:
                        augmented_texts.append(augmented)
                except Exception as e:
                    logger.error(f"Error applying {technique} to text: {e}")

        # Limit the number of augmentations
        max_augmentations = self.config.get("max_augmentations_per_sample", 3)
        return augmented_texts[:max_augmentations]

    def augment_dataset(
        self, texts: List[str], techniques: List[str] = None
    ) -> List[str]:
        """
        Augment a dataset of texts.

        Args:
            texts: List of text samples to augment
            techniques: List of augmentation techniques to apply.
                        If None, uses all available techniques.

        Returns:
            List of original and augmented text samples
        """
        augmented_dataset = texts.copy()
        for text in texts:
            augmented_texts = self.augment_text(text, techniques)
            augmented_dataset.extend(augmented_texts)

        logger.info(
            f"Augmented dataset from {len(texts)} to {len(augmented_dataset)} samples"
        )
        return augmented_dataset

    def _synonym_replacement(self, text: str) -> str:
        """
        Replace words with biblical synonyms.

        Args:
            text: The text to augment

        Returns:
            Text with some words replaced by synonyms
        """
        if not self.synonyms or random.random() > self.config.get(
            "synonym_replacement_prob", 0.3
        ):
            return text

        words = text.split()
        new_words = words.copy()

        # Determine how many words to replace
        n_to_replace = max(1, int(0.1 * len(words)))
        replace_indices = random.sample(
            range(len(words)), min(n_to_replace, len(words))
        )

        for idx in replace_indices:
            word = words[idx].lower().strip(".,;:!?\"'()")
            if word in self.synonyms and self.synonyms[word]:
                synonym = random.choice(self.synonyms[word])

                # Preserve capitalization
                if words[idx][0].isupper():
                    synonym = synonym.capitalize()

                # Preserve punctuation
                for char in ".,;:!?\"'()":
                    if words[idx].endswith(char):
                        synonym += char

                new_words[idx] = synonym

        return " ".join(new_words)

    def _paraphrase(self, text: str) -> str:
        """
        Replace biblical phrases with paraphrases.

        Args:
            text: The text to augment

        Returns:
            Text with some phrases paraphrased
        """
        if not self.paraphrases or random.random() > self.config.get(
            "paraphrase_prob", 0.2
        ):
            return text

        augmented_text = text

        # Sort phrases by length (descending) to avoid overlapping replacements
        phrases = sorted(self.paraphrases.keys(), key=len, reverse=True)

        for phrase in phrases:
            if phrase in augmented_text and self.paraphrases[phrase]:
                paraphrase = random.choice(self.paraphrases[phrase])
                augmented_text = augmented_text.replace(phrase, paraphrase, 1)
                break  # Only replace one phrase per text to maintain readability

        return augmented_text

    def _verse_reference_variation(self, text: str) -> str:
        """
        Vary verse reference formats.

        Args:
            text: The text to augment

        Returns:
            Text with verse references in varied formats
        """
        if random.random() > self.config.get("verse_reference_variation_prob", 0.4):
            return text

        # Patterns to match Bible references
        patterns = [
            (r"(\d*\s*[A-Za-z]+\s+\d+:\d+)", self._vary_reference_format),
            (r"(\d*\s*[A-Za-z]+\s+\d+:\d+-\d+)", self._vary_reference_format),
        ]

        augmented_text = text
        for pattern, transformer in patterns:
            matches = re.findall(pattern, augmented_text)
            for match in matches:
                if random.random() < 0.7:  # Only transform some references
                    varied = transformer(match)
                    augmented_text = augmented_text.replace(match, varied, 1)

        return augmented_text

    def _vary_reference_format(self, reference: str) -> str:
        """
        Helper to vary the format of a Bible reference.

        Args:
            reference: The Bible reference to vary

        Returns:
            Varied format of the reference
        """
        variations = [
            lambda r: r.replace(":", " verse "),
            lambda r: r.replace(":", "."),
            lambda r: re.sub(r"(\d+:\d+)", r"(\1)", r),
            lambda r: re.sub(r"(\d+)-(\d+)", r"\1 to \2", r),
        ]

        transformer = random.choice(variations)
        return transformer(reference)

    def _theological_concept_insertion(self, text: str) -> str:
        """
        Insert theological concepts into the text.

        Args:
            text: The text to augment

        Returns:
            Text with theological concepts inserted
        """
        if random.random() > self.config.get(
            "theological_concept_insertion_prob", 0.15
        ):
            return text

        theological_concepts = [
            "according to Scripture",
            "as the Bible teaches",
            "in biblical context",
            "from a theological perspective",
            "considering the original Greek",
            "in light of the Hebrew understanding",
            "as understood in Christian tradition",
            "from an exegetical standpoint",
            "through proper hermeneutics",
            "in the context of covenant theology",
        ]

        # Insert at beginning or end of a sentence
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if not sentences:
            return text

        insert_idx = random.randint(0, len(sentences) - 1)
        concept = random.choice(theological_concepts)

        # Insert at beginning or end of the chosen sentence
        if random.random() < 0.5:  # Beginning
            sentences[insert_idx] = f"{concept}, {sentences[insert_idx]}"
        else:  # End
            sentences[insert_idx] = re.sub(
                r"([.!?])$", f", {concept}\\1", sentences[insert_idx]
            )

        return " ".join(sentences)


def load_sample_data(file_path: str) -> List[str]:
    """
    Load sample data for augmentation.

    Args:
        file_path: Path to the sample data file

    Returns:
        List of text samples
    """
    try:
        with open(file_path, "r") as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Failed to load sample data from {file_path}: {e}")
        return []


def save_augmented_data(data: List[str], output_path: str) -> bool:
    """
    Save augmented data to file.

    Args:
        data: List of augmented text samples
        output_path: Path to save the augmented data

    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            for sample in data:
                f.write(f"{sample}\n")
        logger.info(f"Saved {len(data)} augmented samples to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save augmented data to {output_path}: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    augmenter = BiblicalDataAugmenter()

    # Example text
    sample_text = "In John 3:16, Jesus explains that God loved the world so much that he gave his only Son."

    # Augment single text
    augmented_samples = augmenter.augment_text(sample_text)
    print(f"Original: {sample_text}")
    for i, augmented in enumerate(augmented_samples, 1):
        print(f"Augmentation {i}: {augmented}")

    # Load and augment dataset
    # sample_data = load_sample_data("data/samples/bible_verses.txt")
    # if sample_data:
    #     augmented_dataset = augmenter.augment_dataset(sample_data)
    #     save_augmented_data(augmented_dataset, "data/augmented/augmented_verses.txt")
