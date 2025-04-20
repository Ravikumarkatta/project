# bible/src/data/pipeline.py
"""
Data pipeline to orchestrate preprocessing, augmentation, and dataset creation for Bible-AI.
"""

import json
import os
import random  # Added import
from typing import Dict, List, Tuple

import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Project-specific imports with error handling
try:
    from src.data.augmentation import BiblicalAugmenter
    from src.data.preprocessing import BiblicalTextPreprocessor
    from src.data.tokenization import BiblicalTokenizer
    from src.data.utils import collate_fn
    from src.model.architecture import BiblicalTransformer, BiblicalTransformerConfig
    from src.model.verse_detector import VerseDetector
    from src.utils.logger import get_logger
except ImportError as e:
    import logging

    logging.basicConfig(level=logging.INFO)
    get_logger = lambda name: logging.getLogger(name)
    logger = get_logger("BibleAIDataPipeline")
    logger.error(f"Failed to import necessary modules: {e}. Pipeline may not function.")
    # Define dummy classes if imports fail, to prevent further NameErrors
    class BiblicalTextPreprocessor:
        def __init__(self, *args, **kwargs): pass
        def process_bible_file(self, *args, **kwargs): return {}
        def save_processed_bible_to_db(self, *args, **kwargs): pass # Corrected method name
        def process_commentary_file(self, *args, **kwargs): return []
        def save_processed_commentaries(self, *args, **kwargs): pass
        def create_verse_aligned_dataset(self, *args, **kwargs): return pd.DataFrame()
        def generate_instruction_data(self, *args, **kwargs): return []
        def create_dataloaders(self, *args, **kwargs): return (None, None) # Dummy DataLoader tuple
        raw_dir = "data/raw"
        processed_dir = "data/processed"

    class BiblicalAugmenter:
        def __init__(self, *args, **kwargs): pass
        def augment_batch(self, *args, **kwargs): return [] # Corrected method name

    class BiblicalTokenizer:
        def __init__(self, *args, **kwargs): pass
        def tokenize(self, *args, **kwargs): return {"input_ids": [], "attention_mask": []}

    class VerseDetector:
         def __init__(self, *args, **kwargs): pass
         def __call__(self, *args, **kwargs): return {"verse_logits": None} # Dummy output

    def collate_fn(*args, **kwargs): return (None, None) # Dummy collate function


logger = get_logger("BibleAIDataPipeline")


class BibleAIDataPipeline:
    """Orchestrates the data pipeline for Bible-AI training."""

    def __init__(self, config_path: str = "config/data_config.json") -> None: # Added return type hint
        """
        Initialize the data pipeline components.

        Args:
            config_path: Path to the main data configuration file.
        """
        try:
            self.preprocessor = BiblicalTextPreprocessor(config_path)
            self.augmenter = BiblicalAugmenter(config_path)
            # Ensure base_tokenizer_name is appropriate for your model
            # Consider making this configurable
            self.tokenizer = BiblicalTokenizer(
                base_tokenizer_name="bert-base-uncased", config_path=config_path
            )
            self.config_path = config_path
            logger.info("BibleAIDataPipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing BibleAIDataPipeline: {e}")
            # Depending on severity, you might want to raise the exception
            # raise

    # Removed the first, incomplete run_pipeline definition and the misplaced code block.
    # Kept and fixed the second definition below.

    def run_pipeline(
        self, augment: bool = True, max_augmentations: int = 3, bible_txt_file: str = "data/raw/bibles/bible.txt"
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Run the full data pipeline using bible.txt: preprocess, augment, tokenize, create DataLoaders.

        Args:
            augment: Whether to apply data augmentation.
            max_augmentations: Maximum number of augmented versions to generate per item.
            bible_txt_file: Path to the primary bible text file to process.

        Returns:
            Tuple of (train_dataloader, val_dataloader).
        """
        logger.info("Starting data pipeline run...")
        try:
            # --- Step 1: Preprocess bible.txt ---
            if not os.path.exists(bible_txt_file):
                 logger.error(f"Bible text file not found: {bible_txt_file}")
                 # Return empty DataLoaders or raise an error
                 return (None, None)

            translation = "RAW" # Assign a generic translation ID for the raw text file
            logger.info(f"Processing Bible file: {bible_txt_file} as translation '{translation}'")
            # Assuming process_bible_file returns a dict structure like {book: {chapter: {verse: text}}}
            bible_data = self.preprocessor.process_bible_file(bible_txt_file, translation)
            if not bible_data:
                 logger.error(f"Failed to process Bible data from {bible_txt_file}")
                 return (None, None)
            # Use the corrected method name
            self.preprocessor.save_processed_bible_to_db(bible_data, translation)
            logger.info(f"Saved processed Bible data for '{translation}' to database.")

            # --- Step 2: Process Commentaries (Optional, adjust as needed) ---
            commentaries = {}
            commentaries_dir = os.path.join(self.preprocessor.raw_dir, "commentaries")
            if os.path.exists(commentaries_dir):
                logger.info(f"Processing commentaries from: {commentaries_dir}")
                for commentary_file in os.listdir(commentaries_dir):
                    if commentary_file.endswith((".xml", ".json", ".txt", ".csv")):
                        source = os.path.splitext(commentary_file)[0]
                        file_path = os.path.join(commentaries_dir, commentary_file)
                        logger.debug(f"Processing commentary file: {file_path}")
                        entries = self.preprocessor.process_commentary_file(file_path, source)
                        if entries:
                            commentaries[source] = entries
                            self.preprocessor.save_processed_commentaries(entries, source)
                logger.info(f"Processed {len(commentaries)} commentary sources.")
            else:
                logger.warning(f"Commentaries directory not found: {commentaries_dir}. Skipping commentary processing.")


            # --- Step 3: Create Verse-Aligned Dataset and Instruction Data ---
            # Ensure the structure passed to create_verse_aligned_dataset is correct
            # It expects a dict where keys are translation IDs
            logger.info("Creating verse-aligned dataset...")
            verse_aligned_df = self.preprocessor.create_verse_aligned_dataset(
                {translation: bible_data}, commentaries
            )
            if verse_aligned_df.empty:
                 logger.warning("Verse-aligned dataset is empty. Instruction generation might fail.")

            logger.info("Generating instruction data...")
            instruction_data = self.preprocessor.generate_instruction_data(verse_aligned_df)
            if not instruction_data:
                 logger.warning("No instruction data generated. Cannot proceed with augmentation or dataloader creation.")
                 return (None, None)
            logger.info(f"Generated {len(instruction_data)} initial instruction examples.")


            # --- Step 4: Augment Instruction Data ---
            if augment:
                logger.info(f"Augmenting instruction data (max_augmentations={max_augmentations})...")
                # Prepare data for augmenter - ensure augment_batch expects this format
                # Assuming augment_batch takes List[Tuple[str, str]] and optional List[str] refs
                qa_pairs = [
                    (item["instruction"] + "\n" + item["input"], item["output"])
                    for item in instruction_data
                ]
                refs = [item["input"] for item in instruction_data] # Assuming 'input' contains the reference

                # Use the corrected method name 'augment_batch'
                augmented_qa = self.augmenter.augment_batch(qa_pairs, refs, intensity=0.2) # Pass intensity if needed

                augmented_instruction_data = []
                for (aug_q, aug_a), orig in zip(augmented_qa, instruction_data):
                    # Be careful with splitting - ensure format is consistent
                    try:
                        instruction, input_text = aug_q.split("\n", 1)
                        augmented_instruction_data.append(
                            {"instruction": instruction, "input": input_text, "output": aug_a, "is_augmented": True}
                        )
                    except ValueError:
                         logger.warning(f"Could not split augmented question: {aug_q}. Skipping augmentation for this item.")
                         # Optionally add original back or just skip
                         # augmented_instruction_data.append(orig)


                logger.info(f"Generated {len(augmented_instruction_data)} augmented examples.")
                instruction_data.extend(augmented_instruction_data)
            else:
                logger.info("Skipping augmentation.")

            # --- Step 5: Split and Save Data ---
            logger.info("Splitting data into training and validation sets...")
            random.shuffle(instruction_data)
            train_size = int(0.8 * len(instruction_data))
            train_data = instruction_data[:train_size]
            val_data = instruction_data[train_size:]
            logger.info(f"Train size: {len(train_data)}, Validation size: {len(val_data)}")

            train_path = os.path.join(self.preprocessor.processed_dir, "train_instruction.json")
            val_path = os.path.join(self.preprocessor.processed_dir, "val_instruction.json")

            try:
                logger.info(f"Saving training data to: {train_path}")
                with open(train_path, "w", encoding="utf-8") as f:
                    json.dump(train_data, f, indent=2)
                logger.info(f"Saving validation data to: {val_path}")
                with open(val_path, "w", encoding="utf-8") as f:
                    json.dump(val_data, f, indent=2)
            except IOError as e:
                 logger.error(f"Failed to save train/validation data: {e}")
                 return (None, None)

            # --- Step 6: Create DataLoaders ---
            # Note: The preprocessor.create_dataloaders method needs to handle the JSON files correctly.
            # It might need adjustment based on its implementation in preprocessing.py
            # (e.g., it might expect a Dataset class instance, not file paths).
            # Assuming create_dataloaders can handle file paths and uses an appropriate Dataset class internally.
            logger.info("Creating DataLoaders...")
            # Make sure the tokenizer passed here is the one intended for the model
            # The BiblicalTokenizer instance `self.tokenizer` might need adjustment
            # if the model expects a standard HF tokenizer.
            # If `create_dataloaders` uses `BibleInstructionDataset` from preprocessing.py,
            # ensure that dataset class uses the correct tokenizer logic.
            train_loader, val_loader = self.preprocessor.create_dataloaders(
                train_path, val_path, self.tokenizer, batch_size=4, max_length=512
            )

            if train_loader and val_loader:
                logger.info("Data pipeline run completed successfully.")
                return train_loader, val_loader
            else:
                 logger.error("Failed to create DataLoaders.")
                 return (None, None)

        except Exception as e:
            logger.exception(f"An error occurred during the data pipeline run: {e}")
            # Return empty DataLoaders or re-raise the exception
            return (None, None)


if __name__ == "__main__":
    logger.info("Running BibleAIDataPipeline as main script.")
    pipeline = BibleAIDataPipeline()
    # Specify the path to your bible.txt file if it's not in the default location
    # Adjust augment and max_augmentations as needed
    train_loader, val_loader = pipeline.run_pipeline(
        augment=True,
        max_augmentations=2,
        bible_txt_file="data/raw/bibles/bible.txt" # Example path
    )

    if train_loader and val_loader:
        print( # Use print for final user output, logger for internal steps
            f"Successfully created DataLoaders.\n"
            f"Training batches: {len(train_loader)}\n"
            f"Validation batches: {len(val_loader)}"
        )
        # Example: Inspect a batch
        # try:
        #     first_train_batch = next(iter(train_loader))
        #     print("\nSample Training Batch Keys:", first_train_batch.keys())
        #     print("Sample Input IDs shape:", first_train_batch['input_ids'].shape)
        # except Exception as e:
        #      print(f"\nCould not inspect training batch: {e}")
    else:
        print("Data pipeline failed to create DataLoaders.")
