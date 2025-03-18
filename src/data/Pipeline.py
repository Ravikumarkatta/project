# bible/src/data/pipeline.py
"""
Data pipeline to orchestrate preprocessing, augmentation, and dataset creation for Bible-AI.
"""

import os
import json
from typing import Dict, List, Tuple
import pandas as pd
from src.data.preprocessing import BiblicalTextPreprocessor
from src.data.augmentation import BiblicalAugmenter
from src.data.tokenization import BiblicalTokenizer
from src.data.utils import collate_fn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

class BibleAIDataPipeline:
    """Orchestrates the data pipeline for Bible-AI training."""

    def __init__(self, config_path: str = "config/data_config.json"):
        self.preprocessor = BiblicalTextPreprocessor(config_path)
        self.augmenter = BiblicalAugmenter(config_path)
        self.tokenizer = BiblicalTokenizer(base_tokenizer_name="bert-base-uncased", config_path=config_path)
        self.config_path = config_path

    def run_pipeline(self, augment: bool = True, max_augmentations: int = 3) -> Tuple[DataLoader, DataLoader]:
        """
        Run the full data pipeline: preprocess, augment, tokenize, and create DataLoaders.
        
        Args:
            augment: Whether to apply data augmentation.
            max_augmentations: Maximum number of augmented versions to generate.
            
        Returns:
            Tuple of (train_dataloader, val_dataloader).
        """
        # Step 1: Preprocess raw data
        bibles = {}
        bibles_dir = os.path.join(self.preprocessor.raw_dir, 'bibles')
        for bible_file in os.listdir(bibles_dir):
            if bible_file.endswith(('.xml', '.json', '.txt')):
                translation = os.path.splitext(bible_file)[0].upper()
                file_path = os.path.join(bibles_dir, bible_file)
                bible_data = self.preprocessor.process_bible_file(file_path, translation)
                bibles[translation] = bible_data
                self.preprocessor.save_processed_bible(bible_data, translation)
        
        commentaries = {}
        commentaries_dir = os.path.join(self.preprocessor.raw_dir, 'commentaries')
        for commentary_file in os.listdir(commentaries_dir):
            if commentary_file.endswith(('.xml', '.json', '.txt', '.csv')):
                source = os.path.splitext(commentary_file)[0]
                file_path = os.path.join(commentaries_dir, commentary_file)
                entries = self.preprocessor.process_commentary_file(file_path, source)
                commentaries[source] = entries
                self.preprocessor.save_processed_commentaries(entries, source)
        
        # Step 2: Create verse-aligned dataset and instruction data
        verse_aligned_df = self.preprocessor.create_verse_aligned_dataset(bibles, commentaries)
        instruction_data = self.preprocessor.generate_instruction_data(verse_aligned_df)
        
        # Step 3: Augment the instruction data
        if augment:
            qa_pairs = [(item["instruction"] + "\n" + item["input"], item["output"]) for item in instruction_data]
            refs = [item["input"] for item in instruction_data]
            augmented_qa = self.augmenter.augment_qa_batch(qa_pairs, refs, intensity=0.2)
            augmented_instruction_data = []
            for (aug_q, aug_a), orig in zip(augmented_qa, instruction_data):
                instruction, input_text = aug_q.split("\n", 1)
                augmented_instruction_data.append({
                    "instruction": instruction,
                    "input": input_text,
                    "output": aug_a
                })
            instruction_data.extend(augmented_instruction_data)
        
        # Step 4: Split into train and validation
        random.shuffle(instruction_data)
        train_size = int(0.8 * len(instruction_data))
        train_data = instruction_data[:train_size]
        val_data = instruction_data[train_size:]
        
        # Save train and validation data
        train_path = os.path.join(self.preprocessor.processed_dir, "train_instruction.json")
        val_path = os.path.join(self.preprocessor.processed_dir, "val_instruction.json")
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2)
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2)
        
        # Step 5: Create DataLoaders
        train_loader, val_loader = self.preprocessor.create_dataloaders(
            train_path, val_path, self.tokenizer, batch_size=4, max_length=512
        )
        return train_loader, val_loader


if __name__ == "__main__":
    pipeline = BibleAIDataPipeline()
    train_loader, val_loader = pipeline.run_pipeline(augment=True, max_augmentations=3)
    print(f"Created DataLoaders with {len(train_loader)} training batches and {len(val_loader)} validation batches.")
