from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.data.preprocessing import (
    BiblicalAugmenter,
    BiblicalTextPreprocessor,
    BiblicalTokenizer,
    VerseDetector,
)


class Pipeline:
    def __init__(self) -> None:
        # Initialize necessary components for the pipeline
        self.text_preprocessor = BiblicalTextPreprocessor()
        self.augmenter = BiblicalAugmenter()
        self.tokenizer = BiblicalTokenizer()
        self.verse_detector = VerseDetector()

    def preprocess_text(self, text: str) -> str:
        # Preprocess a single text string
        processed_text = self.text_preprocessor.preprocess(text)
        return processed_text

    def augment_texts(self, texts: List[str]) -> List[str]:
        # Perform data augmentation on a list of texts
        augmented_texts = self.augmenter.augment_batch(texts)
        return augmented_texts

    def tokenize_texts(self, texts: List[str]) -> List[List[str]]:
        # Tokenize a list of texts into tokens
        tokenized_texts = [self.tokenizer.tokenize(text) for text in texts]
        return tokenized_texts

    def detect_verses(self, text: str) -> List[str]:
        # Detect individual verses in a text
        verses = self.verse_detector.detect(text)
        return verses

    def collate_fn(self, batch: List[Dict[str, Tensor]]) -> Tuple[Tensor, Tensor]:
        # Collate function for DataLoader to batch input and target tensors
        inputs = [item["input"] for item in batch]
        targets = [item["target"] for item in batch]
        input_tensor = torch.stack(inputs)
        target_tensor = torch.stack(targets)
        return input_tensor, target_tensor

    def create_dataloaders(
        self, dataset: Dataset, batch_size: int = 32
    ) -> Tuple[DataLoader[Any], DataLoader[Any]]:
        # Split dataset into training and validation sets, then create DataLoaders
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
        return train_loader, val_loader

    def prepare_dataset(self, file_path: str) -> Dataset:
        # Load and preprocess data from a CSV file into a custom Dataset
        df = pd.read_csv(file_path)
        questions: List[str] = df["question"].tolist()
        answers: List[str] = df["answer"].tolist()

        processed_questions = [self.preprocess_text(q) for q in questions]
        processed_answers = [self.preprocess_text(a) for a in answers]

        tokenized_questions = self.tokenize_texts(processed_questions)
        tokenized_answers = self.tokenize_texts(processed_answers)

        class QADataset(Dataset):
            # Internal dataset class for handling QA pairs
            def __init__(
                self, questions: List[List[str]], answers: List[List[str]]
            ) -> None:
                self.questions = questions
                self.answers = answers

            def __len__(self) -> int:
                # Return the number of samples
                return len(self.questions)

            def __getitem__(self, idx: int) -> Dict[str, Any]:
                # Return one sample pair (input and target)
                return {
                    "input": torch.tensor(self.questions[idx], dtype=torch.long),
                    "target": torch.tensor(self.answers[idx], dtype=torch.long),
                }

        dataset = QADataset(tokenized_questions, tokenized_answers)
        return dataset
