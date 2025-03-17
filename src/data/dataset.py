"""
Dataset classes for Bible AI.

This module provides dataset classes for working with biblical texts,
including verse-level datasets and theological datasets.
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
from torch.utils.data import Dataset

class BibleDataset(Dataset):
    """
    Base class for Bible datasets.
    """
    def __init__(self, data_dir: str):
        """
        Initialize the BibleDataset.
        
        Args:
            data_dir: Directory containing the data files
        """
        self.data_dir = data_dir
        self.data = []
        self.load_data()
        
    def load_data(self) -> None:
        """
        Load data from files. To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement load_data()")
        
    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.
        
        Returns:
            Number of samples
        """
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dictionary containing the sample data
        """
        return self.data[idx]

class VerseDataset(BibleDataset):
    """
    Dataset for Bible verses.
    """
    def __init__(
        self, 
        data_dir: str, 
        translation: str = "KJV", 
        book_filter: Optional[List[str]] = None
    ):
        """
        Initialize the VerseDataset.
        
        Args:
            data_dir: Directory containing the data files
            translation: Bible translation to use
            book_filter: Optional list of books to include
        """
        self.translation = translation
        self.book_filter = book_filter
        super().__init__(data_dir)
        
    def load_data(self) -> None:
        """
        Load verse data from files.
        """
        translation_file = os.path.join(self.data_dir, f"{self.translation}.json")
        
        if not os.path.exists(translation_file):
            raise FileNotFoundError(f"Translation file not found: {translation_file}")
            
        with open(translation_file, 'r', encoding='utf-8') as f:
            bible_data = json.load(f)
            
        self.data = []
        for book, chapters in bible_data.items():
            if self.book_filter and book not in self.book_filter:
                continue
                
            for chapter_num, verses in chapters.items():
                for verse_num, text in verses.items():
                    self.data.append({
                        'book': book,
                        'chapter': int(chapter_num),
                        'verse': int(verse_num),
                        'reference': f"{book} {chapter_num}:{verse_num}",
                        'text': text
                    })
                    
    def get_verse(self, reference: str) -> Optional[Dict[str, Any]]:
        """
        Get a verse by its reference.
        
        Args:
            reference: Verse reference (e.g., "John 3:16")
            
        Returns:
            Dictionary containing the verse data, or None if not found
        """
        for item in self.data:
            if item['reference'] == reference:
                return item
        return None
        
    def get_verses_by_book(self, book: str) -> List[Dict[str, Any]]:
        """
        Get all verses from a specific book.
        
        Args:
            book: Book name
            
        Returns:
            List of verse dictionaries
        """
        return [item for item in self.data if item['book'] == book]
        
    def get_chapter(self, book: str, chapter: int) -> List[Dict[str, Any]]:
        """
        Get all verses from a specific chapter.
        
        Args:
            book: Book name
            chapter: Chapter number
            
        Returns:
            List of verse dictionaries
        """
        return [
            item for item in self.data 
            if item['book'] == book and item['chapter'] == chapter
        ]

class TheologicalDataset(BibleDataset):
    """
    Dataset for theological concepts paired with Bible verses.
    """
    def __init__(
        self, 
        data_dir: str, 
        categories: Optional[List[str]] = None
    ):
        """
        Initialize the TheologicalDataset.
        
        Args:
            data_dir: Directory containing the data files
            categories: Optional list of theological categories to include
        """
        self.categories = categories
        super().__init__(data_dir)
        
    def load_data(self) -> None:
        """
        Load theological data from files.
        """
        theology_file = os.path.join(self.data_dir, "theological_concepts.json")
        
        if not os.path.exists(theology_file):
            raise FileNotFoundError(f"Theology file not found: {theology_file}")
            
        with open(theology_file, 'r', encoding='utf-8') as f:
            theology_data = json.load(f)
            
        self.data = []
        for concept, details in theology_data.items():
            if self.categories and details.get('category') not in self.categories:
                continue
                
            self.data.append({
                'concept': concept,
                'description': details.get('description', ''),
                'category': details.get('category', ''),
                'supporting_verses': details.get('supporting_verses', []),
                'related_concepts': details.get('related_concepts', [])
            })
            
    def get_concept(self, concept_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a theological concept by name.
        
        Args:
            concept_name: Name of the concept
            
        Returns:
            Dictionary containing the concept data, or None if not found
        """
        for item in self.data:
            if item['concept'].lower() == concept_name.lower():
                return item
        return None
        
    def get_concepts_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get all concepts in a specific category.
        
        Args:
            category: Category name
            
        Returns:
            List of concept dictionaries
        """
        return [item for item in self.data if item['category'] == category]
        
    def get_related_verses(self, concept_name: str) -> List[str]:
        """
        Get all verses related to a concept.
        
        Args:
            concept_name: Name of the concept
            
        Returns:
            List of verse references
        """
        concept = self.get_concept(concept_name)
        if concept:
            return concept['supporting_verses']
        return []
