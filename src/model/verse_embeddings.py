from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
from typing import Dict, Optional
from src.bible_manager.storage import BibleStorage
import logging

# Configure logging only if not already configured
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VerseEmbedder:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        bible_id: str = None,
        storage_dir: str = "data/bible_storage",
        embeddings_dir: str = "data/embeddings"
    ):
        """
        Initialize the VerseEmbedder with the specified model and Bible data.

        Args:
            model_name (str): Name of the SentenceTransformer model to use.
            bible_id (str): ID of the Bible data to retrieve from storage.
            storage_dir (str): Directory where Bible data is stored.
            embeddings_dir (str): Directory where embeddings will be saved.
        """
        self.model = SentenceTransformer(model_name)
        self.storage = BibleStorage(storage_dir=storage_dir)
        self.bible_data = self.storage.retrieve_bible(bible_id) if bible_id else {}
        self.data_format = self._detect_format()
        logger.info(f"Detected data format: {self.data_format}")
        
        # Log a sample of the data based on format
        if self.data_format == "nested_dict" and self.bible_data:
            sample_book = list(self.bible_data.keys())[0]
            sample_data = {sample_book: {k: v for k, v in list(self.bible_data[sample_book].items())[:1]}}
            logger.info(f"Sample of retrieved data: {json.dumps(sample_data, indent=2)}")
        elif self.data_format in ["books_list_with_number_dict", "books_list_with_number_list"] and self.bible_data.get("books"):
            logger.info(f"Sample of retrieved data: {json.dumps(list(self.bible_data['books'])[:1], indent=2)}")
        else:
            logger.info("Sample of retrieved data: None")
        
        # Calculate total verses based on data format
        if self.data_format == "nested_dict":
            total_verses = sum(
                len(chapter)
                for book in self.bible_data.values()
                for chapter in book.values()
            )
        elif self.data_format in ["books_list_with_number_dict", "books_list_with_number_list"]:
            total_verses = sum(
                len(chapter["verses"])
                for book in self.bible_data.get("books", [])
                for chapter in book.get("chapters", [])
            )
        else:
            total_verses = 0
        logger.info(f"Total verses in data: {total_verses}")
        
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.bible_data:
            logger.warning("No Bible data loaded; embedding generation will be skipped")
    
    def _detect_format(self) -> str:
        """
        Detect the format of the Bible data.

        Returns:
            str: The detected format ("nested_dict", "books_list_with_number_dict",
                 "books_list_with_number_list", or "unknown").
        """
        if not self.bible_data:
            return "empty"
        
        if isinstance(self.bible_data, dict):
            if "books" in self.bible_data and isinstance(self.bible_data["books"], list):
                if self.bible_data["books"]:
                    book_keys = list(self.bible_data["books"][0].keys())
                    if "chapters" in book_keys:
                        chapters = self.bible_data["books"][0]["chapters"]
                        if isinstance(chapters, list) and chapters:
                            if "number" in chapters[0]:
                                verses = chapters[0].get("verses", {})
                                if isinstance(verses, dict):
                                    return "books_list_with_number_dict"
                                elif isinstance(verses, list):
                                    return "books_list_with_number_list"
                return "books_list"
            else:
                sample_key = list(self.bible_data.keys())[0]
                if isinstance(self.bible_data[sample_key], dict):
                    return "nested_dict"
        return "unknown"
    
    def generate_embeddings(self, version: str) -> Dict[str, list]:
        """
        Generate embeddings for the verses in the Bible data and save them to a file.

        Args:
            version (str): The version of the Bible (e.g., "kjv") for naming the output file.

        Returns:
            Dict[str, list]: A dictionary mapping verse keys to their embeddings.
        """
        if not self.bible_data:
            logger.error("No Bible data available to generate embeddings")
            return {}
        
        verse_texts = []
        verse_keys = []
        
        if self.data_format == "nested_dict":
            for book in self.bible_data:
                for chapter in self.bible_data[book]:
                    chapter_data = self.bible_data[book][chapter]
                    if not isinstance(chapter_data, dict):
                        logger.warning(f"Skipping {book} {chapter}: not a dict")
                        continue
                    for verse, text in chapter_data.items():
                        key = f"{book}_{chapter}_{verse}"
                        verse_texts.append(text)
                        verse_keys.append(key)
        
        elif self.data_format == "books_list_with_number_dict":
            for book_entry in self.bible_data["books"]:
                book_name = book_entry.get("name") or book_entry.get("book")
                if not book_name:
                    logger.warning("Book entry missing 'name' or 'boek' key, skipping")
                    continue
                
                chapters = book_entry.get("chapters", [])
                if not isinstance(chapters, list):
                    logger.warning(f"Chapters for {book_name} is not a list, skipping")
                    continue
                
                for chapter_entry in chapters:
                    chapter_num = chapter_entry.get("number")
                    if not chapter_num:
                        logger.warning(f"Chapter entry in {book_name} missing 'number', skipping")
                        continue
                    
                    verses = chapter_entry.get("verses", {})
                    if not isinstance(verses, dict):
                        logger.warning(f"Verses for {book_name} {chapter_num} is not a dict, skipping")
                        continue
                    
                    for verse_num, text in verses.items():
                        key = f"{book_name}_{chapter_num}_{verse_num}"
                        verse_texts.append(text)
                        verse_keys.append(key)
        
        elif self.data_format == "books_list_with_number_list":
            for book_entry in self.bible_data["books"]:
                book_name = book_entry.get("name") or book_entry.get("book")
                if not book_name:
                    logger.warning("Book entry missing 'name' or 'book' key, skipping")
                    continue
                
                chapters = book_entry.get("chapters", [])
                if not isinstance(chapters, list):
                    logger.warning(f"Chapters for {book_name} is not a list, skipping")
                    continue
                
                for chapter_entry in chapters:
                    chapter_num = chapter_entry.get("number")
                    if not chapter_num:
                        logger.warning(f"Chapter entry in {book_name} missing 'number', skipping")
                        continue
                    
                    verses = chapter_entry.get("verses", [])
                    if not isinstance(verses, list):
                        logger.warning(f"Verses for {book_name} {chapter_num} is not a list, skipping")
                        continue
                    
                    for verse_entry in verses:
                        if not isinstance(verse_entry, dict) or "number" not in verse_entry or "text" not in verse_entry:
                            logger.warning(f"Invalid verse entry in {book_name} {chapter_num}: {verse_entry}")
                            continue
                        verse_num = verse_entry["number"]
                        text = verse_entry["text"]
                        key = f"{book_name}_{chapter_num}_{verse_num}"
                        verse_texts.append(text)
                        verse_keys.append(key)
        
        else:
            logger.error(f"Unsupported data format: {self.data_format}")
            return {}
        
        if not verse_texts:
            logger.error("No verses extracted from Bible data")
            return {}
        
        logger.info(f"Generating embeddings for {len(verse_texts)} verses")
        
        embeddings = self.model.encode(
            verse_texts,
            convert_to_tensor=False,
            show_progress_bar=True,
            batch_size=32
        )
        
        embeddings_dict = {key: embedding.tolist() for key, embedding in zip(verse_keys, embeddings)}
        
        output_path = self.embeddings_dir / f"{version}_embeddings.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(embeddings_dict, f)
        logger.info(f"Saved embeddings to {output_path}")
        
        return embeddings_dict
    
    def load_embeddings(self, version: str) -> Dict[str, list]:
        """
        Load embeddings from a file.

        Args:
            version (str): The version of the Bible (e.g., "kjv") to load embeddings for.

        Returns:
            Dict[str, list]: A dictionary mapping verse keys to their embeddings.
        """
        embeddings_path = self.embeddings_dir / f"{version}_embeddings.json"
        try:
            with embeddings_path.open("r", encoding="utf-8") as f:
                embeddings = json.load(f)
            logger.info(f"Loaded embeddings from {embeddings_path}")
            return embeddings
        except FileNotFoundError:
            logger.warning(f"No embeddings found for version '{version}' at {embeddings_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode embeddings file {embeddings_path}: {str(e)}")
            return {}

if __name__ == "__main__":
    embedder = VerseEmbedder(bible_id="2d4db0be-da7c-4cf2-aa0a-65b8ea930c20")
    embeddings = embedder.generate_embeddings("kjv")
    loaded_embeddings = embedder.load_embeddings("kjv")
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Loaded {len(loaded_embeddings)} embeddings")
    if loaded_embeddings:
        print("Sample embedding for Genesis_50_1:", loaded_embeddings.get("Genesis_50_1")[:5])
    else:
        print("No embeddings loaded to display sample")