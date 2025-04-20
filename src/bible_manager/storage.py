# src/bible_manager/storage.py
"""
BibleStorage Module for Bible-AI
Manages the storage, retrieval, and querying of Bible texts with metadata indexing.
Enhanced for structured JSON support and additional functionality.
"""

import os
import json
import uuid
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import shutil
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger("bible_storage")

class BibleStorage:
    def __init__(self, config_path: Optional[str] = None, storage_dir: str = "data/bible_storage"):
        """Initialize BibleStorage with optional config and storage directory."""
        self.storage_dir = Path(storage_dir)
        self.index_file = self.storage_dir / "index.json"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.config = {}
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                logger.info(f"Loaded storage config from {config_path}")
            except Exception as e:
                logger.error(f"Failed to load config: {str(e)}")
        self.index = self._load_index()
        logger.info(f"BibleStorage initialized with storage_dir: {self.storage_dir}")

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load the index file or return an empty dict if it doesn’t exist."""
        try:
            if self.index_file.exists():
                with self.index_file.open('r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
            return {}

    def _save_index(self) -> None:
        """Save the index to disk with proper formatting."""
        try:
            with self.index_file.open('w', encoding='utf-8') as f:
                json.dump(self.index, f, indent=2)
            logger.debug(f"Index saved to {self.index_file}")
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")
            raise

    def _validate_bible_data(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate the structure of Bible JSON data."""
        try:
            if not isinstance(data, dict) or "books" not in data or "metadata" not in data:
                return False, "Invalid Bible JSON: Missing 'books' or 'metadata'"
            for book in data["books"]:
                if not all(k in book for k in ["name", "code", "chapters"]):
                    return False, f"Invalid book structure in {book.get('name', 'unknown')}"
                for chapter in book["chapters"]:
                    if not all(k in chapter for k in ["chapter", "verses"]):
                        return False, f"Invalid chapter structure in {book['name']} {chapter.get('chapter', 'unknown')}"
                    for verse in chapter["verses"]:
                        if not all(k in verse for k in ["verse", "text"]):
                            return False, f"Invalid verse structure in {book['name']} {chapter['chapter']}:{verse.get('verse', 'unknown')}"
            return True, "Validation successful"
        except Exception as e:
            return False, f"Validation failed: {str(e)}"

    def store_bible(self, file_path: str, metadata: Dict[str, Any]) -> str:
        """Store a Bible JSON file with validation and metadata."""
        file_id = str(uuid.uuid4())
        target_path = self.storage_dir / f"{file_id}.json"
        try:
            # Load and validate the Bible data
            with open(file_path, 'r', encoding='utf-8') as f:
                bible_data = json.load(f)
            is_valid, validation_msg = self._validate_bible_data(bible_data)
            if not is_valid:
                raise ValueError(validation_msg)

            # Update metadata with counts and timestamp
            metadata["uploaded_at"] = datetime.utcnow().isoformat()
            metadata["book_count"] = len(bible_data["books"])
            metadata["chapter_count"] = sum(len(book["chapters"]) for book in bible_data["books"])
            metadata["verse_count"] = sum(
                sum(len(chap["verses"]) for chap in book["chapters"]) for book in bible_data["books"]
            )
            bible_data["metadata"] = metadata

            # Save the validated and updated data
            shutil.copy(file_path, target_path)
            with target_path.open('w', encoding='utf-8') as f:
                json.dump(bible_data, f, indent=2)
            self.index[file_id] = metadata
            self._save_index()
            logger.info(f"Stored Bible with ID {file_id} - Books: {metadata['book_count']}, "
                       f"Chapters: {metadata['chapter_count']}, Verses: {metadata['verse_count']}")
            return file_id
        except Exception as e:
            logger.error(f"Storage failed for {file_path}: {str(e)}")
            raise

    def retrieve_bible(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored Bible by file_id."""
        try:
            file_path = self.storage_dir / f"{file_id}.json"
            if not file_path.exists():
                logger.warning(f"File not found for file_id: {file_id}")
                return None
            with file_path.open('r', encoding='utf-8') as f:
                bible_data = json.load(f)
            logger.info(f"Retrieved Bible text for file_id: {file_id}")
            return bible_data
        except Exception as e:
            logger.error(f"Failed to retrieve Bible file_id {file_id}: {str(e)}")
            return None

    def query_by_metadata(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query stored Bibles by metadata fields."""
        results = []
        for file_id, metadata in self.index.items():
            match = True
            for key, value in query.items():
                if metadata.get(key) != value:
                    match = False
                    break
            if match:
                bible_data = self.retrieve_bible(file_id)
                if bible_data:
                    results.append(bible_data)
        logger.info(f"Found {len(results)} matches for query: {query}")
        return results

    def list_stored_bibles(self) -> List[Dict[str, Any]]:
        """List metadata of all stored Bibles."""
        results = list(self.index.values())
        logger.info(f"Listed {len(results)} stored Bibles")
        return results

    def delete_bible(self, file_id: str) -> bool:
        """Delete a stored Bible by file_id."""
        try:
            file_path = self.storage_dir / f"{file_id}.json"
            if not file_path.exists():
                logger.warning(f"File not found for deletion: {file_id}")
                return False
            file_path.unlink()
            if file_id in self.index:
                del self.index[file_id]
                self._save_index()
            logger.info(f"Deleted Bible text with file_id: {file_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete Bible file_id {file_id}: {str(e)}")
            return False

    def cleanup(self) -> None:
        """Remove orphaned files and index entries."""
        try:
            for filename in os.listdir(self.storage_dir):
                if filename == "index.json":
                    continue
                file_id = os.path.splitext(filename)[0]
                if file_id not in self.index:
                    file_path = self.storage_dir / filename
                    file_path.unlink()
                    logger.info(f"Removed orphaned file: {file_path}")
            for file_id in list(self.index.keys()):
                file_path = self.storage_dir / f"{file_id}.json"
                if not file_path.exists():
                    del self.index[file_id]
                    logger.warning(f"Removed missing file_id from index: {file_id}")
            self._save_index()
            logger.info("Cleanup completed for storage directory")
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")

    def get_bible_stats(self, file_id: str) -> Optional[Dict[str, int]]:
        """Get statistics (book, chapter, verse counts) for a stored Bible."""
        bible_data = self.retrieve_bible(file_id)
        if not bible_data:
            return None
        stats = {
            "book_count": len(bible_data["books"]),
            "chapter_count": sum(len(book["chapters"]) for book in bible_data["books"]),
            "verse_count": sum(
                sum(len(chap["verses"]) for chap in book["chapters"]) for book in bible_data["books"]
            )
        }
        logger.info(f"Stats for file_id {file_id}: {stats}")
        return stats

    def validate_stored_bible(self, file_id: str) -> Tuple[bool, str]:
        """Validate the structure and integrity of a stored Bible."""
        bible_data = self.retrieve_bible(file_id)
        if not bible_data:
            return False, f"No Bible found for file_id: {file_id}"
        return self._validate_bible_data(bible_data)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Manage Bible storage for Bible-AI")
    parser.add_argument("--store", type=str, help="Path to a Bible JSON file to store")
    parser.add_argument("--retrieve", type=str, help="File ID to retrieve a Bible")
    parser.add_argument("--query", type=str, help="JSON query for metadata (e.g., '{\"translation\": \"KJV\"}')")
    parser.add_argument("--list", action="store_true", help="List all stored Bibles")
    parser.add_argument("--delete", type=str, help="File ID to delete a Bible")
    parser.add_argument("--stats", type=str, help="File ID to get Bible stats")
    parser.add_argument("--validate", type=str, help="File ID to validate Bible structure")
    parser.add_argument("--config", type=str, default="config/bible_sources.json", help="Path to configuration file")
    args = parser.parse_args()

    storage = BibleStorage(config_path=args.config)
    try:
        if args.store:
            metadata = {"translation": "Custom", "source": "manual"}
            file_id = storage.store_bible(args.store, metadata)
            print(f"Stored Bible with file_id: {file_id}")
        elif args.retrieve:
            bible_data = storage.retrieve_bible(args.retrieve)
            print(f"Retrieved Bible: {bible_data}")
        elif args.query:
            query = json.loads(args.query)
            results = storage.query_by_metadata(query)
            print(f"Query results: {len(results)} matches")
            for result in results:
                print(result["metadata"])
        elif args.list:
            bibles = storage.list_stored_bibles()
            print(f"Stored Bibles: {len(bibles)}")
            for bible in bibles:
                print(bible)
        elif args.delete:
            success = storage.delete_bible(args.delete)
            print(f"Deletion {'successful' if success else 'failed'}")
        elif args.stats:
            stats = storage.get_bible_stats(args.stats)
            print(f"Stats: {stats}")
        elif args.validate:
            is_valid, message = storage.validate_stored_bible(args.validate)
            print(f"Validation: {'Valid' if is_valid else 'Invalid'} - {message}")
        else:
            parser.print_help()
    finally:
        storage.cleanup()