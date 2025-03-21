# src/bible_manager/storage.py
"""
BibleStorage Module for Bible-AI

Manages the storage, retrieval, and querying of Bible texts with metadata indexing.
"""

import os
import json
import uuid
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import logging
import shutil

# Project-specific imports with fallbacks
try:
    from src.utils.logger import get_logger
except ImportError:
    import logging
    get_logger = lambda name: logging.getLogger(name)

# Initialize logger
logger = get_logger("bible_storage")

"""
    Manages the storage and retrieval of Bible texts in the Bible-AI system.

    Attributes:
        storage_dir: Directory to store Bible files.
        index_file: Path to the metadata index file.
        index: Metadata index mapping file IDs to metadata.
        config: Configuration dictionary.
    """

# src/bible_manager/storage.py (corrected)
class BibleStorage:
    def __init__(self, config_path: Optional[str] = None, storage_dir: str = "data/bible_storage"):
        self.storage_dir = Path(storage_dir)  # Use Path
        self.index_file = self.storage_dir / "index.json"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.config = {}
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                logger.info(f"Loaded storage config from {config_path}")
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
        self.index = self._load_index()
        logger.info(f"BibleStorage initialized with storage_dir: {self.storage_dir}")

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        try:
            if self.index_file.exists():
                with self.index_file.open('r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return {}

    def _save_index(self) -> None:
        try:
            with self.index_file.open('w', encoding='utf-8') as f:
                json.dump(self.index, f, indent=2)
            logger.debug(f"Index saved to {self.index_file}")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise

    def store_bible(self, bible_data: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        file_id = str(uuid.uuid4())
        file_path = self.storage_dir / f"{file_id}.json"
        try:
            with file_path.open('w', encoding='utf-8') as f:
                json.dump(bible_data, f, indent=2)
            self.index[file_id] = metadata
            self._save_index()
            logger.info(f"Stored Bible with ID {file_id}")
            return file_id
        except Exception as e:
            logger.error(f"Storage failed: {str(e)}")
            raise

    def retrieve_bible(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a stored Bible text by its file ID.

        Args:
            file_id: Unique identifier of the Bible file.

        Returns:
            Dict: Bible data if found, None otherwise.
        """
        try:
            file_path = os.path.join(self.storage_dir, f"{file_id}.json")
            if not os.path.exists(file_path):
                logger.warning("File not found for file_id: %s", file_id)
                return None

            with open(file_path, 'r', encoding='utf-8') as f:
                bible_data = json.load(f)
            logger.info("Retrieved Bible text for file_id: %s", file_id)
            return bible_data

        except Exception as e:
            logger.error("Failed to retrieve Bible file_id %s: %s", file_id, e)
            return None

    def query_by_metadata(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query stored Bible texts by metadata fields.

        Args:
            query: Dictionary of metadata fields to match (e.g., {"translation": "NIV"}).

        Returns:
            List: List of matching Bible texts with their metadata.
        """
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
        
        logger.info("Found %d matches for query: %s", len(results), query)
        return results

    def list_stored_bibles(self) -> List[Dict[str, Any]]:
        """
        List all stored Bible texts with their metadata.

        Returns:
            List: List of metadata for all stored Bibles.
        """
        results = list(self.index.values())
        logger.info("Listed %d stored Bibles", len(results))
        return results

    def delete_bible(self, file_id: str) -> bool:
        """
        Delete a stored Bible text by its file ID.

        Args:
            file_id: Unique identifier of the Bible file.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        try:
            file_path = os.path.join(self.storage_dir, f"{file_id}.json")
            if not os.path.exists(file_path):
                logger.warning("File not found for deletion: %s", file_id)
                return False

            os.remove(file_path)
            if file_id in self.index:
                del self.index[file_id]
                self._save_index()
            logger.info("Deleted Bible text with file_id: %s", file_id)
            return True

        except Exception as e:
            logger.error("Failed to delete Bible file_id %s: %s", file_id, e)
            return False

    def cleanup(self) -> None:
        """
        Clean up orphaned files and repair the index.
        """
        try:
            # Check for orphaned files (files not in index)
            for filename in os.listdir(self.storage_dir):
                if filename == "index.json":
                    continue
                file_id = os.path.splitext(filename)[0]
                if file_id not in self.index:
                    file_path = os.path.join(self.storage_dir, filename)
                    os.remove(file_path)
                    logger.info("Removed orphaned file: %s", file_path)

            # Check for missing files (in index but not on disk)
            for file_id in list(self.index.keys()):
                file_path = os.path.join(self.storage_dir, f"{file_id}.json")
                if not os.path.exists(file_path):
                    del self.index[file_id]
                    logger.warning("Removed missing file_id from index: %s", file_id)

            self._save_index()
            logger.info("Cleanup completed for storage directory")

        except Exception as e:
            logger.error("Cleanup failed: %s", e)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manage Bible storage for Bible-AI")
    parser.add_argument("--store", type=str, help="Path to a Bible JSON file to store")
    parser.add_argument("--retrieve", type=str, help="File ID to retrieve a Bible")
    parser.add_argument("--query", type=str, help="JSON query for metadata (e.g., '{\"translation\": \"NIV\"}')")
    parser.add_argument("--list", action="store_true", help="List all stored Bibles")
    parser.add_argument("--delete", type=str, help="File ID to delete a Bible")
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

        else:
            parser.print_help()

    finally:
        storage.cleanup()
