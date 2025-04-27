"""Embedding generation and management pipeline for Bible-AI."""

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    model_name: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    max_length: int = 512
    batch_size: int = 32
    device: str = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    pooling_strategy: str = "mean"  # mean, cls, or max
    normalize: bool = True
    cache_dir: Optional[str] = None


class EmbeddingPipeline:
    """Pipeline for generating and managing text embeddings."""

    def __init__(self, config_path: str = "config/data_config.json"):
        """Initialize embedding pipeline.

        Args:
            config_path: Path to data configuration file
        """
        self.load_config(config_path)
        self.setup_model()
        self.setup_storage()

    def load_config(self, config_path: str):
        """Load pipeline configuration."""
        with open(config_path, "r") as f:
            self.config = json.load(f)

        embedding_config = self.config.get("embedding", {})
        self.embedding_config = EmbeddingConfig(
            model_name=embedding_config.get(
                "model_name", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            max_length=embedding_config.get("max_length", 512),
            batch_size=embedding_config.get("batch_size", 32),
            pooling_strategy=embedding_config.get("pooling_strategy", "mean"),
            normalize=embedding_config.get("normalize", True),
            cache_dir=embedding_config.get("cache_dir", None),
        )

    def setup_model(self):
        """Initialize the embedding model."""
        try:
            self.model = SentenceTransformer(
                self.embedding_config.model_name,
                cache_folder=self.embedding_config.cache_dir,
            )
            self.model.to(self.embedding_config.device)
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise

    def setup_storage(self):
        """Set up embedding storage directories and files."""
        base_dir = self.config.get("data_dir", "data")
        self.embeddings_dir = os.path.join(base_dir, "embeddings")
        os.makedirs(self.embeddings_dir, exist_ok=True)

        # Initialize or load embedding indices
        self.verse_index_path = os.path.join(self.embeddings_dir, "verse_embeddings.h5")
        self.commentary_index_path = os.path.join(
            self.embeddings_dir, "commentary_embeddings.h5"
        )

    def generate_embeddings(
        self, texts: List[str], show_progress: bool = True
    ) -> np.ndarray:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Returns:
            NumPy array of embeddings
        """
        embeddings = []

        # Process in batches
        for i in tqdm(
            range(0, len(texts), self.embedding_config.batch_size),
            disable=not show_progress,
            desc="Generating embeddings",
        ):
            batch = texts[i : i + self.embedding_config.batch_size]

            # Generate embeddings
            with torch.no_grad():
                batch_embeddings = self.model.encode(
                    batch,
                    show_progress_bar=False,
                    normalize_embeddings=self.embedding_config.normalize,
                )
                embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

    def save_verse_embeddings(self, embeddings: np.ndarray, references: List[str]):
        """Save verse embeddings to storage.

        Args:
            embeddings: NumPy array of embeddings
            references: List of verse references corresponding to embeddings
        """
        try:
            with h5py.File(self.verse_index_path, "a") as f:
                # Store embeddings
                if "embeddings" in f:
                    del f["embeddings"]
                f.create_dataset("embeddings", data=embeddings)

                # Store references
                if "references" in f:
                    del f["references"]
                ref_bytes = [ref.encode("utf-8") for ref in references]
                f.create_dataset("references", data=ref_bytes)

            logger.info(
                f"Saved {len(embeddings)} verse embeddings to "
                f"{self.verse_index_path}"
            )
        except Exception as e:
            logger.error(f"Error saving verse embeddings: {e}")
            raise

    def save_commentary_embeddings(self, embeddings: np.ndarray, metadata: List[Dict]):
        """Save commentary embeddings to storage.

        Args:
            embeddings: NumPy array of embeddings
            metadata: List of metadata dicts for each embedding
        """
        try:
            with h5py.File(self.commentary_index_path, "a") as f:
                # Store embeddings
                if "embeddings" in f:
                    del f["embeddings"]
                f.create_dataset("embeddings", data=embeddings)

                # Store metadata
                if "metadata" in f:
                    del f["metadata"]
                metadata_bytes = [json.dumps(m).encode("utf-8") for m in metadata]
                f.create_dataset("metadata", data=metadata_bytes)

            logger.info(
                f"Saved {len(embeddings)} commentary embeddings to "
                f"{self.commentary_index_path}"
            )
        except Exception as e:
            logger.error(f"Error saving commentary embeddings: {e}")
            raise

    def load_verse_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        """Load verse embeddings from storage.

        Returns:
            Tuple of (embeddings array, list of references)
        """
        try:
            with h5py.File(self.verse_index_path, "r") as f:
                embeddings = f["embeddings"][:]
                references = [ref.decode("utf-8") for ref in f["references"][:]]
            return embeddings, references
        except Exception as e:
            logger.error(f"Error loading verse embeddings: {e}")
            return np.array([]), []

    def load_commentary_embeddings(self) -> Tuple[np.ndarray, List[Dict]]:
        """Load commentary embeddings from storage.

        Returns:
            Tuple of (embeddings array, list of metadata dicts)
        """
        try:
            with h5py.File(self.commentary_index_path, "r") as f:
                embeddings = f["embeddings"][:]
                metadata = [json.loads(m.decode("utf-8")) for m in f["metadata"][:]]
            return embeddings, metadata
        except Exception as e:
            logger.error(f"Error loading commentary embeddings: {e}")
            return np.array([]), []

    def search_similar(
        self,
        query: str,
        index: str = "verse",
        top_k: int = 5
    ) -> List[Tuple[float, Union[str, Dict]]]:
        """Search for similar texts using embeddings.

        Args:
            query: Query text to search for
            index: Which embedding index to search ("verse" or "commentary")
            top_k: Number of results to return

        Returns:
            List of (score, reference/metadata) tuples
        """
        # Generate query embedding
        query_embedding = self.generate_embeddings([query], show_progress=False)[0]

        # Load appropriate index
        if index == "verse":
            embeddings, references = self.load_verse_embeddings()
            items = references
        else:
            embeddings, metadata = self.load_commentary_embeddings()
            items = metadata

        if len(embeddings) == 0:
            return []

        # Calculate similarities
        similarities = np.dot(embeddings, query_embedding)

        # Get top k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = [(similarities[i], items[i]) for i in top_indices]

        return results

    def process_verses(self, verses: Dict[str, Dict[int, Dict[int, str]]]):
        """Process and embed all verses.

        Args:
            verses: Dictionary of verses in format {book: {chapter: {verse: text}}}
        """
        # Prepare verse texts and references
        texts = []
        references = []

        for book, chapters in verses.items():
            for chapter, verse_dict in chapters.items():
                for verse, text in verse_dict.items():
                    texts.append(text)
                    references.append(f"{book} {chapter}:{verse}")

        # Generate and save embeddings
        embeddings = self.generate_embeddings(texts)
        self.save_verse_embeddings(embeddings, references)

    def process_commentaries(self, commentaries: Dict[str, List[Dict[str, Any]]]):
        """Process and embed all commentaries.

        Args:
            commentaries: Dictionary of commentaries by source
        """
        texts = []
        metadata = []

        for source, entries in commentaries.items():
            for entry in entries:
                if "content" in entry:
                    texts.append(entry["content"])
                    metadata.append(
                        {
                            "source": source,
                            "reference": entry.get("reference", ""),
                            "collected_at": entry.get("collected_at", ""),
                            **{
                                k: v
                                for k, v in entry.items()
                                if k
                                not in [
                                    "content",
                                    "source",
                                    "reference",
                                    "collected_at",
                                ]
                            },
                        }
                    )

        # Generate and save embeddings
        embeddings = self.generate_embeddings(texts)
        self.save_commentary_embeddings(embeddings, metadata)


def main():
    """Run embedding generation for all data."""
    import asyncio

    from src.data.commentary_collector import CommentaryCollector
    from src.data.preprocessing import BiblicalTextPreprocessor

    # Initialize pipeline
    pipeline = EmbeddingPipeline()

    # Load Bible data
    preprocessor = BiblicalTextPreprocessor()
    bible_data = {}
    bible_files = os.listdir(os.path.join(preprocessor.raw_dir, "bibles"))
    for file in bible_files:
        if file.endswith((".xml", ".json", ".txt")):
            translation = os.path.splitext(file)[0].upper()
            file_path = os.path.join(preprocessor.raw_dir, "bibles", file)
            bible_data[translation] = preprocessor.process_bible_file(
                file_path, translation
            )

    # Process verses
    pipeline.process_verses(bible_data)

    # Load and process commentaries
    collector = CommentaryCollector()
    commentaries = asyncio.run(collector.collect_all())
    pipeline.process_commentaries(commentaries)


if __name__ == "__main__":
    main()
