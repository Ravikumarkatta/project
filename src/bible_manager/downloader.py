# src/bible_manager/downloader.py
"""
Bible Downloader for Bible-AI.

Downloads Bible translations from configured sources and processes them for use in the system.
"""

import os
import requests
import json
import zipfile
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Project-specific imports with fallback
try:
    from src.utils.logger import get_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)
    get_logger = lambda name: logging.getLogger(name)

try:
    from src.bible_manager.converter import BibleConverter
    from src.bible_manager.uploader import BibleUploader
except ImportError as e:
    raise ImportError(f"Failed to import required modules: {e}")

logger = get_logger("bible_manager.downloader")

class BibleDownloader:
    """Downloads and processes Bible translations for Bible-AI."""
    
    # src/bible_manager/downloader.py (updated __init__)
    def __init__(self, config_path: str = "config/bible_sources.json", raw_dir: str = "data/raw/bibles"):
       self.config_path = Path(config_path).resolve()
       self.raw_dir = Path(raw_dir).resolve()
       self.raw_dir.mkdir(parents=True, exist_ok=True)
       self.sources = self._load_sources()
       self.converter = BibleConverter()
       self.uploader = BibleUploader(config_path=str(self.config_path))
       logger.info(f"BibleDownloader initialized with raw_dir: {self.raw_dir}")

    def _load_sources(self) -> Dict[str, Any]:
        """Load Bible sources from config file."""
        try:
            with self.config_path.open('r', encoding='utf-8') as f:
                sources = json.load(f)
            logger.info(f"Loaded {len(sources)} Bible sources")
            return sources
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load sources from {self.config_path}: {str(e)}")
            return {}

    def list_available_versions(self) -> List[str]:
        """List all available Bible versions."""
        versions = list(self.sources.keys())
        logger.info(f"Available versions: {versions}")
        return versions

    def get_version_info(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific Bible version."""
        version_info = self.sources.get(version_id)
        if not version_info:
            logger.warning(f"Version not found: {version_id}")
        return version_info

    def download_version(self, version_id: str, force: bool = False) -> bool:
        """
        Download and process a Bible version.
        
        Args:
            version_id: Bible version identifier.
            force: Force re-download if already exists.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        version_info = self.get_version_info(version_id)
        if not version_info:
            logger.error(f"Unknown version: {version_id}")
            return False

        target_dir = self.raw_dir / version_id
        zip_path = target_dir / f"{version_id}_raw.zip"

        if target_dir.exists() and not force:
            logger.info(f"Version {version_id} already downloaded; skipping (use force=True to override)")
            return True

        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            url = version_info.get("download_url")
            if not url or not url.startswith(("http://", "https://")):
                logger.error(f"Invalid or missing download URL for {version_id}")
                return False

            self._download_file(url, zip_path)
            self._process_zip_file(zip_path, target_dir, version_info)
            converted = self._validate_and_convert(target_dir, version_info)
            if not converted:
                return False

            metadata = {
                "translation": version_id,
                "source": version_info.get("source", "unknown"),
                "downloaded_at": target_dir.stat().st_mtime
            }
            upload_results = self.uploader.upload_directory(str(target_dir), metadata=metadata)
            success = all(result[0] for result in upload_results.values())
            if success:
                logger.info(f"Successfully processed {version_id}")
                return True
            logger.error(f"Upload failed for {version_id}: {upload_results}")
            return False
        except Exception as e:
            logger.error(f"Failed to process {version_id}: {str(e)}")
            return False
        finally:
            self._cleanup_temp_files(target_dir)

    def _download_file(self, url: str, local_path: Path) -> None:
        """Download a file with progress bar and retries."""
        for attempt in range(3):
            try:
                with requests.get(url, stream=True, timeout=10) as r:
                    r.raise_for_status()
                    total_size = int(r.headers.get('content-length', 0))
                    with local_path.open('wb') as f, tqdm(
                        desc=local_path.name,
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024
                    ) as bar:
                        for chunk in r.iter_content(chunk_size=8192):
                            size = f.write(chunk)
                            bar.update(size)
                return
            except requests.RequestException as e:
                logger.warning(f"Download attempt {attempt + 1} failed for {url}: {str(e)}")
                if attempt == 2:
                    raise Exception(f"Download failed after 3 attempts: {str(e)}")

    def _process_zip_file(self, zip_path: Path, target_dir: Path, version_info: Dict[str, Any]) -> None:
        """Extract and organize ZIP contents."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            logger.info(f"Extracted {zip_path} to {target_dir}")

            file_mapping = version_info.get("file_mapping", {})
            for src, dest in file_mapping.items():
                src_path = target_dir / src
                dest_path = target_dir / dest
                if src_path.exists():
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    src_path.rename(dest_path)
                    logger.debug(f"Moved {src} to {dest}")
        except zipfile.BadZipFile as e:
            logger.error(f"Invalid ZIP file {zip_path}: {str(e)}")
            raise

    def _validate_and_convert(self, target_dir: Path, version_info: Dict[str, Any]) -> bool:
        """Validate and convert files to JSON."""
        converted = False
        for file_path in target_dir.iterdir():
            if file_path.is_file():
                try:
                    input_format = self.converter._detect_format(str(file_path))
                    if not input_format:
                        continue
                    bible_data = self.converter._read_file(str(file_path), input_format)
                    if not bible_data or not self.converter._validate_bible_data(bible_data):
                        logger.warning(f"Invalid data in {file_path}")
                        continue
                    output_path = target_dir / f"{file_path.stem}_processed.json"
                    self.converter._write_json(bible_data, str(output_path))
                    converted = True
                    logger.info(f"Converted {file_path} to {output_path}")
                except Exception as e:
                    logger.error(f"Conversion failed for {file_path}: {str(e)}")
        return converted

    def download_multiple_versions(self, version_ids: List[str], force: bool = False) -> Dict[str, bool]:
        """Download multiple versions concurrently."""
        results = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(self.download_version, vid, force): vid for vid in version_ids}
            for future in futures:
                vid = futures[future]
                try:
                    results[vid] = future.result()
                except Exception as e:
                    results[vid] = False
                    logger.error(f"Failed to download {vid}: {str(e)}")
        logger.info(f"Download results: {results}")
        return results

    def is_version_downloaded(self, version_id: str) -> bool:
        """Check if a version is fully processed."""
        target_dir = self.raw_dir / version_id
        return target_dir.exists() and any(f.name.endswith("_processed.json") for f in target_dir.iterdir())

    def _cleanup_temp_files(self, target_dir: Path) -> None:
        """Remove temporary files."""
        for file_path in target_dir.iterdir():
            if file_path.name.endswith("_raw.zip"):
                try:
                    file_path.unlink()
                    logger.debug(f"Removed {file_path}")
                except OSError as e:
                    logger.warning(f"Failed to remove {file_path}: {str(e)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download Bible translations for Bible-AI")
    parser.add_argument("--versions", type=str, help="Comma-separated list of versions (e.g., NIV,ESV)")
    parser.add_argument("--config", type=str, default="config/bible_sources.json", help="Config file path")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    downloader = BibleDownloader(config_path=args.config)
    try:
        if args.versions:
            versions = args.versions.split(',')
            results = downloader.download_multiple_versions(versions, force=args.force)
            for version, success in results.items():
                print(f"{version}: {'Success' if success else 'Failed'}")
        else:
            print("Available versions:", downloader.list_available_versions())
    finally:
        # Cleanup handled per download, no need for explicit call here
        pass