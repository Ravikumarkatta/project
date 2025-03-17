# src/bible_manager/downloader.py
"""
Bible Downloader for Bible-AI.

This module provides functionality for downloading Bible translations
from various sources and preparing them for use in the Bible-AI system.
"""

import os
import json
import requests
import zipfile
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
from tqdm import tqdm
import shutil

# Project-specific imports
try:
    from src.utils.logger import get_logger
except ImportError:
    try:
        from utils.logger import get_logger
    except ImportError:
        import logging
        get_logger = lambda name: logging.getLogger(name)

try:
    from src.bible_manager.converter import BibleConverter
    from src.bible_manager.uploader import BibleUploader
except ImportError as e:
    raise ImportError(f"Failed to import required modules: {e}")

# Initialize logger
logger = get_logger("bible_manager.downloader")

class BibleDownloader:
    """
    Class for downloading Bible translations from various sources.

    Attributes:
        config_path: Path to Bible sources configuration file.
        data_dir: Directory to store downloaded Bible files.
        sources: Dictionary of available Bible sources and their metadata.
        converter: BibleConverter instance for validating and converting downloaded files.
        uploader: BibleUploader instance for storing processed files.
    """
    
    def __init__(self, config_path: Optional[str] = None, data_dir: Optional[str] = None):
        """
        Initialize the Bible downloader.
        
        Args:
            config_path: Path to Bible sources configuration file.
            data_dir: Directory to store downloaded Bible files.
        """
        # Default paths if not provided
        base_path = Path(os.path.abspath(__file__)).parent.parent.parent
        self.config_path = config_path or str(base_path / "config" / "bible_sources.json")
        self.data_dir = data_dir or str(base_path / "data" / "raw" / "bibles")
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load Bible sources configuration
        self.sources = self._load_sources()
        
        # Initialize converter and uploader
        self.converter = BibleConverter(config_path=self.config_path)
        self.uploader = BibleUploader(config_path=self.config_path, upload_dir=str(base_path / "data" / "uploads"))
        logger.info("BibleDownloader initialized with data_dir: %s", self.data_dir)

    def _load_sources(self) -> Dict[str, Any]:
        """
        Load Bible sources configuration from JSON file.
        
        Returns:
            Dictionary of Bible sources and their details.
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    sources = json.load(f)
                    logger.info(f"Loaded {len(sources)} Bible source configurations")
                    return sources
            else:
                logger.warning(f"Bible sources config not found at {self.config_path}")
                return {}
        except Exception as e:
            logger.error(f"Failed to load Bible sources: {str(e)}")
            return {}
            
    def list_available_versions(self) -> List[str]:
        """
        List all available Bible versions that can be downloaded.
        
        Returns:
            List of available Bible version identifiers.
        """
        versions = list(self.sources.keys())
        logger.info("Available Bible versions: %s", versions)
        return versions
    
    def get_version_info(self, version_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific Bible version.
        
        Args:
            version_id: Bible version identifier.
            
        Returns:
            Dictionary with version information or None if not found.
        """
        version_info = self.sources.get(version_id)
        if version_info:
            logger.info("Retrieved info for version: %s", version_id)
        else:
            logger.warning("Version not found: %s", version_id)
        return version_info
    
    def download_version(self, version_id: str, force: bool = False) -> bool:
        """
        Download a specific Bible version and process it for use in Bible-AI.
        
        Args:
            version_id: Bible version identifier.
            force: Whether to force download even if already exists.
            
        Returns:
            True if download and processing were successful, False otherwise.
        """
        # Check if version exists in sources
        version_info = self.sources.get(version_id)
        if not version_info:
            logger.error(f"Unknown Bible version: {version_id}")
            return False
            
        # Determine target path
        target_dir = os.path.join(self.data_dir, version_id)
        if self.is_version_downloaded(version_id) and not force:
            logger.info(f"Bible version {version_id} already exists. Use force=True to redownload.")
            return True
            
        # Ensure target directory exists
        os.makedirs(target_dir, exist_ok=True)
        
        try:
            # Get download URL
            download_url = version_info.get("download_url")
            if not download_url:
                logger.error(f"No download URL for version {version_id}")
                return False
                
            # Download file
            logger.info(f"Downloading Bible version {version_id} from {download_url}")
            local_file = os.path.join(target_dir, f"{version_id}_raw.zip")
            self._download_file(download_url, local_file)
            
            # Process downloaded file based on format
            file_format = version_info.get("format", "unknown")
            if file_format == "zip":
                self._process_zip_file(local_file, target_dir, version_info)
            else:
                logger.warning(f"Unsupported format {file_format} for version {version_id}")
                return False
            
            # Validate and convert the downloaded files
            converted = self._validate_and_convert(target_dir, version_info)
            if not converted:
                logger.error(f"Validation and conversion failed for version {version_id}")
                return False
            
            # Upload to storage
            metadata = {
                "translation": version_id,
                "source": version_info.get("source", "unknown"),
                "downloaded_at": str(os.path.getmtime(target_dir))
            }
            upload_results = self.uploader.upload_directory(target_dir, metadata=metadata)
            success = all(result[0] for result in upload_results.values())
            
            if success:
                logger.info(f"Successfully downloaded and processed Bible version {version_id}")
                return True
            else:
                logger.error(f"Failed to upload processed files for version {version_id}: %s", upload_results)
                return False
                
        except Exception as e:
            logger.error(f"Failed to download version {version_id}: {str(e)}")
            return False
    
    def _download_file(self, url: str, local_path: str) -> None:
        """
        Download a file with a progress bar.
        
        Args:
            url: URL to download from.
            local_path: Local path to save file to.
        """
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                
                with open(local_path, 'wb') as f, tqdm(
                    desc=os.path.basename(local_path),
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        size = f.write(chunk)
                        bar.update(size)
        except Exception as e:
            logger.error(f"Download failed for URL {url}: {e}")
            raise

    def _process_zip_file(self, zip_path: str, target_dir: str, version_info: Dict[str, Any]) -> None:
        """
        Process a downloaded ZIP file containing Bible text.
        
        Args:
            zip_path: Path to the downloaded ZIP file.
            target_dir: Directory to extract to.
            version_info: Version information dictionary.
        """
        try:
            # Extract files
            logger.info(f"Extracting {zip_path} to {target_dir}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
                
            # Process files based on version-specific instructions
            file_mapping = version_info.get("file_mapping", {})
            for src, dest in file_mapping.items():
                src_path = os.path.join(target_dir, src)
                dest_path = os.path.join(target_dir, dest)
                
                if os.path.exists(src_path):
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    os.rename(src_path, dest_path)
                    logger.debug(f"Moved {src} to {dest}")
                
            # Clean up temporary files if specified
            if version_info.get("cleanup_temp", True) and os.path.exists(zip_path):
                os.remove(zip_path)
                logger.debug(f"Removed temporary file {zip_path}")
        except Exception as e:
            logger.error(f"Failed to process ZIP file {zip_path}: {e}")
            raise

    def _validate_and_convert(self, target_dir: str, version_info: Dict[str, Any]) -> bool:
        """
        Validate and convert downloaded Bible files to a standard format.
        
        Args:
            target_dir: Directory containing the downloaded files.
            version_info: Version information dictionary.
            
        Returns:
            bool: True if validation and conversion were successful, False otherwise.
        """
        try:
            converted_files = []
            for filename in os.listdir(target_dir):
                file_path = os.path.join(target_dir, filename)
                if os.path.isfile(file_path):
                    # Detect format
                    input_format = self.converter._detect_format(file_path)
                    if not input_format:
                        logger.warning(f"Could not detect format for {file_path}")
                        continue
                    
                    # Read and validate
                    bible_data = self.converter._read_file(file_path, input_format)
                    if not bible_data.get("books"):
                        logger.warning(f"No valid Bible data in {file_path}")
                        continue
                    
                    # Convert to standard JSON format
                    output_path = os.path.join(target_dir, f"{os.path.splitext(filename)[0]}_processed.json")
                    self.converter._write_json(bible_data, output_path)
                    converted_files.append(output_path)
                    logger.info(f"Converted {file_path} to {output_path}")
            
            return bool(converted_files)
        except Exception as e:
            logger.error(f"Validation and conversion failed in directory {target_dir}: {e}")
            return False

    def download_multiple_versions(self, version_ids: List[str], force: bool = False) -> Dict[str, bool]:
        """
        Download multiple Bible versions.
        
        Args:
            version_ids: List of Bible version identifiers.
            force: Whether to force download even if already exists.
            
        Returns:
            Dictionary mapping version IDs to download success status.
        """
        results = {}
        for version_id in version_ids:
            results[version_id] = self.download_version(version_id, force)
        logger.info("Download results: %s", results)
        return results
        
    def is_version_downloaded(self, version_id: str) -> bool:
        """
        Check if a Bible version is already downloaded.
        
        Args:
            version_id: Bible version identifier.
            
        Returns:
            True if version is downloaded, False otherwise.
        """
        version_dir = os.path.join(self.data_dir, version_id)
        if os.path.exists(version_dir):
            # Check if directory contains processed files
            for filename in os.listdir(version_dir):
                if filename.endswith("_processed.json"):
                    logger.info("Found processed file for version %s", version_id)
                    return True
            logger.warning("Version directory %s exists but no processed files found", version_id)
        return False

    def cleanup(self) -> None:
        """
        Clean up temporary files and directories.
        """
        try:
            for version_id in self.sources.keys():
                version_dir = os.path.join(self.data_dir, version_id)
                if os.path.exists(version_dir):
                    for filename in os.listdir(version_dir):
                        if filename.endswith("_raw.zip"):
                            file_path = os.path.join(version_dir, filename)
                            os.remove(file_path)
                            logger.debug(f"Removed temporary file {file_path}")
            logger.info("Cleanup completed for downloaded Bible versions")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Bible translations for Bible-AI")
    parser.add_argument("--versions", type=str, help="Comma-separated list of Bible versions to download (e.g., NIV,ESV,KJV)")
    parser.add_argument("--config", type=str, default="config/bible_sources.json", help="Path to Bible sources config file")
    parser.add_argument("--force", action="store_true", help="Force re-download even if version exists")
    args = parser.parse_args()

    downloader = BibleDownloader(config_path=args.config)
    try:
        if args.versions:
            version_list = args.versions.split(',')
            results = downloader.download_multiple_versions(version_list, force=args.force)
            for version, success in results.items():
                print(f"Downloaded {version}: {'Success' if success else 'Failed'}")
        else:
            print("Available versions:", downloader.list_available_versions())
            print("Usage: python downloader.py --versions NIV,ESV,KJV [--force]")
    finally:
        downloader.cleanup()
