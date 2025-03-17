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

try:
    from src.utils.logger import get_logger
except ImportError:
    try:
        from utils.logger import get_logger
    except ImportError:
        import logging
        get_logger = lambda name: logging.getLogger(name)

# Initialize logger
logger = get_logger("bible_manager.downloader")

class BibleDownloader:
    """
    Class for downloading Bible translations from various sources.
    """
    
    def __init__(self, config_path: Optional[str] = None, data_dir: Optional[str] = None):
        """
        Initialize the Bible downloader.
        
        Args:
            config_path: Path to Bible sources configuration file
            data_dir: Directory to store downloaded Bible files
        """
        # Default paths if not provided
        base_path = Path(os.path.abspath(__file__)).parent.parent.parent
        
        self.config_path = config_path or str(base_path / "config" / "bible_sources.json")
        self.data_dir = data_dir or str(base_path / "data" / "raw" / "bibles")
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load Bible sources configuration
        self.sources = self._load_sources()
        
    def _load_sources(self) -> Dict[str, Any]:
        """
        Load Bible sources configuration from JSON file.
        
        Returns:
            Dictionary of Bible sources and their details
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
            List of available Bible version identifiers
        """
        return list(self.sources.keys())
    
    def get_version_info(self, version_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific Bible version.
        
        Args:
            version_id: Bible version identifier
            
        Returns:
            Dictionary with version information or None if not found
        """
        return self.sources.get(version_id)
    
    def download_version(self, version_id: str, force: bool = False) -> bool:
        """
        Download a specific Bible version.
        
        Args:
            version_id: Bible version identifier
            force: Whether to force download even if already exists
            
        Returns:
            True if download successful, False otherwise
        """
        # Check if version exists in sources
        version_info = self.sources.get(version_id)
        if not version_info:
            logger.error(f"Unknown Bible version: {version_id}")
            return False
            
        # Determine target path
        target_dir = os.path.join(self.data_dir, version_id)
        if os.path.exists(target_dir) and not force:
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
                
            logger.info(f"Successfully downloaded Bible version {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download version {version_id}: {str(e)}")
            return False
            
    def _download_file(self, url: str, local_path: str) -> None:
        """
        Download a file with progress bar.
        
        Args:
            url: URL to download from
            local_path: Local path to save file to
        """
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
                    
    def _process_zip_file(self, zip_path: str, target_dir: str, version_info: Dict[str, Any]) -> None:
        """
        Process a downloaded ZIP file containing Bible text.
        
        Args:
            zip_path: Path to the downloaded ZIP file
            target_dir: Directory to extract to
            version_info: Version information dictionary
        """
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
            
    def download_multiple_versions(self, version_ids: List[str], force: bool = False) -> Dict[str, bool]:
        """
        Download multiple Bible versions.
        
        Args:
            version_ids: List of Bible version identifiers
            force: Whether to force download even if already exists
            
        Returns:
            Dictionary mapping version IDs to download success status
        """
        results = {}
        for version_id in version_ids:
            results[version_id] = self.download_version(version_id, force)
            
        return results
        
    def is_version_downloaded(self, version_id: str) -> bool:
        """
        Check if a Bible version is already downloaded.
        
        Args:
            version_id: Bible version identifier
            
        Returns:
            True if version is downloaded, False otherwise
        """
        version_dir = os.path.join(self.data_dir, version_id)
        return os.path.exists(version_dir)
