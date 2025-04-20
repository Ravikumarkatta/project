# test.py
"""
Test script for BibleDownloader in Bible-AI.
Downloads and processes Bible versions from config/bible_sources.json.
"""

import argparse
import logging
import traceback
from src.bible_manager.downloader import BibleDownloader
# In test.py
from src.bible_manager.converter import BibleConverter

# Configure basic logging (overridden by src.utils.logger if available)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Test BibleDownloader for Bible-AI")
    parser.add_argument("--versions", type=str, default="kjv", help="Comma-separated list of versions to download (e.g., kjv,web)")
    parser.add_argument("--config", type=str, default="config/bible_sources.json", help="Path to config file")
    parser.add_argument("--force", action="store_true", help="Force re-download even if version exists")
    args = parser.parse_args()

    # Initialize downloader
    try:
        downloader = BibleDownloader(config_path=args.config)
    except Exception as e:
        print(f"Failed to initialize BibleDownloader: {str(e)}")
        traceback.print_exc()
        return

    # List available versions
    try:
        available_versions = downloader.list_available_versions()
        print("Available versions:", available_versions)
    except Exception as e:
        print(f"Error listing versions: {str(e)}")
        traceback.print_exc()
        return

    # Download specified versions
    versions = args.versions.split(",")
    for version in versions:
        version = version.strip()
        if version not in available_versions:
            print(f"Skipping {version}: not found in available versions")
            continue
        
        print(f"\nAttempting to download and process {version}...")
        try:
            success = downloader.download_version(version, force=args.force)
            if success:
                print(f"Successfully downloaded and processed {version}")
            else:
                print(f"Failed to download {version} (check logs for details)")
        except Exception as e:
            print(f"Error downloading {version}: {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    main()