"""
Bible Downloader for Bible-AI.

Downloads Bible translations from configured sources or processes existing files for use in the system.
"""

import json
import os
import shutil
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests  # type: ignore  # Using type: ignore until types-requests is installed
from tqdm import tqdm  # type: ignore  # Using type: ignore until types-tqdm is installed

try:
    from src.utils.logger import get_logger
except ImportError:
    import logging

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

    def __init__(
        self,
        config_path: str = "config/bible_sources.json",
        raw_dir: str = "data/raw/bibles",
    ):
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
            with self.config_path.open("r", encoding="utf-8") as f:
                sources: Dict[str, Any] = json.load(f)
            logger.info(f"Loaded {len(sources.get('sources', sources))} Bible sources")
            return sources.get("sources", sources)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load sources from {self.config_path}: {str(e)}")
            empty_dict: Dict[str, Any] = {}
            return empty_dict

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

    def _parse_custom_text(self, file_path: Path) -> Dict[str, Any]:
        """Parse a custom text file with verse numbers (e.g., '1:1 In the beginning...')."""
        bible_data: Dict[str, Any] = {}
        current_book = None
        current_chapter = None

        try:
            with file_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    if line.startswith("BOOK: "):
                        current_book = line.replace("BOOK: ", "").strip()
                        current_chapter = None
                        logger.debug(f"Found book: {current_book}")
                        continue

                    if line.startswith("CHAPTER: "):
                        current_chapter = line.replace("CHAPTER: ", "").strip()
                        logger.debug(f"Found chapter: {current_chapter}")
                        continue

                    if (
                        current_book
                        and current_chapter
                        and ":" in line
                        and line.split(":")[0].isdigit()
                    ):
                        chapter_num, rest = line.split(":", 1)
                        verse_num, text = rest.split(" ", 1)
                        bible_data.setdefault(current_book, {}).setdefault(
                            chapter_num, {}
                        )[verse_num] = text.strip()
                    else:
                        logger.debug(f"Skipping line: {line}")
            return bible_data
        except Exception as e:
            logger.error(f"Failed to parse custom text file {file_path}: {str(e)}")
            empty_dict: Dict[str, Any] = {}
            return empty_dict

    def _parse_json_structured(self, file_path: Path) -> Dict[str, Any]:
        """Parse a structured JSON file with books, chapters, and verses."""
        bible_data: Dict[str, Any] = {}
        try:
            logger.debug(f"Opening JSON file: {file_path}")
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            logger.debug(f"JSON loaded, top-level keys: {list(data.keys())}")

            # Iterate over book names directly (e.g., "Genesis")
            for book_name in data:
                logger.debug(f"Processing book: {book_name}")
                chapters = data[book_name]
                # Ensure chapters is a dict
                if not isinstance(chapters, dict):
                    logger.warning(
                        f"Chapters for {book_name} is not a dictionary, skipping"
                    )
                    continue
                for chapter_num in chapters:
                    logger.debug(f"Processing chapter: {chapter_num} in {book_name}")
                    verses = chapters[chapter_num]
                for verse_num, verse_text in verses.items():
                    logger.debug(
                        f"Processing verse: {verse_num} - {verse_text[:50] if verse_text else 'None'}"
                    )
                    if not isinstance(verse_text, str):
                        logger.warning(
                            f"Verse text for {book_name} {chapter_num}:{verse_num} is not a string, skipping"
                        )
                        continue
                    bible_data.setdefault(book_name, {}).setdefault(chapter_num, {})[
                        verse_num
                    ] = verse_text.strip()

            logger.info(
                f"Parsed JSON structured file: {len(bible_data)} books processed"
            )
            return bible_data
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding failed for {file_path}: {str(e)}")
            empty_dict: Dict[str, Any] = {}
            return empty_dict
        except Exception as e:
            logger.error(f"Failed to parse JSON structured file {file_path}: {str(e)}")
            empty_dict: Dict[str, Any] = {}
            return empty_dict

    def _parse_gutenberg_text(self, file_path: Path) -> Dict[str, Any]:
        """Parse a Project Gutenberg Bible text file."""
        bible_data: Dict[str, Any] = {}
        current_book = None
        current_chapter = None
        verse_num = 1

        try:
            with file_path.open("r", encoding="utf-8") as f:
                in_bible_text = False
                for line in f:
                    line = line.strip()
                    if not in_bible_text:
                        if "*** START OF" in line:
                            in_bible_text = True
                        continue
                    if "*** END OF" in line:
                        break
                    if not line:
                        continue

                    if line.isupper() and "CHAPTER" not in line:
                        current_book = line
                        current_chapter = None
                        verse_num = 1
                        continue

                    if "CHAPTER" in line:
                        for word in line.split():
                            if word.isdigit():
                                current_chapter = word
                                verse_num = 1
                                break
                        continue

                    if current_book and current_chapter:
                        bible_data.setdefault(current_book, {}).setdefault(
                            current_chapter, {}
                        )[str(verse_num)] = line
                        verse_num += 1
            return bible_data
        except Exception as e:
            logger.error(f"Failed to parse Gutenberg file {file_path}: {str(e)}")
            empty_dict: Dict[str, Any] = {}
            return empty_dict

    def _parse_usfx_xml(self, file_path: Path) -> Dict[str, Any]:
        """Parse USFX XML into downloader format."""
        bible_data: Dict[str, Any] = {}
        current_book = None
        current_chapter = None
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            for elem in root:
                if elem.tag == "book":
                    current_book = elem.get("id")
                elif elem.tag == "c":
                    current_chapter = elem.get("id")
                elif elem.tag == "v" and current_book and current_chapter:
                    verse_num = elem.get("id")
                    text = elem.text.strip() if elem.text else ""
                    bible_data.setdefault(current_book, {}).setdefault(
                        current_chapter, {}
                    )[verse_num] = text
            return bible_data
        except Exception as e:
            logger.error(f"Failed to parse USFX XML {file_path}: {str(e)}")
            empty_dict: Dict[str, Any] = {}
            return empty_dict

    def process_local_file(
        self, version_id: str, local_file_path: str, force: bool = False
    ) -> bool:
        """Process an existing local Bible file."""
        version_info = self.get_version_info(version_id) or {
            "name": version_id,
            "format": "custom_text",
        }
        target_dir = self.raw_dir / version_id
        raw_file = Path(local_file_path).resolve()

        if not raw_file.exists():
            logger.error(f"Local file not found: {raw_file}")
            return False

        if self.is_version_downloaded(version_id) and not force:
            logger.info(f"Version {version_id} already processed")
            return True

        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            target_raw_file = target_dir / f"{version_id}_raw{raw_file.suffix}"
            if raw_file != target_raw_file:
                shutil.copy(raw_file, target_raw_file)
                logger.info(f"Copied local file to {target_raw_file}")
            else:
                logger.info(f"Using existing file at {target_raw_file}")
                target_raw_file = raw_file

            processed_file = self._process_file(
                version_id,
                target_raw_file,
                target_dir,
                version_info["format"],
                version_info,
            )
            if not processed_file:
                logger.error(f"Processing failed for {version_id}")
                return False

            logger.debug(f"Uploading processed file: {processed_file}")
            metadata = {
                "translation": version_id,
                "name": version_info.get("name", version_id),
                "format": version_info["format"],
                "source": "local",
                "downloaded_at": target_raw_file.stat().st_mtime,
            }
            success, message = self.uploader.upload_file(str(processed_file), metadata)
            if success:
                logger.info(f"Successfully uploaded {version_id} with ID: {message}")
            else:
                logger.error(f"Upload failed: {message}")
            return success
        except Exception as e:
            logger.error(f"Failed to process {version_id}: {str(e)}")
            return False
        finally:
            self._cleanup_temp_files(target_dir)

    def download_version(self, version_id: str, force: bool = False) -> bool:
        """Download or process a Bible version locally if download_url is null."""
        version_info = self.get_version_info(version_id)
        if not version_info:
            logger.error(f"Unknown version: {version_id}")
            return False

        target_dir = self.raw_dir / version_id
        format_type = version_info.get("format", "zip").lower()
        raw_file = (
            target_dir
            / f"{version_id}_raw{'txt' if format_type in ['custom_text', 'gutenberg'] else '.json' if format_type == 'json_structured' else '.zip'}"
        )

        if self.is_version_downloaded(version_id) and not force:
            logger.info(f"Version {version_id} already processed")
            return True

        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            url = version_info.get("download_url")
            if url and url.startswith(("http://", "https://")):
                self._download_file(url, raw_file)
                logger.info(f"Downloaded {version_id} to {raw_file}")
            elif raw_file.exists():
                logger.info(f"Using existing local file {raw_file}")
            else:
                logger.error(f"No download URL and local file not found: {raw_file}")
                return False

            processed_file = self._process_file(
                version_id, raw_file, target_dir, format_type, version_info
            )
            if not processed_file:
                logger.error(f"Processing failed for {version_id}")
                return False

            logger.debug(f"Uploading processed file: {processed_file}")
            metadata = {
                "translation": version_id,
                "name": version_info.get("name", version_id),
                "format": format_type,
                "source": str(raw_file) if not url else url,
                "downloaded_at": raw_file.stat().st_mtime,
            }
            success, message = self.uploader.upload_file(str(processed_file), metadata)
            if success:
                logger.info(f"Successfully uploaded {version_id} with ID: {message}")
            else:
                logger.error(f"Upload failed: {message}")
            return success
        except Exception as e:
            logger.error(f"Failed to process {version_id}: {str(e)}")
            return False
        finally:
            self._cleanup_temp_files(target_dir)

    def _download_file(self, url: str, local_path: Path) -> None:
        """Download a file with progress bar."""
        with requests.get(url, stream=True, timeout=10) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            with local_path.open("wb") as f, tqdm(
                desc=local_path.name,
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)

    def _process_file(
        self,
        version_id: str,
        raw_file: Path,
        target_dir: Path,
        format_type: str,
        version_info: Dict[str, Any],
    ) -> Optional[Path]:
        """Process a downloaded or local file."""
        processed_file = target_dir / f"{version_id}_processed.json"
        logger.debug(f"Starting processing: {raw_file}, format: {format_type}")
        try:
            if format_type == "custom_text":
                logger.debug("Parsing as custom_text")
                bible_data = self._parse_custom_text(raw_file)
            elif format_type == "json_structured":
                logger.debug("Parsing as json_structured")
                bible_data = self._parse_json_structured(raw_file)
            elif format_type == "gutenberg":
                logger.debug("Parsing as gutenberg")
                bible_data = self._parse_gutenberg_text(raw_file)
            elif format_type in ["usfx", "zip"]:
                logger.debug("Processing as USFX/ZIP")
                with zipfile.ZipFile(raw_file) as zip_ref:
                    zip_ref.extractall(target_dir)
                content_file = (
                    target_dir / "WEBUSFX.xml"
                    if version_id == "web"
                    else next(iter(target_dir.glob("*.xml")), Path(""))
                )
                if not content_file or not content_file.exists():
                    raise ValueError("No valid USFX content file found in ZIP")
                bible_data = self._parse_usfx_xml(content_file)
            else:
                logger.error(f"Unsupported format: {format_type}")
                return None

            if not bible_data:
                logger.error(f"No data processed from {raw_file}")
                return None

            with processed_file.open("w", encoding="utf-8") as f:
                json.dump(bible_data, f, indent=2)
            logger.debug(f"Processed file created: {processed_file}")
            return processed_file
        except Exception as e:
            logger.error(f"Processing failed for {raw_file}: {str(e)}")
            return None

    def download_multiple_versions(
        self, version_ids: List[str], force: bool = False
    ) -> Dict[str, bool]:
        """Download multiple versions concurrently."""
        results = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self.download_version, vid, force): vid
                for vid in version_ids
            }
            for future in futures:
                results[futures[future]] = future.result()
        return results

    def is_version_downloaded(self, version_id: str) -> bool:
        """Check if a version is fully processed."""
        return (self.raw_dir / version_id / f"{version_id}_processed.json").exists()

    def _cleanup_temp_files(self, target_dir: Path) -> None:
        """Remove temporary raw files."""
        for file_path in target_dir.glob("*_raw.*"):
            try:
                file_path.unlink()
                logger.debug(f"Cleaned up: {file_path}")
            except OSError as e:
                logger.warning(f"Failed to remove {file_path}: {str(e)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download Bible translations for Bible-AI"
    )
    parser.add_argument("--versions", type=str, help="Comma-separated list of versions")
    parser.add_argument("--config", type=str, default="config/bible_sources.json")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--local-file", type=str, help="Path to local Bible file")
    args = parser.parse_args()

    downloader = BibleDownloader(config_path=args.config)
    if args.local_file and args.versions:
        success = downloader.process_local_file(
            args.versions.split(",")[0], args.local_file, args.force
        )
        print(f"Local file: {'Success' if success else 'Failed'}")
    elif args.versions:
        results = downloader.download_multiple_versions(
            args.versions.split(","), args.force
        )
        for version, success in results.items():
            print(f"{version}: {'Success' if success else 'Failed'}")
    else:
        print("Available versions:", downloader.list_available_versions())
