# test.py
import logging
logging.basicConfig(level=logging.INFO)
from src.bible_manager.downloader import BibleDownloader

downloader = BibleDownloader(config_path="config/bible_sources.json")
print("Downloader initialized")