#!/usr/bin/env python3
"""Script to verify the integrity and completeness of biblical data."""

import os
import json
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.data.preprocessing import BiblicalTextPreprocessor
from src.utils.logger import setup_logger

logger = setup_logger("data_verification", "logs/verification.log")

def verify_bible_data():
    """Verify the integrity and completeness of biblical data."""
    config_path = os.path.join(project_root, "config", "data_config.json")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        return False
    
    # Initialize preprocessor
    preprocessor = BiblicalTextPreprocessor()
    
    # Check raw bible data exists
    raw_dir = os.path.join(project_root, "data", "raw", "bibles")
    if not os.path.exists(raw_dir):
        logger.error("Raw bible data directory not found")
        return False
    
    # Check for required bible translations
    required_translations = config.get("validation", {}).get("required_translations", ["KJV"])
    missing_translations = []
    
    for translation in required_translations:
        translation_files = [
            f for f in os.listdir(raw_dir)
            if f.lower().startswith(translation.lower())
        ]
        if not translation_files:
            missing_translations.append(translation)
    
    if missing_translations:
        logger.error(f"Missing required translations: {', '.join(missing_translations)}")
        return False
    
    # Verify processed data
    processed_dir = os.path.join(project_root, "data", "processed")
    if not os.path.exists(processed_dir):
        logger.error("Processed data directory not found")
        return False
    
    # Verify minimum verses per book
    min_verses = config.get("validation", {}).get("min_verses_per_book", 10)
    required_books = config.get("validation", {}).get("required_books", [
        "Genesis", "Exodus", "Psalms", "Matthew", "John", "Romans", "Revelation"
    ])
    
    try:
        for translation in required_translations:
            bible_path = os.path.join(raw_dir, f"{translation.lower()}.json")
            if not os.path.exists(bible_path):
                continue
                
            with open(bible_path, 'r') as f:
                bible_data = json.load(f)
            
            for book in required_books:
                if book not in bible_data:
                    logger.error(f"Required book {book} not found in {translation}")
                    return False
                
                total_verses = sum(
                    len(verses) 
                    for chapter in bible_data[book].values() 
                    for verses in chapter.values()
                )
                
                if total_verses < min_verses:
                    logger.error(
                        f"Book {book} in {translation} has insufficient verses "
                        f"({total_verses} < {min_verses})"
                    )
                    return False
    
    except Exception as e:
        logger.error(f"Error verifying bible data: {e}")
        return False
    
    logger.info("Biblical data verification completed successfully")
    return True

if __name__ == "__main__":
    success = verify_bible_data()
    sys.exit(0 if success else 1)