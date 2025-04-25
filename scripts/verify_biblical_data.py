#!/usr/bin/env python
# scripts/verify_biblical_data.py

"""Script to verify the integrity of biblical data."""
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.bible_manager.validator import BibleDataValidator
from src.utils.logger import setup_logger

logger = setup_logger("bible_verification", "logs/data_verification.log")


def verify_bible_data():
    """Verify the integrity of biblical data files."""
    try:
        validator = BibleDataValidator()

        # Check raw data
        raw_data_status = validator.check_raw_data()
        if not raw_data_status:
            logger.error("Raw biblical data verification failed")
            sys.exit(1)

        # Check processed data
        processed_data_status = validator.check_processed_data()
        if not processed_data_status:
            logger.error("Processed biblical data verification failed")
            sys.exit(1)

        # Verify verse references
        verse_ref_status = validator.verify_verse_references()
        if not verse_ref_status:
            logger.error("Verse reference verification failed")
            sys.exit(1)

        logger.info("All biblical data verification checks passed")
        return True

    except Exception as e:
        logger.error(f"Biblical data verification failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    verify_bible_data()
