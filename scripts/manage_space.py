#!/usr/bin/env python3
"""Script to manage disk space for low-resource environments."""

import gzip
import logging
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiskSpaceManager:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.log_dir = self.project_root / "logs"
        self.data_dir = self.project_root / "data"
        self.downloads_dir = self.project_root / "downloads"

    def compress_old_logs(self, days_threshold: int = 7):
        """Compress log files older than specified days."""
        for log_file in self.log_dir.glob("*.log"):
            if not log_file.is_file():
                continue

            file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
            if datetime.now() - file_time > timedelta(days=days_threshold):
                try:
                    with open(log_file, "rb") as f_in:
                        with gzip.open(f"{log_file}.gz", "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(log_file)
                    logger.info(f"Compressed {log_file}")
                except Exception as e:
                    logger.error(f"Error compressing {log_file}: {e}")

    def cleanup_downloads(self):
        """Remove temporary download files."""
        try:
            shutil.rmtree(self.downloads_dir)
            os.makedirs(self.downloads_dir)
            logger.info("Cleaned downloads directory")
        except Exception as e:
            logger.error(f"Error cleaning downloads: {e}")

    def cleanup_cached_data(self):
        """Remove cached data files that can be regenerated."""
        cache_patterns = [
            "**/__pycache__",
            "**/.pytest_cache",
            "**/.mypy_cache",
            "**/node_modules/.cache",
            "frontend/build",
            "frontend/dist",
        ]

        for pattern in cache_patterns:
            for path in self.project_root.glob(pattern):
                try:
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                    logger.info(f"Removed cache: {path}")
                except Exception as e:
                    logger.error(f"Error removing cache {path}: {e}")

    def run_maintenance(self):
        """Run all maintenance tasks."""
        logger.info("Starting disk space maintenance...")
        self.compress_old_logs()
        self.cleanup_downloads()
        self.cleanup_cached_data()
        logger.info("Disk space maintenance completed")


def main():
    manager = DiskSpaceManager(os.getcwd())
    manager.run_maintenance()


if __name__ == "__main__":
    main()
