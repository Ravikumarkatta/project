# bible/src/utils/logger.py
"""Logging utility module."""

import json
import logging
import os
from pathlib import Path
# 2.2: Import Optional and Any
from typing import Optional, Dict, Any


# 2.1: Add type hint for 'level' parameter
def setup_logger(
    name: str, log_file: Optional[str] = None, level: int = logging.INFO
) -> logging.Logger:
    """Set up a logger with file and console handlers.

    Args:
        name: Name of the logger
        log_file: Path to log file. If None, only console logging is enabled
        level: Logging level (e.g., logging.INFO, logging.DEBUG)

    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times if logger already exists
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)

    # Create formatters
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Create file handler if log file specified
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        try:
            os.makedirs(log_path.parent, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except OSError as e:
            # Log error if directory creation or file handler setup fails
            logger.error(f"Failed to set up file handler for {log_file}: {e}")


    return logger


# 2.2: Change config_path type hint to Optional[str]
# 2.3: Change return type hint to dict[str, Any]
def load_logging_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load logging configuration from file.

    Args:
        config_path: Path to logging config JSON file

    Returns:
        Dictionary containing logging configuration
    """
    default_config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": "logs/app.log", # Default log file path
                "mode": "a",
            },
        },
        "loggers": {
            # Root logger configuration
            "": {"handlers": ["console", "file"], "level": "INFO", "propagate": True}
        },
    }

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path) as f:
                loaded_config = json.load(f)
                # Basic validation could be added here if needed
                return loaded_config
        except json.JSONDecodeError as e:
            print(f"Error decoding logging config file {config_path}: {e}")
            return default_config
        except Exception as e:
            print(f"Error loading logging config from {config_path}: {e}")
            return default_config
    elif config_path:
        print(f"Logging config file not found: {config_path}. Using default config.")

    return default_config

# Optional: Function to get logger using the loaded config (if you prefer this pattern)
# def get_logger_from_config(name: str, config_path: Optional[str] = None) -> logging.Logger:
#     config = load_logging_config(config_path)
#     logging.config.dictConfig(config)
#     return logging.getLogger(name)

# Simpler function to just get a logger instance (often sufficient)
def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance by name. Assumes setup_logger or config has been called.

    Args:
        name: Name of the logger.

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)
