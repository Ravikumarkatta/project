import logging
from datetime import datetime
import os

def setup_logger(log_dir):
    """Set up logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('bible_ai')
    logger.setLevel(logging.INFO)
    
    # File handler
    log_file = os.path.join(log_dir, 'training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
