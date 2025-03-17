"""
Utility module for providing common functionality across the application.

This module provides access to various utility functions and classes that
are used throughout the project, including logging, security, and theological
checks.
"""

# Import core utilities
from .logger import setup_logger, get_logger
from .security import check_security, sanitize_input

# Import theological checks if available
try:
    from .theological_checks import validate_theological_content
except ImportError:
    # Create a dummy function if the module is not yet implemented
    def validate_theological_content(*args, **kwargs):
        """
        Placeholder for theological content validation.
        
        This is a stub that will be replaced once the theological_checks module
        is implemented.
        """
        return True

# Version information
__version__ = '0.1.0'

# Define what should be imported with "from utils import *"
__all__ = [
    'setup_logger',
    'get_logger',
    'check_security',
    'sanitize_input',
    'validate_theological_content',
]
