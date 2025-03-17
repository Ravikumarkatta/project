"""
Logger module for Bible-AI application monitoring.

This module provides structured logging functionality with various severity levels,
formatting options, and theological context tracking for the Bible-AI platform.
"""

import logging
import os
import sys
import time
import json
from datetime import datetime
from typing import Optional, Union, Dict, Any, List

# Default log format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

class BibleAILogger:
    """
    Logger class for handling Bible-AI application logs with different severity levels,
    formats, and theological context tracking.
    """
    
    def __init__(
        self,
        name: str = "bible-ai",
        level: int = logging.INFO,
        log_format: str = DEFAULT_FORMAT,
        log_file: Optional[str] = None,
        console_output: bool = True,
        config_file: Optional[str] = None
    ):
        """
        Initialize logger with customizable parameters.
        
        Args:
            name: Logger name (default: "bible-ai")
            level: Logging level (default: logging.INFO)
            log_format: Format for log messages (default: DEFAULT_FORMAT)
            log_file: Optional file path to write logs to
            console_output: Whether to print logs to console (default: True)
            config_file: Optional path to logging configuration file
        """
        # Load config if provided
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    # Override parameters with config values
                    name = config.get('name', name)
                    level_str = config.get('level', 'INFO')
                    level = getattr(logging, level_str.upper(), level)
                    log_format = config.get('format', log_format)
                    log_file = config.get('log_file', log_file)
                    console_output = config.get('console_output', console_output)
            except Exception as e:
                # If config loading fails, use defaults and log the error
                print(f"Error loading logger config: {e}")
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Clear any existing handlers
        self.logger.handlers = []
        
        # Create formatter
        formatter = logging.Formatter(log_format)
        
        # Add console handler if requested
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Add file handler if log_file is specified
        if log_file:
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # For tracking theological context
        self.theological_context = {}
    
    def set_theological_context(self, **kwargs):
        """
        Set theological context for subsequent log messages.
        
        Args:
            **kwargs: Key-value pairs representing theological context
                (e.g., denomination, doctrine_area, bible_version)
        """
        self.theological_context.update(kwargs)
    
    def clear_theological_context(self):
        """Clear the theological context."""
        self.theological_context = {}
    
    def debug(self, message: str, **kwargs):
        """Log debug message with optional key-value pairs."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with optional key-value pairs."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional key-value pairs."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with optional key-value pairs."""
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with optional key-value pairs."""
        self._log(logging.CRITICAL, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs):
        """
        Internal method to log messages with contextual data.
        
        Args:
            level: Logging level
            message: Log message
            **kwargs: Additional key-value pairs to include in log
        """
        # Combine theological context with provided kwargs
        context = {**self.theological_context, **kwargs}
        
        if context:
            # Format extra context data as key-value pairs
            context_str = " ".join([f"{k}={v}" for k, v in context.items()])
            full_message = f"{message} - {context_str}"
        else:
            full_message = message
            
        self.logger.log(level, full_message)
    
    def log_verse_access(self, reference: str, translation: str, context: Optional[str] = None):
        """
        Log Bible verse access with specific formatting.
        
        Args:
            reference: Bible reference (e.g., "John 3:16")
            translation: Bible translation (e.g., "NIV", "ESV")
            context: Optional context of the access (e.g., "search", "study")
        """
        self.info(f"Verse access: {reference}", 
                 translation=translation, 
                 context=context or "direct")
    
    def log_theological_check(self, check_type: str, passed: bool, details: Optional[str] = None):
        """
        Log theological check results.
        
        Args:
            check_type: Type of theological check (e.g., "doctrine", "hermeneutics")
            passed: Whether the check passed
            details: Optional details about the check
        """
        status = "PASSED" if passed else "FAILED"
        self.info(f"Theological check {status}: {check_type}", 
                 check_type=check_type,
                 passed=passed,
                 details=details or "No details provided")
    
    def timed(self, message: str = "Operation completed", **kwargs):
        """
        Context manager for timing operations.
        
        Usage:
            with logger.timed("Bible search query"):
                results = search_bible(query)
                
        Args:
            message: Description of the operation being timed
            **kwargs: Additional context for the log
        """
        class TimedContext:
            def __init__(self, logger, message, context):
                self.logger = logger
                self.message = message
                self.context = context
                
            def __enter__(self):
                self.start = time.time()
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                elapsed = time.time() - self.start
                if exc_type:
                    self.logger.error(f"{self.message} failed after {elapsed:.3f}s", 
                                     error=str(exc_val),
                                     **self.context)
                else:
                    self.logger.info(f"{self.message} in {elapsed:.3f}s",
                                    **self.context)
                
        return TimedContext(self, message, kwargs)


def get_logger(
    name: str = "bible-ai",
    level: Union[str, int] = logging.INFO,
    log_file: Optional[str] = None,
    config_file: Optional[str] = None
) -> BibleAILogger:
    """
    Factory function to create a configured logger instance.
    
    Args:
        name: Logger name
        level: Logging level (can be string like "INFO" or integer constants)
        log_file: Optional path to log file
        config_file: Optional path to logging configuration file
        
    Returns:
        Configured BibleAILogger instance
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())
        
    # Create filename with timestamp if log_file is just a directory
    if log_file and os.path.isdir(log_file):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_file, f"{name}_{timestamp}.log")
        
    return BibleAILogger(name=name, level=level, log_file=log_file, config_file=config_file)


# Create a default logger instance
default_logger = get_logger()

# Convenience methods using the default logger
debug = default_logger.debug
info = default_logger.info
warning = default_logger.warning
error = default_logger.error
critical = default_logger.critical
log_verse_access = default_logger.log_verse_access
log_theological_check = default_logger.log_theological_check
timed = default_logger.timed
set_theological_context = default_logger.set_theological_context
clear_theological_context = default_logger.clear_theological_context


class LoggingMiddleware:
    """
    Middleware class for web application request logging.
    Can be used with FastAPI, Flask, or other web frameworks.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the middleware with a logger instance.
        
        Args:
            logger: Optional logger instance (uses default_logger if not specified)
        """
        self.logger = logger or default_logger
    
    async def log_request(self, request, call_next):
        """
        Log information about incoming requests and responses.
        
        Args:
            request: The HTTP request object
            call_next: Callable to process the request
            
        Returns:
            The HTTP response object
        """
        start_time = time.time()
        
        # Log request information
        self.logger.info(
            f"Request started: {request.method} {request.url.path}",
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if hasattr(request, 'client') else None
        )
        
        # Process the request
        try:
            response = await call_next(request)
            
            # Log response information
            process_time = time.time() - start_time
            self.logger.info(
                f"Request completed in {process_time:.3f}s: {response.status_code}",
                status_code=response.status_code,
                process_time=f"{process_time:.3f}s"
            )
            
            return response
        except Exception as e:
            # Log any exceptions
            process_time = time.time() - start_time
            self.logger.error(
                f"Request failed in {process_time:.3f}s: {str(e)}",
                error=str(e),
                process_time=f"{process_time:.3f}s"
            )
            raise
