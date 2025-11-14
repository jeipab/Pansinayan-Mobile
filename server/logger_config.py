"""
Logging configuration for Pansinayan server.
"""

import logging
import sys
from pathlib import Path


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration with file and console handlers.
    
    Args:
        log_level: Logging level (e.g., "INFO", "DEBUG", "WARNING")
        
    Returns:
        Configured logger instance
        
    Note:
        If logs directory creation fails, falls back to console-only logging.
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    
    # Try to create logs directory and file handler
    logs_dir = Path("logs")
    try:
        logs_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(logs_dir / "server.log")
        handlers.append(file_handler)
    except (OSError, PermissionError) as e:
        # Fallback to console-only logging if directory creation fails
        print(f"Warning: Could not create logs directory: {e}. Using console logging only.")
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )
    
    # Set third-party loggers to WARNING to reduce noise
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured at {log_level} level")
    
    return logger