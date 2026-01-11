"""Logging configuration."""

import logging
import sys
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_dir: str = "./logs"):
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
    """
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{log_dir}/app.log"),
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level={log_level}")
