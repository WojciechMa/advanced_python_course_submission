"""Configuration loader."""

import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/pipeline_config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.warning(f"Config file not found: {config_path}")
        return {}
    
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config
