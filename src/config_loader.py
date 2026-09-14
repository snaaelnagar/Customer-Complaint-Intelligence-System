"""
config_loader.py

Loads project configuration from config.yaml.
"""

import sys
from pathlib import Path

import yaml

from src.logger import logger
from src.exception import CustomException


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config.yaml"


def load_config() -> dict:
    """Load config.yaml safely."""
    try:
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(f"config.yaml not found at: {CONFIG_PATH}")

        with open(CONFIG_PATH, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if not config:
            raise ValueError("config.yaml is empty")

        logger.info("config.yaml loaded successfully")
        return config

    except Exception as e:
        logger.exception("Failed to load config.yaml")
        raise CustomException(e, sys)


config = load_config()