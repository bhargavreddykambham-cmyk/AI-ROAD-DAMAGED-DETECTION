"""
============================================================
AI Road Damage Detection — utils/logger.py
Centralized logging configuration
============================================================
"""

import sys
import os
from loguru import logger
from pathlib import Path


def setup_logger(
    log_dir: str = "logs",
    log_level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "30 days",
) -> None:
    """Configure loguru logger for the project."""
    Path(log_dir).mkdir(exist_ok=True)

    logger.remove()

    # Console handler
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | "
               "<cyan>{module}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
        colorize=True,
    )

    # File handler
    logger.add(
        os.path.join(log_dir, "road_damage_{time:YYYY-MM-DD}.log"),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | "
               "{module}:{function}:{line} | {message}",
        rotation=rotation,
        retention=retention,
        compression="zip",
    )

    logger.info("Logger initialized")


def get_logger():
    return logger


# Auto-setup on import
setup_logger()