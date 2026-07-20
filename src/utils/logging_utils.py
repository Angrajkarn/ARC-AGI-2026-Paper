"""Structured logging utilities using loguru with rich formatting."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str | Path] = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """Configure the global loguru logger.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to write logs to file.
        rotation: Log file rotation policy.
        retention: Log file retention policy.
    """
    logger.remove()  # Remove default handler

    # Console handler with rich formatting
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    logger.add(sys.stderr, format=fmt, level=log_level, colorize=True)

    # File handler (optional)
    if log_file is not None:
        file_fmt = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
        logger.add(
            log_file,
            format=file_fmt,
            level=log_level,
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
        )


def get_logger(name: str):
    """Get a named logger (loguru uses the global logger; this returns it bound to the name).

    Args:
        name: Module name for log context.

    Returns:
        A bound loguru logger instance.
    """
    return logger.bind(module=name)


# Initialize with sensible defaults on import
setup_logger()

__all__ = ["setup_logger", "get_logger", "logger"]
