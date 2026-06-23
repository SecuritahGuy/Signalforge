from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Literal

_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def setup_logging(
    *,
    name: str = "signalforge",
    level: str | int = "INFO",
    log_file: str | Path | None = None,
    module_levels: dict[str, str | int] | None = None,
) -> logging.Logger:
    """Configure structured logging for the signalforge package.

    Parameters
    ----------
    name : logger name (default "signalforge").
    level : log level string or int (default "INFO").
    log_file : optional path to a log file.
    module_levels : optional dict mapping module names to log levels.
    """
    resolved_level = _LOG_LEVELS.get(level.upper(), level) if isinstance(level, str) else level

    logger = logging.getLogger(name)
    logger.setLevel(resolved_level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(resolved_level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if module_levels:
        for mod_name, mod_level in module_levels.items():
            mod_level_resolved = _LOG_LEVELS.get(mod_level.upper(), mod_level) if isinstance(mod_level, str) else mod_level
            logging.getLogger(mod_name).setLevel(mod_level_resolved)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a module, inheriting root signalforge config."""
    return logging.getLogger(name)


class LogContext:
    """Context manager for adding structured context to log messages.

    Usage::

        with LogContext(symbol="AAPL", strategy="momentum"):
            logger.info("processing symbol")
    """

    def __init__(self, **kwargs: object) -> None:
        self._kwargs = kwargs

    def __enter__(self) -> LogContext:
        return self

    def __exit__(self, *args: object) -> None:
        pass
