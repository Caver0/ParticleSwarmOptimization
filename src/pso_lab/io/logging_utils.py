from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import re
import sys

def _sanitize_logger_name(name: str) -> str:
    sanitized_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return sanitized_name or "pso_lab"


def _build_log_path(name: str, log_dir: str | Path) -> Path:
    logs_dir = Path(log_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return logs_dir / f"{_sanitize_logger_name(name)}_{timestamp}.log"


def setup_logger(
    name: str = "pso_lab",
    level: int = logging.INFO,
    log_dir: str | Path = "logs",
) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt = "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    log_path = _build_log_path(name, log_dir)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    setattr(logger, "log_file_path", log_path)

    logger.info("Log file created at %s", log_path)
    return logger
