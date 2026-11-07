"""Structured logging configuration with credential masking and safety controls."""

import logging
import re
import sys
from typing import Optional

# Regex patterns to detect potential API keys and tokens in log messages
SECRET_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9_-]{20,}", re.IGNORECASE),
    re.compile(r"(api[_-]?key|token|bearer|secret)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{12,})['\"]?", re.IGNORECASE),
]

class SecretMaskingFilter(logging.Filter):
    """Logging filter that scrubs suspected API keys, tokens, and credentials."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self._sanitize(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: self._sanitize(v) if isinstance(v, str) else v for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(self._sanitize(v) if isinstance(v, str) else v for v in record.args)
        return True

    @staticmethod
    def _sanitize(text: str) -> str:
        sanitized = text
        for pattern in SECRET_PATTERNS:
            sanitized = pattern.sub("[REDACTED_SECRET]", sanitized)
        return sanitized

def setup_logger(name: str = "rag_cot", level: Optional[str] = None) -> logging.Logger:
    """Configures and returns a structured logger with safety filters applied."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handler.addFilter(SecretMaskingFilter())
        logger.addHandler(handler)

    log_level = level or "INFO"
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    return logger

# Module-level default logger
logger = setup_logger()

