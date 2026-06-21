"""Structured logging for Vigil. One log line per state transition; never a bare print."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any


class _KeyValueFormatter(logging.Formatter):
    """Compact, greppable structured lines: `ts level logger msg key=value ...`."""

    def format(self, record: logging.LogRecord) -> str:
        base = (
            f"{self.formatTime(record, '%Y-%m-%dT%H:%M:%S')} {record.levelname:<5} "
            f"{record.name} {record.getMessage()}"
        )
        extra = getattr(record, "vigil_fields", None)
        if extra:
            kv = " ".join(f"{k}={_render(v)}" for k, v in extra.items())
            return f"{base} {kv}"
        return base


def _render(v: Any) -> str:
    if isinstance(v, (dict, list)):
        return json.dumps(v, separators=(",", ":"))
    s = str(v)
    return f'"{s}"' if " " in s else s


_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_KeyValueFormatter())
    root = logging.getLogger("vigil")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(f"vigil.{name}")


def set_level(level_name: str) -> None:
    """Apply a settings-driven log level once config has been read."""
    configure_logging()
    logging.getLogger("vigil").setLevel(level_name.upper())


def log_event(logger: logging.Logger, level: int, msg: str, **fields: Any) -> None:
    """Emit one structured line with key=value fields."""
    logger.log(level, msg, extra={"vigil_fields": fields})
