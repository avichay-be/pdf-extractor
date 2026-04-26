"""
Helpers for structured and decorated logging.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any


def sanitize_log_fields(**fields: Any) -> dict[str, Any]:
    """Drop empty fields so log payloads stay compact."""
    return {
        key: value
        for key, value in fields.items()
        if value is not None
    }


def build_log_extra(**fields: Any) -> dict[str, dict[str, Any]]:
    """Build the ``extra`` payload expected by the application formatters."""
    extra_fields = sanitize_log_fields(**fields)
    if not extra_fields:
        return {}
    return {"extra_fields": extra_fields}


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    **fields: Any,
) -> None:
    """Log a normalized event with structured fields."""
    logger.log(level, event, extra=build_log_extra(**fields))


def fingerprint_text(text: str, length: int = 12) -> str:
    """Return a short stable fingerprint for text without logging the text itself."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:length]


def render_extra_fields(extra_fields: dict[str, Any]) -> str:
    """Render structured fields for plain-text logs."""
    parts: list[str] = []
    for key in sorted(extra_fields):
        value = extra_fields[key]
        if isinstance(value, (dict, list, tuple)):
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
        elif isinstance(value, str):
            rendered = value if value and value.replace("_", "").replace("-", "").isalnum() else json.dumps(value, ensure_ascii=False)
        else:
            rendered = str(value)
        parts.append(f"{key}={rendered}")
    return " ".join(parts)
