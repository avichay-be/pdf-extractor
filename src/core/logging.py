import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict

from src.core.config import settings
from src.core.error_handling import request_id_var


class RequestIDFilter(logging.Filter):
    """
    Inject the current request_id (from ContextVar) into log records.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        request_id = request_id_var.get()
        record.request_id = request_id if request_id else ""
        return True

class JSONFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings after parsing the LogRecord.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }

        if settings.LOG_INCLUDE_REQUEST_ID and getattr(record, "request_id", ""):
            log_obj["request_id"] = record.request_id
        
        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_obj.update(record.extra_fields)
            
        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_obj)

def setup_logging() -> None:
    """
    Configure logging for the application.
    Uses JSON logging if configured, otherwise standard text logging.
    """
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.DEBUG)
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    
    if settings.LOG_FORMAT.lower() == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + (
                " - request_id=%(request_id)s" if settings.LOG_INCLUDE_REQUEST_ID else ""
            )
        )
        
    handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
        
    root_logger.addHandler(handler)

    # Inject request id filter so formatters can include it
    if settings.LOG_INCLUDE_REQUEST_ID:
        handler.addFilter(RequestIDFilter())
    
    # Set level for specific loggers
    logging.getLogger("uvicorn.access").setLevel(log_level)
    logging.getLogger("uvicorn.error").setLevel(log_level)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # Reduce noise from httpx
