"""
Shared constants for PDF extraction.

This module consolidates constants used across the codebase to ensure
consistency and make it easier to modify common values.
"""

# Markdown formatting
MARKDOWN_SECTION_SEPARATOR = "\n\n---\n\n"
MARKDOWN_PAGE_HEADER_TEMPLATE = "# Page {page_number}\n\n"

# HTTP defaults
DEFAULT_HTTP_TIMEOUT = 120.0
DEFAULT_HEALTH_CHECK_TIMEOUT = 10.0
DEFAULT_KEEPALIVE_CONNECTIONS = 10
DEFAULT_MAX_CONNECTIONS = 20

# Pagination
DEFAULT_MAX_PAGES_PER_CHUNK = 15
MISTRAL_MAX_PAGES_LIMIT = 30  # API hard limit
