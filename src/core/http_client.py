"""
Shared HTTP client utilities: configured AsyncClient and retry helper.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Iterable, Optional

import httpx

from src.core.config import settings
from src.core.error_handling import request_id_var

logger = logging.getLogger(__name__)


def get_async_client(timeout: Optional[float] = None) -> httpx.AsyncClient:
    """Create a configured AsyncClient with shared limits/timeouts."""
    return httpx.AsyncClient(
        timeout=timeout or settings.HTTP_CLIENT_TIMEOUT,
        limits=httpx.Limits(
            max_keepalive_connections=settings.HTTP_MAX_KEEPALIVE_CONNECTIONS,
            max_connections=settings.HTTP_MAX_CONNECTIONS,
        ),
    )


async def request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    json: Any = None,
    data: Any = None,
    files: Any = None,
    timeout: Optional[float] = None,
    retry_statuses: Iterable[int] | None = None,
    max_attempts: Optional[int] = None,
) -> httpx.Response:
    """
    Perform an HTTP request with bounded retries and exponential backoff.

    - Honors Retry-After headers for 429/503 when present.
    - Retries connection errors and configured status codes.
    """
    attempts = max_attempts or settings.HTTP_RETRY_ATTEMPTS
    statuses = tuple(retry_statuses or settings.HTTP_RETRY_STATUSES)
    backoff_base = settings.HTTP_RETRY_BACKOFF_SECONDS
    req_id = request_id_var.get()

    for attempt in range(1, attempts + 1):
        try:
            response = await client.request(
                method,
                url,
                headers=headers,
                json=json,
                data=data,
                files=files,
                timeout=timeout or settings.HTTP_CLIENT_TIMEOUT,
            )

            if response.status_code not in statuses:
                return response

            retry_after = _get_retry_after_seconds(response)
            if attempt == attempts:
                return response

            delay = retry_after or backoff_base * math.pow(2, attempt - 1)
            logger.warning(
                f"[{req_id}] Retryable status {response.status_code} on {method} {url} "
                f"attempt {attempt}/{attempts}, sleeping {delay:.2f}s"
            )
            await asyncio.sleep(delay)
            continue

        except httpx.HTTPError as exc:
            if attempt == attempts:
                logger.error(f"[{req_id}] HTTP error after {attempts} attempts: {exc}")
                raise
            delay = backoff_base * math.pow(2, attempt - 1)
            logger.warning(
                f"[{req_id}] HTTP error on attempt {attempt}/{attempts}: {exc}; sleeping {delay:.2f}s"
            )
            await asyncio.sleep(delay)
            continue

    raise RuntimeError("request_with_retry exhausted attempts")


def _get_retry_after_seconds(response: httpx.Response) -> Optional[float]:
    retry_after = response.headers.get("Retry-After")
    if not retry_after:
        return None
    try:
        return float(retry_after)
    except ValueError:
        return None


@asynccontextmanager
async def get_managed_client(
    persistent_client: Optional[httpx.AsyncClient] = None,
    timeout: Optional[float] = None
):
    """
    Context manager for HTTP client lifecycle.

    Automatically handles creation and cleanup of clients.
    Reuses persistent client if provided, creates temporary otherwise.

    Args:
        persistent_client: Optional persistent client to reuse
        timeout: Optional timeout override

    Yields:
        httpx.AsyncClient instance

    Example:
        async with get_managed_client(self._client, self.timeout) as client:
            response = await client.post(url, ...)
    """
    should_close = persistent_client is None
    client = persistent_client or get_async_client(timeout=timeout)
    try:
        yield client
    finally:
        if should_close:
            await client.aclose()


class RateLimiter:
    """Generic rate limiter for API clients."""

    def __init__(self, min_interval: float):
        """
        Initialize rate limiter.

        Args:
            min_interval: Minimum seconds between requests
        """
        self.min_interval = min_interval
        self._last_request_time = 0.0
        self._lock = asyncio.Lock()

    async def wait_if_needed(self):
        """Wait if necessary to maintain rate limit."""
        async with self._lock:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
            self._last_request_time = time.time()
