"""
Unit tests for Mistral client rate limiting.
"""
import asyncio
import time

import pytest

from src.core.config import settings
from src.services.mistral_client import MistralDocumentClient


@pytest.mark.asyncio
async def test_rate_limiting_serializes_concurrent_requests(monkeypatch):
    """Concurrent rate-limit checks should be spaced by the configured interval."""
    previous_interval = settings.MISTRAL_MIN_REQUEST_INTERVAL
    monkeypatch.setattr(settings, "MISTRAL_MIN_REQUEST_INTERVAL", 0.05)
    client = MistralDocumentClient(api_key="test-key")

    start = time.perf_counter()
    await asyncio.gather(
        client._enforce_rate_limit(),
        client._enforce_rate_limit(),
        client._enforce_rate_limit(),
    )
    elapsed = time.perf_counter() - start

    assert elapsed >= settings.MISTRAL_MIN_REQUEST_INTERVAL * 2
    assert settings.MISTRAL_MIN_REQUEST_INTERVAL == 0.05
    monkeypatch.setattr(settings, "MISTRAL_MIN_REQUEST_INTERVAL", previous_interval)
