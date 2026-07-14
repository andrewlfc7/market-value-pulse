from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass

import httpx


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/139.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class RateLimiter:
    """Global request-start rate limiter shared by all workers."""

    def __init__(self, requests_per_minute: int) -> None:
        self._minimum_interval = 60.0 / requests_per_minute
        self._lock = asyncio.Lock()
        self._last_started_at = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            sleep_for = self._minimum_interval - (now - self._last_started_at)
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            self._last_started_at = time.monotonic()


@dataclass(frozen=True)
class FetchResult:
    url: str
    content: bytes
    status_code: int
    attempts: int
    content_type: str


class FetchError(RuntimeError):
    def __init__(
        self,
        *,
        url: str,
        attempts: int,
        message: str,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.url = url
        self.attempts = attempts
        self.status_code = status_code


class TransfermarktHttpClient:
    def __init__(
        self,
        *,
        concurrency: int,
        requests_per_minute: int,
        timeout_seconds: float,
        max_retries: int,
    ) -> None:
        self._semaphore = asyncio.Semaphore(concurrency)
        self._rate_limiter = RateLimiter(requests_per_minute)
        self._max_retries = max_retries
        self._client = httpx.AsyncClient(
            headers=DEFAULT_HEADERS,
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=True,
        )

    async def __aenter__(self) -> "TransfermarktHttpClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self._client.aclose()

    async def get(self, url: str, *, accept_json: bool = False) -> FetchResult:
        headers = {"Accept": "application/json"} if accept_json else None
        last_error: Exception | None = None
        last_status: int | None = None

        for attempt in range(1, self._max_retries + 2):
            await self._rate_limiter.wait()

            try:
                async with self._semaphore:
                    response = await self._client.get(url, headers=headers)

                last_status = response.status_code
                if response.status_code == 200:
                    return FetchResult(
                        url=str(response.url),
                        content=response.content,
                        status_code=response.status_code,
                        attempts=attempt,
                        content_type=response.headers.get("content-type", ""),
                    )

                if response.status_code not in RETRYABLE_STATUS_CODES:
                    raise FetchError(
                        url=url,
                        attempts=attempt,
                        status_code=response.status_code,
                        message=f"Non-retryable HTTP {response.status_code}",
                    )

                last_error = RuntimeError(f"HTTP {response.status_code}")
            except FetchError:
                raise
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc

            if attempt <= self._max_retries:
                backoff = min(60.0, (2 ** (attempt - 1)) + random.uniform(0.25, 1.5))
                await asyncio.sleep(backoff)

        raise FetchError(
            url=url,
            attempts=self._max_retries + 1,
            status_code=last_status,
            message=str(last_error or "Unknown request failure"),
        )
