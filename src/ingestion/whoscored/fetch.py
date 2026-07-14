from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any, Callable

from playwright.async_api import Browser, Page, async_playwright

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
BLOCKED_RESOURCE_TYPES = {"image", "media", "font", "stylesheet"}
BLOCKED_HOST_PARTS = {
    "doubleclick.net",
    "googletagmanager.com",
    "google-analytics.com",
    "googlesyndication.com",
    "facebook.net",
}


@dataclass(frozen=True)
class MatchFetchRequest:
    match_id: int
    match_url: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class MatchFetchResult:
    match_id: int
    match_url: str
    html: str
    status_code: int | None
    elapsed_ms: int
    attempts: int
    error: str | None

    @property
    def succeeded(self) -> bool:
        return self.error is None and "matchCentreData" in self.html


async def _configure_page(page: Page) -> None:
    async def route_request(route) -> None:
        request = route.request
        url = request.url.lower()
        if request.resource_type in BLOCKED_RESOURCE_TYPES or any(
            part in url for part in BLOCKED_HOST_PARTS
        ):
            await route.abort()
        else:
            await route.continue_()

    await page.route("**/*", route_request)


async def _fetch_one(
    page: Page,
    request: MatchFetchRequest,
    *,
    timeout_ms: int,
    max_retries: int,
) -> MatchFetchResult:
    started = time.monotonic()
    status_code: int | None = None
    html = ""
    error: str | None = None
    attempts = 0
    for attempt in range(1, max_retries + 2):
        attempts = attempt
        error = None
        html = ""
        try:
            response = await page.goto(
                request.match_url,
                wait_until="domcontentloaded",
                timeout=timeout_ms,
            )
            status_code = response.status if response else None
            try:
                await page.wait_for_function(
                    "document.documentElement.innerHTML.includes('matchCentreData')",
                    timeout=min(timeout_ms, 20_000),
                )
            except Exception:
                pass
            html = await page.content()
            if "matchCentreData" not in html:
                error = "Page loaded without matchCentreData"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            try:
                html = await page.content()
            except Exception:
                html = ""
        if error is None:
            break
        if attempt <= max_retries:
            await asyncio.sleep(min(8.0, 1.5 * (2 ** (attempt - 1))))
    return MatchFetchResult(
        match_id=request.match_id,
        match_url=request.match_url,
        html=html,
        status_code=status_code,
        elapsed_ms=round((time.monotonic() - started) * 1000),
        attempts=attempts,
        error=error,
    )


async def _worker(
    browser: Browser,
    queue: asyncio.Queue[MatchFetchRequest | None],
    results: list[MatchFetchResult],
    *,
    timeout_ms: int,
    delay_ms: int,
    max_retries: int,
    on_result: Callable[[MatchFetchResult], None] | None,
) -> None:
    context = await browser.new_context(
        user_agent=USER_AGENT,
        viewport={"width": 1440, "height": 1100},
        extra_http_headers={
            "accept-language": "en-US,en;q=0.9",
            "referer": "https://www.whoscored.com/",
        },
    )
    page = await context.new_page()
    await _configure_page(page)
    while True:
        request = await queue.get()
        try:
            if request is None:
                break
            result = await _fetch_one(
                page,
                request,
                timeout_ms=timeout_ms,
                max_retries=max_retries,
            )
            results.append(result)
            if on_result is not None:
                on_result(result)
            if delay_ms:
                jitter = random.randint(0, max(1, delay_ms // 3))
                await asyncio.sleep((delay_ms + jitter) / 1000)
        finally:
            queue.task_done()
    await context.close()


async def fetch_match_pages(
    requests: list[MatchFetchRequest],
    *,
    workers: int = 2,
    timeout_ms: int = 45_000,
    delay_ms: int = 1_000,
    max_retries: int = 2,
    headful: bool = False,
    on_result: Callable[[MatchFetchResult], None] | None = None,
) -> list[MatchFetchResult]:
    queue: asyncio.Queue[MatchFetchRequest | None] = asyncio.Queue()
    for request in requests:
        queue.put_nowait(request)
    worker_count = max(1, min(workers, len(requests))) if requests else 0
    for _ in range(worker_count):
        queue.put_nowait(None)
    results: list[MatchFetchResult] = []
    if not requests:
        return results
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=not headful)
        tasks = [
            asyncio.create_task(
                _worker(
                    browser,
                    queue,
                    results,
                    timeout_ms=timeout_ms,
                    delay_ms=delay_ms,
                    max_retries=max_retries,
                    on_result=on_result,
                )
            )
            for _ in range(worker_count)
        ]
        await queue.join()
        await asyncio.gather(*tasks)
        await browser.close()
    return sorted(results, key=lambda result: result.match_id)
