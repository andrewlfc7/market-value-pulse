from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from playwright.sync_api import BrowserContext, Page

from ingestion.common import atomic_write_bytes
from ingestion.whoscored.discovery import discovered_matches_as_dicts

BLOCKED_RESOURCE_TYPES = {"image", "media", "font"}
BLOCKED_HOST_PARTS = {
    "doubleclick.net", "googletagmanager.com", "google-analytics.com",
    "googlesyndication.com", "facebook.net", "oddschecker.com",
}
MONTHS = {
    "January":"Jan", "Jan":"Jan", "February":"Feb", "Feb":"Feb",
    "March":"Mar", "Mar":"Mar", "April":"Apr", "Apr":"Apr",
    "May":"May", "June":"Jun", "Jun":"Jun", "July":"Jul", "Jul":"Jul",
    "August":"Aug", "Aug":"Aug", "September":"Sep", "Sept":"Sep", "Sep":"Sep",
    "October":"Oct", "Oct":"Oct", "November":"Nov", "Nov":"Nov",
    "December":"Dec", "Dec":"Dec",
}
DATE_HEADER_RE = re.compile(
    r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+"
    r"([A-Za-z]{3,9})\s+\d{1,2}\s+(\d{4})"
)


@dataclass(frozen=True)
class CalendarDiscovery:
    rows: list[dict[str, Any]]
    pages_collected: int
    first_calendar_label: str | None
    last_calendar_label: str | None


def configure_page(page: Page) -> None:
    def route_request(route) -> None:
        request = route.request
        url = request.url.lower()
        if request.resource_type in BLOCKED_RESOURCE_TYPES or any(
            part in url for part in BLOCKED_HOST_PARTS
        ):
            route.abort()
        else:
            route.continue_()
    page.route("**/*", route_request)


def dismiss_overlays(page: Page) -> None:
    for selector in (
        "button:has-text('Accept')", "button:has-text('Accept All')",
        "button:has-text('Agree')", "[aria-label='Close']",
    ):
        try:
            locator = page.locator(selector).first
            if locator.count() and locator.is_visible(timeout=300):
                locator.click(timeout=700, force=True)
        except Exception:
            pass


def _calendar_label(page: Page) -> str | None:
    for selector in ("#toggleCalendar span.toggleDatePicker", "#toggleCalendar span"):
        try:
            text = page.locator(selector).first.text_content(timeout=800)
            if text:
                parts = text.replace("▼", "").replace("▲", "").strip().split()
                month = MONTHS.get(parts[0], parts[0]) if parts else None
                year = next((x for x in parts if len(x) == 4 and x.isdigit()), None)
                return f"{month} {year}" if month and year else text.strip()
        except Exception:
            pass
    return None


def _fixture_month(page: Page) -> str | None:
    labels = [
        f"{MONTHS.get(month, month[:3])} {year}"
        for month, year in DATE_HEADER_RE.findall(page.content())
    ]
    return Counter(labels).most_common(1)[0][0] if labels else None


def _signature(page: Page) -> str:
    try:
        return page.eval_on_selector_all(
            "a[href]",
            "els => els.map(a => a.getAttribute('href')).filter(h => h && h.toLowerCase().includes('/matches/')).sort().join('|')",
        ) or ""
    except Exception:
        return ""


def _force_selected_month(page: Page, label: str | None, before: str) -> bool:
    if not label:
        return False
    month = label.split()[0]
    try:
        page.locator("#toggleCalendar").first.click(timeout=2_000, force=True)
        target = page.get_by_text(month, exact=True)
        for index in range(target.count()):
            locator = target.nth(index)
            if locator.is_visible(timeout=300):
                locator.click(timeout=1_000, force=True)
                break
        else:
            return False
    except Exception:
        return False
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        time.sleep(0.4)
        if _signature(page) not in {"", before}:
            return True
    return False


def _advance(page: Page, selector: str, wait_seconds: float) -> bool:
    dismiss_overlays(page)
    before_signature = _signature(page)
    before_label = _calendar_label(page)
    locator = page.locator(selector).first
    try:
        if not locator.count():
            return False
        locator.scroll_into_view_if_needed(timeout=1_000)
        locator.click(timeout=2_000, force=True)
    except Exception:
        try:
            clicked = page.evaluate(
                "sel => {const el=document.querySelector(sel); if(!el) return false; el.click(); return true;}",
                selector,
            )
            if not clicked:
                return False
        except Exception:
            return False
    deadline = time.monotonic() + 15
    label_changed = False
    while time.monotonic() < deadline:
        time.sleep(0.4)
        dismiss_overlays(page)
        current_label = _calendar_label(page)
        label_changed = label_changed or current_label != before_label
        current_signature = _signature(page)
        if current_signature and current_signature != before_signature:
            time.sleep(wait_seconds)
            return True
    if label_changed:
        return _force_selected_month(page, _calendar_label(page), before_signature)
    return False


def _open_fixture_page(
    context: BrowserContext,
    fixture_url: str,
    *,
    timeout_ms: int,
    wait_seconds: float,
) -> Page:
    page = context.new_page()
    configure_page(page)
    page.goto(fixture_url, wait_until="domcontentloaded", timeout=timeout_ms)
    try:
        page.wait_for_selector(
            "#dayChangeBtn-prev, #dayChangeBtn-next",
            timeout=min(timeout_ms, 15_000),
        )
    except Exception:
        pass
    time.sleep(wait_seconds)
    dismiss_overlays(page)
    return page


def _collect(
    page: Page,
    *,
    fixture_url: str,
    direction: str,
    step: int,
    raw_directory: Path,
) -> tuple[list[dict[str, Any]], str]:
    page_html = page.content()
    calendar_label = _calendar_label(page)
    fixture_month = _fixture_month(page)
    label = fixture_month or calendar_label or f"{direction}-{step}"
    safe_label = re.sub(r"[^A-Za-z0-9]+", "-", label).strip("-").lower()
    atomic_write_bytes(
        raw_directory / f"{direction}_{step:03d}_{safe_label}.html",
        page_html.encode("utf-8"),
    )
    rows = discovered_matches_as_dicts(page_html, page.url)
    for row in rows:
        row.update({
            "fixture_url": fixture_url,
            "calendar_label": calendar_label,
            "fixture_month_label": fixture_month,
            "direction": direction,
            "step": step,
        })
    return rows, label


def discover_fixture_calendar(
    context: BrowserContext,
    *,
    fixture_url: str,
    raw_directory: Path,
    max_previous: int = 60,
    max_next: int = 12,
    timeout_ms: int = 60_000,
    wait_seconds: float = 1.0,
) -> CalendarDiscovery:
    raw_directory.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    labels: list[str] = []
    seen_signatures: set[str] = set()
    pages = 0

    def record(page: Page, direction: str, step: int) -> tuple[bool, int]:
        nonlocal pages
        signature = _signature(page)
        repeated = bool(signature) and signature in seen_signatures
        if signature:
            seen_signatures.add(signature)
        rows, label = _collect(
            page,
            fixture_url=fixture_url,
            direction=direction,
            step=step,
            raw_directory=raw_directory,
        )
        pages += 1
        labels.append(label)
        all_rows.extend(rows)
        return repeated, len(rows)

    page = _open_fixture_page(
        context, fixture_url, timeout_ms=timeout_ms, wait_seconds=wait_seconds
    )
    record(page, "current", 0)
    empty_or_repeat = 0
    for step in range(1, max_next + 1):
        if not _advance(page, "#dayChangeBtn-next", wait_seconds):
            break
        repeated, count = record(page, "next", step)
        empty_or_repeat = empty_or_repeat + 1 if repeated or count == 0 else 0
        if empty_or_repeat >= 2:
            break
    page.close()

    page = _open_fixture_page(
        context, fixture_url, timeout_ms=timeout_ms, wait_seconds=wait_seconds
    )
    empty_or_repeat = 0
    for step in range(1, max_previous + 1):
        if not _advance(page, "#dayChangeBtn-prev", wait_seconds):
            break
        repeated, count = record(page, "previous", step)
        empty_or_repeat = empty_or_repeat + 1 if repeated or count == 0 else 0
        if empty_or_repeat >= 3:
            break
    page.close()

    unique = {int(row["match_id"]): row for row in all_rows}
    rows = [unique[key] for key in sorted(unique)]
    return CalendarDiscovery(
        rows=rows,
        pages_collected=pages,
        first_calendar_label=labels[-1] if labels else None,
        last_calendar_label=labels[0] if labels else None,
    )
