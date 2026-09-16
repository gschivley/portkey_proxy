#!/usr/bin/env python3
"""
Usage & cost tracking for the Portkey proxy (used by proxy.py and
usage_report.py).

Every completed chat request is appended as one JSON line to a usage log
(default: usage.jsonl next to this script) with its token counts and the
dollar cost computed from Portkey's public model-pricing API:

    GET {PRICING_BASE_URL}/pricing/<provider>/<model>
    (default PRICING_BASE_URL = https://api.portkey.ai/model-configs)

Cost is computed at record time; each record also stores the rates that were
applied, so history never changes when prices do. Records whose model has no
known price are stored with cost_usd=null and counted separately in reports.

Monthly totals are compared against MONTHLY_BUDGET_USD (default 250, the
Princeton AI Sandbox cap). Crossing a budget/5 milestone (or having <= $25 /
<= $10 left) logs a warning. Tracking is best-effort: it never raises into
the request path.

Configuration (environment / .env, read at import time):
    USAGE_TRACKING       "0"/"false" to disable recording (default on)
    USAGE_LOG_PATH       JSONL log path (default: usage.jsonl next to script)
    MONTHLY_BUDGET_USD   monthly budget for reports/warnings (default 250)
    PRICING_BASE_URL     pricing API base (default https://api.portkey.ai/model-configs)
    PRICING_CACHE_PATH   pricing cache file (default .pricing_cache.json)
    PRICING_TTL_HOURS    pricing cache TTL (default 24)

Price units: the pricing API returns the per-token USD price scaled x100
(verified: openai/gpt-4o -> 0.00025 == $2.50/1M; openai/gpt-5 -> 0.000125 ==
$1.25/1M; anthropic/claude-3-5-sonnet -> 0.0003 == $3.00/1M). Hence
PRICE_SCALE below: cost_usd = tokens * price / PRICE_SCALE.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

SCRIPT_DIR = Path(__file__).resolve().parent

# API price values are per-token USD price x100 (see module docstring).
PRICE_SCALE = 100.0

USAGE_TRACKING = os.environ.get("USAGE_TRACKING", "1").lower() not in (
    "0",
    "false",
    "no",
    "off",
)
USAGE_LOG_PATH = Path(os.environ.get("USAGE_LOG_PATH") or SCRIPT_DIR / "usage.jsonl")
MONTHLY_BUDGET_USD = float(os.environ.get("MONTHLY_BUDGET_USD") or 250)
PRICING_BASE_URL = (
    os.environ.get("PRICING_BASE_URL") or "https://api.portkey.ai/model-configs"
).rstrip("/")
PRICING_CACHE_PATH = Path(
    os.environ.get("PRICING_CACHE_PATH") or SCRIPT_DIR / ".pricing_cache.json"
)
PRICING_TTL_HOURS = float(os.environ.get("PRICING_TTL_HOURS") or 24)

_MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")

log = logging.getLogger("usage")


def current_month() -> str:
    """Calendar month bucket (local time), e.g. '2026-09'."""
    return time.strftime("%Y-%m")


# --------------------------------------------------------------------------- #
# Pricing cache
# --------------------------------------------------------------------------- #


class PricingCache:
    """Lazy, file-cached access to Portkey's public per-model pricing API."""

    def __init__(
        self, base_url: str, path: Path, ttl_hours: float, timeout: float = 10.0
    ) -> None:
        self.base_url = base_url
        self.path = path
        self.ttl_seconds = ttl_hours * 3600
        self.timeout = timeout
        self._lock = threading.Lock()
        self._data: Optional[Dict[str, Dict[str, Any]]] = None

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if self._data is not None:
            return self._data
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._data = raw if isinstance(raw, dict) else {}
        except (OSError, ValueError):
            self._data = {}
        return self._data

    def _save(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            tmp.write_text(json.dumps(self._data or {}, indent=1), encoding="utf-8")
            tmp.replace(self.path)
        except OSError as exc:
            log.warning("could not write pricing cache %s: %s", self.path, exc)

    def _fetch(self, provider: str, model: str) -> Optional[Dict[str, float]]:
        url = f"{self.base_url}/pricing/{provider}/{model}"
        try:
            resp = httpx.get(url, timeout=self.timeout)
        except Exception as exc:  # noqa: BLE001 - never break the request path
            log.warning("pricing fetch failed for %s/%s: %s", provider, model, exc)
            return None
        if resp.status_code == 404:
            return {}  # known-missing model; cached as a negative result
        if resp.status_code != 200:
            log.warning("pricing fetch %s -> HTTP %s", url, resp.status_code)
            return None
        try:
            payload = resp.json()
        except ValueError:
            log.warning("pricing fetch %s -> invalid JSON", url)
            return None
        return _rates_from_payload(payload)

    def rates(self, provider: str, model: str) -> Optional[Dict[str, float]]:
        """Rate dict for provider/model (empty dict = no pricing known).

        Fresh cache first; otherwise re-fetch. Stale cache beats no cache
        when a refresh fails. None only when the model was never seen.
        """
        if not provider or not model:
            return None
        key = f"{provider}/{model}"
        with self._lock:
            cache = self._load()
            entry = cache.get(key)
            now = time.time()
            fresh = entry and (now - entry.get("fetched_at", 0) < self.ttl_seconds)
            if not fresh:
                fetched = self._fetch(provider, model)
                if fetched is not None:
                    entry = {"fetched_at": now, "rates": fetched}
                    cache[key] = entry
                    self._save()
            if not entry:
                return None
            return dict(entry.get("rates") or {})


def _rates_from_payload(payload: Any) -> Dict[str, float]:
    """Extract the token rate dict from a pricing API response body."""
    out: Dict[str, float] = {}
    if not isinstance(payload, dict):
        return out
    prices = payload.get("pay_as_you_go")
    if not isinstance(prices, dict):
        return out
    key_map = {
        "request_token": "input",
        "response_token": "output",
        "cache_read_input_token": "cache_read_input",
        "cache_write_input_token": "cache_write_input",
    }
    for src, dst in key_map.items():
        cell = prices.get(src)
        price = cell.get("price") if isinstance(cell, dict) else None
        if isinstance(price, (int, float)):
            out[dst] = float(price)
    return out


_PRICING = PricingCache(PRICING_BASE_URL, PRICING_CACHE_PATH, PRICING_TTL_HOURS)


# --------------------------------------------------------------------------- #
# Cost computation
# --------------------------------------------------------------------------- #


def _to_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def extract_tokens(usage: Dict[str, Any]) -> Dict[str, int]:
    """Normalize an OpenAI usage object into the token counts we track."""
    prompt = _to_int(usage.get("prompt_tokens"))
    details = usage.get("prompt_tokens_details")
    cached = _to_int(details.get("cached_tokens")) if isinstance(details, dict) else 0
    cached = min(cached, prompt)
    completion = _to_int(usage.get("completion_tokens"))
    total = _to_int(usage.get("total_tokens")) or (prompt + completion)
    cdetails = usage.get("completion_tokens_details")
    reasoning = (
        _to_int(cdetails.get("reasoning_tokens")) if isinstance(cdetails, dict) else 0
    )
    return {
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "total_tokens": total,
    }


def compute_cost(rates: Dict[str, float], tokens: Dict[str, int]) -> float:
    """USD cost for one call: (uncached input + cache-read + output) / scale."""
    uncached = tokens["prompt_tokens"] - tokens["cached_tokens"]
    input_rate = rates.get("input", 0.0)
    cache_rate = rates.get("cache_read_input", input_rate)
    cost = (
        uncached * input_rate
        + tokens["cached_tokens"] * cache_rate
        + tokens["completion_tokens"] * rates.get("output", 0.0)
    ) / PRICE_SCALE
    return round(cost, 8)


# --------------------------------------------------------------------------- #
# Usage log (append-only JSONL)
# --------------------------------------------------------------------------- #

_LOG_LOCK = threading.Lock()
# Running total for the current month, kept in memory; re-derived from the
# log whenever the process starts up in or rolls into a new month.
_MONTH_STATE: Dict[str, Any] = {"month": None, "total": 0.0}


def _append_line(record: Dict[str, Any]) -> None:
    try:
        with USAGE_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
    except OSError as exc:
        log.warning("could not append to usage log %s: %s", USAGE_LOG_PATH, exc)


def iter_records(month: Optional[str] = None) -> List[Dict[str, Any]]:
    """Read the usage log (optionally filtered to one 'YYYY-MM' bucket)."""
    try:
        lines = USAGE_LOG_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out: List[Dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if not isinstance(rec, dict):
            continue
        if month is not None and rec.get("month") != month:
            continue
        out.append(rec)
    return out


def month_total(month: str) -> float:
    return round(sum(r.get("cost_usd") or 0.0 for r in iter_records(month)), 8)


def _budget_alerts(prev: float, new: float, budget: float) -> List[str]:
    """Warnings for milestones crossed between prev and new monthly spend."""
    alerts: List[str] = []
    if budget <= 0 or new <= prev:
        return alerts
    step = budget / 5
    for k in range(1, 6):
        threshold = round(step * k, 8)
        if prev < threshold <= new:
            alerts.append(
                f"budget: monthly spend ${new:.2f} crossed ${threshold:.2f} "
                f"of the ${budget:.2f} allowance"
            )
    for remaining in (25.0, 10.0):
        threshold = round(budget - remaining, 8)
        if threshold > 0 and prev < threshold <= new:
            alerts.append(
                f"budget: ${remaining:.0f} or less remaining "
                f"(spent ${new:.2f} of ${budget:.2f})"
            )
    return alerts


def record(
    alias: str,
    upstream: str,
    provider: Optional[str],
    usage: Optional[Dict[str, Any]],
    stream: bool = False,
    pricing_model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Append one completed call to the usage log. Returns the record or None.

    No-op when tracking is disabled or the response carried no usage block.
    Cost is None (with a log warning) when the model has no known pricing.
    pricing_model overrides the catalog name used for the price lookup
    (some upstream deployment names differ from their pricing API names).
    """
    if not USAGE_TRACKING or not isinstance(usage, dict) or not usage:
        return None
    tokens = extract_tokens(usage)
    priced_as = pricing_model or upstream
    rates: Optional[Dict[str, float]] = None
    cost: Optional[float] = None
    if provider:
        rates = _PRICING.rates(provider, priced_as)
        if rates:
            cost = compute_cost(rates, tokens)
    if cost is None:
        log.warning(
            "usage: no pricing for %s/%s; recorded without cost", provider, priced_as
        )
    record_line = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "month": current_month(),
        "model": alias,
        "upstream": upstream,
        "provider": provider,
        "pricing_model": priced_as,
        "stream": bool(stream),
        **tokens,
        "rates": rates or None,
        "cost_usd": cost,
    }
    with _LOG_LOCK:
        month = record_line["month"]
        if _MONTH_STATE["month"] != month:
            # Cold start (or new month): derive the running total from the log.
            _MONTH_STATE["month"] = month
            _MONTH_STATE["total"] = month_total(month)
        prev = _MONTH_STATE["total"]
        _append_line(record_line)
        if cost is not None:
            _MONTH_STATE["total"] = round(prev + cost, 8)
        new = _MONTH_STATE["total"]
        for alert in _budget_alerts(prev, new, MONTHLY_BUDGET_USD):
            log.warning(alert)
    return record_line


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def monthly_report(month: Optional[str] = None) -> Dict[str, Any]:
    """Aggregated usage & spend for one calendar month."""
    month = month or current_month()
    if not _MONTH_RE.match(month):
        raise ValueError(f"invalid month {month!r}; expected YYYY-MM")
    records = iter_records(month)
    per_model: Dict[str, Dict[str, Any]] = {}
    spent = 0.0
    without_pricing = 0
    for rec in records:
        name = str(rec.get("model") or "unknown")
        row = per_model.setdefault(
            name,
            {
                "model": name,
                "upstream": rec.get("upstream"),
                "provider": rec.get("provider"),
                "requests": 0,
                "prompt_tokens": 0,
                "cached_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "requests_without_pricing": 0,
            },
        )
        row["requests"] += 1
        for field in (
            "prompt_tokens",
            "cached_tokens",
            "completion_tokens",
            "total_tokens",
        ):
            row[field] += _to_int(rec.get(field))
        cost = rec.get("cost_usd")
        if isinstance(cost, (int, float)):
            row["cost_usd"] = round(row["cost_usd"] + cost, 8)
            spent += cost
        else:
            row["requests_without_pricing"] += 1
            without_pricing += 1
    budget = MONTHLY_BUDGET_USD
    spent = round(spent, 8)
    return {
        "month": month,
        "log_path": str(USAGE_LOG_PATH),
        "budget_usd": budget,
        "spent_usd": spent,
        "remaining_usd": round(budget - spent, 8),
        "pct_used": round(spent / budget * 100, 2) if budget else None,
        "over_budget": spent > budget if budget else None,
        "requests": len(records),
        "requests_without_pricing": without_pricing,
        "models": sorted(
            per_model.values(), key=lambda r: (-r["cost_usd"], r["model"])
        ),
    }
