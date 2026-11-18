#!/usr/bin/env python3
"""Check Portkey model availability and pricing for every proxy alias.

The Portkey catalog has **no list endpoint** — every discovery path
(`/`, `/models`, `/providers`, `/pricing`, `/pricing/openai`) returns
`400 Invalid pricing config path. Expected: /model-configs/pricing/<provider>/<model>`.
So this script enumerates the aliases in `proxy.MODEL_REGISTRY` and probes each
one individually.

Usage (run from anywhere; the script locates `proxy.py` itself):

    ./.venv/bin/python .github/skills/check-portkey-models/scripts/check_models.py
    ./.venv/bin/python .../check_models.py --no-live      # pricing only, no API calls
    ./.venv/bin/python .../check_models.py --json         # machine-readable
    ./.venv/bin/python .../check_models.py --model gpt-6.1-sol --model glm-5

Exit code is 0 unless a pricing lookup failed for a network/HTTP reason, so the
report is usable in CI without failing on legitimately unpriced models.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx


def _find_repo_root() -> Path:
    """Walk up from this script until the directory containing proxy.py."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "proxy.py").exists():
            return parent
    raise SystemExit("could not locate proxy.py above this script")


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Importing proxy loads .env and builds MODEL_REGISTRY at import time.
# Never read .env directly — this is the supported way to see the live config.
import proxy  # noqa: E402

# The catalog returns per-token USD scaled x100, so $/1M = price * 1e6 / 100.
PRICE_TO_PER_MILLION = 1e6 / 100.0

PRICE_KEY_MAP = {
    "request_token": "input",
    "response_token": "output",
    "cache_read_input_token": "cache_read_input",
    "cache_write_input_token": "cache_write_input",
}

# The gateway's error for a deployment that does not exist.
UNKNOWN_MODEL_MARKER = "unknown-model"

PROBE_PROMPT = "Reply with the single word: ok"


# --------------------------------------------------------------------------- #
# Pricing
# --------------------------------------------------------------------------- #


def fetch_pricing(
    provider: str, model: str, base_url: str, timeout: float = 15.0
) -> Dict[str, Any]:
    """Look up one provider/model in the public pricing catalog.

    Returns {"status": ok|missing|empty|error, "detail": str, "rates": {...}}
    where rates are already converted to USD per 1M tokens.
    """
    url = f"{base_url}/pricing/{provider}/{model}"
    try:
        resp = httpx.get(url, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - report, never raise
        return {"status": "error", "detail": str(exc)[:160], "rates": {}}
    if resp.status_code == 404:
        return {"status": "missing", "detail": "404 not in catalog", "rates": {}}
    if resp.status_code != 200:
        return {
            "status": "error",
            "detail": f"HTTP {resp.status_code}",
            "rates": {},
        }
    try:
        payload = resp.json()
    except ValueError:
        return {"status": "error", "detail": "invalid JSON", "rates": {}}

    prices = payload.get("pay_as_you_go") or {}
    rates: Dict[str, float] = {}
    for src, dst in PRICE_KEY_MAP.items():
        cell = prices.get(src)
        price = cell.get("price") if isinstance(cell, dict) else None
        if isinstance(price, (int, float)):
            rates[dst] = round(float(price) * PRICE_TO_PER_MILLION, 6)
    if not rates:
        return {"status": "empty", "detail": "no pay_as_you_go rates", "rates": {}}
    return {"status": "ok", "detail": "", "rates": rates}


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #


def probe_live(
    alias: str,
    entry: Dict[str, Any],
    portkey_client: Any,
    nrp_client: Any,
) -> Dict[str, Any]:
    """Send one tiny chat request to the deployment the proxy would call.

    Probes `upstream` (what actually leaves the proxy), not the alias.
    Deliberately sends NO max-token / sampling params: reasoning models reject
    a tiny `max_completion_tokens` ("Could not finish the message because
    max_tokens or model output limit was reached") and gpt-* deployments reject
    sampling knobs, so omitting them is the only shape that works everywhere.
    """
    if entry.get("api_source") == "nrp":
        client = nrp_client
        if client is None:
            return {"status": "skipped", "detail": "NRP_API_KEY not set"}
    else:
        client = portkey_client
        if client is None:
            return {
                "status": "skipped",
                "detail": "no PORTKEY_API_KEY / AI_SANDBOX_KEY",
            }

    upstream = entry.get("upstream", alias)
    try:
        resp = client.chat.completions.create(
            model=upstream,
            messages=[{"role": "user", "content": PROBE_PROMPT}],
        )
    except Exception as exc:  # noqa: BLE001 - report, never raise
        msg = str(exc)
        if UNKNOWN_MODEL_MARKER in msg:
            return {"status": "unavailable", "detail": "unknown-model"}
        return {
            "status": "error",
            "detail": f"{type(exc).__name__}: {msg[:160]}",
        }
    finish = resp.choices[0].finish_reason if resp.choices else "?"
    return {"status": "available", "detail": f"finish={finish}"}


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "-"
    if value == 0:
        return "0"
    if value < 0.01:
        return f"{value:.4f}"
    return f"{value:.2f}"


def build_rows(
    aliases: List[str],
    base_url: str,
    live: bool,
    portkey_client: Any,
    nrp_client: Any,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for alias in aliases:
        entry = proxy.MODEL_REGISTRY[alias]
        provider = entry.get("provider")
        # The proxy prices `pricing_model or upstream`; mirror that exactly.
        priced_as = entry.get("pricing_model") or entry.get("upstream", alias)
        zero_cost = entry.get("track_usage", True) is False

        if zero_cost:
            pricing = {
                "status": "zero-cost",
                "detail": "track_usage=False (not billed)",
                "rates": {},
            }
        elif not provider:
            pricing = {"status": "error", "detail": "no provider set", "rates": {}}
        else:
            pricing = fetch_pricing(provider, priced_as, base_url)

        availability = (
            probe_live(alias, entry, portkey_client, nrp_client)
            if live
            else {"status": "skipped", "detail": "--no-live"}
        )

        rows.append(
            {
                "alias": alias,
                "provider": provider,
                "upstream": entry.get("upstream", alias),
                "priced_as": priced_as,
                "pricing_model_override": bool(entry.get("pricing_model")),
                "zero_cost": zero_cost,
                "pricing": pricing,
                "availability": availability,
            }
        )
    return rows


def print_markdown(rows: List[Dict[str, Any]], base_url: str) -> None:
    print(f"Pricing catalog: {base_url}")
    print(f"Aliases checked: {len(rows)}")
    print()
    header = (
        "| Alias | Provider | Priced as | In $/1M | CacheRd | CacheWr "
        "| Out $/1M | Pricing | Live |"
    )
    print(header)
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |")
    for row in rows:
        rates = row["pricing"]["rates"]
        print(
            f"| `{row['alias']}` "
            f"| {row['provider'] or '-'} "
            f"| `{row['priced_as']}` "
            f"| {_fmt(rates.get('input'))} "
            f"| {_fmt(rates.get('cache_read_input'))} "
            f"| {_fmt(rates.get('cache_write_input'))} "
            f"| {_fmt(rates.get('output'))} "
            f"| {row['pricing']['status']} "
            f"| {row['availability']['status']} |"
        )

    warnings = collect_warnings(rows)
    if warnings:
        print()
        print("## Warnings")
        for line in warnings:
            print(f"- {line}")


def collect_warnings(rows: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for row in rows:
        alias = row["alias"]
        pricing = row["pricing"]
        avail = row["availability"]

        if pricing["status"] == "missing":
            if row["zero_cost"]:
                continue  # expected: NRP models are not in the catalog
            if row["pricing_model_override"]:
                out.append(
                    f"`{alias}`: pricing_model `{row['priced_as']}` 404s — the "
                    "override itself is stale; find the current catalog name."
                )
            else:
                out.append(
                    f"`{alias}`: upstream `{row['priced_as']}` 404s in the "
                    "catalog. If the deployment is live, set "
                    f"`{alias.upper().replace('.', '').replace('-', '_')}_PRICING_MODEL` "
                    "to the catalog name (cost_usd is null until then)."
                )
        elif pricing["status"] == "error":
            out.append(f"`{alias}`: pricing lookup failed — {pricing['detail']}")
        elif pricing["status"] == "empty":
            out.append(
                f"`{alias}`: catalog entry for `{row['priced_as']}` has no "
                "pay_as_you_go rates."
            )

        if avail["status"] == "unavailable":
            out.append(
                f"`{alias}`: deployment `{row['upstream']}` is NOT available "
                "(gateway returned unknown-model)."
            )
        elif avail["status"] == "error":
            out.append(f"`{alias}`: live probe failed — {avail['detail']}")
        elif avail["status"] == "skipped" and avail["detail"] != "--no-live":
            out.append(f"`{alias}`: live probe skipped — {avail['detail']}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check Portkey model availability and pricing per alias"
    )
    parser.add_argument(
        "--no-live",
        action="store_true",
        help="skip the live availability probe (pricing only, no API calls)",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "--model",
        action="append",
        metavar="ALIAS",
        help="only check this alias (repeatable)",
    )
    parser.add_argument(
        "--base-url",
        default=getattr(proxy, "PRICING_BASE_URL", None)
        or "https://api.portkey.ai/model-configs",
        help="pricing catalog base URL",
    )
    args = parser.parse_args()

    aliases = sorted(proxy.MODEL_REGISTRY)
    if args.model:
        unknown = [m for m in args.model if m not in proxy.MODEL_REGISTRY]
        if unknown:
            print(
                f"unknown alias(es): {', '.join(unknown)}\n"
                f"available: {', '.join(aliases)}",
                file=sys.stderr,
            )
            return 2
        aliases = [m for m in aliases if m in set(args.model)]

    portkey_client = None
    nrp_client = None
    if not args.no_live:
        key = getattr(proxy, "PORTKEY_API_KEY", "") or ""
        if key:
            portkey_client = proxy.get_portkey_client(key)
        if getattr(proxy, "NRP_API_KEY", ""):
            nrp_client = proxy.get_nrp_client()

    rows = build_rows(
        aliases, args.base_url.rstrip("/"), not args.no_live, portkey_client, nrp_client
    )

    if args.json:
        print(json.dumps({"base_url": args.base_url, "models": rows}, indent=2))
    else:
        print_markdown(rows, args.base_url)

    # Non-zero only for real lookup failures, not for unpriced models.
    return 1 if any(r["pricing"]["status"] == "error" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
