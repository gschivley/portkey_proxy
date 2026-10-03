#!/usr/bin/env python3
"""
Portkey AI Sandbox -> local OpenAI-compatible HTTP proxy for GitHub Copilot.

Exposes:
    POST /v1/chat/completions     (also /chat/completions)
    POST /v1/responses            (also /responses; OpenAI Responses API)
    GET  /v1/models               (also /models)
    GET  /v1/usage                (also /usage; monthly cost/usage report)
    GET  /health, GET /

Incoming OpenAI-compatible requests are translated into calls to the
`portkey_ai` Python SDK; SDK responses (including streaming SSE) are
converted back into OpenAI JSON/SSE. Model names, parameters, errors and
usage metadata are preserved.

Both wire APIs are served and the client chooses; a chat response is never
handed back as a `response` object (the shapes are not interchangeable). The
one deliberate exception is transparent: a chat request carrying function tools
can be *served* through the Responses API and translated back into
chat-completion shape — see TOOLS_VIA_RESPONSES. That matters because the
GPT-5.6 family rejects tools + reasoning on /v1/chat/completions (adjusted via
TOOLS_REASONING_EFFORT) and the gpt-6 family rejects tools there in every
shape, while both accept them on /v1/responses.

Authentication: the API key configured in the Copilot app's model provider
settings is used. The app sends it on every request as
`Authorization: Bearer <key>`; the proxy reads it (or `x-portkey-api-key`)
and forwards it to Portkey through the SDK. PORTKEY_API_KEY / AI_SANDBOX_KEY
in the environment or .env is an optional fallback for requests that send
no key.

Configuration (local .env file next to this script, or environment):
    PORTKEY_API_KEY   optional fallback key for requests that send no API key
    AI_SANDBOX_KEY    fallback alias for PORTKEY_API_KEY
    PORTKEY_BASE_URL  upstream gateway base, default https://api.portkey.ai/v1
    PROXY_HOST        bind host, default 127.0.0.1
    PROXY_PORT        bind port, default 8090
    PORTKEY_TIMEOUT_MS   upstream timeout sent to the Portkey gateway, in
                           milliseconds (unset: gateway default applies;
                           never put seconds here — 600 means 600 ms)
    INCLUDE_STREAM_USAGE   "1" to request usage in streaming responses (default)
    RESPONSES_ENABLED      "0" to disable POST /v1/responses (default on; when
                           off the route answers the original 404 stub)
    TOOLS_REASONING_EFFORT strip|none|keep for requests with function tools
                           (default strip; only applied to models whose
                           upstream rejects tools + reasoning — the GPT-5.6
                           family — and the alternative is auto-tried)
    TOOLS_VIA_RESPONSES  auto|always|never (default auto). When a chat request
                           carries function tools, serve it through the
                           Responses API and translate the reply back into
                           chat-completion shape. "auto" only bridges models
                           whose upstream has no working chat shape for tools
                           at all (the gpt-6 family); "always" also bridges the
                           GPT-5.6 family, which keeps reasoning intact; "never"
                           forwards the upstream 400 instead. Needs
                           RESPONSES_ENABLED=1.
    TOOLS_VIA_RESPONSES  auto|always|never (default auto). Serve a chat request
                           that carries function tools through the Responses
                           API and translate the reply back into
                           chat-completion shape. "auto" only bridges models
                           whose upstream has no working chat shape for tools
                           at all (the gpt-6 family); "always" also bridges the
                           GPT-5.6 family, which keeps reasoning intact; "never"
                           forwards the upstream 400 instead. Needs
                           RESPONSES_ENABLED=1.
    GPT56_SOL_UPSTREAM_MODEL     optional upstream model for gpt-5.6-sol
    GPT56_TERRA_UPSTREAM_MODEL   optional upstream model for gpt-5.6-terra
    GPT6_ASTRA_UPSTREAM_MODEL    optional upstream model for gpt-6-astra
    GPT6_SOL_UPSTREAM_MODEL      optional upstream model for gpt-6.1-sol
    GPT6_LUNA_UPSTREAM_MODEL     optional upstream model for gpt-6-luna
    GEMINI38_FLASH_UPSTREAM_MODEL  optional upstream model for gemini-3.8-flash
    CLAUDE_OPUS_55_UPSTREAM_MODEL     optional upstream model for claude-opus-5-5
    CLAUDE_SONNET_55_UPSTREAM_MODEL   optional upstream model for claude-sonnet-5-5
    CLAUDE_FABLE_51_UPSTREAM_MODEL    optional upstream model for claude-fable-5-1
    GPT6_ASTRA_REASONING_EFFORTS   comma-separated reasoning_effort values the
                           upstream accepts for gpt-6-astra (default
                           low,medium,high,xhigh — 'none' is rejected there)
    GEMINI38_FLASH_REASONING_EFFORTS
    CLAUDE_OPUS_55_REASONING_EFFORTS
    CLAUDE_SONNET_55_REASONING_EFFORTS
    CLAUDE_FABLE_51_REASONING_EFFORTS
                           comma-separated accepted reasoning_effort values
                           for those models (default: unset, so whatever
                           the client sends is forwarded untouched)
    GEMINI38_FLASH_PRICING_MODEL   pricing-catalog model name for
                           gemini-3.8-flash (default gemini-3.8-flash-gt-128k)
    GPT6_SOL_PRICING_MODEL pricing-catalog model name for gpt-6.1-sol
                           (default gpt-6-sol — the catalog has no entry under
                           the deployment name)

Free-first routing (see routing.py; opt-in, GET /v1/router):
    ROUTER_ENABLED           "1" to accept the virtual model aliases
                             auto-free / auto-hard / auto-review and to apply
                             the paid-approval + budget gates to explicit paid
                             models too (default off: only named models work)
    ROUTER_CONFIG_PATH       optional JSON file with pools / capabilities /
                             native_only_params / max_attempts / thresholds
    ROUTER_FREE_POOL         free (NRP) aliases, in order
                             (default qwen3,deepseek-v4-flash,glm-5)
    ROUTER_PAID_POOL         paid (Portkey) aliases, in order
    ROUTER_HARD_POOL         pool for hard tasks (default: paid pool)
    ROUTER_REVIEW_POOL       pool for review tasks (default: paid pool)
    ROUTER_MAX_ATTEMPTS      upstream dispatches per request (default 4)
    ROUTER_DEADLINE_SECONDS  stop starting new attempts after this long
                             (default 120)
    PAID_APPROVAL_THRESHOLD_USD
                             recorded monthly spend at/above which every paid
                             request needs proxy_routing.paid_approved (230);
                             must be below MONTHLY_BUDGET_USD, at which point
                             routing becomes free-only

Usage & cost tracking (see usage.py; GET /v1/usage, or run usage_report.py):
    USAGE_TRACKING       "0" to disable recording usage/cost (default on)
    USAGE_LOG_PATH       JSONL usage log path (default usage.jsonl next to
                           this script; gitignored)
    MONTHLY_BUDGET_USD   monthly budget for reports/warnings (default 250)
    PRICING_BASE_URL     pricing API base (default
                           https://api.portkey.ai/model-configs)
    PRICING_CACHE_PATH   pricing cache file (default .pricing_cache.json)
    PRICING_TTL_HOURS    pricing cache TTL in hours (default 24)

Run:
    python proxy.py                      # serves on 127.0.0.1:8090
    python proxy.py --host 0.0.0.0 --port 8090
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, NamedTuple, Optional, Tuple

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from portkey_ai import Portkey
from portkey_ai._vendor.openai import OpenAI
from portkey_ai._vendor.openai.resources.chat.completions.completions import (
    Completions as _VendoredChatCompletions,
)
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

PROXY_VERSION = "1.7.0"

# --------------------------------------------------------------------------- #
# Environment / .env loading
# --------------------------------------------------------------------------- #

SCRIPT_DIR = Path(__file__).resolve().parent

try:  # python-dotenv if installed; otherwise fall back to a tiny parser
    from dotenv import load_dotenv as _dotenv_load
except ImportError:  # pragma: no cover
    _dotenv_load = None


def _load_env_files(script_dir: Path) -> None:
    for path in (script_dir / ".env", Path.cwd() / ".env"):
        if not path.is_file():
            continue
        if _dotenv_load is not None:
            _dotenv_load(path, override=False)
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_env_files(SCRIPT_DIR)

# Imported after .env loading: usage.py reads its config at import time.
import usage as usage_mod  # noqa: E402

# Also imported after .env loading: routing.py reads ROUTER_* config lazily and
# needs usage.py's MONTHLY_BUDGET_USD for the budget gate.
import routing as routing_mod  # noqa: E402

PORTKEY_API_KEY = os.environ.get("PORTKEY_API_KEY") or os.environ.get("AI_SANDBOX_KEY")
_raw_base_url = os.environ.get("PORTKEY_BASE_URL") or "https://api.portkey.ai/v1"
PORTKEY_BASE_URL = _raw_base_url.rstrip("/")
if not re.search(r"/v\d+$", PORTKEY_BASE_URL):
    PORTKEY_BASE_URL += "/v1"

# NRP gateway: a separate OpenAI-compatible upstream (https://ellm.nrp-nautilus.io)
# that serves its own models (qwen3, deepseek-v4-flash, glm-5). Unlike Portkey,
# NRP uses a fixed API key from the environment (NRP_API_KEY) that is not
# taken from the request, and it authenticates with standard
# `Authorization: Bearer <key>` — the Portkey SDK masks that header and sends
# the key as `x-portkey-api-key` instead, which the NRP gateway rejects with
# 403. NRP requests therefore go through a plain OpenAI SDK client (see
# get_nrp_client). NRP models are zero-cost, so they are never recorded in the
# usage report (see the track_usage flag on MODEL_REGISTRY entries).
NRP_BASE_URL = (os.environ.get("NRP_BASE_URL") or "https://ellm.nrp-nautilus.io/v1").rstrip("/")
if not re.search(r"/v\d+$", NRP_BASE_URL):
    NRP_BASE_URL += "/v1"
NRP_API_KEY = os.environ.get("NRP_API_KEY", "").strip()

PROXY_HOST = os.environ.get("PROXY_HOST", "127.0.0.1")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "8090"))
# Per-request upstream timeout sent to the Portkey gateway, in MILLISECONDS
# (via x-portkey-request-timeout). Unset by default so the gateway's own
# default applies. Note: the SDK treats this value as ms — a seconds-style
# value such as 600 yields a 408 "Request exceeded the timeout" after 600ms.
_timeout_raw = os.environ.get("PORTKEY_TIMEOUT_MS", "").strip()
PORTKEY_TIMEOUT_MS = int(_timeout_raw) if _timeout_raw else None
INCLUDE_STREAM_USAGE = os.environ.get("INCLUDE_STREAM_USAGE", "1").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# POST /v1/responses is backed by the SDK's `client.responses.create`, which the
# AI Sandbox does support (verified live 2026-09-29) despite the older FAQ note.
RESPONSES_ENABLED = os.environ.get("RESPONSES_ENABLED", "1").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# Some upstreams (the GPT-5.6 family) reject chat-completions requests that
# combine function tools with reasoning ("... set reasoning_effort to 'none'").
# For those models only: when function tools are present, "strip" (default)
# removes the reasoning params, "none" sends an explicit
# reasoning_effort="none", "keep" forwards unchanged. If the first variant is
# rejected with the tools+reasoning 400, the other variant is tried once
# automatically. Models that accept tools + reasoning (gpt-6-astra,
# gemini-3.8-flash) are not adjusted this way — see REASONING_PROFILES.
TOOLS_REASONING_EFFORT = (
    (os.environ.get("TOOLS_REASONING_EFFORT") or "strip").strip().lower()
)
if TOOLS_REASONING_EFFORT not in ("strip", "none", "keep"):
    logging.warning(
        "Invalid TOOLS_REASONING_EFFORT=%r; falling back to 'strip'",
        TOOLS_REASONING_EFFORT,
    )
    TOOLS_REASONING_EFFORT = "strip"
# How /v1/chat/completions requests that carry function tools are served. See
# the module docstring and _chat_to_responses_kwargs for why the bridge exists.
TOOLS_VIA_RESPONSES = (os.environ.get("TOOLS_VIA_RESPONSES") or "auto").strip().lower()
if TOOLS_VIA_RESPONSES not in ("auto", "always", "never"):
    logging.warning(
        "Invalid TOOLS_VIA_RESPONSES=%r; falling back to 'auto'", TOOLS_VIA_RESPONSES
    )
    TOOLS_VIA_RESPONSES = "auto"

# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
# "gpt-5.6-sol" / "gpt-5.6-terra" are the names the Copilot app will use.
# They are passed to Portkey as-is (they may be Portkey deployment aliases).
# To route them to a different upstream model, set e.g.
#   GPT56_SOL_UPSTREAM_MODEL=claude-sonnet-5-5
# or edit "upstream" below.
#
# Each model also declares a reasoning profile: upstreams disagree about
# 'reasoning_effort', so the proxy adapts per model instead of applying one
# workaround globally.
#   efforts        values the upstream accepts (None = unknown, forward as-is)
#   fallback       value substituted when the client sends an unsupported one
#   tools_conflict True when the upstream rejects tools + reasoning together
#   tools_on_chat  False when NO chat-completions request shape accepts
#                    function tools at all, so tools must go via /v1/responses

REASONING_PROFILES: Dict[str, Dict[str, Any]] = {
    # Rejects tools + reasoning in /v1/chat/completions (not on /v1/responses)
    # and asks for reasoning_effort="none" instead; "none" is valid here, so
    # the strip/none retry ladder has a working shape to fall back to.
    "gpt-5.6": {
        "efforts": ["none", "low", "medium", "high", "xhigh"],
        "fallback": "none",
        "tools_conflict": True,
        "tools_on_chat": True,
    },
    # No working chat shape for tools (verified live 2026-09-29): tools +
    # reasoning 400s, reasoning_effort="none" 400s because 'none' is not a
    # valid effort here, and omitting reasoning entirely still 400s since the
    # upstream applies an implicit default effort. Tools only work on
    # /v1/responses, hence tools_on_chat=False -> TOOLS_VIA_RESPONSES bridges.
    "gpt-6": {
        "efforts": ["low", "medium", "high", "xhigh"],
        "fallback": "low",
        "tools_conflict": False,
        "tools_on_chat": False,
    },
    # Gemini-style upstreams: tools + reasoning is fine and the effort
    # vocabulary is theirs to define, so values pass through untouched
    # unless an explicit list is configured via the model's efforts_env.
    "flexible": {
        "efforts": None,
        "fallback": None,
        "tools_conflict": False,
        "tools_on_chat": True,
    },
}

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "gpt-5.6-sol": {
        "upstream": os.environ.get("GPT56_SOL_UPSTREAM_MODEL", "gpt-5.6-sol"),
        "description": "Portkey/AI Sandbox model exposed to the Copilot app",
        "reasoning": "gpt-5.6",
        "provider": "openai",
    },
    "gpt-5.6-terra": {
        "upstream": os.environ.get("GPT56_TERRA_UPSTREAM_MODEL", "gpt-5.6-terra"),
        "description": "Portkey/AI Sandbox model exposed to the Copilot app",
        "reasoning": "gpt-5.6",
        "provider": "openai",
    },
    # Named gpt-6 but behaves like the GPT-5.6 family (verified live 2026-10-02):
    # it accepts reasoning_effort="none" (astra/6.1-sol reject it) and takes
    # function tools on /v1/chat/completions once reasoning is set to "none",
    # so the strip -> none retry ladder works here and no bridging is needed.
    # Its advertised effort vocabulary is exactly the gpt-5.6 profile's.
    "gpt-6-luna": {
        "upstream": os.environ.get("GPT6_LUNA_UPSTREAM_MODEL", "gpt-6-luna"),
        "description": "Portkey/AI Sandbox model exposed to the Copilot app",
        "reasoning": "gpt-5.6",
        "provider": "openai",
    },
    "gpt-6-astra": {
        "upstream": os.environ.get("GPT6_ASTRA_UPSTREAM_MODEL", "gpt-6-astra"),
        "description": "Portkey/AI Sandbox model exposed to the Copilot app",
        "reasoning": "gpt-6",
        "efforts_env": "GPT6_ASTRA_REASONING_EFFORTS",
        "provider": "openai",
    },
    "gpt-6.1-sol": {
        "upstream": os.environ.get("GPT6_SOL_UPSTREAM_MODEL", "gpt-6.1-sol"),
        "description": "Portkey/AI Sandbox model exposed to the Copilot app",
        "reasoning": "gpt-6",
        "efforts_env": "GPT6_ASTRA_REASONING_EFFORTS",
        "provider": "openai",
        # The pricing catalog lists this deployment as "gpt-6-sol"; looking it
        # up under the request name 404s and the call is logged without a cost.
        "pricing_model": os.environ.get("GPT6_SOL_PRICING_MODEL", "gpt-6-sol"),
    },
    "gemini-3.8-flash": {
        "upstream": os.environ.get("GEMINI38_FLASH_UPSTREAM_MODEL", "gemini-3.8-flash"),
        "description": "Portkey/AI Sandbox model exposed to the Copilot app",
        "reasoning": "flexible",
        "efforts_env": "GEMINI38_FLASH_REASONING_EFFORTS",
        "provider": "google",
        # The pricing catalog lists this model under a longer name.
        "pricing_model": os.environ.get(
            "GEMINI38_FLASH_PRICING_MODEL", "gemini-3.8-flash-gt-128k"
        ),
    },
    "claude-opus-5-5": {
        "upstream": os.environ.get("CLAUDE_OPUS_55_UPSTREAM_MODEL", "claude-opus-5-5"),
        "description": "Anthropic Claude Opus 5.5 via the AI Sandbox",
        "reasoning": "flexible",
        "efforts_env": "CLAUDE_OPUS_55_REASONING_EFFORTS",
        "provider": "anthropic",
    },
    "claude-sonnet-5-5": {
        "upstream": os.environ.get(
            "CLAUDE_SONNET_55_UPSTREAM_MODEL", "claude-sonnet-5-5"
        ),
        "description": "Anthropic Claude Sonnet 5.5 via the AI Sandbox",
        "reasoning": "flexible",
        "efforts_env": "CLAUDE_SONNET_55_REASONING_EFFORTS",
        "provider": "anthropic",
    },
    "claude-fable-5-1": {
        "upstream": os.environ.get(
            "CLAUDE_FABLE_51_UPSTREAM_MODEL", "claude-fable-5-1"
        ),
        "description": "Anthropic Claude Fable 5.1 via the AI Sandbox",
        "reasoning": "flexible",
        "efforts_env": "CLAUDE_FABLE_51_REASONING_EFFORTS",
        "provider": "anthropic",
    },
    # NRP gateway models. api_source routes them to NRP_BASE_URL with the
    # NRP_API_KEY (not the per-request key); track_usage=False keeps them out
    # of the usage/cost report because they are zero-cost.
    "qwen3": {
        "upstream": os.environ.get("QWEN3_UPSTREAM_MODEL", "qwen3"),
        "description": "Qwen3 via the NRP gateway (zero-cost, not usage-tracked)",
        "reasoning": "flexible",
        "api_source": "nrp",
        "responses_via_chat": True,
        "track_usage": False,
        "provider": "openai",
        # Vision verified live 2026-10-02 (Qwen3.8-Flash-Next-FP8 is a
        # multimodal "Causal Language Model with Vision Encoder"; the NRP
        # catalog lists image + video input), so no `capabilities` entry is
        # needed and image requests stay on qwen3 by default.
    },
    "deepseek-v4-flash": {
        "upstream": os.environ.get(
            "DEEPSEEK_V4_FLASH_UPSTREAM_MODEL", "deepseek-v4-flash"
        ),
        "description": "DeepSeek V4 Flash via the NRP gateway (zero-cost, not usage-tracked)",
        "reasoning": "flexible",
        "api_source": "nrp",
        "responses_via_chat": True,
        "track_usage": False,
        "provider": "openai",
    },
    "glm-5": {
        "upstream": os.environ.get("GLM5_UPSTREAM_MODEL", "glm-5"),
        "description": "GLM-5 via the NRP gateway (zero-cost, not usage-tracked)",
        "reasoning": "flexible",
        "api_source": "nrp",
        "responses_via_chat": True,
        "track_usage": False,
        "provider": "openai",
    },
}


def _parse_efforts(raw: Optional[str]) -> Optional[List[str]]:
    values = [v.strip().lower() for v in (raw or "").split(",") if v.strip()]
    return values or None


def reasoning_caps(alias: str) -> Dict[str, Any]:
    """Reasoning capabilities of a model alias, with env overrides applied."""
    entry = MODEL_REGISTRY.get(alias, {})
    profile = REASONING_PROFILES.get(
        entry.get("reasoning", "flexible"), REASONING_PROFILES["flexible"]
    )
    caps: Dict[str, Any] = {
        "efforts": list(profile["efforts"]) if profile["efforts"] else None,
        "fallback": profile["fallback"],
        "tools_conflict": bool(profile["tools_conflict"]),
        "tools_on_chat": bool(profile.get("tools_on_chat", True)),
    }
    override = _parse_efforts(os.environ.get(entry.get("efforts_env", ""), ""))
    if override:
        caps["efforts"] = override
        if caps["fallback"] not in override:
            caps["fallback"] = next((v for v in override if v != "none"), override[0])
    return caps


def _effort_supported(caps: Dict[str, Any], effort: str) -> bool:
    return caps["efforts"] is None or effort in caps["efforts"]


# --------------------------------------------------------------------------- #
# Reasoning parameter handling (per model)
# --------------------------------------------------------------------------- #
# Two independent upstream quirks are handled here:
#   1. A model may reject specific *values* of reasoning_effort (gpt-6-astra
#      rejects "none"). Those are rewritten to the model's fallback before the
#      request goes out — see _sanitize_reasoning.
#   2. A model may reject reasoning *entirely* when function tools are present
#      (the GPT-5.6 family, on /v1/chat/completions only). Those requests are
#      adjusted per TOOLS_REASONING_EFFORT and retried once with the
#      alternative variant — see _tools_reasoning_plan. This conflict does not
#      exist on /v1/responses, where the same models accept tools + reasoning,
#      so the responses path never applies _tools_reasoning_plan.


def _sanitize_reasoning(
    kwargs: Dict[str, Any], caps: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Replace reasoning effort values the model rejects with its fallback.

    Returns (adjusted kwargs, note) where note describes the rewrite and is
    None when nothing changed. Models with an unknown vocabulary (efforts=None,
    e.g. gemini) are forwarded untouched.
    """
    efforts = caps["efforts"]
    if not efforts:
        return kwargs, None
    fallback = caps["fallback"] or efforts[0]
    notes: List[str] = []

    def fix(value: Any) -> Any:
        if isinstance(value, str) and value.strip().lower() not in efforts:
            notes.append(f"{value.strip().lower()!r} -> {fallback!r}")
            return fallback
        return value

    adjusted = dict(kwargs)
    if "reasoning_effort" in adjusted:
        adjusted["reasoning_effort"] = fix(adjusted["reasoning_effort"])
    reasoning = adjusted.get("reasoning")
    if isinstance(reasoning, dict) and "effort" in reasoning:
        repaired = dict(reasoning)
        repaired["effort"] = fix(repaired["effort"])
        adjusted["reasoning"] = repaired
    if not notes:
        return kwargs, None
    return adjusted, "; ".join(notes)


def _apply_tools_reasoning_mode(kwargs: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Copy of kwargs with reasoning params adjusted per mode.

    'strip' removes reasoning entirely; 'none' sets an explicit
    reasoning_effort="none" (per the upstream's own guidance).
    """
    adjusted = dict(kwargs)
    if mode == "strip":
        adjusted.pop("reasoning_effort", None)
        reasoning = adjusted.get("reasoning")
        if isinstance(reasoning, dict):
            without_effort = {k: v for k, v in reasoning.items() if k != "effort"}
            if without_effort:
                adjusted["reasoning"] = without_effort
            else:
                adjusted.pop("reasoning", None)
    elif mode == "none":
        adjusted["reasoning_effort"] = "none"
        if isinstance(adjusted.get("reasoning"), dict):
            adjusted.pop("reasoning", None)  # avoid two conflicting shapes
    return adjusted


def _tools_reasoning_plan(
    kwargs: Dict[str, Any], caps: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Ordered request variants to try for a request with function tools."""
    if not kwargs.get("tools"):
        return [kwargs]
    if not caps["tools_conflict"]:
        # This model accepts tools + reasoning (gpt-6-astra, gemini): send the
        # request as the client wrote it, keeping a defensive strip-retry in
        # case the upstream still reports a tools+reasoning conflict.
        variants = [kwargs, _apply_tools_reasoning_mode(kwargs, "strip")]
    elif TOOLS_REASONING_EFFORT == "keep":
        variants = [kwargs]
    else:
        mode = TOOLS_REASONING_EFFORT
        if mode == "none" and not _effort_supported(caps, "none"):
            mode = "strip"  # this model has no "none" to offer
        variants = [_apply_tools_reasoning_mode(kwargs, mode)]
        alternate = "none" if mode == "strip" else "strip"
        if alternate == "strip" or _effort_supported(caps, "none"):
            variants.append(_apply_tools_reasoning_mode(kwargs, alternate))
    deduped: List[Dict[str, Any]] = []
    for variant in variants:
        if variant not in deduped:
            deduped.append(variant)
    return deduped


def _is_tools_reasoning_conflict(status: int, message: str) -> bool:
    """True for the upstream's tools+reasoning 400."""
    text = (message or "").lower()
    return status == 400 and "reasoning" in text and "tool" in text


def _reasoning_desc(kwargs: Dict[str, Any]) -> str:
    if "reasoning_effort" in kwargs:
        return f"reasoning_effort={kwargs['reasoning_effort']!r}"
    reasoning = kwargs.get("reasoning")
    if isinstance(reasoning, dict):
        if "effort" in reasoning:
            return f"reasoning.effort={reasoning['effort']!r}"
        return "reasoning(no effort)"
    return "no reasoning params"


# --------------------------------------------------------------------------- #
# Errors (OpenAI-style)
# --------------------------------------------------------------------------- #


class ApiError(Exception):
    """Raise to return an OpenAI-shaped error response."""

    def __init__(
        self,
        status_code: int,
        message: str,
        err_type: str = "invalid_request_error",
        code: Optional[str] = None,
        param: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.err_type = err_type
        self.code = code
        self.param = param


def _exception_details(
    exc: Exception,
) -> Tuple[int, str, str, Optional[str], Optional[str]]:
    """Map SDK/HTTP exceptions onto (status, message, type, code, param)."""
    name = type(exc).__name__
    if name == "_AttemptFailure":
        return exc.status_code, str(exc), exc.err_type, exc.code, exc.param
    status_map = {
        "AuthenticationError": (401, "authentication_error"),
        "PermissionDeniedError": (403, "permission_error"),
        "RateLimitError": (429, "rate_limit_error"),
        "BadRequestError": (400, "invalid_request_error"),
        "UnprocessableEntityError": (422, "invalid_request_error"),
        "NotFoundError": (404, "invalid_request_error"),
        "ConflictError": (409, "invalid_request_error"),
        "InternalServerError": (500, "server_error"),
        "APIConnectionError": (503, "api_error"),
        "APITimeoutError": (504, "api_error"),
        "APIResponseValidationError": (502, "api_error"),
        "APIError": (502, "api_error"),
    }
    if name in status_map:
        status, err_type = status_map[name]
    else:
        status = getattr(exc, "status_code", 500) or 500
        err_type = "api_error"

    message = getattr(exc, "message", None) or str(exc) or "Request failed."

    # Prefer structured error payload from the upstream body, if available.
    body = getattr(exc, "body", None)
    error_obj: Any = {}
    if isinstance(body, dict):
        candidate = body.get("error")
        error_obj = candidate if isinstance(candidate, dict) else body
    elif body is not None and hasattr(body, "get"):
        candidate = body.get("error")
        error_obj = candidate if isinstance(candidate, dict) else {}

    message = error_obj.get("message") or message
    code = error_obj.get("code")
    param = error_obj.get("param")
    err_type = error_obj.get("type") or err_type
    return status, message, err_type, code, param


def _error_payload(
    message: str, err_type: str, code: Optional[str], param: Optional[str]
) -> Dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": err_type,
            "param": param,
            "code": code,
        }
    }


# --------------------------------------------------------------------------- #
# Responses API request handling
# --------------------------------------------------------------------------- #
# /v1/responses is a different wire format from /v1/chat/completions, so it has
# its own small translation layer. Deliberately *not* shared with the chat path:
#   - reasoning is nested only (`reasoning: {"effort": ...}`); a flat
#     `reasoning_effort` is a chat-completions parameter.
#   - function tools are flat (`{"type": "function", "name", "parameters"}`),
#     not nested under a `function` key.
#   - the tools+reasoning conflict does not exist here, so _tools_reasoning_plan
#     is never applied and reasoning is preserved for the GPT-5.6 family.
# The SDK's responses.create() takes a keyword whitelist and folds anything else
# into `extra_body`, so unknown client parameters still reach the gateway.

_CHAT_ONLY_MESSAGE_KEYS = ("tool_calls", "function_call", "tool_call_id")
_EASY_INPUT_ROLES = ("user", "assistant", "system", "developer")


def _messages_are_easy_input(messages: Any) -> bool:
    """True when a chat `messages` array maps 1:1 onto Responses `input`.

    Only plain role+content messages do: tool results, tool calls and function
    calls have different item shapes on the Responses API and must not be
    renamed into `input` as if they were messages.
    """
    if not isinstance(messages, list) or not messages:
        return False
    for message in messages:
        if not isinstance(message, dict):
            return False
        if any(key in message for key in _CHAT_ONLY_MESSAGE_KEYS):
            return False
        if message.get("role") not in _EASY_INPUT_ROLES:
            return False
    return True


def _responses_kwargs(
    body: Dict[str, Any], caps: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Translate a /v1/responses request body into SDK kwargs.

    Returns (kwargs, note); `note` describes any chat-shaped input that had to
    be converted and is None when the body was already Responses-native.
    """
    kwargs = {k: v for k, v in body.items() if k not in ("model", "stream")}
    notes: List[str] = []

    # `stream_options` is a chat-completions parameter used to opt into usage in
    # the stream. Responses reports usage on `response.completed` regardless,
    # and the upstream rejects the parameter outright, so drop it rather than
    # fail a request that is otherwise valid.
    if kwargs.pop("stream_options", None) is not None:
        notes.append("dropped stream_options (Responses always reports usage)")

    # A flat reasoning_effort belongs to chat completions; the Responses API
    # reads reasoning out of the nested object. Normalise so mixed clients get
    # their effort honoured instead of a confusing upstream 400.
    effort = kwargs.pop("reasoning_effort", None)
    if effort is not None:
        reasoning = kwargs.get("reasoning")
        if isinstance(reasoning, dict) and "effort" in reasoning:
            notes.append(
                f"dropped reasoning_effort={effort!r} in favour of "
                f"reasoning.effort={reasoning['effort']!r}"
            )
        else:
            merged = dict(reasoning) if isinstance(reasoning, dict) else {}
            merged["effort"] = effort
            kwargs["reasoning"] = merged
            notes.append(f"reasoning_effort={effort!r} -> reasoning.effort")

    # Chat-shaped `messages` become Responses `input` when they are plain
    # role+content items (all four roles are valid there, so it is a rename).
    if "messages" in kwargs:
        messages = kwargs.pop("messages")
        if "input" in kwargs:
            # `messages` is not a Responses parameter at all; forwarding it in
            # the body would only confuse the gateway. Prefer the native key.
            notes.append("dropped 'messages' in favour of 'input'")
        elif _messages_are_easy_input(messages):
            kwargs["input"] = messages
            notes.append("messages -> input")
        else:
            raise ApiError(
                400,
                "This /v1/responses request carries a chat-style 'messages' "
                "array that cannot be mapped onto Responses input items "
                "(tool calls, tool results and function calls have their own "
                "item shapes). Send 'input' in the Responses format, or use "
                "POST /v1/chat/completions.",
                param="messages",
                code="unsupported_input_shape",
            )

    kwargs, rewrite = _sanitize_reasoning(kwargs, caps)
    if rewrite:
        notes.append(f"reasoning effort {rewrite}")
    return kwargs, "; ".join(notes) or None


# --------------------------------------------------------------------------- #
# Bridge: serve chat-completions tool calls through the Responses API
# --------------------------------------------------------------------------- #
# Some upstreams (the gpt-6 family) have no chat-completions request shape that
# accepts function tools at all — see REASONING_PROFILES["gpt-6"]. The Sandbox
# error message offers two escape hatches that contradict each other ("use
# /v1/responses or set reasoning_effort to 'none'", while 'none' is rejected for
# those models), and omitting reasoning entirely still 400s because the upstream
# applies an implicit default effort. /v1/responses does accept tools +
# reasoning, so a chat client can still use those models if the proxy translates
# the request over and the answer back.
#
# The client keeps speaking chat completions: same route, same request body,
# same response shape (tool calls still nested under `function`), same SSE chunk
# format and `data: [DONE]`. Only the upstream call changes.
# Controlled by TOOLS_VIA_RESPONSES (auto | always | never).


def _chat_content_to_input(content: Any) -> Any:
    """Map chat message content onto Responses input content parts."""
    if isinstance(content, str) or content is None:
        return content if content is not None else ""
    if not isinstance(content, list):
        return content
    parts: List[Dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            parts.append({"type": "input_text", "text": str(part)})
            continue
        kind = part.get("type")
        if kind == "text":
            parts.append({"type": "input_text", "text": part.get("text", "")})
        elif kind in ("input_text", "input_image"):
            parts.append(dict(part))  # already Responses-shaped
        elif kind == "image_url":
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            mapped: Dict[str, Any] = {"type": "input_image"}
            if url:
                mapped["image_url"] = url
            if isinstance(image, dict) and image.get("detail"):
                mapped["detail"] = image["detail"]
            parts.append(mapped)
        else:
            parts.append(dict(part))
    return parts


_CHAT_INPUT_ROLES = ("user", "assistant", "system", "developer")


def _chat_messages_to_input(messages: List[Any]) -> List[Dict[str, Any]]:
    """Convert a chat `messages` array into Responses input items.

    Tool calls and tool results are why a plain rename is not enough: an
    assistant `tool_calls` message becomes one `function_call` item per call and
    a `tool` message becomes a `function_call_output` item. Without that the
    model would never see the tool output it is being asked about.
    """
    items: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role in ("tool", "function"):
            output = message.get("content")
            item: Dict[str, Any] = {
                "type": "function_call_output",
                "call_id": message.get("tool_call_id")
                or message.get("call_id")
                or "call_unknown",
                "output": output if isinstance(output, str) else json.dumps(output),
            }
            if message.get("name"):
                item["name"] = message["name"]
            items.append(item)
            continue
        tool_calls = message.get("tool_calls") or message.get("function_call")
        if tool_calls:
            content = message.get("content")
            if content:
                items.append(
                    {
                        "role": "assistant",
                        "type": "message",
                        "content": _chat_content_to_input(content),
                    }
                )
            calls = tool_calls if isinstance(tool_calls, list) else [tool_calls]
            for call in calls:
                if not isinstance(call, dict):
                    continue
                fn = (
                    call["function"] if isinstance(call.get("function"), dict) else call
                )
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call.get("id") or fn.get("id") or "call_unknown",
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments") or "{}",
                    }
                )
            continue
        if role in _CHAT_INPUT_ROLES:
            entry: Dict[str, Any] = {
                "role": role,
                "type": "message",
                "content": _chat_content_to_input(message.get("content")),
            }
            if message.get("phase") in ("commentary", "final_answer"):
                entry["phase"] = message["phase"]
            items.append(entry)
    return items


def _chat_tools_to_responses(tools: Any) -> Any:
    """Un-nest chat function tools (`{"function": {...}}`) for Responses."""
    if not isinstance(tools, list):
        return tools
    flat: List[Dict[str, Any]] = []
    for tool in tools:
        if (
            isinstance(tool, dict)
            and tool.get("type") == "function"
            and isinstance(tool.get("function"), dict)
        ):
            fn = tool["function"]
            converted: Dict[str, Any] = {"type": "function"}
            for key in ("name", "description", "parameters", "strict"):
                if key in fn:
                    converted[key] = fn[key]
            flat.append(converted)
        else:
            flat.append(tool if not isinstance(tool, dict) else dict(tool))
    return flat


def _chat_tool_choice_to_responses(choice: Any) -> Any:
    """Map chat `tool_choice` onto the Responses shape."""
    if isinstance(choice, str) or choice is None:
        return choice
    if isinstance(choice, dict) and choice.get("type") == "function":
        fn = choice["function"] if isinstance(choice.get("function"), dict) else {}
        name = fn.get("name") or choice.get("name")
        if name:
            return {"type": "function", "name": name}
    return choice


# Chat parameters with no Responses equivalent: dropped, and logged as such.
# `stream_options` matters most -- chat clients (and INCLUDE_STREAM_USAGE) send
# it to request usage in the stream, but the Responses API reports usage on the
# `response.completed` event unconditionally and the upstream rejects the
# parameter outright ("Unknown parameter: 'stream_options.include_usage'").
_BRIDGE_DROPPED_PARAMS = (
    "stream_options",
    "frequency_penalty",
    "presence_penalty",
    "logprobs",
    "top_logprobs",
    "n",
    "seed",
    "logit_bias",
    "modalities",
    "audio",
    "prediction",
    "web_search_options",
)


def _chat_to_responses_kwargs(
    body: Dict[str, Any], caps: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Translate a /v1/chat/completions body into Responses kwargs."""
    kwargs: Dict[str, Any] = {}
    notes: List[str] = []
    for key, value in body.items():
        if key in ("model", "stream"):
            continue
        if key == "messages":
            kwargs["input"] = _chat_messages_to_input(
                value if isinstance(value, list) else []
            )
        elif key == "tools":
            kwargs["tools"] = _chat_tools_to_responses(value)
        elif key == "tool_choice":
            kwargs["tool_choice"] = _chat_tool_choice_to_responses(value)
        elif key in ("max_tokens", "max_completion_tokens"):
            kwargs["max_output_tokens"] = value
        elif key == "reasoning_effort":
            reasoning = dict(kwargs.get("reasoning") or {})
            reasoning["effort"] = value
            kwargs["reasoning"] = reasoning
        elif key == "response_format" and isinstance(value, dict):
            # Chat json_* modes map onto Responses text.format.
            if value.get("type") in ("json_object", "json_schema"):
                text = dict(kwargs.get("text") or {})
                text["format"] = dict(value)
                kwargs["text"] = text
            else:
                notes.append("dropped response_format")
        elif key in _BRIDGE_DROPPED_PARAMS:
            # Includes stream_options: chat clients send it to opt into usage,
            # but Responses reports usage on response.completed anyway.
            notes.append(f"dropped {key}")
        else:
            kwargs[key] = value  # temperature, top_p, stop, user, metadata, ...
    kwargs, rewrite = _sanitize_reasoning(kwargs, caps)
    if rewrite:
        notes.append(f"reasoning effort {rewrite}")
    return kwargs, "; ".join(notes) or None


def _responses_output_text(data: Dict[str, Any]) -> str:
    """Concatenate every output_text part of a Response's message items."""
    chunks: List[str] = []
    for item in data.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
    return "".join(chunks)


def _responses_to_chat_usage(usage: Any) -> Optional[Dict[str, Any]]:
    """Re-express Responses usage with chat-completions token names."""
    if not isinstance(usage, dict) or not usage:
        return None
    out: Dict[str, Any] = {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }
    for src, dst in (
        ("input_tokens_details", "prompt_tokens_details"),
        ("output_tokens_details", "completion_tokens_details"),
    ):
        if isinstance(usage.get(src), dict):
            out[dst] = dict(usage[src])
    return out


def _bridge_finish_reason(
    data: Dict[str, Any], tool_calls: List[Dict[str, Any]]
) -> str:
    status = data.get("status")
    if status == "incomplete":
        reason = (data.get("incomplete_details") or {}).get("reason")
        return "length" if reason == "max_output_tokens" else "content_filter"
    if status in ("failed", "cancelled"):
        return "stop"
    return "tool_calls" if tool_calls else "stop"


def _bridge_chat_id(response_id: Any) -> str:
    """Render a Responses id (`resp_...` / `resp-...`) as a chat completion id."""
    text = str(response_id or "")
    if text.startswith("resp"):
        rest = text[4:]
        rest = rest[1:] if rest[:1] in ("_", "-") else rest
        return f"chatcmpl-{rest}"
    return text or "chatcmpl-0"


def _responses_to_chat_completion(data: Dict[str, Any], alias: str) -> Dict[str, Any]:
    """Convert a Responses body into an OpenAI chat.completion."""
    tool_calls: List[Dict[str, Any]] = []
    for item in data.get("output") or []:
        if isinstance(item, dict) and item.get("type") == "function_call":
            tool_calls.append(
                {
                    "id": item.get("call_id")
                    or item.get("id")
                    or f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments") or "",
                    },
                }
            )
    text = _responses_output_text(data)
    message: Dict[str, Any] = {
        "role": "assistant",
        # Chat convention: content is null on a tool-only turn.
        "content": text if text else (None if tool_calls else ""),
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    created = data.get("created_at")
    return {
        "id": _bridge_chat_id(data.get("id")),
        "object": "chat.completion",
        "created": (
            int(created) if isinstance(created, (int, float)) else int(time.time())
        ),
        "model": alias,
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": _bridge_finish_reason(data, tool_calls),
            }
        ],
        "usage": _responses_to_chat_usage(data.get("usage")),
    }


class _ResponsesToChatSSE:
    """Stateful translator: Responses SSE events -> chat.completion chunks.

    Reproduces the frame sequence a chat stream sends — a role delta first,
    then content and/or `tool_calls` deltas, a finish_reason frame, an optional
    usage frame, and `data: [DONE]` — so an unmodified chat client can drive a
    bridged request.
    """

    def __init__(self, alias: str) -> None:
        self.alias = alias
        self.started = False
        self.response_id = "chatcmpl-0"
        self.created = int(time.time())
        self.tool_indexes: Dict[Any, int] = {}
        self.next_tool_index = 0
        self.finish_reason: Optional[str] = None
        self.usage: Optional[Dict[str, Any]] = None

    def _chunk(self, delta: Dict[str, Any], finish_reason: Optional[str] = None) -> str:
        self.started = True
        return sse_frame(
            {
                **self._base(),
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "logprobs": None,
                        "finish_reason": finish_reason,
                    }
                ],
            }
        )

    def _base(self) -> Dict[str, Any]:
        return {
            "id": self.response_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.alias,
        }

    def _start(self) -> List[str]:
        """The opening role delta, emitted once and only if nothing came first."""
        return (
            [] if self.started else [self._chunk({"role": "assistant", "content": ""})]
        )

    def _tool_index(self, key: Any) -> int:
        if key is None:
            return self.next_tool_index
        if key not in self.tool_indexes:
            self.tool_indexes[key] = self.next_tool_index
            self.next_tool_index += 1
        return self.tool_indexes[key]

    def consume(self, event: Dict[str, Any]) -> List[str]:
        """Translate one Responses event into zero or more chat chunks."""
        kind = event.get("type")
        if kind in ("response.created", "response.in_progress", "response.queued"):
            self._capture_ids(event.get("response"))
            return []
        if kind == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") != "function_call":
                return []
            index = self._tool_index(item.get("call_id") or event.get("output_index"))
            self.finish_reason = "tool_calls"
            return self._start() + [
                self._chunk(
                    {
                        "tool_calls": [
                            {
                                "index": index,
                                "id": item.get("call_id") or f"call_{index}",
                                "type": "function",
                                "function": {
                                    "name": item.get("name", ""),
                                    "arguments": "",
                                },
                            }
                        ]
                    }
                )
            ]
        if kind == "response.output_text.delta":
            delta = event.get("delta")
            if not isinstance(delta, str) or not delta:
                return []
            return self._start() + [self._chunk({"content": delta})]
        if kind == "response.function_call_arguments.delta":
            delta = event.get("delta")
            if not isinstance(delta, str) or not delta:
                return []
            index = self._tool_index(event.get("item_id") or event.get("output_index"))
            self.finish_reason = "tool_calls"
            return self._start() + [
                self._chunk(
                    {"tool_calls": [{"index": index, "function": {"arguments": delta}}]}
                )
            ]
        if kind in ("response.completed", "response.incomplete", "response.failed"):
            self._finish_from(event)
            return []
        return []

    def _capture_ids(self, response: Any) -> None:
        if not isinstance(response, dict):
            return
        if response.get("id"):
            self.response_id = _bridge_chat_id(response.get("id"))
        created = response.get("created_at")
        if isinstance(created, (int, float)):
            self.created = int(created)

    def _finish_from(self, event: Dict[str, Any]) -> None:
        response = event.get("response")
        if not isinstance(response, dict):
            return
        self._capture_ids(response)
        self.usage = _responses_to_chat_usage(response.get("usage"))
        tool_calls = [
            item
            for item in response.get("output") or []
            if isinstance(item, dict) and item.get("type") == "function_call"
        ]
        if event.get("type") == "response.completed":
            self.finish_reason = "tool_calls" if tool_calls else "stop"
        else:
            self.finish_reason = _bridge_finish_reason(response, tool_calls)

    def finish(self) -> List[str]:
        """Closing frames: finish_reason, optional usage, and [DONE]."""
        out = self._start()
        out.append(self._chunk({}, self.finish_reason or "stop"))
        if self.usage:
            out.append(sse_frame({**self._base(), "choices": [], "usage": self.usage}))
        out.append("data: [DONE]\n\n")
        return out


# --------------------------------------------------------------------------- #
# Reverse bridge: serve /v1/responses traffic through chat completions
# --------------------------------------------------------------------------- #
# The NRP gateway (ellm.nrp-nautilus.io) exposes /v1/responses but does not
# implement it: the Responses body fails its chat-shaped validation with a
# 400 ("chat-completions request must have at least one message"). Copilot
# speaks the Responses API, so for entries with `responses_via_chat` the
# proxy converts the request into chat-completions kwargs, calls the chat
# endpoint, and translates the reply (JSON or SSE) back into Responses
# shapes. Clients keep talking Responses and never learn the difference.
# --------------------------------------------------------------------------- #


# Responses parameters with no chat-completions equivalent: dropped (logged).
_NRP_DROPPED_PARAMS = (
    "store",
    "background",
    "include",
    "prompt",
    "prompt_cache_key",
    "safety_identifier",
    "service_tier",
    "truncation",
    "max_tool_output_tokens",
    "top_logprobs",
    "previous_response_id",
)


def _responses_content_to_chat_parts(content: Any) -> List[Dict[str, Any]]:
    """Map Responses message content parts onto their chat equivalents."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    parts: List[Dict[str, Any]] = []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        text = part.get("text") or part.get("refusal") or ""
        if ptype in ("input_text", "output_text", "text"):
            parts.append({"type": "text", "text": text})
        elif ptype == "refusal":
            parts.append({"type": "refusal", "refusal": part.get("refusal") or ""})
        elif ptype in ("input_image", "image_url"):
            image = (
                part.get("image_url")
                if isinstance(part.get("image_url"), dict)
                else {"url": part.get("image_url") or part.get("image") or ""}
            )
            parts.append({"type": "image_url", "image_url": image})
        elif ptype in ("input_audio", "audio"):
            parts.append({"type": "input_audio", "input_audio": part.get("input_audio") or part.get("audio") or {}})
        elif ptype in ("file", "input_file"):
            parts.append({"type": "image_url", "image_url": part.get("file") or {}})
    return parts


def _responses_input_to_messages(
    inp: Any, notes: List[str]
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Convert Responses `input` into chat `messages`.

    Returns (messages, tool_call_id -> function name), the map used to
    re-attach `name` to tool results when converting back.
    """
    if inp is None:
        return [], {}
    if isinstance(inp, str):
        return [{"role": "user", "content": inp}], {}
    if not isinstance(inp, list):
        raise ApiError(
            400,
            "The NRP bridge expects 'input' to be a string or an array of "
            "Responses items.",
            param="input",
            code="unsupported_input_shape",
        )
    messages: List[Dict[str, Any]] = []
    call_names: Dict[str, str] = {}
    for item in inp:
        if not isinstance(item, dict):
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
            continue
        itype = item.get("type")
        role = item.get("role") or "user"
        if itype in (None, "message"):
            messages.append(
                {
                    "role": role,
                    "content": _responses_content_to_chat_parts(
                        item.get("content")
                    ),
                }
            )
        elif itype == "function_call":
            call_id = item.get("call_id") or item.get("id") or f"call_{len(call_names)}"
            call_names[call_id] = item.get("name", "")
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": item.get("name", ""),
                                "arguments": item.get("arguments") or "{}",
                            },
                        }
                    ],
                }
            )
        elif itype == "function_call_output":
            call_id = item.get("call_id") or "call_unknown"
            out = item.get("output")
            msg: Dict[str, Any] = {
                "role": "tool",
                "tool_call_id": call_id,
                "content": out if isinstance(out, str) else json.dumps(out),
            }
            if item.get("name") or call_names.get(call_id):
                msg["name"] = item.get("name") or call_names[call_id]
            messages.append(msg)
        elif itype == "item_reference":
            notes.append("dropped item_reference input")
        elif itype in ("reasoning", "local_shell_call", "web_search_call", "compaction"):
            notes.append(f"dropped '{itype}' input item")
        else:
            notes.append(f"dropped unknown input item '{itype}'")
    return messages, call_names


def _responses_tool_to_chat(tool: Any) -> Any:
    """Re-nest a flat Responses function tool into the chat shape."""
    if isinstance(tool, dict) and tool.get("type") == "function" and "name" in tool:
        inner = {k: v for k, v in tool.items() if k != "type"}
        return {"type": "function", "function": inner}
    return tool


def _responses_to_chat_kwargs(
    body: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, str], Optional[str]]:
    """Translate a /v1/responses request body into chat-completions kwargs.

    Returns (kwargs, tool_call_names, note). `tool_call_names` maps call ids
    to function names for converting assistant function_call inputs.
    """
    kwargs: Dict[str, Any] = {}
    notes: List[str] = []
    call_names: Dict[str, str] = {}

    inp = body.get("input")
    messages, call_names = _responses_input_to_messages(inp, notes)
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions:
        messages.insert(0, {"role": "system", "content": instructions})
    if messages:
        kwargs["messages"] = messages

    for key, value in body.items():
        if key in ("model", "stream", "input", "instructions"):
            continue
        if key in _NRP_DROPPED_PARAMS:
            notes.append(f"dropped {key}")
        elif key == "tools":
            kwargs["tools"] = [_responses_tool_to_chat(t) for t in value] if isinstance(value, list) else value
        elif key == "tool_choice":
            if isinstance(value, dict) and value.get("type") == "function":
                kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": value.get("name", "")},
                }
            elif isinstance(value, dict) and value.get("type") in (
                "shell",
                "custom",
                "file_search",
                "web_search_preview",
            ):
                raise ApiError(
                    400,
                    f"The NRP bridge cannot map tool_choice {value.get('type')!r} "
                    "onto chat completions.",
                    param="tool_choice",
                    code="unsupported_tool_choice",
                )
            else:
                kwargs["tool_choice"] = value
        elif key == "max_output_tokens":
            kwargs["max_completion_tokens"] = value
        elif key == "reasoning":
            effort = value.get("effort") if isinstance(value, dict) else None
            if effort is not None and effort != "none":
                kwargs["reasoning_effort"] = effort
            elif effort == "none":
                notes.append("dropped reasoning.effort=none")
        elif key == "text":
            fmt = value.get("format") if isinstance(value, dict) else None
            if isinstance(fmt, dict) and fmt.get("type") in ("json_object", "json_schema"):
                kwargs["response_format"] = dict(fmt)
            else:
                notes.append("dropped text")
        elif key == "metadata" and isinstance(value, dict):
            # Chat only accepts string-valued metadata; keep it if it fits.
            if all(isinstance(v, str) for v in value.values()):
                kwargs["metadata"] = value
            else:
                notes.append("dropped metadata (non-string values)")
        else:
            kwargs[key] = value  # temperature, top_p, user, stop, ...
    if not kwargs.get("messages"):
        raise ApiError(
            400,
            "The NRP bridge needs 'input' (or 'instructions') to build chat "
            "messages; the request carried neither.",
            param="input",
        )
    return kwargs, call_names, "; ".join(notes) or None


def _chat_message_to_output_items(
    message: Dict[str, Any], item_id_prefix: str = "msg"
) -> List[Dict[str, Any]]:
    """Split a chat assistant message into Responses output items."""
    items: List[Dict[str, Any]] = []
    text = message.get("content")
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning and not text and not message.get("tool_calls"):
        # Reasoning-only turns carry no user-visible output in Responses.
        text = None
    if text:
        items.append(
            {
                "id": f"{item_id_prefix}_0",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "logprobs": []}],
            }
        )
    for i, call in enumerate(message.get("tool_calls") or []):
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        call_id = call.get("id") or f"call_{i}"
        items.append(
            {
                "id": f"fc_{call_id}",
                "type": "function_call",
                "call_id": call_id,
                "name": fn.get("name", ""),
                "arguments": fn.get("arguments") or "",
                "status": "completed",
            }
        )
    return items


def _chat_usage_to_responses(usage: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(usage, dict) or not usage:
        return None
    out = {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }
    for src, dst in (
        ("prompt_tokens_details", "input_tokens_details"),
        ("completion_tokens_details", "output_tokens_details"),
    ):
        if isinstance(usage.get(src), dict):
            out[dst] = dict(usage[src])
    return out


def _chat_to_responses_completion(data: Dict[str, Any], alias: str) -> Dict[str, Any]:
    """Convert a chat.completion body into a Responses `response` object."""
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    output = _chat_message_to_output_items(message)
    finish = choice.get("finish_reason")
    status = "incomplete" if finish == "length" else "completed"
    created = data.get("created")
    return {
        "id": data.get("id") or f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": int(created) if isinstance(created, (int, float)) else int(time.time()),
        "status": status,
        "model": alias,
        "output": output,
        "output_text": _chat_message_text(message),
        "error": None,
        "incomplete_details": (
            {"reason": "max_output_tokens"} if status == "incomplete" else None
        ),
        "usage": _chat_usage_to_responses(data.get("usage")),
    }


def _chat_message_text(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    return ""


class _ChatToResponsesSSE:
    """Stateful translator: chat.completion SSE chunks -> Responses events.

    Reproduces the event sequence a Responses stream sends: response.created,
    response.in_progress, message/function-call items with text and argument
    deltas, their done events, and a final response.completed carrying usage.
    No `data: [DONE]` sentinel (Responses streams do not use one).
    """

    def __init__(self, alias: str, upstream: str) -> None:
        self.alias = alias
        self.upstream = upstream
        self.response_id = f"resp_{uuid.uuid4().hex}"
        self.created = int(time.time())
        self.seq = 0
        self.msg_item_id = "msg_0"
        self.text_open = False
        self.text_index = 0
        self.text_done = False
        self.tools: Dict[int, Dict[str, Any]] = {}
        self.tool_order: List[int] = []
        self.finish_reason: Optional[str] = None
        self.usage: Optional[Dict[str, Any]] = None
        self.emitted_text: List[str] = []

    def start_events(self) -> List[str]:
        """response.created + response.in_progress, emitted up front."""
        response = self._skeleton("in_progress", output=[])
        return [
            self._event({"type": "response.created", "response": response}),
            self._event({"type": "response.in_progress", "response": response}),
        ]

    def _skeleton(
        self, status: str, output: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        return {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created,
            "status": status,
            "error": None,
            "incomplete_details": None,
            "model": self.alias,
            "output": output,
            "output_text": "".join(self.emitted_text),
            "usage": self.usage,
        }

    def _event(self, payload: Dict[str, Any]) -> str:
        payload.setdefault("sequence_number", self.seq)
        self.seq += 1
        return sse_event_frame(payload)

    def _open_text(self) -> List[str]:
        if self.text_open:
            return []
        self.text_open = True
        return [
            self._event(
                {
                    "type": "response.output_item.added",
                    "output_index": self.text_index,
                    "item": {
                        "id": self.msg_item_id,
                        "type": "message",
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [],
                    },
                }
            ),
            self._event(
                {
                    "type": "response.content_part.added",
                    "item_id": self.msg_item_id,
                    "output_index": self.text_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "logprobs": []},
                }
            ),
        ]

    def consume(self, chunk: Dict[str, Any]) -> List[str]:
        """Translate one chat.completion.chunk dict into Responses events."""
        out: List[str] = []
        if chunk.get("id") and chunk["id"] != self.response_id and self.seq <= 2:
            # Keep the client-visible id aligned with the upstream stream.
            self.response_id = chunk["id"]
        if isinstance(chunk.get("usage"), dict) and chunk["usage"]:
            self.usage = _chat_usage_to_responses(chunk["usage"])
        choices = chunk.get("choices") or []
        if not choices:
            return out
        choice = choices[0]
        delta = choice.get("delta") or {}
        if isinstance(delta.get("content"), str) and delta["content"]:
            out += self._open_text()
            self.emitted_text.append(delta["content"])
            out.append(
                self._event(
                    {
                        "type": "response.output_text.delta",
                        "item_id": self.msg_item_id,
                        "output_index": self.text_index,
                        "content_index": 0,
                        "delta": delta["content"],
                    }
                )
            )
        for call in delta.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            index = call.get("index")
            if index is None:
                index = len(self.tool_order)
            if index not in self.tools:
                self.tools[index] = {
                    "item_id": f"fc_{index}",
                    "call_id": call.get("id") or f"call_{index}",
                    "name": (call.get("function") or {}).get("name", ""),
                    "output_index": self.text_index + 1 + len(self.tool_order),
                }
                self.tool_order.append(index)
                out.append(
                    self._event(
                        {
                            "type": "response.output_item.added",
                            "output_index": self.tools[index]["output_index"],
                            "item": {
                                "id": self.tools[index]["item_id"],
                                "type": "function_call",
                                "call_id": self.tools[index]["call_id"],
                                "name": self.tools[index]["name"],
                                "arguments": "",
                                "status": "in_progress",
                            },
                        }
                    )
                )
            args = (call.get("function") or {}).get("arguments")
            if isinstance(args, str) and args:
                info = self.tools[index]
                info["arguments"] = info.get("arguments", "") + args
                out.append(
                    self._event(
                        {
                            "type": "response.function_call_arguments.delta",
                            "item_id": info["item_id"],
                            "output_index": info["output_index"],
                            "call_id": info["call_id"],
                            "delta": args,
                        }
                    )
                )
        if choice.get("finish_reason"):
            self.finish_reason = choice["finish_reason"]
        return out

    def finish(self) -> List[str]:
        """Close open items and emit the terminal response.completed."""
        out: List[str] = []
        output: List[Dict[str, Any]] = []
        if self.text_open and not self.text_done:
            text = "".join(self.emitted_text)
            out.append(
                self._event(
                    {
                        "type": "response.output_text.done",
                        "item_id": self.msg_item_id,
                        "output_index": self.text_index,
                        "content_index": 0,
                        "text": text,
                    }
                )
            )
            out.append(
                self._event(
                    {
                        "type": "response.content_part.done",
                        "item_id": self.msg_item_id,
                        "output_index": self.text_index,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": text, "logprobs": []},
                    }
                )
            )
            out.append(
                self._event(
                    {
                        "type": "response.output_item.done",
                        "output_index": self.text_index,
                        "item": {
                            "id": self.msg_item_id,
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {"type": "output_text", "text": text, "logprobs": []}
                            ],
                        },
                    }
                )
            )
            self.text_done = True
            output.append(
                {
                    "id": self.msg_item_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "logprobs": []}],
                }
            )
        for index in self.tool_order:
            info = self.tools[index]
            args = info.get("arguments", "")
            item = {
                "id": info["item_id"],
                "type": "function_call",
                "call_id": info["call_id"],
                "name": info["name"],
                "arguments": args,
                "status": "completed",
            }
            out.append(
                self._event(
                    {
                        "type": "response.function_call_arguments.done",
                        "item_id": info["item_id"],
                        "output_index": info["output_index"],
                        "call_id": info["call_id"],
                        "arguments": args,
                    }
                )
            )
            out.append(
                self._event(
                    {
                        "type": "response.output_item.done",
                        "output_index": info["output_index"],
                        "item": item,
                    }
                )
            )
            output.append(item)
        status = "incomplete" if self.finish_reason == "length" else "completed"
        response = self._skeleton(status, output)
        if status == "incomplete":
            response["incomplete_details"] = {"reason": "max_output_tokens"}
        out.append(self._event({"type": "response.completed", "response": response}))
        return out





# --------------------------------------------------------------------------- #
# Serialization helpers
# --------------------------------------------------------------------------- #


def to_json_dict(obj: Any) -> Dict[str, Any]:
    """Convert SDK pydantic objects to plain JSON dicts."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", exclude_none=True)
    if hasattr(obj, "dict"):
        return obj.dict(exclude_none=True)
    if isinstance(obj, dict):
        return obj
    return json.loads(json.dumps(obj, default=str))


def sse_frame(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def sse_event_frame(payload: Dict[str, Any]) -> str:
    """SSE frame for the Responses API, which names each event.

    Chat-completions chunks carry no `type`, so they stay plain `data:` frames;
    Responses stream events are typed (response.created,
    response.output_text.delta, ...) and clients expect the `event:` line too.
    """
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    event_type = payload.get("type")
    if isinstance(event_type, str) and event_type:
        return f"event: {event_type}\ndata: {body}\n\n"
    return f"data: {body}\n\n"


def _track_usage(alias: str, usage_data: Any, stream: bool) -> None:
    """Best-effort usage/cost recording; never breaks the request path."""
    try:
        entry = MODEL_REGISTRY.get(alias, {})
        if entry.get("track_usage", True) is False:
            return
        usage_mod.record(
            alias,
            entry.get("upstream", alias),
            entry.get("provider"),
            usage_data,
            stream=stream,
            pricing_model=entry.get("pricing_model"),
        )
    except Exception:  # noqa: BLE001
        logging.exception("usage tracking failed for model=%s", alias)


# --------------------------------------------------------------------------- #
# Portkey clients (one per API key, cached)
# --------------------------------------------------------------------------- #

FALLBACK_API_KEY = PORTKEY_API_KEY  # optional; per-request keys take precedence

_CLIENT_CACHE: Dict[Tuple[str, str], Any] = {}
_CLIENT_CACHE_LOCK = threading.Lock()
MAX_CACHED_CLIENTS = 16


def _cache_client(cache_key: Tuple[str, str], client: Any) -> Any:
    """Store a freshly built client, evicting the oldest past the cap.

    Caller must hold _CLIENT_CACHE_LOCK.
    """
    _CLIENT_CACHE[cache_key] = client
    while len(_CLIENT_CACHE) > MAX_CACHED_CLIENTS:
        oldest = next(iter(_CLIENT_CACHE))
        evicted = _CLIENT_CACHE.pop(oldest)
        try:
            evicted.close()
        except Exception:  # noqa: BLE001 - best effort
            pass
    return client


def get_portkey_client(api_key: str) -> Portkey:
    """Return a cached Portkey SDK client bound to the given API key."""
    cache_key = (api_key, PORTKEY_BASE_URL)
    with _CLIENT_CACHE_LOCK:
        client = _CLIENT_CACHE.get(cache_key)
        if client is None:
            client = _cache_client(
                cache_key,
                Portkey(
                    api_key=api_key,
                    base_url=PORTKEY_BASE_URL,
                    request_timeout=PORTKEY_TIMEOUT_MS,
                ),
            )
        return client


def get_nrp_client() -> OpenAI:
    """Return a cached plain OpenAI client for the NRP gateway.

    NRP is a vanilla OpenAI-compatible endpoint that authenticates with
    `Authorization: Bearer <NRP_API_KEY>`. The Portkey SDK masks that header
    and sends the key as `x-portkey-api-key` instead, which the NRP gateway
    rejects with 403, so NRP requests bypass the SDK. The OpenAI client
    exposes the same `chat.completions` / `responses` surface, so the rest
    of the proxy is unchanged.
    """
    cache_key = (NRP_API_KEY, NRP_BASE_URL)
    with _CLIENT_CACHE_LOCK:
        client = _CLIENT_CACHE.get(cache_key)
        if client is None:
            client = _cache_client(
                cache_key, OpenAI(api_key=NRP_API_KEY, base_url=NRP_BASE_URL)
            )
        return client


_NRP_CHAT_PARAMS = frozenset(
    p
    for p in inspect.signature(_VendoredChatCompletions.create).parameters
    if p != "self"
)


def _nrp_chat_kwargs(
    client: Any, kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """Fold parameters the vendored OpenAI client doesn't accept into extra_body.

    The Portkey SDK forwards unknown chat parameters into the request body by
    itself, but the plain OpenAI client used for NRP raises TypeError on them
    (e.g. a stray `mock_upstream_mode` from the test suite, or any future
    chat parameter). Only the NRP client needs this; Portkey requests pass
    through untouched.
    """
    if not isinstance(client, OpenAI):
        return kwargs
    unknown = {k: v for k, v in kwargs.items() if k not in _NRP_CHAT_PARAMS}
    if not unknown:
        return kwargs
    kept = {k: v for k, v in kwargs.items() if k in _NRP_CHAT_PARAMS}
    body = dict(kept.get("extra_body") or {})
    body.update(unknown)
    kept["extra_body"] = body
    return kept


def resolve_api_key(request: Request) -> Optional[str]:
    """API key from the Copilot app (Authorization: Bearer ...), then
    x-portkey-api-key, then the optional fallback key from .env/env."""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth[len("bearer ") :].strip()
        if token:
            return token
    header_key = request.headers.get("x-portkey-api-key", "").strip()
    if header_key:
        return header_key
    return FALLBACK_API_KEY


@asynccontextmanager
async def lifespan(_: FastAPI):
    logging.info(
        "Portkey proxy v%s ready: upstream=%s models=%s fallback_key=%s "
        "tools_reasoning=%s responses=%s nrp=%s",
        PROXY_VERSION,
        PORTKEY_BASE_URL,
        ", ".join(MODEL_REGISTRY),
        "configured" if FALLBACK_API_KEY else "none (per-request keys only)",
        TOOLS_REASONING_EFFORT,
        "enabled" if RESPONSES_ENABLED else "disabled",
        f"{NRP_BASE_URL} (key={'set' if NRP_API_KEY else 'unset'})",
    )
    yield
    with _CLIENT_CACHE_LOCK:
        while _CLIENT_CACHE:
            _, client = _CLIENT_CACHE.popitem()
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="Portkey OpenAI-compatible proxy",
    description="Local OpenAI-compatible endpoint backed by the Portkey AI Sandbox SDK.",
    version=PROXY_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def access_log(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    elapsed_ms = (time.monotonic() - start) * 1000
    logging.info(
        "%s %s -> %s (%.0f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# --------------------------- free-first routing ---------------------------- #


def _build_router() -> Optional[routing_mod.Router]:
    """Instantiate the router from env/config, or None when disabled.

    A misconfigured router aborts startup: silently ignoring a broken
    ROUTER_CONFIG_PATH would route traffic against rules nobody reviewed.
    """
    if not routing_mod.is_enabled():
        return None
    try:
        router = routing_mod.Router(
            MODEL_REGISTRY,
            config_path=routing_mod.config_path(),
            responses_enabled=RESPONSES_ENABLED,
        )
    except ValueError as exc:
        raise SystemExit(f"ROUTER_ENABLED=1 but the router is misconfigured: {exc}")
    logging.info(
        "free-first router enabled: free=%s approval_at=$%s budget=$%s tracking=%s",
        ",".join(router.pools["free"]) or "none",
        f"{router.approval_threshold_usd:g}",
        f"{router.monthly_budget_usd:g}",
        "on" if usage_mod.USAGE_TRACKING else "OFF",
    )
    return router


ROUTER: Optional[routing_mod.Router] = _build_router()


class _AttemptFailure(Exception):
    """One candidate model's failure, raised while the response is rewindable.

    Only raised when the caller asked for `surface_errors`, so the router can
    try another model. Once bytes have reached the client the legacy SSE error
    frames are used and nothing is retried.
    """

    def __init__(
        self,
        status_code: int,
        message: str,
        err_type: str = "api_error",
        code: Optional[str] = None,
        param: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.err_type = err_type
        self.code = code
        self.param = param

    @classmethod
    def from_exception(cls, exc: BaseException) -> "_AttemptFailure":
        status, message, err_type, code, param = _exception_details(exc)
        return cls(status, message, err_type, code, param)

    @classmethod
    def from_api_error(cls, exc: ApiError) -> "_AttemptFailure":
        return cls(exc.status_code, exc.message, exc.err_type, exc.code, exc.param)

    @property
    def retryable(self) -> bool:
        """Whether a different candidate could serve this request.

        Transient upstream failures and request-shape rejections qualify. Auth
        errors do not (they belong to the credential, not the model), and
        quota exhaustion is handled by the budget gate rather than by retrying.
        """
        status, message = self.status_code, str(self)
        if routing_mod.is_quota_exhaustion(status, message, self.code):
            return True
        if status in (404, 408, 425, 429) or status >= 500:
            return True
        if _is_tools_reasoning_conflict(status, message):
            return True
        return status == 400 and bool(_CAPABILITY_ERROR_RE.search(message))

    @property
    def policy(self) -> bool:
        """True when the paid tier itself is unavailable, not the request.

        These failures are what justify reverting a hard/review route to the
        free pool: another paid model would hit the same wall.
        """
        return (
            self.status_code == 402
            or self.code in ("budget_exhausted", "insufficient_quota")
            or routing_mod.is_quota_exhaustion(
                self.status_code, str(self), self.code
            )
        )

    def as_json(self) -> JSONResponse:
        return JSONResponse(
            status_code=self.status_code,
            content=_error_payload(str(self), self.err_type, self.code, self.param),
        )

    def as_sse(self, wire: str = "chat") -> StreamingResponse:
        """Serialize a terminal failure in the caller's wire protocol."""
        payload = _error_payload(str(self), self.err_type, self.code, self.param)["error"]
        frames = (
            [sse_event_frame({"type": "error", "message": str(self), "code": self.code, "param": self.param})]
            if wire == "responses"
            else [sse_frame({"error": payload}), "data: [DONE]\n\n"]
        )
        return StreamingResponse(
            iter(frames),
            media_type="text/event-stream",
            headers=sse_headers(),
        )


_CAPABILITY_ERROR_RE = re.compile(
    r"unsupported|not supported|does not support|unrecognized"
    r"|unknown (?:parameter|model)|invalid (?:parameter|model)"
    r"|extra inputs are not permitted|is not one of",
    re.IGNORECASE,
)


class _Candidate(NamedTuple):
    alias: str
    upstream: str
    client: Any
    source: str


class _Route(NamedTuple):
    """A resolved request: the ordered models allowed to serve it."""

    requested: str
    virtual: bool
    candidates: List[_Candidate]
    body: Dict[str, Any]
    reason: str
    scope: Optional[str]
    # Free models to try if a paid route fails on a budget/quota signal.
    free_fallback: List[_Candidate] = []


def sse_headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    if extra:
        headers.update(extra)
    return headers


def route_headers(route: _Route, cand: _Candidate) -> Optional[Dict[str, str]]:
    """Tell the client which concrete model actually served a virtual alias."""
    if not route.virtual:
        return None
    return {"X-Proxy-Model": cand.alias, "X-Proxy-Route": route.requested}


def _no_credentials_error() -> ApiError:
    return ApiError(
        401,
        "No API key provided. Enter your Portkey / AI Sandbox API key in the "
        "Copilot app's model provider settings (it is sent as "
        "'Authorization: ******'), or set PORTKEY_API_KEY in the "
        "proxy's environment or .env file as a fallback.",
        err_type="authentication_error",
        code="missing_api_key",
    )


def _no_nrp_key_error() -> ApiError:
    return ApiError(
        401,
        "NRP_API_KEY is not set. Add NRP_API_KEY=... to the proxy's environment "
        "or .env file to use NRP models.",
        err_type="authentication_error",
        code="missing_api_key",
    )


def _model_not_found_error(model: str) -> ApiError:
    available = sorted(MODEL_REGISTRY)
    if ROUTER is not None:
        available = sorted(available + list(routing_mod.VIRTUAL_ALIASES))
    return ApiError(
        404,
        f"The model `{model}` does not exist or you do not have access to it. "
        f"Available models: {', '.join(available)}.",
        code="model_not_found",
        param="model",
    )


async def _validated_request(request: Request, wire: str = "chat") -> _Route:
    """Parse, authenticate and route a request body for both completions routes.

    Returns the ordered models allowed to serve the request (one entry for an
    explicit alias, or while the router is disabled), plus the body with any
    `proxy_routing` hints removed. Raises ApiError on any problem: bad JSON,
    missing/unknown model, no API key, or a request policy gates refuse.
    """
    try:
        body = await request.json()
    except Exception:
        raise ApiError(400, "Request body is not valid JSON.")
    if not isinstance(body, dict):
        raise ApiError(400, "Request body must be a JSON object.")

    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise ApiError(
            400,
            "You must specify a 'model' in the request body.",
            param="model",
        )

    virtual = ROUTER is not None and model in routing_mod.VIRTUAL_ALIASES
    if model not in MODEL_REGISTRY and not virtual:
        raise _model_not_found_error(model)

    # Routing hints are a proxy-only field: they are always removed from the
    # forwarded body, and only validated while the router is on so a disabled
    # router keeps accepting (and stripping) exactly what it used to.
    try:
        hints, body = routing_mod.parse_hints(body, enforce=ROUTER is not None)
    except routing_mod.RoutingError as exc:
        raise ApiError(exc.status_code, exc.message, exc.err_type, exc.code, exc.param)

    entry = MODEL_REGISTRY.get(model, {})
    is_nrp_model = entry.get("api_source") == "nrp"
    # An explicit NRP model never needs a Portkey key; anything else (including
    # a virtual alias, which may resolve to a paid model) resolves one.
    api_key = None if is_nrp_model else resolve_api_key(request)

    if ROUTER is None:
        # Legacy resolution, unchanged: one model, credential errors win.
        if is_nrp_model:
            # NRP models use a fixed key from the environment (not the request
            # key) and are served by a plain OpenAI client (****** against
            # the NRP gateway; the Portkey SDK's x-portkey-api-key scheme gets
            # 403s.
            if not NRP_API_KEY:
                raise _no_nrp_key_error()
            client = get_nrp_client()
        else:
            if not api_key:
                raise _no_credentials_error()
            client = get_portkey_client(api_key)
        return _Route(
            requested=model,
            virtual=False,
            candidates=[
                _Candidate(
                    model,
                    entry["upstream"],
                    client,
                    entry.get("api_source", "portkey"),
                )
            ],
            body=body,
            reason="",
            scope=routing_mod.Router.scope(api_key, "portkey") if api_key else None,
        )

    try:
        plan = ROUTER.plan(
            model,
            wire=wire,
            body=body,
            hints=hints,
            have_portkey_key=bool(api_key),
            have_nrp_key=bool(NRP_API_KEY),
            portkey_scope=routing_mod.Router.scope(api_key, "portkey") if api_key else None,
        )
    except routing_mod.RoutingError as exc:
        raise ApiError(exc.status_code, exc.message, exc.err_type, exc.code, exc.param)

    def _reachable(aliases: List[str]) -> List[_Candidate]:
        """Candidates for a virtual route, skipping models it cannot call."""
        out: List[_Candidate] = []
        for alias in aliases:
            ent = MODEL_REGISTRY[alias]
            if ent.get("api_source") == "nrp":
                if NRP_API_KEY:
                    out.append(
                        _Candidate(alias, ent["upstream"], get_nrp_client(), "nrp")
                    )
            elif api_key:
                out.append(
                    _Candidate(
                        alias,
                        ent["upstream"],
                        get_portkey_client(api_key),
                        "portkey",
                    )
                )
        return out

    candidates: List[_Candidate] = []
    for alias in plan.candidates:
        cand_entry = MODEL_REGISTRY[alias]
        if cand_entry.get("api_source") == "nrp":
            if not NRP_API_KEY:
                # An explicit NRP model keeps its legacy 401; a virtual route
                # simply drops the unreachable model from the list.
                if not plan.virtual:
                    raise _no_nrp_key_error()
                continue
            candidates.append(
                _Candidate(alias, cand_entry["upstream"], get_nrp_client(), "nrp")
            )
            continue
        if not api_key:
            if not plan.virtual:
                raise _no_credentials_error()
            continue
        candidates.append(
            _Candidate(
                alias, cand_entry["upstream"], get_portkey_client(api_key), "portkey"
            )
        )
    if not candidates:
        raise ApiError(
            401,
            "No credentials are configured for any model the router may use. "
            "Set NRP_API_KEY for the free models, and send a Portkey key (or "
            "set PORTKEY_API_KEY) for the paid ones.",
            err_type="authentication_error",
            code="missing_api_key",
        )

    logging.info("router: %s", plan.reason())
    free_fallback: List[_Candidate] = []
    if plan.virtual and plan.gate in ("hard", "review") and plan.pinned is None:
        # Keep the free tier behind a paid-only route: the budget policy is to
        # serve the request for free rather than fail it when the paid wallet
        # runs out mid-flight.
        chosen = {c.alias for c in candidates}
        free_fallback = [
            cand
            for cand in _reachable(
                ROUTER.free_fallback(
                    wire=wire,
                    body=body,
                    have_portkey_key=bool(api_key),
                    have_nrp_key=bool(NRP_API_KEY),
                    portkey_scope=(
                        routing_mod.Router.scope(api_key, "portkey") if api_key else None
                    ),
                )
            )
            if cand.alias not in chosen
        ]
    return _Route(
        requested=model,
        virtual=plan.virtual,
        candidates=candidates,
        body=body,
        reason=plan.reason(),
        scope=routing_mod.Router.scope(api_key, "portkey") if api_key else None,
        free_fallback=free_fallback,
    )


def _note_route_failure(
    route: _Route, cand: _Candidate, failure: _AttemptFailure
) -> None:
    """Restrict the rest of this request to free models on verified exhaustion.

    Only the paid upstream can prove the monthly budget is gone; an NRP 402/429
    is unrelated to the Portkey bill and is ignored here.
    """
    if ROUTER is None or cand.source != "portkey":
        return
    if not routing_mod.is_quota_exhaustion(
        failure.status_code, str(failure), failure.code
    ):
        return
    ROUTER.mark_exhausted(route.scope or ROUTER.scope(None, "portkey"), str(failure))


async def _prime_stream(frames: Any) -> Tuple[Optional[str], Optional[_AttemptFailure]]:
    """Pull the first SSE frame before committing to this attempt.

    Returns (frame, None) once the upstream has proved it can serve the
    request, or (None, failure) while the response is still rewindable. An
    empty stream counts as a failure so the next candidate is tried. Nothing
    reaches the client before this resolves, which is what keeps the fallback
    honest: no replay after streaming begins.
    """
    try:
        first = await run_in_threadpool(next, frames, None)
    except _AttemptFailure as exc:
        return None, exc
    except Exception as exc:  # noqa: BLE001
        return None, _AttemptFailure.from_exception(exc)
    if first is None:
        return None, _AttemptFailure(502, "Upstream returned an empty stream.")
    return first, None


def _frames_with_head(head: str, rest: Any) -> Generator[str, None, None]:
    try:
        yield head
        yield from rest
    finally:
        _close_stream(rest)


def _close_stream(stream: Any) -> None:
    close = getattr(stream, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            logging.warning("Could not close upstream stream", exc_info=True)


def _payload_failure(data: Dict[str, Any], status: int = 502) -> Optional[_AttemptFailure]:
    error = data.get("error")
    if data.get("type") == "error":
        error = data
    elif data.get("type") == "response.failed":
        error = (data.get("response") or {}).get("error")
    if not isinstance(error, dict):
        return None
    return _AttemptFailure(
        status, str(error.get("message") or "Upstream stream failed."),
        str(error.get("type") or "api_error"), error.get("code"), error.get("param"),
    )


def _observe_route_response(route: _Route, cand: _Candidate, response: Response) -> Response:
    """Observe explicit failures and committed stream errors without replay."""
    if ROUTER is None or cand.source != "portkey" or isinstance(response, StreamingResponse):
        return response
    if response.status_code >= 400:
        failure = _payload_failure(json.loads(response.body), response.status_code)
        if failure:
            _note_route_failure(route, cand, failure)
    return response


async def _run_routed(route: _Route, dispatch: Any, *, stream: bool, wire: str = "chat") -> Response:
    """Try the route's candidates in order, at most max_attempts times.

    `dispatch(cand, headers)` returns a committed Response or raises
    _AttemptFailure / ApiError. The first non-retryable failure ends the loop;
    if every attempt fails the last failure is returned in the shape the
    client asked for (JSON body, or an SSE error stream).
    """
    assert ROUTER is not None
    started = time.monotonic()
    limit = ROUTER.max_attempts
    exhausted_scope = route.scope or ROUTER.scope(None, "portkey")
    last_failure: Optional[_AttemptFailure] = None
    tried = 0
    candidates = list(route.candidates)
    fallback_added = False
    for cand in candidates:
        if tried >= limit:
            break
        if time.monotonic() - started >= ROUTER.deadline_seconds:
            logging.warning("router: deadline exceeded after %d attempt(s)", tried)
            break
        if cand.source == "portkey" and ROUTER.scope_exhausted(exhausted_scope):
            # The paid wallet is empty: don't burn another round trip on it.
            last_failure = _AttemptFailure(
                402,
                f"Skipped paid model `{cand.alias}`: the upstream reported the "
                "monthly quota is exhausted.",
                code="budget_exhausted",
            )
            if not fallback_added:
                candidates.extend(route.free_fallback)
                fallback_added = True
            continue
        tried += 1
        try:
            response = await dispatch(cand, route_headers(route, cand))
            return _observe_route_response(route, cand, response)
        except ApiError as exc:
            failure = _AttemptFailure.from_api_error(exc)
        except _AttemptFailure as exc:
            failure = exc
        last_failure = failure
        _note_route_failure(route, cand, failure)
        if not failure.retryable:
            break
        if cand.source == "portkey" and failure.policy and not fallback_added:
            candidates.extend(route.free_fallback)
            fallback_added = True
        logging.warning(
            "router: model=%s failed (status=%s): %s",
            cand.alias,
            failure.status_code,
            str(failure)[:300],
        )
    if last_failure is None:
        raise ApiError(
            400,
            "The router found no model eligible to serve this request.",
            code="no_eligible_model",
        )
    return last_failure.as_sse(wire) if stream else last_failure.as_json()


# ----------------------------- chat completions ---------------------------- #


def _sse_stream(
    client: Portkey,
    upstream: str,
    alias: str,
    attempts: List[Dict[str, Any]],
    surface_errors: bool = False,
    failure_callback: Any = None,
) -> Generator[str, None, None]:
    """Run the SDK stream and re-emit OpenAI SSE chunks.

    `attempts` holds one or two request-kwargs variants; when function tools
    are present, the second variant is tried if the upstream rejects the
    first with the tools+reasoning conflict (before any chunk was sent).

    With `surface_errors` (routed requests only) a failure that happens
    before the first chunk is raised instead of being streamed, so the router
    can still switch models; after that point the response is committed and
    the legacy error frame is used.
    """
    yielded_any = False
    recorded_usage = False
    stream = None
    try:
        stream = client.chat.completions.create(
            model=upstream, stream=True, **attempts[0]
        )
        for chunk in stream:
            data = to_json_dict(chunk)
            failure = _payload_failure(data)
            if failure:
                raise failure
            yielded_any = True
            data["model"] = alias  # report the name the client asked for
            if (
                not recorded_usage
                and isinstance(data.get("usage"), dict)
                and data["usage"]
            ):
                recorded_usage = True
                _track_usage(alias, data["usage"], stream=True)
            yield sse_frame(data)
        if surface_errors and not yielded_any:
            raise _AttemptFailure(502, "Upstream returned an empty stream.")
    except Exception as exc:  # noqa: BLE001 - translate every failure into SSE
        status, message, err_type, code, param = _exception_details(exc)
        if (
            not yielded_any
            and len(attempts) > 1
            and _is_tools_reasoning_conflict(status, message)
        ):
            logging.info(
                "Upstream rejected stream (tools+reasoning: %s); retrying with %s",
                message,
                _reasoning_desc(attempts[1]),
            )
            yield from _sse_stream(
                client, upstream, alias, attempts[1:], surface_errors, failure_callback
            )
            return
        if surface_errors and not yielded_any:
            raise _AttemptFailure(status, message, err_type, code, param)
        if failure_callback:
            failure_callback(_AttemptFailure(status, message, err_type, code, param))
        if status >= 500:
            logging.exception("Upstream stream failed (status=%s)", status)
        else:
            logging.warning("Upstream stream failed (status=%s): %s", status, message)
        yield sse_frame(
            {"error": _error_payload(message, err_type, code, param)["error"]}
        )
    finally:
        _close_stream(stream)
    yield "data: [DONE]\n\n"


def _bridge_chat_to_responses(kwargs: Dict[str, Any], caps: Dict[str, Any]) -> bool:
    """Whether a chat request with tools should be served via /v1/responses.

    auto   -> only models with no working chat shape for tools (gpt-6 family)
    always -> every request that carries function tools, so the GPT-5.6 family
              keeps its reasoning instead of having it stripped
    never  -> never bridge; the upstream 400 is forwarded to the client
    """
    if not RESPONSES_ENABLED or TOOLS_VIA_RESPONSES == "never":
        return False
    if not kwargs.get("tools"):
        return False
    if TOOLS_VIA_RESPONSES == "always":
        return True
    return not caps["tools_on_chat"]


def _sse_bridged_stream(
    client: Portkey,
    upstream: str,
    alias: str,
    kwargs: Dict[str, Any],
    surface_errors: bool = False,
    failure_callback: Any = None,
) -> Generator[str, None, None]:
    """Stream the Responses API and re-emit chat-completion SSE chunks.

    The client sees an ordinary chat stream: role delta, content/tool_calls
    deltas, finish_reason, optional usage, then `data: [DONE]`.

    `surface_errors` raises pre-commit failures (see `_sse_stream`).
    """
    translator = _ResponsesToChatSSE(alias)
    recorded_usage = False
    yielded_any = False
    stream = None
    try:
        stream = client.responses.create(model=upstream, stream=True, **kwargs)
        for event in stream:
            data = to_json_dict(event)
            failure = _payload_failure(data)
            if failure:
                raise failure
            if not recorded_usage:
                usage: Any = data.get("usage")
                if not isinstance(usage, dict) or not usage:
                    nested = data.get("response")
                    if isinstance(nested, dict):
                        usage = nested.get("usage")
                if isinstance(usage, dict) and usage:
                    recorded_usage = True
                    _track_usage(alias, usage, stream=True)
            for frame in translator.consume(data):
                yielded_any = True
                yield frame
        if surface_errors and not yielded_any:
            raise _AttemptFailure(502, "Upstream returned an empty stream.")
    except Exception as exc:  # noqa: BLE001 - translate every failure into SSE
        status, message, err_type, code, param = _exception_details(exc)
        if surface_errors and not yielded_any:
            raise _AttemptFailure(status, message, err_type, code, param)
        if failure_callback:
            failure_callback(_AttemptFailure(status, message, err_type, code, param))
        if status >= 500:
            logging.exception("Bridged responses stream failed (status=%s)", status)
        else:
            logging.warning(
                "Bridged responses stream failed (status=%s): %s", status, message
            )
        yield sse_frame(
            {"error": _error_payload(message, err_type, code, param)["error"]}
        )
        yield "data: [DONE]\n\n"
        return
    finally:
        _close_stream(stream)
    for frame in translator.finish():
        yield frame


async def _dispatch_chat(
    cand: _Candidate,
    body: Dict[str, Any],
    extra_headers: Optional[Dict[str, str]] = None,
    *,
    surface_errors: bool = False,
    failure_callback: Any = None,
) -> Response:
    """Send one chat-completions request to one concrete model.

    Returns a committed response. With `surface_errors` set (routed requests),
    any failure that happens before the client is committed raises
    `_AttemptFailure` instead, so the router can try another model.
    """
    alias, upstream, client = cand.alias, cand.upstream, cand.client
    stream = bool(body.get("stream", False))

    # Forward all remaining parameters untouched (temperature, max_tokens,
    # tools, tool_choice, top_p, stop, seed, response_format, ...).
    kwargs = {k: v for k, v in body.items() if k not in ("model", "stream")}
    if stream and INCLUDE_STREAM_USAGE:
        kwargs.setdefault("stream_options", {"include_usage": True})
    kwargs = _nrp_chat_kwargs(client, kwargs)

    # Per-model reasoning handling: rewrite effort values the model rejects
    # (gpt-6-astra has no "none"), and build the ordered list of request
    # variants for models that reject tools + reasoning together (GPT-5.6).
    caps = reasoning_caps(alias)
    kwargs, rewrite = _sanitize_reasoning(kwargs, caps)
    if rewrite:
        logging.info(
            "model=%s: reasoning_effort %s (unsupported upstream)", alias, rewrite
        )

    # Some upstreams have no chat-completions shape for function tools at all
    # (gpt-6 family); serve those through /v1/responses and translate back, so
    # the client still gets an ordinary chat.completion.
    if _bridge_chat_to_responses(kwargs, caps):
        resp_kwargs, drop_note = _chat_to_responses_kwargs(body, caps)
        logging.info(
            "chat request bridged via responses: model=%s stream=%s tools=%s "
            "reasoning=%s%s",
            alias,
            stream,
            len(resp_kwargs.get("tools") or []),
            _reasoning_desc(resp_kwargs),
            f" ({drop_note})" if drop_note else "",
        )
        if not stream:
            try:
                bridged = await run_in_threadpool(
                    client.responses.create, model=upstream, **resp_kwargs
                )
            except Exception as exc:  # noqa: BLE001
                status, message, err_type, code, param = _exception_details(exc)
                logging.warning(
                    "Bridged responses error (status=%s, model=%s): %s",
                    status,
                    alias,
                    message,
                )
                if surface_errors:
                    raise _AttemptFailure(status, message, err_type, code, param)
                return JSONResponse(
                    status_code=status,
                    content=_error_payload(message, err_type, code, param),
                )
            raw = to_json_dict(bridged)
            _track_usage(alias, raw.get("usage"), stream=False)
            return JSONResponse(
                _responses_to_chat_completion(raw, alias), headers=extra_headers
            )
        frames = _sse_bridged_stream(
            client, upstream, alias, resp_kwargs, surface_errors, failure_callback
        )
        if surface_errors:
            first, failure = await _prime_stream(frames)
            if failure is not None:
                raise failure
            frames = _frames_with_head(first, frames)
        return StreamingResponse(
            frames,
            media_type="text/event-stream",
            headers=sse_headers(extra_headers),
        )

    attempts = _tools_reasoning_plan(kwargs, caps)
    reasoning_in = {
        k: kwargs[k] for k in ("reasoning_effort", "reasoning") if k in kwargs
    }
    logging.info(
        "chat request: model=%s stream=%s tools=%s reasoning_in=%s",
        alias,
        stream,
        bool(kwargs.get("tools")),
        reasoning_in or "none",
    )
    if (
        caps["tools_conflict"]
        and kwargs.get("tools")
        and reasoning_in
        and attempts[0] != kwargs
    ):
        logging.info(
            "function tools present: reasoning %s -> %s",
            reasoning_in,
            _reasoning_desc(attempts[0]),
        )

    if not stream:
        data: Optional[Dict[str, Any]] = None
        failure: Tuple[int, str, str, Optional[str], Optional[str]] = (
            500,
            "Upstream request failed.",
            "api_error",
            None,
            None,
        )
        for attempt_kwargs in attempts:
            try:
                response = await run_in_threadpool(
                    client.chat.completions.create, model=upstream, **attempt_kwargs
                )
                data = to_json_dict(response)
                break
            except Exception as exc:  # noqa: BLE001
                status, message, err_type, code, param = _exception_details(exc)
                failure = (status, message, err_type, code, param)
                logging.warning(
                    "Upstream error (status=%s, model=%s): %s", status, alias, message
                )
                if attempt_kwargs is not attempts[-1] and _is_tools_reasoning_conflict(
                    status, message
                ):
                    logging.info("Retrying with %s", _reasoning_desc(attempts[-1]))
                    continue
                break
        if data is None:
            status, message, err_type, code, param = failure
            if surface_errors:
                raise _AttemptFailure(status, message, err_type, code, param)
            return JSONResponse(
                status_code=status,
                content=_error_payload(message, err_type, code, param),
            )
        data["model"] = alias
        data.setdefault("object", "chat.completion")
        _track_usage(alias, data.get("usage"), stream=False)
        return JSONResponse(data, headers=extra_headers)

    # Streaming: the response is only returned once the attempt is committed,
    # so a routed request can still fall back; a non-routed one keeps the
    # legacy behaviour of error frames inside the stream.
    frames = _sse_stream(client, upstream, alias, attempts, surface_errors, failure_callback)
    if surface_errors:
        first, stream_failure = await _prime_stream(frames)
        if stream_failure is not None:
            raise stream_failure
        frames = _frames_with_head(first, frames)
    return StreamingResponse(
        frames,
        media_type="text/event-stream",
        headers=sse_headers(extra_headers),
    )


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    route = await _validated_request(request, "chat")
    if route.virtual:
        return await _run_routed(
            route,
            lambda cand, headers: _dispatch_chat(
                cand, route.body, headers, surface_errors=True,
                failure_callback=lambda failure: _note_route_failure(route, cand, failure),
            ),
            stream=bool(route.body.get("stream", False)),
        )
    cand = route.candidates[0]
    response = await _dispatch_chat(
        cand, route.body, route_headers(route, cand),
        failure_callback=lambda failure: _note_route_failure(route, cand, failure),
    )
    return _observe_route_response(route, cand, response)


# ------------------------------- usage report ------------------------------ #


@app.get("/v1/usage")
@app.get("/usage")
async def get_usage(month: Optional[str] = None):
    """Monthly usage & cost summary (per model + total vs the budget)."""
    try:
        return usage_mod.monthly_report(month)
    except ValueError as exc:
        raise ApiError(400, str(exc), param="month")


# ------------------------------- models list ------------------------------- #


def _model_entry(name: str, created: int, owned_by: str) -> Dict[str, Any]:
    return {
        "id": name,
        "object": "model",
        "created": created,
        "owned_by": owned_by,
        "permission": [],
    }


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    created_now = int(time.time())
    data = [
        _model_entry(name, created_now, "princeton-ai-sandbox")
        for name in sorted(MODEL_REGISTRY)
    ]
    if ROUTER is not None:
        # The virtual aliases are requestable models, so advertise them too.
        data += [
            _model_entry(alias, created_now, "portkey-proxy-router")
            for alias in routing_mod.VIRTUAL_ALIASES
        ]
    return {"object": "list", "data": data}


@app.get("/v1/router")
@app.get("/router")
async def router_status():
    """Which models each virtual alias resolves to, and the gate state.

    Reports config, spend, and exhaustion only — never request content.
    """
    if ROUTER is None:
        return {"enabled": False}
    return ROUTER.status()


@app.post("/v1/router/reset")
@app.post("/router/reset")
async def router_reset():
    """Clear the paid-scope exhaustion note (e.g. after the quota is raised)."""
    if ROUTER is None:
        return {"enabled": False}
    ROUTER.reset_exhaustion()
    return {"enabled": True, "exhausted_scopes": []}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": PROXY_VERSION,
        "models": sorted(MODEL_REGISTRY)
        + (sorted(routing_mod.VIRTUAL_ALIASES) if ROUTER is not None else []),
        "fallback_key_configured": bool(FALLBACK_API_KEY),
        "router_enabled": ROUTER is not None,
    }


@app.get("/")
async def root():
    return {
        "service": "portkey-openai-proxy",
        "status": "ok",
        "version": PROXY_VERSION,
        "models": sorted(MODEL_REGISTRY),
        "endpoints": {
            "chat_completions": "POST /v1/chat/completions",
            "responses": "POST /v1/responses",
            "models": "GET /v1/models",
            "usage": "GET /v1/usage",
            "router": "GET /v1/router",
        },
    }


# ------------------------------- responses API ----------------------------- #


def _rewrite_response_models(payload: Dict[str, Any], alias: str) -> Dict[str, Any]:
    """Report the alias the client asked for, wherever a model name appears.

    Responses events carry the model both at the top level (the non-streaming
    body) and nested inside the `response` object (response.created,
    response.completed, ...). Only existing keys are rewritten: delta events
    have no model field and must not grow one.
    """
    if "model" in payload:
        payload["model"] = alias
    nested = payload.get("response")
    if isinstance(nested, dict) and "model" in nested:
        nested["model"] = alias
    return payload


def _sse_responses_stream(
    client: Portkey,
    upstream: str,
    alias: str,
    kwargs: Dict[str, Any],
    surface_errors: bool = False,
    failure_callback: Any = None,
) -> Generator[str, None, None]:
    """Run the SDK Responses stream and re-emit typed OpenAI SSE events.

    Unlike chat completions there is no `data: [DONE]` sentinel: the stream
    ends with a `response.completed` event, which also carries the usage block.
    There is no tools+reasoning conflict on this endpoint, so a single request
    variant is used (no _tools_reasoning_plan retry ladder).

    `surface_errors` raises pre-commit failures so the router can switch
    models (see `_sse_stream`).
    """
    recorded_usage = False
    yielded_any = False
    stream = None
    try:
        stream = client.responses.create(model=upstream, stream=True, **kwargs)
        for event in stream:
            data = _rewrite_response_models(to_json_dict(event), alias)
            failure = _payload_failure(data)
            if failure:
                raise failure
            if not recorded_usage:
                usage: Any = data.get("usage")
                if not isinstance(usage, dict) or not usage:
                    nested = data.get("response")
                    if isinstance(nested, dict):
                        usage = nested.get("usage")
                if isinstance(usage, dict) and usage:
                    recorded_usage = True
                    _track_usage(alias, usage, stream=True)
            yielded_any = True
            yield sse_event_frame(data)
        if surface_errors and not yielded_any:
            raise _AttemptFailure(502, "Upstream returned an empty stream.")
    except Exception as exc:  # noqa: BLE001 - translate every failure into SSE
        status, message, err_type, code, param = _exception_details(exc)
        if surface_errors and not yielded_any:
            raise _AttemptFailure(status, message, err_type, code, param)
        if failure_callback:
            failure_callback(_AttemptFailure(status, message, err_type, code, param))
        if status >= 500:
            logging.exception("Upstream responses stream failed (status=%s)", status)
        else:
            logging.warning(
                "Upstream responses stream failed (status=%s): %s", status, message
            )
        yield sse_event_frame(
            {
                "type": "error",
                "message": message,
                "code": code,
                "param": param,
            }
        )
    finally:
        _close_stream(stream)


def _sse_nrp_responses_stream(
    client: OpenAI,
    upstream: str,
    alias: str,
    kwargs: Dict[str, Any],
    surface_errors: bool = False,
    failure_callback: Any = None,
) -> Generator[str, None, None]:
    """Stream NRP chat completions and re-emit them as Responses SSE events.

    The upstream has no working /v1/responses, so the reply is translated from
    chat chunks. Like real Responses streams it ends with `response.completed`
    (carrying usage) and never sends `data: [DONE]`.

    The synthetic start events are normally emitted first, but a routed
    attempt (`surface_errors`) holds them back until the upstream has accepted
    the request: sending them would commit the response and make a model
    switch impossible.
    """
    translator = _ChatToResponsesSSE(alias, upstream)
    kwargs = _nrp_chat_kwargs(client, kwargs)
    emitted = False
    stream = None
    if not surface_errors:
        for frame in translator.start_events():
            yield frame
        emitted = True
    try:
        stream = client.chat.completions.create(
            model=upstream, stream=True, **kwargs
        )
        for chunk in stream:
            data = to_json_dict(chunk)
            failure = _payload_failure(data)
            if failure:
                raise failure
            if not emitted:
                # Creating an iterator is not acceptance: wait for upstream data.
                emitted = True
                yield from translator.start_events()
            for frame in translator.consume(data):
                yield frame
        if surface_errors and not emitted:
            raise _AttemptFailure(502, "Upstream returned an empty stream.")
    except Exception as exc:  # noqa: BLE001 - translate every failure into SSE
        status, message, err_type, code, param = _exception_details(exc)
        if surface_errors and not emitted:
            raise _AttemptFailure(status, message, err_type, code, param)
        if failure_callback:
            failure_callback(_AttemptFailure(status, message, err_type, code, param))
        if status >= 500:
            logging.exception("NRP bridge stream failed (status=%s)", status)
        else:
            logging.warning("NRP bridge stream failed (status=%s): %s", status, message)
        yield sse_event_frame(
            {"type": "error", "message": message, "code": code, "param": param}
        )
        return
    finally:
        _close_stream(stream)
    for frame in translator.finish():
        yield frame


def _response_id_from_frame(frame: str) -> Optional[str]:
    """Extract a response id from one SSE frame, if it carries one."""
    for line in frame.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            payload = json.loads(line[len("data:") :].strip())
        except Exception:  # noqa: BLE001 - `data: [DONE]` and partial frames
            return None
        if not isinstance(payload, dict):
            continue
        for candidate in (payload, payload.get("response")):
            if isinstance(candidate, dict):
                rid = candidate.get("id")
                if isinstance(rid, str) and rid:
                    return rid
    return None


def _pin_response_stream(frames: Any, alias: str) -> Generator[str, None, None]:
    """Remember which model owns a streamed Responses thread.

    A later request carrying `previous_response_id` must go back to the same
    model, so the id is learned from the first frame that contains it.
    """
    pinned = False
    try:
        for frame in frames:
            if not pinned:
                rid = _response_id_from_frame(frame)
                if rid:
                    if ROUTER is not None:
                        ROUTER.remember_response(rid, alias)
                    pinned = True
            yield frame
    finally:
        _close_stream(frames)


async def _dispatch_responses(
    cand: _Candidate,
    body: Dict[str, Any],
    extra_headers: Optional[Dict[str, str]] = None,
    *,
    surface_errors: bool = False,
    pin: bool = False,
    failure_callback: Any = None,
) -> Response:
    """Send one /v1/responses request to one concrete model.

    See `_dispatch_chat` for the `surface_errors` contract; `pin` records the
    response id against the winning model for stateful continuations.
    """
    alias, upstream, client = cand.alias, cand.upstream, cand.client
    stream = bool(body.get("stream", False))

    def _remember(response_id: Any) -> None:
        if pin and ROUTER is not None:
            ROUTER.remember_response(response_id, alias)

    if MODEL_REGISTRY[alias].get("responses_via_chat"):
        # The upstream has no working Responses API (NRP's /v1/responses is a
        # broken passthrough); serve this request via chat completions and
        # translate the reply back. The client never learns the difference.
        kwargs, _names, note = _responses_to_chat_kwargs(body)
        if stream and INCLUDE_STREAM_USAGE:
            kwargs.setdefault("stream_options", {"include_usage": True})
        logging.info(
            "responses request bridged via chat: model=%s stream=%s tools=%s%s",
            alias,
            stream,
            bool(kwargs.get("tools")),
            f" ({note})" if note else "",
        )
        if not stream:
            kwargs = _nrp_chat_kwargs(client, kwargs)
            try:
                response = await run_in_threadpool(
                    client.chat.completions.create, model=upstream, **kwargs
                )
            except Exception as exc:  # noqa: BLE001
                status, message, err_type, code, param = _exception_details(exc)
                logging.warning(
                    "NRP bridge error (status=%s, model=%s): %s",
                    status,
                    alias,
                    message,
                )
                if surface_errors:
                    raise _AttemptFailure(status, message, err_type, code, param)
                return JSONResponse(
                    status_code=status,
                    content=_error_payload(message, err_type, code, param),
                )
            data = _chat_to_responses_completion(to_json_dict(response), alias)
            _track_usage(alias, data.get("usage"), stream=False)
            _remember(data.get("id"))
            return JSONResponse(data, headers=extra_headers)
        frames = _sse_nrp_responses_stream(
            client, upstream, alias, kwargs, surface_errors, failure_callback
        )
        if surface_errors:
            first, failure = await _prime_stream(frames)
            if failure is not None:
                raise failure
            frames = _frames_with_head(first, frames)
        if pin:
            frames = _pin_response_stream(frames, alias)
        return StreamingResponse(
            frames,
            media_type="text/event-stream",
            headers=sse_headers(extra_headers),
        )

    caps = reasoning_caps(alias)
    kwargs, note = _responses_kwargs(body, caps)
    if note:
        logging.info("responses request: model=%s: %s", alias, note)
    logging.info(
        "responses request: model=%s stream=%s tools=%s reasoning=%s",
        alias,
        stream,
        bool(kwargs.get("tools")),
        _reasoning_desc(kwargs),
    )

    if not stream:
        try:
            response = await run_in_threadpool(
                client.responses.create, model=upstream, **kwargs
            )
        except Exception as exc:  # noqa: BLE001
            status, message, err_type, code, param = _exception_details(exc)
            logging.warning(
                "Upstream responses error (status=%s, model=%s): %s",
                status,
                alias,
                message,
            )
            if surface_errors:
                raise _AttemptFailure(status, message, err_type, code, param)
            return JSONResponse(
                status_code=status,
                content=_error_payload(message, err_type, code, param),
            )
        data = _rewrite_response_models(to_json_dict(response), alias)
        data.setdefault("object", "response")
        _track_usage(alias, data.get("usage"), stream=False)
        _remember(data.get("id"))
        return JSONResponse(data, headers=extra_headers)

    # Streaming: headers are sent before the request is made, so upstream
    # errors are surfaced as a typed SSE error event. A routed attempt primes
    # the stream first so it can still fall back to another model.
    frames = _sse_responses_stream(client, upstream, alias, kwargs, surface_errors, failure_callback)
    if surface_errors:
        first, failure = await _prime_stream(frames)
        if failure is not None:
            raise failure
        frames = _frames_with_head(first, frames)
    if pin:
        frames = _pin_response_stream(frames, alias)
    return StreamingResponse(
        frames,
        media_type="text/event-stream",
        headers=sse_headers(extra_headers),
    )


@app.post("/v1/responses")
@app.post("/responses")
async def responses_create(request: Request):
    if not RESPONSES_ENABLED:
        raise ApiError(
            404,
            "The OpenAI Responses API is disabled on this proxy "
            "(set RESPONSES_ENABLED=1 to enable it). Use "
            "POST /v1/chat/completions instead.",
            code="unsupported_endpoint",
        )
    route = await _validated_request(request, "responses")
    if route.virtual:
        return await _run_routed(
            route,
            lambda cand, headers: _dispatch_responses(
                cand, route.body, headers, surface_errors=True, pin=True,
                failure_callback=lambda failure: _note_route_failure(route, cand, failure),
            ),
            stream=bool(route.body.get("stream", False)),
            wire="responses",
        )
    cand = route.candidates[0]
    response = await _dispatch_responses(
        cand, route.body, route_headers(route, cand),
        failure_callback=lambda failure: _note_route_failure(route, cand, failure),
    )
    return _observe_route_response(route, cand, response)


# ------------------------------ error handlers ----------------------------- #


@app.exception_handler(ApiError)
async def handle_api_error(_: Request, exc: ApiError):
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_payload(exc.message, exc.err_type, exc.code, exc.param),
    )


@app.exception_handler(StarletteHTTPException)
async def handle_http_error(_: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_payload(str(exc.detail), "invalid_request_error", None, None),
    )


@app.exception_handler(Exception)
async def handle_unexpected(_: Request, exc: Exception):
    if isinstance(exc, (ValueError, TypeError)):
        return JSONResponse(
            status_code=400,
            content=_error_payload(str(exc), "invalid_request_error", None, None),
        )
    logging.exception("Unhandled proxy error")
    return JSONResponse(
        status_code=500,
        content=_error_payload(
            "Internal server error while contacting the upstream model.",
            "server_error",
            None,
            None,
        ),
    )


# --------------------------------- main ------------------------------------ #


def _masked_key(key: Optional[str]) -> str:
    if not key:
        return "<not set>"
    return key[:6] + "..." + key[-4:] if len(key) > 12 else "***"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        # The router logs at import time, which makes an argument-less
        # basicConfig() install a WARNING-level handler first; force=True
        # keeps INFO visible either way.
        force=True,
    )
    parser = argparse.ArgumentParser(description="Portkey OpenAI-compatible proxy")
    parser.add_argument(
        "--host", default=PROXY_HOST, help="bind host (default: %(default)s)"
    )
    parser.add_argument(
        "--port", type=int, default=PROXY_PORT, help="bind port (default: %(default)s)"
    )
    args = parser.parse_args()

    print("=" * 72)
    print(f" Portkey OpenAI-compatible proxy v{PROXY_VERSION}")
    print(f"   Local endpoint : http://{args.host}:{args.port}/v1")
    print(f"   Portkey gateway: {PORTKEY_BASE_URL}")
    print(f"   NRP gateway    : {NRP_BASE_URL} (key {'set' if NRP_API_KEY else 'NOT set'})")
    print("   Auth           : API key from each request (set it in the Copilot app)")
    print(f"   Fallback key   : {_masked_key(FALLBACK_API_KEY)}")
    print(f"   Models         : {', '.join(sorted(MODEL_REGISTRY))}")
    print(f"   Streaming usage: {'on' if INCLUDE_STREAM_USAGE else 'off'}")
    print(
        "   Responses API  : "
        + ("enabled (POST /v1/responses)" if RESPONSES_ENABLED else "disabled")
    )
    print(
        "   Usage tracking: "
        + (
            f"on -> {usage_mod.USAGE_LOG_PATH} "
            f"(budget ${usage_mod.MONTHLY_BUDGET_USD:.0f}/mo)"
            if usage_mod.USAGE_TRACKING
            else "off"
        )
    )
    print(
        "   Gateway timeout: "
        + (
            "not sent (gateway default applies)"
            if PORTKEY_TIMEOUT_MS is None
            else f"{PORTKEY_TIMEOUT_MS} ms"
        )
    )
    print("=" * 72)
    print(" In the Copilot app model provider settings, use:")
    print(f"   Base URL : http://{args.host}:{args.port}/v1")
    print(
        "   Wire API : Completions (recommended) or Responses"
        if RESPONSES_ENABLED
        else "   Wire API : Completions   (Responses is disabled on this proxy)"
    )
    print("   API key  : <your Portkey / AI Sandbox key>")
    print()
    print(" Or for GitHub Copilot CLI, export:")
    print("   export COPILOT_PROVIDER_TYPE=openai")
    print(f"   export COPILOT_PROVIDER_BASE_URL=http://{args.host}:{args.port}/v1")
    print("   export COPILOT_PROVIDER_API_KEY=<your Portkey / AI Sandbox key>")
    print("   export COPILOT_MODEL=gpt-5.6-sol")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
