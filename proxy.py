#!/usr/bin/env python3
"""
Portkey AI Sandbox -> local OpenAI-compatible HTTP proxy for GitHub Copilot.

Exposes:
    POST /v1/chat/completions     (also /chat/completions)
    GET  /v1/models               (also /models)
    GET  /health, GET /

Incoming OpenAI-compatible requests are translated into calls to the
`portkey_ai` Python SDK; SDK responses (including streaming SSE) are
converted back into OpenAI JSON/SSE. Model names, parameters, errors and
usage metadata are preserved.

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
    TOOLS_REASONING_EFFORT strip|none|keep for requests with function tools
                           (default strip; only applied to models whose
                           upstream rejects tools + reasoning — the GPT-5.6
                           family — and the alternative is auto-tried)
    GPT56_SOL_UPSTREAM_MODEL     optional upstream model for gpt-5.6-sol
    GPT56_TERRA_UPSTREAM_MODEL   optional upstream model for gpt-5.6-terra
    GPT56_LUNA_UPSTREAM_MODEL   optional upstream model for gpt-5.6-luna
    GPT6_ASTRA_UPSTREAM_MODEL    optional upstream model for gpt-6-astra
    GEMINI38_FLASH_UPSTREAM_MODEL  optional upstream model for gemini-3.8-flash
    CLAUDE_OPUS_48_UPSTREAM_MODEL  optional upstream model for claude-opus-4-8
    CLAUDE_FABLE_5_UPSTREAM_MODEL  optional upstream model for claude-fable-5
    GPT6_ASTRA_REASONING_EFFORTS   comma-separated reasoning_effort values the
                           upstream accepts for gpt-6-astra (default
                           low,medium,high,xhigh — 'none' is rejected there)
    GEMINI38_FLASH_REASONING_EFFORTS
    CLAUDE_OPUS_48_REASONING_EFFORTS
    CLAUDE_FABLE_5_REASONING_EFFORTS
                           comma-separated accepted reasoning_effort values
                           for those models (default: unset, so whatever
                           the client sends is forwarded untouched)

Run:
    python proxy.py                      # serves on 127.0.0.1:8090
    python proxy.py --host 0.0.0.0 --port 8090
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from portkey_ai import Portkey
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

PROXY_VERSION = "1.4.0"

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

PORTKEY_API_KEY = os.environ.get("PORTKEY_API_KEY") or os.environ.get("AI_SANDBOX_KEY")
_raw_base_url = os.environ.get("PORTKEY_BASE_URL") or "https://api.portkey.ai/v1"
PORTKEY_BASE_URL = _raw_base_url.rstrip("/")
if not re.search(r"/v\d+$", PORTKEY_BASE_URL):
    PORTKEY_BASE_URL += "/v1"
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

# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
# "gpt-5.6-sol" / "gpt-5.6-terra" are the names the Copilot app will use.
# They are passed to Portkey as-is (they may be Portkey deployment aliases).
# To route them to a different upstream model, set e.g.
#   GPT56_SOL_UPSTREAM_MODEL=claude-sonnet-4-6
# or edit "upstream" below.
#
# Each model also declares a reasoning profile: upstreams disagree about
# 'reasoning_effort', so the proxy adapts per model instead of applying one
# workaround globally.
#   efforts        values the upstream accepts (None = unknown, forward as-is)
#   fallback       value substituted when the client sends an unsupported one
#   tools_conflict True when the upstream rejects tools + reasoning together

REASONING_PROFILES: Dict[str, Dict[str, Any]] = {
    # Rejects tools + reasoning in /v1/chat/completions and asks for
    # reasoning_effort="none" instead; "none" is a valid value here.
    "gpt-5.6": {
        "efforts": ["none", "low", "medium", "high", "xhigh"],
        "fallback": "none",
        "tools_conflict": True,
    },
    # Accepts tools + reasoning, but 'none' is not a valid effort:
    # "Supported values are: 'low', 'medium', 'high', and 'xhigh'".
    "gpt-6": {
        "efforts": ["low", "medium", "high", "xhigh"],
        "fallback": "low",
        "tools_conflict": False,
    },
    # Gemini-style upstreams: tools + reasoning is fine and the effort
    # vocabulary is theirs to define, so values pass through untouched
    # unless an explicit list is configured via the model's efforts_env.
    "flexible": {"efforts": None, "fallback": None, "tools_conflict": False},
}

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "gpt-5.6-sol": {
        "upstream": os.environ.get("GPT56_SOL_UPSTREAM_MODEL", "gpt-5.6-sol"),
        "description": "Portkey/AI Sandbox model exposed to the Copilot app",
        "reasoning": "gpt-5.6",
    },
    "gpt-5.6-terra": {
        "upstream": os.environ.get("GPT56_TERRA_UPSTREAM_MODEL", "gpt-5.6-terra"),
        "description": "Portkey/AI Sandbox model exposed to the Copilot app",
        "reasoning": "gpt-5.6",
    },
    "gpt-5.6-luna": {
        "upstream": os.environ.get("GPT56_LUNA_UPSTREAM_MODEL", "gpt-5.6-luna"),
        "description": "Portkey/AI Sandbox model exposed to the Copilot app",
        "reasoning": "gpt-5.6",
    },
    "gpt-6-astra": {
        "upstream": os.environ.get("GPT6_ASTRA_UPSTREAM_MODEL", "gpt-6-astra"),
        "description": "Portkey/AI Sandbox model exposed to the Copilot app",
        "reasoning": "gpt-6",
        "efforts_env": "GPT6_ASTRA_REASONING_EFFORTS",
    },
    "gemini-3.8-flash": {
        "upstream": os.environ.get("GEMINI38_FLASH_UPSTREAM_MODEL", "gemini-3.8-flash"),
        "description": "Portkey/AI Sandbox model exposed to the Copilot app",
        "reasoning": "flexible",
        "efforts_env": "GEMINI38_FLASH_REASONING_EFFORTS",
    },
    "claude-opus-4-8": {
        "upstream": os.environ.get("CLAUDE_OPUS_48_UPSTREAM_MODEL", "claude-opus-4-8"),
        "description": "Anthropic Claude Opus 4.8 via the AI Sandbox",
        "reasoning": "flexible",
        "efforts_env": "CLAUDE_OPUS_48_REASONING_EFFORTS",
    },
    "claude-fable-5": {
        "upstream": os.environ.get("CLAUDE_FABLE_5_UPSTREAM_MODEL", "claude-fable-5"),
        "description": "Anthropic Claude Fable 5 via the AI Sandbox",
        "reasoning": "flexible",
        "efforts_env": "CLAUDE_FABLE_5_REASONING_EFFORTS",
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
#      (the GPT-5.6 family). Those requests are adjusted per
#      TOOLS_REASONING_EFFORT and retried once with the alternative variant —
#      see _tools_reasoning_plan. /v1/responses is not supported by the
#      Sandbox, so there is no alternative endpoint to fall back on.


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


# --------------------------------------------------------------------------- #
# Portkey clients (one per API key, cached)
# --------------------------------------------------------------------------- #

FALLBACK_API_KEY = PORTKEY_API_KEY  # optional; per-request keys take precedence

_CLIENT_CACHE: Dict[str, Portkey] = {}
_CLIENT_CACHE_LOCK = threading.Lock()
MAX_CACHED_CLIENTS = 16


def get_portkey_client(api_key: str) -> Portkey:
    """Return a cached Portkey SDK client bound to the given API key."""
    with _CLIENT_CACHE_LOCK:
        client = _CLIENT_CACHE.get(api_key)
        if client is None:
            client = Portkey(
                api_key=api_key,
                base_url=PORTKEY_BASE_URL,
                request_timeout=PORTKEY_TIMEOUT_MS,
            )
            _CLIENT_CACHE[api_key] = client
            while len(_CLIENT_CACHE) > MAX_CACHED_CLIENTS:
                oldest = next(iter(_CLIENT_CACHE))
                evicted = _CLIENT_CACHE.pop(oldest)
                try:
                    evicted.close()
                except Exception:  # noqa: BLE001
                    pass
        return client


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
        "Portkey proxy v%s ready: upstream=%s models=%s fallback_key=%s tools_reasoning=%s",
        PROXY_VERSION,
        PORTKEY_BASE_URL,
        ", ".join(MODEL_REGISTRY),
        "configured" if FALLBACK_API_KEY else "none (per-request keys only)",
        TOOLS_REASONING_EFFORT,
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


# ----------------------------- chat completions ---------------------------- #


def _sse_stream(
    client: Portkey,
    upstream: str,
    alias: str,
    attempts: List[Dict[str, Any]],
) -> Generator[str, None, None]:
    """Run the SDK stream and re-emit OpenAI SSE chunks.

    `attempts` holds one or two request-kwargs variants; when function tools
    are present, the second variant is tried if the upstream rejects the
    first with the tools+reasoning conflict (before any chunk was sent).
    """
    yielded_any = False
    try:
        stream = client.chat.completions.create(
            model=upstream, stream=True, **attempts[0]
        )
        for chunk in stream:
            yielded_any = True
            data = to_json_dict(chunk)
            data["model"] = alias  # report the name the client asked for
            yield sse_frame(data)
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
            yield from _sse_stream(client, upstream, alias, attempts[1:])
            return
        if status >= 500:
            logging.exception("Upstream stream failed (status=%s)", status)
        else:
            logging.warning("Upstream stream failed (status=%s): %s", status, message)
        yield sse_frame(
            {"error": _error_payload(message, err_type, code, param)["error"]}
        )
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise ApiError(400, "Request body is not valid JSON.")
    if not isinstance(body, dict):
        raise ApiError(400, "Request body must be a JSON object.")

    api_key = resolve_api_key(request)
    if not api_key:
        raise ApiError(
            401,
            "No API key provided. Enter your Portkey / AI Sandbox API key in the "
            "Copilot app's model provider settings (it is sent as "
            "'Authorization: Bearer <key>'), or set PORTKEY_API_KEY in the "
            "proxy's environment or .env file as a fallback.",
            err_type="authentication_error",
            code="missing_api_key",
        )
    client = get_portkey_client(api_key)

    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise ApiError(
            400,
            "You must specify a 'model' in the request body.",
            param="model",
        )
    if model not in MODEL_REGISTRY:
        raise ApiError(
            404,
            f"The model `{model}` does not exist or you do not have access to it. "
            f"Available models: {', '.join(sorted(MODEL_REGISTRY))}.",
            code="model_not_found",
            param="model",
        )

    alias = model
    upstream = MODEL_REGISTRY[model]["upstream"]
    stream = bool(body.get("stream", False))

    # Forward all remaining parameters untouched (temperature, max_tokens,
    # tools, tool_choice, top_p, stop, seed, response_format, ...).
    kwargs = {k: v for k, v in body.items() if k not in ("model", "stream")}
    if stream and INCLUDE_STREAM_USAGE:
        kwargs.setdefault("stream_options", {"include_usage": True})

    # Per-model reasoning handling: rewrite effort values the model rejects
    # (gpt-6-astra has no "none"), and build the ordered list of request
    # variants for models that reject tools + reasoning together (GPT-5.6).
    caps = reasoning_caps(alias)
    kwargs, rewrite = _sanitize_reasoning(kwargs, caps)
    if rewrite:
        logging.info(
            "model=%s: reasoning_effort %s (unsupported upstream)", alias, rewrite
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
            return JSONResponse(
                status_code=status,
                content=_error_payload(message, err_type, code, param),
            )
        data["model"] = alias
        data.setdefault("object", "chat.completion")
        return JSONResponse(data)

    # Streaming: headers are sent before the request is made, so upstream
    # errors are surfaced as SSE error frames followed by [DONE].
    return StreamingResponse(
        _sse_stream(client, upstream, alias, attempts),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ------------------------------- models list ------------------------------- #


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    created_now = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "created": created_now,
                "owned_by": "princeton-ai-sandbox",
                "permission": [],
            }
            for name in sorted(MODEL_REGISTRY)
        ],
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": PROXY_VERSION,
        "models": sorted(MODEL_REGISTRY),
        "fallback_key_configured": bool(FALLBACK_API_KEY),
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
            "models": "GET /v1/models",
        },
    }


@app.post("/v1/responses")
@app.post("/responses")
async def responses_not_supported():
    raise ApiError(
        404,
        "The OpenAI Responses API is not supported by this proxy (the AI Sandbox "
        "only supports chat completions). Use POST /v1/chat/completions instead.",
        code="unsupported_endpoint",
    )


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
    print("   Auth           : API key from each request (set it in the Copilot app)")
    print(f"   Fallback key   : {_masked_key(FALLBACK_API_KEY)}")
    print(f"   Models         : {', '.join(sorted(MODEL_REGISTRY))}")
    print(f"   Streaming usage: {'on' if INCLUDE_STREAM_USAGE else 'off'}")
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
    print("   Wire API : Completions   (the AI Sandbox does not support Responses)")
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
