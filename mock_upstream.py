#!/usr/bin/env python3
"""
Mock OpenAI-compatible upstream used by test_proxy.py (development helper).

Records every request body/headers to _captured.json so the integration
test can assert that the proxy forwards auth, model rewrites, parameters,
streaming, and errors exactly as expected.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

CAPTURE = Path(__file__).resolve().parent / "_captured.json"

app = FastAPI()


def _record(request: Request, body: Dict[str, Any]) -> None:
    CAPTURE.write_text(
        json.dumps(
            {"headers": {k: v for k, v in request.headers.items()}, "body": body},
            indent=2,
        ),
        encoding="utf-8",
    )


def _sse(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


TOOLS_REASONING_ERROR = (
    "Function tools with reasoning_effort are not supported for {model} in "
    "/v1/chat/completions. To use function tools, use /v1/responses or set "
    "reasoning_effort to 'none'."
)

# gpt-6-astra-style rejection: tools + reasoning is fine, but 'none' is not a
# valid reasoning_effort value for that model family.
UNSUPPORTED_EFFORT_ERROR = (
    "'reasoning_effort' does not support '{effort}' with this model. "
    "Supported values are: 'low', 'medium', 'high', and 'xhigh'."
)

# Fixed pricing served at /pricing/<provider>/<model> for usage-tracking
# tests (Portkey pricing-API shape; anthropic is intentionally absent so the
# "model has no known price" path can be exercised).
MOCK_PRICING: Dict[str, Dict[str, float]] = {
    "openai": {
        "request_token": 0.0004,
        "response_token": 0.002,
        "cache_read_input_token": 0.00004,
        "cache_write_input_token": 0.0005,
    },
    "google": {
        "request_token": 0.000075,
        "response_token": 0.000375,
        "cache_read_input_token": 0.0000075,
    },
}


@app.get("/pricing/{provider}/{model}")
async def pricing(provider: str, model: str):
    rates = MOCK_PRICING.get(provider)
    if not rates:
        return _error(404, "Pricing config not found for the given model")
    payg = {k: {"price": v} for k, v in rates.items()}
    return {"pay_as_you_go": payg, "currency": "USD"}


def _error(
    status: int, message: str, param: Any = None, code: Any = None
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": param,
                "code": code,
            }
        },
    )


def _tools_reasoning_error(model: str) -> JSONResponse:
    return _error(400, TOOLS_REASONING_ERROR.format(model=model))


def _unsupported_effort_error(effort: Any) -> JSONResponse:
    return _error(
        400, UNSUPPORTED_EFFORT_ERROR.format(effort=effort), param="reasoning_effort"
    )


def _effort_of(body: Dict[str, Any]) -> Any:
    effort = body.get("reasoning_effort")
    if effort is None and isinstance(body.get("reasoning"), dict):
        effort = body.get("reasoning").get("effort")
    return effort


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    # Test-only directive selecting the upstream's validation personality.
    mode = body.pop("mock_upstream_mode", None)
    # Test-only directive: report this many cached input tokens in usage.
    cached_tokens = body.pop("mock_cached_tokens", None)
    _record(request, body)
    model = body.get("model", "unknown")
    now = int(time.time())

    has_tools = bool(body.get("tools"))
    effort = _effort_of(body)
    if mode == "reject_with_reasoning" and has_tools and effort is not None:
        # Upstream rejects tools whenever any reasoning effort is present.
        return _tools_reasoning_error(model)
    if mode == "reject_tools_unless_none" and has_tools and effort != "none":
        # Upstream requires an explicit reasoning_effort="none" with tools.
        return _tools_reasoning_error(model)
    if mode == "reject_none_effort" and effort == "none":
        # gpt-6-astra-style upstream: tools + reasoning is fine, but 'none'
        # is not a supported reasoning_effort value.
        return _unsupported_effort_error("none")

    # Configurable failure modes used by the integration test.
    if body.get("fail") == "auth":
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "message": "Invalid API key provided.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_api_key",
                }
            },
        )
    if body.get("fail") == "rate":
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "message": "Rate limit reached for the AI Sandbox.",
                    "type": "rate_limit_error",
                    "param": None,
                    "code": "rate_limit_exceeded",
                }
            },
        )

    usage: Dict[str, Any] = {
        "prompt_tokens": 12,
        "completion_tokens": 8,
        "total_tokens": 20,
    }
    if cached_tokens is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}

    if not body.get("stream"):
        return {
            "id": "cmpl-mock-0001",
            "object": "chat.completion",
            "created": now,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from mock upstream!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }

    def stream():
        base = {
            "id": "cmpl-mock-0001",
            "object": "chat.completion.chunk",
            "created": now,
            "model": model,
        }
        yield _sse(
            {
                **base,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
        )
        yield _sse(
            {
                **base,
                "choices": [{"index": 0, "delta": {"content": "Hello "}, "finish_reason": None}],
            }
        )
        yield _sse(
            {
                **base,
                "choices": [
                    {"index": 0, "delta": {"content": "from mock upstream!"}, "finish_reason": None}
                ],
            }
        )
        if (body.get("stream_options") or {}).get("include_usage"):
            yield _sse({**base, "choices": [], "usage": usage})
        yield _sse(
            {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9077)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
