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
                    "message": {
                        "role": "assistant",
                        "content": "Hello from mock upstream!",
                    },
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
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
        )
        yield _sse(
            {
                **base,
                "choices": [
                    {"index": 0, "delta": {"content": "Hello "}, "finish_reason": None}
                ],
            }
        )
        yield _sse(
            {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "from mock upstream!"},
                        "finish_reason": None,
                    }
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


# --------------------------------------------------------------------------- #
# Responses API (/v1/responses)
# --------------------------------------------------------------------------- #
# Unlike the chat-completions mock above, this route deliberately ACCEPTS
# function tools together with reasoning — matching the real upstream, where
# the tools+reasoning conflict exists only on /v1/chat/completions. That lets
# the tests assert the proxy forwards reasoning untouched on this endpoint.


def _sse_event(payload: Dict[str, Any]) -> str:
    """Typed SSE frame, as the Responses API emits them (no [DONE] sentinel)."""
    event = payload.get("type")
    data = f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
    return f"event: {event}\n{data}" if event else data


@app.post("/v1/responses")
@app.post("/responses")
async def responses(request: Request):
    body = await request.json()
    mode = body.pop("mock_upstream_mode", None)
    cached_tokens = body.pop("mock_cached_tokens", None)
    # Test-only directive: answer with a function_call output item instead of a
    # message, so the chat<->responses bridge can be exercised end to end.
    tool_call = body.pop("mock_tool_call", None)
    _record(request, body)
    model = body.get("model", "unknown")
    now = int(time.time())

    # Only the effort-vocabulary personality applies here; there is no
    # tools+reasoning conflict on this endpoint.
    if mode == "reject_none_effort" and _effort_of(body) == "none":
        return _unsupported_effort_error("none")

    # The real upstream rejects chat-completions-only parameters on this
    # endpoint (verified 2026-09-29): "Unknown parameter:
    # 'stream_options.include_usage'". Mirroring it keeps the bridge honest.
    if isinstance(body.get("stream_options"), dict):
        return _error(
            400,
            "openai error: Unknown parameter: 'stream_options.include_usage'.",
            param="stream_options.include_usage",
            code="unknown_parameter",
        )

    if body.get("fail") == "auth":
        return _error(401, "Invalid API key provided.", code="invalid_api_key")
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
        "input_tokens": 12,
        "output_tokens": 8,
        "total_tokens": 20,
        "output_tokens_details": {"reasoning_tokens": 3},
    }
    if cached_tokens is not None:
        usage["input_tokens_details"] = {"cached_tokens": cached_tokens}

    if isinstance(tool_call, dict):
        name = tool_call.get("name", "get_weather")
        arguments = tool_call.get("arguments") or '{"city":"Paris"}'
        call_id = tool_call.get("call_id") or "call-mock-0001"
        output_items = [
            {
                "id": "fc-mock-0001",
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": arguments,
                "status": "completed",
            }
        ]
    else:
        output_items = [
            {
                "id": "msg-mock-0001",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Hello from mock upstream!",
                        "annotations": [],
                    }
                ],
            }
        ]

    response_obj = {
        "id": "resp-mock-0001",
        "object": "response",
        "created_at": now,
        "status": "completed",
        "model": model,
        "output": output_items,
        "usage": usage,
    }

    if not body.get("stream"):
        return response_obj

    def stream():
        seq = 0

        def event(payload: Dict[str, Any]) -> str:
            nonlocal seq
            payload = {"sequence_number": seq, **payload}
            seq += 1
            return _sse_event(payload)

        def in_progress() -> Dict[str, Any]:
            return {
                **response_obj,
                "status": "in_progress",
                "output": [],
                "usage": None,
            }

        yield event({"type": "response.created", "response": in_progress()})
        yield event({"type": "response.in_progress", "response": in_progress()})

        if isinstance(tool_call, dict):
            # Real upstreams add the item with empty arguments, then stream the
            # arguments as deltas -- the same split the bridge must reassemble.
            yield event(
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "id": "fc-mock-0001",
                        "type": "function_call",
                        "call_id": output_items[0]["call_id"],
                        "name": output_items[0]["name"],
                        "arguments": "",
                        "status": "in_progress",
                    },
                }
            )
            arguments = output_items[0]["arguments"]
            step = max(1, len(arguments) // 2)
            for start in range(0, len(arguments), step):
                yield event(
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": "fc-mock-0001",
                        "output_index": 0,
                        "delta": arguments[start : start + step],
                    }
                )
            yield event(
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": "fc-mock-0001",
                    "output_index": 0,
                    "arguments": arguments,
                }
            )
            yield event(
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": output_items[0],
                }
            )
            yield event({"type": "response.completed", "response": response_obj})
            return

        yield event(
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": "msg-mock-0001",
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                },
            }
        )
        for delta in ("Hello ", "from mock upstream!"):
            yield event(
                {
                    "type": "response.output_text.delta",
                    "item_id": "msg-mock-0001",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": delta,
                }
            )
        yield event(
            {
                "type": "response.output_text.done",
                "item_id": "msg-mock-0001",
                "output_index": 0,
                "content_index": 0,
                "text": "Hello from mock upstream!",
            }
        )
        yield event(
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": response_obj["output"][0],
            }
        )
        yield event({"type": "response.completed", "response": response_obj})

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
