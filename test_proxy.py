#!/usr/bin/env python3
"""
End-to-end integration test for proxy.py.

Boots a mock Portkey-compatible upstream (mock_upstream.py) and the real
proxy (proxy.py) as subprocesses, then verifies:

  - model listing (gpt-5.6-sol / gpt-5.6-terra / gpt-5.6-luna)
  - non-streaming chat completions (JSON shape, alias -> upstream rewrite,
    auth injection, parameter forwarding, usage metadata)
  - streaming chat completions (SSE chunks, [DONE], final usage)
  - error translation (unknown model, missing model, upstream 401, bad JSON)
  - per-request API keys (Copilot app style Authorization: Bearer ...),
    x-portkey-api-key passthrough, and 401 when no key is available
  - per-model reasoning handling: GPT-5.6 strips reasoning with tools and
    auto-retries with reasoning_effort="none"; gpt-6-astra keeps tools+reasoning
    but rewrites unsupported values (e.g. "none" -> "low", or via a configured
    vocabulary); gemini-3.8-flash forwards effort values untouched; the
    claude-opus-4-8 / claude-fable-5 aliases use the same flexible profile

Run with the same interpreter that has the dependencies:
    ./.venv/bin/python test_proxy.py
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent
PY = sys.executable
CAPTURE = ROOT / "_captured.json"

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  PASS  {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL  {name}  {detail}")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_ready(url: str, timeout: float = 25.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=1.0).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.2)
    return False


def parse_sse(text: str) -> list[dict]:
    frames = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block or not block.startswith("data:"):
            continue
        data = block[len("data:") :].strip()
        if data == "[DONE]":
            frames.append({"done": True})
        else:
            frames.append(json.loads(data))
    return frames


def main() -> int:
    mock_port = free_port()
    proxy_port = free_port()

    mock_log = open(ROOT / "_mock.log", "w")
    proxy_log = open(ROOT / "_proxy.log", "w")
    proxy2: subprocess.Popen | None = None
    proxy2_log = None

    mock = subprocess.Popen(
        [PY, str(ROOT / "mock_upstream.py"), "--port", str(mock_port)],
        stdout=mock_log,
        stderr=subprocess.STDOUT,
        cwd=ROOT,
    )

    env = dict(os.environ)
    env.update(
        {
            "PORTKEY_API_KEY": "sk-portkey-test",
            "PORTKEY_BASE_URL": f"http://127.0.0.1:{mock_port}",  # no /v1: proxy must normalize
            "GPT56_SOL_UPSTREAM_MODEL": "upstream-sol",
            "GPT56_TERRA_UPSTREAM_MODEL": "upstream-terra",
            "GPT56_LUNA_UPSTREAM_MODEL": "upstream-luna",
            "GPT6_ASTRA_UPSTREAM_MODEL": "upstream-astra",
            "GEMINI38_FLASH_UPSTREAM_MODEL": "upstream-gemini",
            "CLAUDE_OPUS_48_UPSTREAM_MODEL": "upstream-opus",
            "CLAUDE_FABLE_5_UPSTREAM_MODEL": "upstream-fable",
            "INCLUDE_STREAM_USAGE": "1",
        }
    )
    proxy = subprocess.Popen(
        [PY, str(ROOT / "proxy.py"), "--port", str(proxy_port)],
        stdout=proxy_log,
        stderr=subprocess.STDOUT,
        cwd=ROOT,
        env=env,
    )

    base = f"http://127.0.0.1:{proxy_port}"
    try:
        if not wait_ready(f"{base}/health"):
            print("ERROR: proxy did not become ready; see _proxy.log")
            print(open(ROOT / "_proxy.log").read()[-2000:])
            return 1

        client = httpx.Client(base_url=base, timeout=30.0)

        # 1. Models list ---------------------------------------------------
        r = client.get("/v1/models")
        ids = [m["id"] for m in r.json().get("data", [])]
        check(
            "v1/models lists all models",
            r.status_code == 200
            and "gpt-5.6-sol" in ids
            and "gpt-5.6-terra" in ids
            and "gpt-5.6-luna" in ids
            and "gpt-6-astra" in ids
            and "gemini-3.8-flash" in ids
            and "claude-opus-4-8" in ids
            and "claude-fable-5" in ids,
            str(r.json()),
        )
        r = client.get("/models")
        check("bare /models also works", r.status_code == 200)
        r = client.get("/health")
        check(
            "health reports proxy version",
            isinstance(r.json().get("version"), str),
            r.text,
        )

        # 2. Non-streaming -------------------------------------------------
        req_body = {
            "model": "gpt-5.6-sol",
            "messages": [{"role": "user", "content": "Hi there"}],
            "temperature": 0.7,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "f",
                        "description": "d",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        }
        r = client.post("/v1/chat/completions", json=req_body)
        j = r.json()
        check("non-stream status 200", r.status_code == 200, r.text)
        check("non-stream object", j.get("object") == "chat.completion", str(j))
        check(
            "non-stream client-facing model",
            j.get("model") == "gpt-5.6-sol",
            str(j.get("model")),
        )
        check(
            "non-stream content",
            j["choices"][0]["message"]["content"] == "Hello from mock upstream!",
            str(j),
        )
        check(
            "non-stream usage forwarded",
            j.get("usage")
            == {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
            str(j.get("usage")),
        )

        cap = json.loads(CAPTURE.read_text())
        check(
            "upstream got rewritten model",
            cap["body"].get("model") == "upstream-sol",
            str(cap["body"].get("model")),
        )
        # The portkey_ai SDK authenticates via the x-portkey-api-key header.
        check(
            "upstream got Portkey api key header",
            cap["headers"].get("x-portkey-api-key") == "sk-portkey-test",
            str(cap["headers"].get("x-portkey-api-key")),
        )
        # No gateway timeout header must be sent unless PORTKEY_TIMEOUT_MS is
        # set (a seconds-style value there caused upstream 408s).
        check(
            "no gateway timeout header by default",
            "x-portkey-request-timeout" not in cap["headers"],
            str({k: v for k, v in cap["headers"].items() if "timeout" in k}),
        )
        check(
            "temperature forwarded",
            cap["body"].get("temperature") == 0.7,
            str(cap["body"]),
        )
        check(
            "tools forwarded",
            cap["body"].get("tools") == req_body["tools"],
            str(cap["body"].get("tools")),
        )

        # 2b. reasoning params + function tools (GPT-5.6 restriction) -----
        # Default: strip reasoning when tools are present; auto-retry with an
        # explicit reasoning_effort="none" if the upstream still rejects.
        r = client.post(
            "/v1/chat/completions", json={**req_body, "reasoning_effort": "high"}
        )
        check("tools + reasoning_effort -> 200", r.status_code == 200, r.text)
        cap = json.loads(CAPTURE.read_text())
        check(
            "reasoning_effort stripped when tools present",
            "reasoning_effort" not in cap["body"],
            str(cap["body"].get("reasoning_effort")),
        )

        r = client.post(
            "/v1/chat/completions",
            json={**req_body, "reasoning": {"effort": "high", "summary": "auto"}},
        )
        check("tools + reasoning dict -> 200", r.status_code == 200, r.text)
        cap = json.loads(CAPTURE.read_text())
        check(
            "reasoning.effort stripped when tools present",
            isinstance(cap["body"].get("reasoning"), dict)
            and "effort" not in cap["body"]["reasoning"],
            str(cap["body"].get("reasoning")),
        )

        # Upstream personality 1: rejects tools whenever reasoning is present.
        r = client.post(
            "/v1/chat/completions",
            json={
                **req_body,
                "reasoning_effort": "high",
                "mock_upstream_mode": "reject_with_reasoning",
            },
        )
        check("strip passes a presence-checking upstream", r.status_code == 200, r.text)

        # Upstream personality 2: requires an explicit reasoning_effort=none.
        r = client.post(
            "/v1/chat/completions",
            json={**req_body, "mock_upstream_mode": "reject_tools_unless_none"},
        )
        check(
            "auto-retry with explicit none (non-stream)", r.status_code == 200, r.text
        )
        cap = json.loads(CAPTURE.read_text())
        check(
            "retry forwarded reasoning_effort=none",
            cap["body"].get("reasoning_effort") == "none",
            str(cap["body"].get("reasoning_effort")),
        )

        r = client.post(
            "/v1/chat/completions",
            json={
                **req_body,
                "stream": True,
                "mock_upstream_mode": "reject_tools_unless_none",
            },
        )
        frames = parse_sse(r.text)
        content = "".join(
            c.get("delta", {}).get("content", "")
            for f in frames
            if f.get("choices") and not f.get("done")
            for c in f["choices"]
        )
        check(
            "auto-retry with explicit none (stream)",
            r.status_code == 200 and content == "Hello from mock upstream!",
            r.text[:200],
        )

        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-5.6-sol",
                "messages": [{"role": "user", "content": "Hi"}],
                "reasoning_effort": "high",
            },
        )
        check("reasoning_effort without tools -> 200", r.status_code == 200, r.text)
        cap = json.loads(CAPTURE.read_text())
        check(
            "reasoning_effort untouched without tools",
            cap["body"].get("reasoning_effort") == "high",
            str(cap["body"].get("reasoning_effort")),
        )

        r = client.post(
            "/v1/chat/completions", json=req_body
        )  # tools, no reasoning_effort
        check("tools without reasoning_effort -> 200", r.status_code == 200, r.text)
        cap = json.loads(CAPTURE.read_text())
        check(
            "no reasoning_effort injected when absent",
            "reasoning_effort" not in cap["body"],
            str(cap["body"].get("reasoning_effort")),
        )

        # 2c. gpt-5.6-luna ------------------------------------------------
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-5.6-luna",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        check("gpt-5.6-luna -> 200", r.status_code == 200, r.text)
        check(
            "gpt-5.6-luna client-facing model",
            r.json().get("model") == "gpt-5.6-luna",
            r.text,
        )
        cap = json.loads(CAPTURE.read_text())
        check(
            "gpt-5.6-luna mapped to upstream-luna",
            cap["body"].get("model") == "upstream-luna",
            str(cap["body"].get("model")),
        )

        # 2d. gpt-6-astra / gemini: reasoning is model-specific -------------
        # gpt-6-astra accepts tools + reasoning together, so nothing is
        # stripped; only values the model rejects ('none') are rewritten.
        astra_body = {
            "model": "gpt-6-astra",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": req_body["tools"],
            "reasoning_effort": "high",
        }
        r = client.post("/v1/chat/completions", json=astra_body)
        check("astra: tools + reasoning_effort -> 200", r.status_code == 200, r.text)
        check(
            "astra: client-facing model", r.json().get("model") == "gpt-6-astra", r.text
        )
        cap = json.loads(CAPTURE.read_text())
        check(
            "astra: mapped to upstream-astra",
            cap["body"].get("model") == "upstream-astra",
            str(cap["body"].get("model")),
        )
        check(
            "astra: reasoning_effort kept (not stripped) with tools",
            cap["body"].get("reasoning_effort") == "high",
            str(cap["body"].get("reasoning_effort")),
        )

        # 'none' is not a valid astra effort -> rewritten to the fallback.
        r = client.post(
            "/v1/chat/completions",
            json={**astra_body, "reasoning_effort": "none"},
        )
        check("astra: reasoning_effort=none -> 200", r.status_code == 200, r.text)
        cap = json.loads(CAPTURE.read_text())
        check(
            "astra: none rewritten to fallback 'low'",
            cap["body"].get("reasoning_effort") == "low",
            str(cap["body"].get("reasoning_effort")),
        )

        # Same for the nested reasoning.effort shape.
        r = client.post(
            "/v1/chat/completions",
            json={
                **astra_body,
                "reasoning": {"effort": "none", "summary": "auto"},
            },
        )
        check("astra: reasoning.effort=none -> 200", r.status_code == 200, r.text)
        cap = json.loads(CAPTURE.read_text())
        check(
            "astra: reasoning.effort=none rewritten to 'low'",
            isinstance(cap["body"].get("reasoning"), dict)
            and cap["body"]["reasoning"].get("effort") == "low",
            str(cap["body"].get("reasoning")),
        )
        check(
            "astra: other reasoning keys preserved",
            cap["body"].get("reasoning", {}).get("summary") == "auto",
            str(cap["body"].get("reasoning")),
        )

        # xhigh is supported by astra and must pass through unchanged.
        r = client.post(
            "/v1/chat/completions",
            json={**astra_body, "reasoning_effort": "xhigh"},
        )
        cap = json.loads(CAPTURE.read_text())
        check(
            "astra: xhigh forwarded unchanged",
            r.status_code == 200 and cap["body"].get("reasoning_effort") == "xhigh",
            str(cap["body"].get("reasoning_effort")),
        )

        # An upstream that rejects 'none' outright must still be satisfied.
        r = client.post(
            "/v1/chat/completions",
            json={
                **astra_body,
                "reasoning_effort": "none",
                "mock_upstream_mode": "reject_none_effort",
            },
        )
        check(
            "astra: none rejected upstream -> rewritten and 200",
            r.status_code == 200,
            r.text,
        )
        cap = json.loads(CAPTURE.read_text())
        check(
            "astra: rewrite reached upstream as 'low'",
            cap["body"].get("reasoning_effort") == "low",
            str(cap["body"].get("reasoning_effort")),
        )

        # Streaming path gets the same treatment.
        r = client.post(
            "/v1/chat/completions",
            json={
                **astra_body,
                "reasoning_effort": "none",
                "stream": True,
                "mock_upstream_mode": "reject_none_effort",
            },
        )
        frames = parse_sse(r.text)
        content = "".join(
            c.get("delta", {}).get("content", "")
            for f in frames
            if f.get("choices") and not f.get("done")
            for c in f["choices"]
        )
        check(
            "astra: stream rewrites none and succeeds",
            r.status_code == 200 and content == "Hello from mock upstream!",
            r.text[:200],
        )

        # Configurable vocabulary via env is exercised in the second proxy;
        # by default gemini's effort values are forwarded untouched.
        gemini_body = {
            "model": "gemini-3.8-flash",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": req_body["tools"],
            "reasoning_effort": "minimal",
        }
        r = client.post("/v1/chat/completions", json=gemini_body)
        check("gemini: tools + reasoning -> 200", r.status_code == 200, r.text)
        cap = json.loads(CAPTURE.read_text())
        check(
            "gemini: unknown effort forwarded untouched",
            cap["body"].get("reasoning_effort") == "minimal",
            str(cap["body"].get("reasoning_effort")),
        )
        r = client.post(
            "/v1/chat/completions",
            json={**gemini_body, "reasoning_effort": "none"},
        )
        cap = json.loads(CAPTURE.read_text())
        check(
            "gemini: none forwarded untouched",
            r.status_code == 200 and cap["body"].get("reasoning_effort") == "none",
            str(cap["body"].get("reasoning_effort")),
        )

        # Claude models: flexible profile, like gemini. Tools + reasoning are
        # forwarded as written (no strip), and effort values pass through.
        for alias, upstream_name in (
            ("claude-opus-4-8", "upstream-opus"),
            ("claude-fable-5", "upstream-fable"),
        ):
            claude_body = {
                "model": alias,
                "messages": [{"role": "user", "content": "Hi"}],
                "tools": req_body["tools"],
                "reasoning_effort": "high",
            }
            r = client.post("/v1/chat/completions", json=claude_body)
            check(f"{alias}: tools + reasoning -> 200", r.status_code == 200, r.text)
            check(
                f"{alias}: client-facing model", r.json().get("model") == alias, r.text
            )
            cap = json.loads(CAPTURE.read_text())
            check(
                f"{alias}: mapped to {upstream_name}",
                cap["body"].get("model") == upstream_name,
                str(cap["body"].get("model")),
            )
            check(
                f"{alias}: keeps reasoning alongside tools",
                cap["body"].get("reasoning_effort") == "high",
                str(cap["body"].get("reasoning_effort")),
            )
            r = client.post(
                "/v1/chat/completions",
                json={**claude_body, "reasoning_effort": "minimal"},
            )
            cap = json.loads(CAPTURE.read_text())
            check(
                f"{alias}: unknown effort forwarded untouched",
                r.status_code == 200
                and cap["body"].get("reasoning_effort") == "minimal",
                str(cap["body"].get("reasoning_effort")),
            )
            r = client.post(
                "/v1/chat/completions", json={**claude_body, "stream": True}
            )
            frames = parse_sse(r.text)
            check(
                f"{alias}: streams with alias model",
                r.status_code == 200 and frames[0].get("model") == alias,
                str(frames[0].get("model")),
            )

        # GPT-5.6 must keep its original behaviour: reasoning stripped with tools.
        r = client.post(
            "/v1/chat/completions",
            json={**astra_body, "model": "gpt-5.6-sol", "reasoning_effort": "high"},
        )
        cap = json.loads(CAPTURE.read_text())
        check(
            "gpt-5.6: still strips reasoning with tools",
            r.status_code == 200 and "reasoning_effort" not in cap["body"],
            str(cap["body"].get("reasoning_effort")),
        )

        # 3. Streaming ----------------------------------------------------
        r = client.post("/v1/chat/completions", json={**req_body, "stream": True})
        frames = parse_sse(r.text)
        content = "".join(
            c.get("delta", {}).get("content", "")
            for f in frames
            if f.get("choices") and not f.get("done")
            for c in f["choices"]
        )
        check(
            "stream status 200 + text/event-stream",
            r.status_code == 200
            and r.headers["content-type"].startswith("text/event-stream"),
            r.headers.get("content-type"),
        )
        check("stream contains [DONE]", frames[-1].get("done") is True, str(frames[-1]))
        check(
            "stream reassembles content",
            content == "Hello from mock upstream!",
            repr(content),
        )
        check(
            "stream chunk model is alias",
            frames[0].get("model") == "gpt-5.6-sol",
            str(frames[0].get("model")),
        )
        usage_frames = [f for f in frames if f.get("usage")]
        check(
            "stream forwards usage (include_usage)",
            len(usage_frames) == 1 and usage_frames[0]["usage"]["total_tokens"] == 20,
            str(usage_frames),
        )
        cap = json.loads(CAPTURE.read_text())
        check(
            "stream upsteam model rewritten",
            cap["body"].get("model") == "upstream-sol",
            str(cap["body"].get("model")),
        )
        check(
            "stream flag forwarded",
            cap["body"].get("stream") is True,
            str(cap["body"].get("stream")),
        )

        # 4. Errors -------------------------------------------------------
        r = client.post(
            "/v1/chat/completions", json={"model": "does-not-exist", "messages": []}
        )
        check("unknown model -> 404", r.status_code == 404, str(r.status_code))
        check(
            "unknown model -> OpenAI error",
            r.json().get("error", {}).get("code") == "model_not_found",
            r.text,
        )

        r = client.post("/v1/chat/completions", json={"messages": []})
        check(
            "missing model -> 400",
            r.status_code == 400 and r.json()["error"]["param"] == "model",
            r.text,
        )

        r = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-5.6-terra", "messages": [], "fail": "auth"},
        )
        check(
            "upstream 401 -> 401 with OpenAI error",
            r.status_code == 401
            and r.json()["error"]["message"] == "Invalid API key provided.",
            r.text,
        )

        r = client.post(
            "/v1/chat/completions",
            content=b"{not json",
            headers={"content-type": "application/json"},
        )
        check(
            "invalid JSON -> 400", r.status_code == 400 and "error" in r.json(), r.text
        )

        r = client.post("/v1/responses", json={"model": "gpt-5.6-sol"})
        check(
            "responses API -> 404 with message",
            r.status_code == 404 and "chat completions" in r.json()["error"]["message"],
        )

        # 5. Per-request API key (the key set in the Copilot app) ---------
        r = client.post(
            "/v1/chat/completions",
            json=req_body,
            headers={"Authorization": "Bearer sk-from-copilot-app"},
        )
        check("per-request bearer key accepted", r.status_code == 200, r.text)
        cap = json.loads(CAPTURE.read_text())
        check(
            "per-request key forwarded upstream (beats fallback)",
            cap["headers"].get("x-portkey-api-key") == "sk-from-copilot-app",
            str(cap["headers"].get("x-portkey-api-key")),
        )

        r = client.post(
            "/v1/chat/completions",
            json=req_body,
            headers={"x-portkey-api-key": "sk-header-key"},
        )
        check("x-portkey-api-key incoming accepted", r.status_code == 200, r.text)
        cap = json.loads(CAPTURE.read_text())
        check(
            "x-portkey-api-key forwarded upstream",
            cap["headers"].get("x-portkey-api-key") == "sk-header-key",
            str(cap["headers"].get("x-portkey-api-key")),
        )

        # 6. Proxy with no fallback key: the app key is required ----------
        proxy2_port = free_port()
        env2 = dict(os.environ)
        env2.update(
            {
                "PORTKEY_API_KEY": "",
                "AI_SANDBOX_KEY": "",
                "PORTKEY_BASE_URL": f"http://127.0.0.1:{mock_port}",
                "INCLUDE_STREAM_USAGE": "1",
                "TOOLS_REASONING_EFFORT": "none",
                "PORTKEY_TIMEOUT_MS": "300000",
                # Vocabulary override: the proxy must honour configured values
                # and pick its fallback from them (never 'none' for astra).
                "GPT6_ASTRA_REASONING_EFFORTS": "minimal,medium,xhigh",
            }
        )
        proxy2_log = open(ROOT / "_proxy2.log", "w")
        proxy2 = subprocess.Popen(
            [PY, str(ROOT / "proxy.py"), "--port", str(proxy2_port)],
            stdout=proxy2_log,
            stderr=subprocess.STDOUT,
            cwd=ROOT,
            env=env2,
        )
        base2 = f"http://127.0.0.1:{proxy2_port}"
        check("proxy starts with no key configured", wait_ready(f"{base2}/health"))
        client2 = httpx.Client(base_url=base2, timeout=30.0)

        r = client2.post("/v1/chat/completions", json=req_body)
        check(
            "no key anywhere -> 401",
            r.status_code == 401 and r.json()["error"]["code"] == "missing_api_key",
            r.text,
        )
        r = client2.post(
            "/v1/chat/completions",
            json=req_body,
            headers={"Authorization": "Bearer sk-app-key"},
        )
        check("app key works without any .env key", r.status_code == 200, r.text)
        cap = json.loads(CAPTURE.read_text())
        check(
            "app key forwarded upstream",
            cap["headers"].get("x-portkey-api-key") == "sk-app-key",
            str(cap["headers"].get("x-portkey-api-key")),
        )

        # TOOLS_REASONING_EFFORT=none mode: explicit reasoning_effort=none.
        r = client2.post(
            "/v1/chat/completions",
            json={**req_body, "reasoning_effort": "high"},
            headers={"Authorization": "Bearer sk-app-key"},
        )
        check(
            "mode=none: tools + reasoning_effort -> 200", r.status_code == 200, r.text
        )
        cap = json.loads(CAPTURE.read_text())
        check(
            "mode=none: explicit reasoning_effort=none forwarded",
            cap["body"].get("reasoning_effort") == "none",
            str(cap["body"].get("reasoning_effort")),
        )
        check(
            "configured timeout forwarded in ms",
            cap["headers"].get("x-portkey-request-timeout") == "300000",
            str(cap["headers"].get("x-portkey-request-timeout")),
        )

        # TOOLS_REASONING_EFFORT=none must not force 'none' onto models that
        # reject it: astra keeps its reasoning effort even with tools present.
        # ("medium" is valid under this proxy's GPT6_ASTRA_REASONING_EFFORTS.)
        r = client2.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-6-astra",
                "messages": [{"role": "user", "content": "Hi"}],
                "tools": req_body["tools"],
                "reasoning_effort": "medium",
            },
            headers={"Authorization": "Bearer sk-app-key"},
        )
        cap = json.loads(CAPTURE.read_text())
        check(
            "mode=none does not force none onto astra",
            r.status_code == 200 and cap["body"].get("reasoning_effort") == "medium",
            r.text,
        )

        # GPT6_ASTRA_REASONING_EFFORTS=minimal,medium,xhigh: unsupported values
        # map to the first configured non-'none' effort.
        r = client2.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-6-astra",
                "messages": [{"role": "user", "content": "Hi"}],
                "reasoning_effort": "none",
            },
            headers={"Authorization": "Bearer sk-app-key"},
        )
        cap = json.loads(CAPTURE.read_text())
        check(
            "env effort list drives the fallback",
            r.status_code == 200 and cap["body"].get("reasoning_effort") == "minimal",
            str(cap["body"].get("reasoning_effort")),
        )
        r = client2.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-6-astra",
                "messages": [{"role": "user", "content": "Hi"}],
                "reasoning_effort": "medium",
            },
            headers={"Authorization": "Bearer sk-app-key"},
        )
        cap = json.loads(CAPTURE.read_text())
        check(
            "env effort list keeps supported values",
            r.status_code == 200 and cap["body"].get("reasoning_effort") == "medium",
            str(cap["body"].get("reasoning_effort")),
        )

    finally:
        procs = [(proxy, proxy_log), (mock, mock_log)]
        if proxy2 is not None:
            procs.append((proxy2, proxy2_log))
        for proc, _ in procs:
            proc.terminate()
        for proc, log in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            log.close()
        if CAPTURE.exists():
            CAPTURE.unlink()

    print()
    print(f"  {len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("FAILED:", ", ".join(FAILED))
        return 1
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
