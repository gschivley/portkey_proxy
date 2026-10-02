#!/usr/bin/env python3
"""
End-to-end integration test for proxy.py.

Boots a mock Portkey-compatible upstream (mock_upstream.py) and the real
proxy (proxy.py) as subprocesses, then verifies:

  - model listing (gpt-5.6-sol / gpt-5.6-terra / gpt-6-luna)
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
    claude-opus-5-5 / claude-sonnet-5-5 / claude-fable-5-1 aliases use the same
    flexible profile
  - usage & cost tracking (usage.py): JSONL records for stream + non-stream
    calls with costs from the (mocked) pricing API, cached-token split,
    pricing_model overrides, unknown-price fallback, GET /usage aggregation,
    and budget-milestone warning logs

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
        if not block:
            continue
        # Responses API frames carry an `event: <type>` line before the data;
        # chat-completion chunks do not. Collect every data: line in the block.
        data_lines = [
            line[len("data:") :].strip()
            for line in block.splitlines()
            if line.strip().startswith("data:")
        ]
        if not data_lines:
            continue
        data = "\n".join(data_lines)
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
    proxy3: subprocess.Popen | None = None
    proxy3_log = None
    proxy4: subprocess.Popen | None = None
    proxy4_log = None

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
            "GPT6_LUNA_UPSTREAM_MODEL": "upstream-luna",
            "GPT6_ASTRA_UPSTREAM_MODEL": "upstream-astra",
            "GEMINI38_FLASH_UPSTREAM_MODEL": "upstream-gemini",
            "CLAUDE_OPUS_55_UPSTREAM_MODEL": "upstream-opus",
            "CLAUDE_SONNET_55_UPSTREAM_MODEL": "upstream-sonnet",
            "CLAUDE_FABLE_51_UPSTREAM_MODEL": "upstream-fable",
            "INCLUDE_STREAM_USAGE": "1",
            # Blank the NRP key so a developer's real .env / shell export can
            # never leak into the default proxies (section 10 sets its own).
            "NRP_API_KEY": "",
            # Usage tracking gets its own proxy instance (section 9); keep the
            # default proxies from writing the repo's usage.jsonl or hitting
            # the real pricing API.
            "USAGE_TRACKING": "0",
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
            and "gpt-6-luna" in ids
            and "gpt-6-astra" in ids
            and "gpt-6.1-sol" in ids
            and "gemini-3.8-flash" in ids
            and "claude-opus-5-5" in ids
            and "claude-sonnet-5-5" in ids
            and "claude-fable-5-1" in ids,
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

        # 2c. gpt-6-luna ---------------------------------------------------
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-6-luna",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        check("gpt-6-luna -> 200", r.status_code == 200, r.text)
        check(
            "gpt-6-luna client-facing model",
            r.json().get("model") == "gpt-6-luna",
            r.text,
        )
        cap = json.loads(CAPTURE.read_text())
        check(
            "gpt-6-luna mapped to upstream-luna",
            cap["body"].get("model") == "upstream-luna",
            str(cap["body"].get("model")),
        )

        # gpt-6-luna is named like the gpt-6 family but behaves like GPT-5.6
        # (verified live 2026-10-02): 'none' is a valid effort and function
        # tools work on chat once reasoning is "none". So tools + reasoning is
        # stripped, and against an upstream that demands an explicit "none" the
        # retry ladder satisfies it -- no /v1/responses bridging.
        luna_tools = {
            "model": "gpt-6-luna",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": req_body["tools"],
            "reasoning_effort": "high",
        }
        r = client.post("/v1/chat/completions", json=luna_tools)
        cap = json.loads(CAPTURE.read_text())
        check(
            "gpt-6-luna: tools + reasoning -> 200 with reasoning stripped",
            r.status_code == 200 and "reasoning_effort" not in cap["body"],
            str(cap["body"].get("reasoning_effort")),
        )
        check(
            "gpt-6-luna: served on chat, not bridged to responses",
            "messages" in cap["body"] and "input" not in cap["body"],
            str(sorted(cap["body"])),
        )
        r = client.post(
            "/v1/chat/completions",
            json={**luna_tools, "mock_upstream_mode": "reject_tools_unless_none"},
        )
        cap = json.loads(CAPTURE.read_text())
        check(
            "gpt-6-luna: auto-retries with reasoning_effort=none",
            r.status_code == 200 and cap["body"].get("reasoning_effort") == "none",
            str(cap["body"].get("reasoning_effort")),
        )
        # 'none' is valid here (unlike astra), so it must survive untouched.
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-6-luna",
                "messages": [{"role": "user", "content": "Hi"}],
                "reasoning_effort": "none",
            },
        )
        cap = json.loads(CAPTURE.read_text())
        check(
            "gpt-6-luna: reasoning_effort=none forwarded (valid effort)",
            r.status_code == 200 and cap["body"].get("reasoning_effort") == "none",
            str(cap["body"].get("reasoning_effort")),
        )

        # 2d. gpt-6-astra / gemini: reasoning is model-specific -------------
        # gpt-6-astra has NO working chat-completions shape for function tools
        # (tools+reasoning 400s, 'none' 400s, and stripping reasoning still
        # 400s), so a tools-bearing chat request is bridged through
        # /v1/responses: reasoning is preserved as the nested shape the client
        # asked for, and the reply is translated back to chat-completion form.
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
        check(
            "astra: bridged reply keeps chat.completion shape",
            r.json().get("object") == "chat.completion"
            and r.json()["choices"][0]["message"]["role"] == "assistant",
            r.text,
        )
        cap = json.loads(CAPTURE.read_text())
        check(
            "astra: mapped to upstream-astra",
            cap["body"].get("model") == "upstream-astra",
            str(cap["body"].get("model")),
        )
        check(
            "astra: bridged to the responses endpoint",
            "input" in cap["body"] and "messages" not in cap["body"],
            str(sorted(cap["body"])),
        )
        check(
            "astra: tools flattened for responses",
            bool(cap["body"].get("tools"))
            and cap["body"]["tools"][0].get("name") == "f"
            and "function" not in cap["body"]["tools"][0],
            str(cap["body"].get("tools")),
        )
        check(
            "astra: reasoning_effort kept (not stripped) with tools",
            cap["body"].get("reasoning") == {"effort": "high"},
            str(cap["body"].get("reasoning")),
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
            (cap["body"].get("reasoning") or {}).get("effort") == "low",
            str(cap["body"].get("reasoning")),
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
            r.status_code == 200
            and (cap["body"].get("reasoning") or {}).get("effort") == "xhigh",
            str(cap["body"].get("reasoning")),
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
            (cap["body"].get("reasoning") or {}).get("effort") == "low",
            str(cap["body"].get("reasoning")),
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
            ("claude-opus-5-5", "upstream-opus"),
            ("claude-sonnet-5-5", "upstream-sonnet"),
            ("claude-fable-5-1", "upstream-fable"),
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

        # (/v1/responses is a real endpoint — covered in section 7.)

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
                "USAGE_TRACKING": "0",
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
            r.status_code == 200
            and (cap["body"].get("reasoning") or {}).get("effort") == "medium",
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

        # 7. Responses API (/v1/responses) --------------------------------
        # Responses-native request: nested reasoning, FLAT function tools.
        resp_body = {
            "model": "gpt-5.6-sol",
            "input": "Hi there",
            "reasoning": {"effort": "medium"},
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
        }
        r = client.post("/v1/responses", json=resp_body)
        j = r.json()
        check("responses non-stream status 200", r.status_code == 200, r.text)
        check("responses non-stream object", j.get("object") == "response", str(j))
        check(
            "responses client-facing model",
            j.get("model") == "gpt-5.6-sol",
            str(j.get("model")),
        )
        check(
            "responses output text",
            j.get("output", [{}])[0].get("content", [{}])[0].get("text")
            == "Hello from mock upstream!",
            str(j.get("output")),
        )
        check(
            "responses usage passthrough",
            j.get("usage", {}).get("input_tokens") == 12
            and j.get("usage", {}).get("output_tokens") == 8,
            str(j.get("usage")),
        )
        cap = json.loads(CAPTURE.read_text())
        check(
            "responses model rewritten upstream",
            cap["body"].get("model") == "upstream-sol",
            str(cap["body"].get("model")),
        )
        check(
            "responses input forwarded",
            cap["body"].get("input") == "Hi there",
            str(cap["body"].get("input")),
        )
        check(
            "responses flat tools forwarded untouched",
            cap["body"].get("tools", [{}])[0].get("name") == "get_weather",
            str(cap["body"].get("tools")),
        )
        # The headline difference from chat completions: the GPT-5.6 family
        # accepts tools + reasoning here, so reasoning must NOT be stripped.
        check(
            "responses keeps reasoning with tools (gpt-5.6)",
            cap["body"].get("reasoning") == {"effort": "medium"},
            str(cap["body"].get("reasoning")),
        )
        check(
            "responses forwards auth key",
            cap["headers"].get("x-portkey-api-key") == "sk-portkey-test",
            str(cap["headers"].get("x-portkey-api-key")),
        )
        r = client.post("/responses", json=resp_body)
        check("bare /responses also works", r.status_code == 200, r.text)

        # A flat reasoning_effort (chat-completions style) is normalised into
        # the nested shape the Responses API reads.
        r = client.post(
            "/v1/responses",
            json={"model": "gpt-5.6-terra", "input": "Hi", "reasoning_effort": "high"},
        )
        cap = json.loads(CAPTURE.read_text())
        check(
            "flat reasoning_effort -> reasoning.effort",
            r.status_code == 200
            and cap["body"].get("reasoning") == {"effort": "high"}
            and "reasoning_effort" not in cap["body"],
            str(cap["body"]),
        )

        # Chat-style `messages` become Responses `input` when they are plain
        # role+content items.
        r = client.post(
            "/v1/responses",
            json={
                "model": "gpt-5.6-terra",
                "messages": [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "Hi"},
                ],
            },
        )
        cap = json.loads(CAPTURE.read_text())
        check(
            "messages -> input conversion",
            r.status_code == 200
            and cap["body"].get("input", [{}])[0].get("role") == "system"
            and "messages" not in cap["body"],
            str(cap["body"]),
        )

        # Chat tool items have no Responses message shape: reject clearly
        # instead of silently mangling them.
        r = client.post(
            "/v1/responses",
            json={
                "model": "gpt-5.6-terra",
                "messages": [
                    {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
                ],
            },
        )
        check(
            "chat tool messages -> 400 unsupported_input_shape",
            r.status_code == 400
            and r.json()["error"].get("code") == "unsupported_input_shape",
            r.text,
        )

        # When a body carries both, the native Responses key wins and the
        # chat-only one is dropped rather than forwarded to the gateway.
        r = client.post(
            "/v1/responses",
            json={
                "model": "gpt-5.6-terra",
                "input": "native",
                "messages": [{"role": "user", "content": "chat"}],
            },
        )
        cap = json.loads(CAPTURE.read_text())
        check(
            "input wins over messages",
            r.status_code == 200
            and cap["body"].get("input") == "native"
            and "messages" not in cap["body"],
            str(cap["body"]),
        )

        # Per-model effort vocabulary still applies: astra rejects 'none'.
        r = client.post(
            "/v1/responses",
            json={
                "model": "gpt-6-astra",
                "input": "Hi",
                "reasoning": {"effort": "none"},
                "mock_upstream_mode": "reject_none_effort",
            },
        )
        cap = json.loads(CAPTURE.read_text())
        check(
            "responses rewrites unsupported effort (astra none -> low)",
            r.status_code == 200 and cap["body"]["reasoning"]["effort"] == "low",
            str(cap["body"].get("reasoning")),
        )

        # Flexible models forward an unknown effort untouched.
        r = client.post(
            "/v1/responses",
            json={
                "model": "gemini-3.8-flash",
                "input": "Hi",
                "reasoning": {"effort": "minimal"},
            },
        )
        cap = json.loads(CAPTURE.read_text())
        check(
            "responses forwards unknown effort for flexible models",
            r.status_code == 200 and cap["body"]["reasoning"]["effort"] == "minimal",
            str(cap["body"].get("reasoning")),
        )

        # Streaming: typed events, text reassembles, usage arrives once, and
        # there is no chat-style [DONE] sentinel.
        r = client.post("/v1/responses", json={**resp_body, "stream": True})
        check(
            "responses stream content type",
            r.status_code == 200
            and r.headers.get("content-type", "").startswith("text/event-stream"),
            str(r.headers.get("content-type")),
        )
        frames = parse_sse(r.text)
        deltas = [
            f.get("delta", "")
            for f in frames
            if f.get("type") == "response.output_text.delta"
        ]
        check(
            "responses stream reassembles text",
            "".join(deltas) == "Hello from mock upstream!",
            str(deltas),
        )
        completed = [f for f in frames if f.get("type") == "response.completed"]
        check(
            "responses stream ends with response.completed",
            len(completed) == 1 and bool(completed[0].get("response")),
            str([f.get("type") for f in frames]),
        )
        check(
            "responses stream has no [DONE] frame",
            not any(f.get("done") for f in frames),
            str([f.get("type") for f in frames]),
        )
        check(
            "responses stream rewrites model to alias",
            completed and completed[0]["response"].get("model") == "gpt-5.6-sol",
            str(completed),
        )
        usage_frames = [
            f
            for f in frames
            if isinstance(f.get("usage"), dict)
            or isinstance((f.get("response") or {}).get("usage"), dict)
        ]
        check(
            "responses stream carries usage exactly once",
            len(usage_frames) == 1
            and usage_frames[0]["response"]["usage"].get("input_tokens") == 12,
            str([f.get("type") for f in usage_frames]),
        )
        cap = json.loads(CAPTURE.read_text())
        check(
            "responses stream forwarded upstream with model rewrite",
            cap["body"].get("model") == "upstream-sol",
            str(cap["body"].get("model")),
        )

        # Errors on this endpoint are OpenAI-shaped too.
        r = client.post("/v1/responses", json={"model": "does-not-exist", "input": "x"})
        check(
            "responses unknown model -> 404",
            r.status_code == 404 and r.json()["error"]["code"] == "model_not_found",
            r.text,
        )
        r = client.post("/v1/responses", json={"input": "x"})
        check(
            "responses missing model -> 400",
            r.status_code == 400 and r.json()["error"]["param"] == "model",
            r.text,
        )
        r = client.post(
            "/v1/responses",
            json={"model": "gpt-5.6-sol", "input": "x", "fail": "auth"},
        )
        check(
            "responses upstream 401 -> 401 OpenAI error",
            r.status_code == 401
            and r.json()["error"]["message"] == "Invalid API key provided.",
            r.text,
        )
        r = client.post(
            "/v1/responses",
            content=b"{not json",
            headers={"content-type": "application/json"},
        )
        check(
            "responses invalid JSON -> 400",
            r.status_code == 400 and "error" in r.json(),
            r.text,
        )

        # 8. Chat -> Responses bridge (models with no chat shape for tools) --
        # gpt-6-astra tools requests are bridged above; here we verify the
        # tool-call round trip comes back in chat form.
        bridge_body = {
            "model": "gpt-6-astra",
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "tools": req_body["tools"],
            "mock_tool_call": {"name": "f", "arguments": '{"city":"Paris"}'},
        }
        r = client.post("/v1/chat/completions", json=bridge_body)
        j = r.json()
        msg = (j.get("choices") or [{}])[0].get("message", {})
        check("bridge: tool call -> 200", r.status_code == 200, r.text)
        check(
            "bridge: finish_reason is tool_calls",
            (j.get("choices") or [{}])[0].get("finish_reason") == "tool_calls",
            str(j.get("choices")),
        )
        check(
            "bridge: tool_calls nested under function (chat shape)",
            bool(msg.get("tool_calls"))
            and msg["tool_calls"][0]["function"]["name"] == "f"
            and msg["tool_calls"][0]["function"]["arguments"] == '{"city":"Paris"}',
            str(msg),
        )
        check(
            "bridge: content null on a tool-only turn",
            msg.get("content") is None,
            str(msg.get("content")),
        )
        check(
            "bridge: usage renamed to chat token names",
            j.get("usage", {}).get("prompt_tokens") == 12
            and j.get("usage", {}).get("completion_tokens") == 8,
            str(j.get("usage")),
        )
        check(
            "bridge: id/object/model are chat-shaped",
            j.get("object") == "chat.completion"
            and str(j.get("id", "")).startswith("chatcmpl")
            and j.get("model") == "gpt-6-astra",
            str({k: j.get(k) for k in ("object", "id", "model")}),
        )

        # Streaming bridge: deltas reassemble into one tool call, then
        # finish_reason and [DONE] exactly like a chat stream.
        r = client.post("/v1/chat/completions", json={**bridge_body, "stream": True})
        frames = parse_sse(r.text)
        check(
            "bridge stream: ends with [DONE]",
            bool(frames) and frames[-1].get("done") is True,
            str([f.get("type") for f in frames])[:200],
        )
        check(
            "bridge stream: no responses-style events leak through",
            not any(str(f.get("type", "")).startswith("response.") for f in frames),
            str([f.get("type") for f in frames])[:200],
        )
        deltas = [
            c.get("delta", {})
            for f in frames
            if not f.get("done")
            for c in f.get("choices") or []
        ]
        rebuilt = {"args": "", "name": None, "id": None}
        for delta in deltas:
            for call in delta.get("tool_calls") or []:
                if call.get("id"):
                    rebuilt["id"] = call["id"]
                fn = call.get("function") or {}
                if fn.get("name"):
                    rebuilt["name"] = fn["name"]
                rebuilt["args"] += fn.get("arguments") or ""
        check(
            "bridge stream: tool call reassembled from argument deltas",
            rebuilt["name"] == "f"
            and rebuilt["args"] == '{"city":"Paris"}'
            and bool(rebuilt["id"]),
            str(rebuilt),
        )
        check(
            "bridge stream: role delta first",
            deltas and deltas[0].get("role") == "assistant",
            str(deltas[:1]),
        )
        finish = [
            c.get("finish_reason")
            for f in frames
            if not f.get("done")
            for c in f.get("choices") or []
            if c.get("finish_reason")
        ]
        check(
            "bridge stream: finish_reason tool_calls emitted",
            finish == ["tool_calls"],
            str(finish),
        )
        usage_frames = [f for f in frames if f.get("usage")]
        check(
            "bridge stream: exactly one usage frame in chat names",
            len(usage_frames) == 1
            and usage_frames[0]["usage"].get("prompt_tokens") == 12,
            str(usage_frames),
        )

        # Copilot sends `stream_options` on every streaming chat request, but
        # the Responses upstream rejects it outright -- the bridge must drop it
        # (usage still arrives, via response.completed).
        r = client.post(
            "/v1/chat/completions",
            json={
                **astra_body,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        frames = parse_sse(r.text)
        check(
            "bridge stream: chat stream_options dropped, still 200",
            r.status_code == 200
            and not any("error" in f for f in frames)
            and frames
            and frames[-1].get("done") is True,
            r.text[:200],
        )
        check(
            "bridge stream: usage still captured without stream_options",
            any(f.get("usage") for f in frames),
            str([f.get("usage") for f in frames if f.get("usage")]),
        )
        cap = json.loads(CAPTURE.read_text())
        check(
            "bridge: stream_options never reaches the responses upstream",
            "stream_options" not in cap["body"],
            str(sorted(cap["body"])),
        )
        r = client.post(
            "/v1/chat/completions",
            json={**astra_body, "stream_options": {"include_usage": True}},
        )
        check(
            "bridge non-stream: stream_options dropped",
            r.status_code == 200 and r.json().get("object") == "chat.completion",
            r.text[:200],
        )

        # Multi-turn tool history must survive the conversion, or the model
        # never sees the tool output it is being asked about.
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-6-astra",
                "messages": [
                    {"role": "user", "content": "Weather?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "f",
                                    "arguments": '{"city":"Paris"}',
                                },
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
                    {"role": "user", "content": "thanks"},
                ],
                "tools": req_body["tools"],
            },
        )
        cap = json.loads(CAPTURE.read_text())
        items = cap["body"].get("input") or []
        kinds = [i.get("type") or i.get("role") for i in items]
        check(
            "bridge: assistant tool_calls -> function_call item",
            any(
                i.get("type") == "function_call"
                and i.get("call_id") == "call_1"
                and i.get("name") == "f"
                for i in items
            ),
            str(kinds),
        )
        check(
            "bridge: tool result -> function_call_output item",
            any(
                i.get("type") == "function_call_output"
                and i.get("call_id") == "call_1"
                and i.get("output") == "sunny"
                for i in items
            ),
            str(kinds),
        )
        check(
            "bridge: max_tokens -> max_output_tokens",
            r.status_code == 200,
            r.text,
        )
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-6-astra",
                "messages": [{"role": "user", "content": "Hi"}],
                "tools": req_body["tools"],
                "max_tokens": 128,
                "frequency_penalty": 0.5,
            },
        )
        cap = json.loads(CAPTURE.read_text())
        check(
            "bridge: max_tokens renamed, chat-only params dropped",
            cap["body"].get("max_output_tokens") == 128
            and "frequency_penalty" not in cap["body"]
            and "max_tokens" not in cap["body"],
            str(cap["body"]),
        )

        # 9. Usage & cost tracking (dedicated proxy, isolated log/cache) ---
        usage_log = ROOT / "_usage_test.jsonl"
        pricing_cache = ROOT / "_pricing_cache_test.json"
        for stale in (usage_log, pricing_cache, ROOT / "_proxy3.log"):
            if stale.exists():
                stale.unlink()
        month_now = time.strftime("%Y-%m")
        seed = {
            "ts": f"{month_now}-01T00:00:00",
            "month": month_now,
            "model": "seed-model",
            "upstream": "seed-upstream",
            "provider": "openai",
            "pricing_model": "seed-upstream",
            "stream": False,
            "prompt_tokens": 100,
            "cached_tokens": 0,
            "completion_tokens": 50,
            "reasoning_tokens": 0,
            "total_tokens": 150,
            "rates": {"input": 0.0004, "output": 0.002},
            "cost_usd": 49.9999,
        }
        usage_log.write_text(json.dumps(seed) + "\n", encoding="utf-8")

        env3 = dict(env)
        env3.update(
            {
                "USAGE_TRACKING": "1",
                "USAGE_LOG_PATH": str(usage_log),
                "PRICING_CACHE_PATH": str(pricing_cache),
                "PRICING_BASE_URL": f"http://127.0.0.1:{mock_port}",
            }
        )
        proxy3_port = free_port()
        proxy3_log = open(ROOT / "_proxy3.log", "w")
        proxy3 = subprocess.Popen(
            [PY, str(ROOT / "proxy.py"), "--port", str(proxy3_port)],
            stdout=proxy3_log,
            stderr=subprocess.STDOUT,
            cwd=ROOT,
            env=env3,
        )
        base3 = f"http://127.0.0.1:{proxy3_port}"
        check("usage: tracking proxy starts", wait_ready(f"{base3}/health"))
        client3 = httpx.Client(base_url=base3, timeout=30.0)

        def usage_records() -> list[dict]:
            return [
                json.loads(line)
                for line in usage_log.read_text().splitlines()
                if line.strip()
            ]

        # Non-streaming: token counts + cost from mock pricing (cached split).
        mock_in = (8 * 0.0004 + 4 * 0.00004 + 8 * 0.002) / 100
        r = client3.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-5.6-sol",
                "messages": [{"role": "user", "content": "Hi"}],
                "mock_cached_tokens": 4,
            },
        )
        check("usage: non-stream request ok", r.status_code == 200, r.text)
        sol = [x for x in usage_records() if x["model"] == "gpt-5.6-sol"]
        check("usage: one non-stream record", len(sol) == 1, str(len(sol)))
        if sol:
            s = sol[0]
            check(
                "usage: record identifies model/provider",
                s["upstream"] == "upstream-sol"
                and s["provider"] == "openai"
                and s["stream"] is False,
                str(s),
            )
            check(
                "usage: token counts incl. cached",
                s["prompt_tokens"] == 12
                and s["cached_tokens"] == 4
                and s["completion_tokens"] == 8,
                str(s),
            )
            check(
                "usage: cost computed from pricing API rates",
                s["cost_usd"] is not None and abs(s["cost_usd"] - mock_in) < 1e-9,
                str(s.get("cost_usd")),
            )
            check(
                "usage: applied rates stored on record",
                bool(s.get("rates")) and s["rates"]["input"] == 0.0004,
                str(s.get("rates")),
            )

        # Streaming: usage captured from the final SSE chunk.
        mock_nosplit = (12 * 0.0004 + 8 * 0.002) / 100
        r = client3.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-5.6-terra",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
        check("usage: stream request ok", r.status_code == 200, r.text)
        terr = [x for x in usage_records() if x["model"] == "gpt-5.6-terra"]
        check(
            "usage: stream record from final usage chunk",
            len(terr) == 1
            and terr[0]["stream"] is True
            and terr[0]["cost_usd"] is not None
            and abs(terr[0]["cost_usd"] - mock_nosplit) < 1e-9,
            str(terr),
        )

        # Unknown pricing: request succeeds, cost stored as null.
        r = client3.post(
            "/v1/chat/completions",
            json={
                "model": "claude-fable-5-1",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        check("usage: unpriced model request ok", r.status_code == 200, r.text)
        fab = [x for x in usage_records() if x["model"] == "claude-fable-5-1"]
        check(
            "usage: unpriced model recorded with cost null",
            len(fab) == 1 and fab[0]["cost_usd"] is None,
            str(fab),
        )

        # gemini is priced under its catalog name (pricing_model override).
        r = client3.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-3.8-flash",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        check("usage: gemini request ok", r.status_code == 200, r.text)
        gem = [x for x in usage_records() if x["model"] == "gemini-3.8-flash"]
        mock_gem = (12 * 0.000075 + 8 * 0.000375) / 100
        check(
            "usage: gemini priced under gt-128k catalog name",
            len(gem) == 1
            and gem[0]["pricing_model"] == "gemini-3.8-flash-gt-128k"
            and gem[0]["cost_usd"] is not None
            and abs(gem[0]["cost_usd"] - mock_gem) < 1e-9,
            str(gem),
        )

        # Responses API: input_tokens/output_tokens are normalised into the
        # same canonical schema, so cost matches the chat equivalent.
        r = client3.post(
            "/v1/responses",
            json={
                "model": "gpt-6-luna",
                "input": "Hi",
                "mock_cached_tokens": 4,
            },
        )
        check("usage: responses request ok", r.status_code == 200, r.text)
        luna = [x for x in usage_records() if x["model"] == "gpt-6-luna"]
        check("usage: responses recorded once", len(luna) == 1, str(len(luna)))
        if luna:
            ln = luna[0]
            check(
                "usage: responses tokens normalised to chat schema",
                ln["prompt_tokens"] == 12
                and ln["cached_tokens"] == 4
                and ln["completion_tokens"] == 8
                and ln["reasoning_tokens"] == 3,
                str(ln),
            )
            check(
                "usage: responses cost equals chat equivalent",
                ln["cost_usd"] is not None and abs(ln["cost_usd"] - mock_in) < 1e-9,
                str(ln.get("cost_usd")),
            )

        # gpt-6.1-sol is priced under its catalog name (the deployment name
        # 404s upstream, which previously left cost_usd null).
        r = client3.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-6.1-sol",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        check("usage: gpt-6.1-sol request ok", r.status_code == 200, r.text)
        s61 = [x for x in usage_records() if x["model"] == "gpt-6.1-sol"]
        mock_61 = (12 * 0.0004 + 8 * 0.002) / 100
        check(
            "usage: gpt-6.1-sol priced under gpt-6-sol catalog name",
            len(s61) == 1
            and s61[0]["pricing_model"] == "gpt-6-sol"
            and s61[0]["cost_usd"] is not None
            and abs(s61[0]["cost_usd"] - mock_61) < 1e-9,
            str(s61),
        )

        # Cache-write tokens are a priced subset of the input: plain input,
        # cache-write and cache-read are each billed at their own rate.
        r = client3.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-6-astra",
                "messages": [{"role": "user", "content": "Hi"}],
                "mock_cached_tokens": 4,
                "mock_cache_write_tokens": 3,
            },
        )
        check("usage: cache-write request ok", r.status_code == 200, r.text)
        cw = [x for x in usage_records() if x["model"] == "gpt-6-astra"]
        mock_cw = (5 * 0.0004 + 3 * 0.0005 + 4 * 0.00004 + 8 * 0.002) / 100
        check(
            "usage: cache_write_tokens tracked and billed at its own rate",
            len(cw) == 1
            and cw[0]["cache_write_tokens"] == 3
            and cw[0]["cached_tokens"] == 4
            and cw[0]["cost_usd"] is not None
            and abs(cw[0]["cost_usd"] - mock_cw) < 1e-9,
            str(cw),
        )
        check(
            "usage: cache-write rate stored on the record",
            (cw[0].get("rates") or {}).get("cache_write_input") == 0.0005,
            str(cw[0].get("rates")),
        )

        # GET /v1/usage and /usage aggregation (seed + 7 tracked requests).
        spent = (
            49.9999 + mock_in + mock_nosplit + mock_gem + mock_in + mock_61 + mock_cw
        )
        r = client3.get("/v1/usage")
        rep = r.json() if r.status_code == 200 else {}
        check("usage: GET /v1/usage -> 200", r.status_code == 200, r.text)
        check(
            "usage: month, request count and spend",
            rep.get("month") == month_now
            and rep.get("requests") == 8
            and abs(rep.get("spent_usd", 0) - spent) < 1e-6,
            str(rep),
        )
        check(
            "usage: budget/remaining math",
            rep.get("budget_usd") == 250.0
            and abs(rep.get("remaining_usd", 0) - (250.0 - spent)) < 1e-6
            and rep.get("over_budget") is False,
            str(rep),
        )
        check(
            "usage: per-model rows",
            {m["model"] for m in rep.get("models", [])}
            == {
                "seed-model",
                "gpt-5.6-sol",
                "gpt-5.6-terra",
                "gpt-6-luna",
                "gpt-6.1-sol",
                "gpt-6-astra",
                "claude-fable-5-1",
                "gemini-3.8-flash",
            },
            str(rep.get("models")),
        )
        check(
            "usage: unpriced requests counted",
            rep.get("requests_without_pricing") == 1,
            str(rep.get("requests_without_pricing")),
        )
        r = client3.get("/usage")
        check(
            "usage: bare /usage aggregates the same",
            r.status_code == 200 and r.json().get("requests") == 8,
            r.text,
        )
        r = client3.get("/v1/usage", params={"month": "2020-01"})
        check(
            "usage: other month is empty",
            r.status_code == 200 and r.json().get("requests") == 0,
            r.text,
        )
        r = client3.get("/v1/usage", params={"month": "nope"})
        check(
            "usage: invalid month -> 400",
            r.status_code == 400 and r.json()["error"].get("param") == "month",
            r.text,
        )

        # Budget milestone warning (seed spend 49.99 crossed $50.00).
        p3log = (ROOT / "_proxy3.log").read_text(encoding="utf-8", errors="replace")
        check(
            "usage: $50 crossing warning logged",
            "crossed $50.00" in p3log,
            p3log[-300:],
        )
        check(
            "usage: missing-pricing warning logged",
            "no pricing for anthropic/upstream-fable" in p3log,
            p3log[-300:],
        )

        # Pricing cache persisted on disk, incl. the negative anthropic entry.
        try:
            cache = json.loads(pricing_cache.read_text(encoding="utf-8"))
        except OSError:
            cache = {}
        check(
            "usage: pricing cache populated (incl. negative)",
            "openai/upstream-sol" in cache
            and (cache.get("anthropic/upstream-fable") or {}).get("rates") == {},
            str(sorted(cache)),
        )

        # ------------------------------------------------------------ #
        # 10. NRP api_source: Bearer auth + Responses-via-chat bridge ---
        # ------------------------------------------------------------ #
        # The default proxy has no NRP key; a dedicated proxy points
        # NRP_BASE_URL at the mock (the mock ignores auth but records the
        # headers, which is exactly what the 403 regression needs).
        # The registry always lists the aliases; without a key, requests to
        # them are rejected with 401 (the client is never built).
        r = client.get("/v1/models")
        check(
            "nrp: aliases listed even without a key",
            r.status_code == 200
            and {"qwen3", "deepseek-v4-flash", "glm-5"}
            <= {m["id"] for m in r.json()["data"]},
            str(sorted(m["id"] for m in r.json()["data"]))[:200],
        )
        r = client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "x"}]},
        )
        check(
            "nrp: chat without NRP_API_KEY -> 401",
            r.status_code == 401 and "NRP_API_KEY" in r.text,
            r.text,
        )
        r = client.post("/v1/responses", json={"model": "glm-5", "input": "x"})
        check(
            "nrp: responses without NRP_API_KEY -> 401",
            r.status_code == 401,
            r.text,
        )

        env4 = dict(env)
        env4.update(
            {
                "NRP_API_KEY": "nrp-test-key",
                "NRP_BASE_URL": f"http://127.0.0.1:{mock_port}/v1",
            }
        )
        proxy4_port = free_port()
        stale4 = ROOT / "_proxy4.log"
        if stale4.exists():
            stale4.unlink()
        proxy4_log = open(stale4, "w")
        proxy4 = subprocess.Popen(
            [PY, str(ROOT / "proxy.py"), "--port", str(proxy4_port)],
            stdout=proxy4_log,
            stderr=subprocess.STDOUT,
            cwd=ROOT,
            env=env4,
        )
        base4 = f"http://127.0.0.1:{proxy4_port}"
        wait_ready(base4)
        client4 = httpx.Client(base_url=base4, timeout=30.0)

        r = client4.get("/v1/models")
        check(
            "nrp: models listed with NRP_API_KEY",
            r.status_code == 200
            and {"qwen3", "deepseek-v4-flash", "glm-5"}
            <= {m["id"] for m in r.json()["data"]},
            str(sorted(m["id"] for m in r.json()["data"]))[:200],
        )

        # --- plain chat: Bearer auth, no portkey headers, alias rewrite ---
        r = client4.post(
            "/v1/chat/completions",
            json={
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        cap = json.loads(CAPTURE.read_text())
        check(
            "nrp: chat returns 200",
            r.status_code == 200
            and r.json()["choices"][0]["message"]["content"]
            == "Hello from mock upstream!",
            r.text,
        )
        check(
            "nrp: chat rewrites model to alias",
            r.json().get("model") == "deepseek-v4-flash",
            str(r.json().get("model")),
        )
        check(
            "nrp: chat uses Bearer auth with NRP key",
            cap["headers"].get("authorization") == "Bearer nrp-test-key",
            str(cap["headers"].get("authorization"))[:20],
        )
        check(
            "nrp: no x-portkey-* headers leak to NRP",
            not any(k.startswith("x-portkey") for k in cap["headers"]),
            str([k for k in cap["headers"] if k.startswith("x-portkey")]),
        )
        check(
            "nrp: chat rewrites alias to upstream model",
            cap["body"].get("model") == "deepseek-v4-flash",
            str(cap["body"].get("model")),
        )

        # --- responses -> chat bridge: non-streaming, text ---------------
        r = client4.post(
            "/v1/responses",
            json={"model": "deepseek-v4-flash", "input": "Hi"},
        )
        cap = json.loads(CAPTURE.read_text())
        check(
            "nrp: responses bridged to chat endpoint",
            r.status_code == 200 and cap["path"] == "/v1/chat/completions",
            str(cap.get("path")),
        )
        check(
            "nrp: bridged input becomes chat messages",
            cap["body"].get("messages") == [{"role": "user", "content": "Hi"}],
            str(cap["body"].get("messages")),
        )
        check(
            "nrp: bridged response is a Responses envelope",
            r.json().get("object") == "response"
            and r.json().get("status") == "completed"
            and r.json().get("model") == "deepseek-v4-flash"
            and r.json()["output"][0]["content"][0]["text"]
            == "Hello from mock upstream!",
            json.dumps(r.json())[:300],
        )
        check(
            "nrp: bridged envelope keeps upstream id and resp_ fallback works",
            r.json().get("id") == "cmpl-mock-0001",
            str(r.json().get("id")),
        )

        # --- responses bridge: tools + tool results ----------------------
        r = client4.post(
            "/v1/responses",
            json={
                "model": "deepseek-v4-flash",
                "instructions": "Be brief.",
                "input": [
                    {"role": "user", "content": "Weather?"},
                    {
                        "type": "function_call",
                        "name": "get_weather",
                        "call_id": "call_1",
                        "arguments": '{"city":"Princeton"}',
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "21C and sunny",
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    }
                ],
                "tool_choice": "auto",
                "max_output_tokens": 512,
                "reasoning": {"effort": "high"},
                "mock_chat_tool_call": True,
            },
        )
        cap = json.loads(CAPTURE.read_text())
        check(
            "nrp: bridged tools re-nested under function",
            cap["body"]["tools"]
            == [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
            str(cap["body"].get("tools")),
        )
        check(
            "nrp: bridged instructions becomes system message",
            cap["body"]["messages"][0]
            == {"role": "system", "content": "Be brief."},
            str(cap["body"]["messages"][:1]),
        )
        check(
            "nrp: bridged function_call becomes assistant tool_calls",
            cap["body"]["messages"][2]
            == {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Princeton"}',
                        },
                    }
                ],
            },
            str(cap["body"]["messages"][2:3]),
        )
        check(
            "nrp: bridged function_call_output becomes tool message",
            cap["body"]["messages"][3]
            == {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "21C and sunny",
                "name": "get_weather",
            },
            str(cap["body"]["messages"][3:4]),
        )
        check(
            "nrp: bridged param renames (max_output_tokens, reasoning_effort)",
            cap["body"].get("max_completion_tokens") == 512
            and cap["body"].get("reasoning_effort") == "high"
            and "max_output_tokens" not in cap["body"]
            and "reasoning" not in cap["body"],
            str(cap["body"]),
        )
        out = r.json()["output"]
        check(
            "nrp: bridged tool call converted back to function_call item",
            out[0] == {
                "id": "fc_call_mock_1",
                "type": "function_call",
                "status": "completed",
                "call_id": "call_mock_1",
                "name": "get_weather",
                "arguments": '{"city": "Princeton"}',
            },
            json.dumps(out)[:300],
        )

        # Copilot resends assistant Responses output items as conversation history.
        for stream_history in (False, True):
            history = [
                {"role": "user", "content": [{"type": "input_text", "text": "Hi"}]},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello", "annotations": []}],
                },
                {"role": "assistant", "content": "Earlier reply"},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "refusal", "refusal": "Cannot do that."}],
                },
                {"role": "user", "content": "Continue"},
            ]
            r = client4.post(
                "/v1/responses",
                json={"model": "qwen3", "input": history, "stream": stream_history},
            )
            cap = json.loads(CAPTURE.read_text())
            messages = cap["body"]["messages"]
            check(
                f"nrp: history uses chat text/refusal parts (stream={stream_history})",
                messages[0]["content"] == [{"type": "text", "text": "Hi"}]
                and messages[1]["content"] == [{"type": "text", "text": "Hello"}]
                and messages[2]["content"] == [{"type": "text", "text": "Earlier reply"}]
                and messages[3]["content"] == [{"type": "refusal", "refusal": "Cannot do that."}],
                str(messages),
            )
            result = parse_sse(r.text)[-1] if stream_history else r.json()
            check(
                f"nrp: assistant history completes (stream={stream_history})",
                r.status_code == 200
                and (
                    result.get("type") == "response.completed"
                    if stream_history
                    else result.get("status") == "completed"
                ),
                r.text[-500:],
            )

        # --- responses bridge: streaming text -----------------------------
        r = client4.post(
            "/v1/responses",
            json={"model": "glm-5", "input": "Hi", "stream": True},
        )
        frames = parse_sse(r.text)
        types = [f.get("type") for f in frames]
        check(
            "nrp: stream is typed Responses events",
            r.status_code == 200 and types[0] == "response.created",
            str(types[:4]),
        )
        text = "".join(
            f.get("delta", "") for f in frames if f.get("type") == "response.output_text.delta"
        )
        check(
            "nrp: stream reassembles bridged text",
            text == "Hello from mock upstream!",
            repr(text),
        )
        check(
            "nrp: stream ends with response.completed",
            types[-1] == "response.completed"
            and frames[-1]["response"]["model"] == "glm-5",
            str(types[-2:]),
        )
        check(
            "nrp: stream has no [DONE] frame",
            not any(f.get("done") for f in frames),
            str(types[-1]),
        )
        check(
            "nrp: stream model rewritten to alias",
            frames[0]["response"]["model"] == "glm-5",
            json.dumps(frames[0])[:150],
        )

        # --- responses bridge: streaming tool call ------------------------
        r = client4.post(
            "/v1/responses",
            json={
                "model": "qwen3",
                "input": "Weather?",
                "stream": True,
                "mock_chat_tool_call": True,
            },
        )
        frames = parse_sse(r.text)
        types = [f.get("type") for f in frames]
        args = "".join(
            f.get("delta", "")
            for f in frames
            if f.get("type") == "response.function_call_arguments.delta"
        )
        check(
            "nrp: streamed tool call argument deltas reassemble",
            args == '{"city": "Princeton"}',
            repr(args),
        )
        check(
            "nrp: streamed tool call emits output_item.added",
            "response.output_item.added" in types,
            str(types[:6]),
        )
        completed = [f for f in frames if f.get("type") == "response.completed"]
        fc = [
            item
            for item in completed[0]["response"]["output"]
            if item.get("type") == "function_call"
        ]
        check(
            "nrp: completed response carries function_call item",
            len(fc) == 1
            and fc[0]["name"] == "get_weather"
            and fc[0]["call_id"] == "call_mock_1",
            json.dumps(completed[0]["response"]["output"])[:250],
        )

        # --- chat client with tool history still works via plain chat -----
        r = client4.post(
            "/v1/chat/completions",
            json={
                "model": "deepseek-v4-flash",
                "messages": [
                    {"role": "user", "content": "Weather?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "get_weather", "parameters": {}},
                    }
                ],
            },
        )
        cap = json.loads(CAPTURE.read_text())
        check(
            "nrp: chat tool round trip forwarded untouched",
            r.status_code == 200 and len(cap["body"]["messages"]) == 3,
            str(cap["body"].get("messages"))[:200],
        )

        # --- missing NRP key on the dedicated proxy: 401 path ------------
        r = client4.post(
            "/v1/responses",
            json={"model": "gpt-5.6-sol", "input": "Hi"},
        )
        check(
            "nrp proxy: portkey models still work",
            r.status_code == 200 and r.json()["object"] == "response",
            r.text[:150],
        )

        p4log = (ROOT / "_proxy4.log").read_text(encoding="utf-8", errors="replace")
        check(
            "nrp: bridge log line emitted",
            "responses request bridged via chat" in p4log,
            p4log[-300:],
        )
        # Match the actual 403 signals (httpx's "403 Forbidden" response line,
        # the proxy's "status=403" warning). A bare "403" substring false-positives
        # on uvicorn PIDs/ports, e.g. "Started server process [40363]".
        check(
            "nrp: no 403 in proxy log",
            "403 Forbidden" not in p4log and "status=403" not in p4log,
            [ln for ln in p4log.splitlines() if "403 Forbidden" in ln or "status=403" in ln][:3],
        )

        client4.close()

    finally:
        procs = [(proxy, proxy_log), (mock, mock_log)]
        if proxy2 is not None:
            procs.append((proxy2, proxy2_log))
        if proxy3 is not None:
            procs.append((proxy3, proxy3_log))
        if proxy4 is not None:
            procs.append((proxy4, proxy4_log))
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
        for leftover in (ROOT / "_usage_test.jsonl", ROOT / "_pricing_cache_test.json"):
            if leftover.exists():
                leftover.unlink()

    print()
    print(f"  {len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("FAILED:", ", ".join(FAILED))
        return 1
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
