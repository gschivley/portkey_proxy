# Portkey → OpenAI-compatible local proxy for GitHub Copilot

A small local HTTP proxy that lets the GitHub Copilot app talk to the
Princeton AI Sandbox (Portkey gateway). It exposes an OpenAI-compatible
API and translates requests into calls to the official `portkey_ai`
Python SDK, translating responses (including streaming SSE) back into
OpenAI JSON/SSE.

Exposed models (by default):

| Alias (what Copilot requests) | Upstream model |
|-------------------------------|----------------|
| `gpt-5.6-sol`                 | `gpt-5.6-sol`  |
| `gpt-5.6-terra`               | `gpt-5.6-terra`|
| `gpt-5.6-luna`                | `gpt-5.6-luna` |
| `gpt-6-astra`                 | `gpt-6-astra`  |
| `gemini-3.8-flash`            | `gemini-3.8-flash` |
| `claude-opus-4-8`             | `claude-opus-4-8` |
| `claude-fable-5`              | `claude-fable-5` |

The aliases are passed through to Portkey as-is. If an alias is not a real
deployment name on the sandbox, map it to a real model by setting the matching
`*_UPSTREAM_MODEL` variable (e.g. `GPT56_SOL_UPSTREAM_MODEL`,
`GPT6_ASTRA_UPSTREAM_MODEL`, `GEMINI38_FLASH_UPSTREAM_MODEL`,
`CLAUDE_OPUS_48_UPSTREAM_MODEL`, `CLAUDE_FABLE_5_UPSTREAM_MODEL`; or editing
`MODEL_REGISTRY` in `proxy.py`) — e.g. `claude-sonnet-4-6`.

## Endpoints

- `POST /v1/chat/completions` (also mounted at `/chat/completions`) — OpenAI chat completions, streaming (`"stream": true`) and non-streaming.
- `GET /v1/models` (also at `/models`) — lists the available models.
- `GET /health`, `GET /` — liveness info.
- `POST /v1/responses` — returns a 404 with a clear message (the Sandbox does not support the Responses API).

Everything requested is forwarded: parameters (`temperature`, `max_tokens`,
`top_p`, `tools`, `tool_choice`, `stop`, `seed`, `response_format`, …),
model names, errors (translated to OpenAI error JSON with the upstream status code),
and usage metadata (including in the final streaming chunk when
`INCLUDE_STREAM_USAGE=1`).

**Authentication**: the API key you enter in the Copilot app's model provider
settings is sent with every request as `Authorization: Bearer <key>`; the
proxy reads it and forwards it to Portkey through the SDK (as
`x-portkey-api-key`). `PORTKEY_API_KEY` in `.env` is an *optional* fallback
used only for requests that carry no key — requests with neither get a 401.

## Setup

```bash
# 1. Create a virtualenv and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. (Optional) add a fallback key to .env — not needed if you enter the
#    key in the Copilot app:
#    PORTKEY_API_KEY=<your key>

# 3. Start the proxy
python proxy.py
```

You should see:

```
 Portkey OpenAI-compatible proxy
   Local endpoint : http://127.0.0.1:8090/v1
   ...
```

## Point the Copilot app at the proxy

In the model provider settings:

| Setting  | Value |
|----------|-------|
| Base URL | `http://127.0.0.1:8090/v1` |
| Wire API | **Completions** |
| API key  | your Portkey / AI Sandbox key |

Use **Completions**, not Responses: the AI Sandbox only supports the
`chat.completions` method — per the Princeton FAQ, *"The Responses API from
the newer SDK is not yet supported and attempts to use this may result in a
404 error."* The proxy implements `POST /v1/chat/completions` (streaming and
non-streaming) and returns a clear 404 on `/v1/responses`.

For GitHub Copilot CLI the same settings map to environment variables:

```bash
export COPILOT_PROVIDER_TYPE=openai
export COPILOT_PROVIDER_BASE_URL=http://127.0.0.1:8090/v1
export COPILOT_PROVIDER_API_KEY=<your Portkey / AI Sandbox key>
export COPILOT_MODEL=gpt-5.6-sol        # or gpt-5.6-terra / gpt-5.6-luna /
                                        # gpt-6-astra / gemini-3.8-flash /
                                        # claude-opus-4-8 / claude-fable-5
copilot
```

Notes:

- The proxy serves its routes at both `/v1/chat/completions` and
  `/chat/completions`, so a base URL with or without `/v1` both work.
- Copilot requires models to support **tool calling and streaming**.
  If the upstream model errors on `tools`, the error is forwarded verbatim.

## Quick sanity check

```bash
# Non-streaming
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-5.6-sol","messages":[{"role":"user","content":"Say hi"}]}'

# Streaming
curl -N http://127.0.0.1:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-5.6-terra","stream":true,"messages":[{"role":"user","content":"Say hi"}]}'
```

## Tests

Runs the proxy against a mock upstream (no real key needed):

```bash
./.venv/bin/python test_proxy.py
```

## Troubleshooting

Reasoning (`reasoning_effort`) is handled **per model**, because the upstreams
disagree about it. Each model in `MODEL_REGISTRY` declares a profile from
`REASONING_PROFILES`:

| Profile | Models | Behaviour |
| --------- | -------- | ----------- |
| `gpt-5.6` | `gpt-5.6-sol/terra/luna` | Rejects function `tools` + reasoning together. The proxy strips reasoning when tools are present, and auto-retries with `reasoning_effort: "none"` if the upstream still complains. `"none"` **is** a valid value here. |
| `gpt-6` | `gpt-6-astra` | Accepts tools + reasoning, but `"none"` is **not** a valid effort — supported values are `low`, `medium`, `high`, `xhigh`. The proxy leaves reasoning intact and rewrites any unsupported value (e.g. a client-sent `"none"`) to the fallback `low`. |
| `flexible` | `gemini-3.8-flash` | Accepts tools + reasoning and its own effort vocabulary, so values are forwarded untouched unless you configure an explicit list. |

**"Function tools with reasoning_effort are not supported ... set reasoning_effort to 'none'"**

The GPT-5.6-family upstream rejects `/v1/chat/completions` requests that
combine function `tools` with reasoning, and Copilot sends both. Since
`/v1/responses` is not supported by the Sandbox, the proxy handles this
automatically (only for the `gpt-5.6` profile):

1. When function tools are present, it removes the reasoning parameters
   (`reasoning_effort` / `reasoning.effort`) from the request.
2. If the upstream still rejects that with the tools+reasoning 400, it
   retries once with an explicit `reasoning_effort: "none"` — covering
   upstreams that require the value to be present.

Both steps are visible in the proxy log (`chat request: ...`,
`Retrying with ...`). Step 1 is controlled by `TOOLS_REASONING_EFFORT` in
`.env`: `strip` (default) | `none` (send an explicit `"none"` first) |
`keep` (forward unchanged, reproduces the 400). This setting never forces
`"none"` onto a model that rejects it (e.g. `gpt-6-astra`).

**"'reasoning_effort' does not support 'none' with this model"**

That's `gpt-6-astra` (or another model whose vocabulary excludes `"none"`). The
proxy rewrites unsupported values to the model's fallback automatically. If a
model's real vocabulary differs, set it explicitly, e.g.:

```bash
GPT6_ASTRA_REASONING_EFFORTS=low,medium,high,xhigh
GEMINI38_FLASH_REASONING_EFFORTS=minimal,low,medium,high
```

The first configured non-`none` value becomes the fallback for rejected inputs.

**Still seeing the error?**

- Confirm you are on the current build: `curl http://127.0.0.1:8090/health`
  must report a `"version"` field (v1.1.0+ has the handling above).
- Reproduce the failing request and check the proxy log. If there is no
  `chat request: ...` line for it, the app is not sending that request to
  the proxy — check the provider base URL and that the provider/model is
  actually selected in the app.
- If the log shows the retry also failing (`Retrying with ...` followed by
  the same upstream 400), the upstream rejects both representations; share
  those log lines.

**"408 Request exceeded the timeout sent in the request: 600ms"**

The proxy was sending a per-request timeout of `600` to the Portkey gateway,
which is interpreted as **600 milliseconds** (`x-portkey-request-timeout:
600`), when it was meant as seconds. v1.2.0 no longer sends a timeout header
by default, so the gateway's own default applies. If you want to control it,
set `PORTKEY_TIMEOUT_MS` in **milliseconds** (e.g. `300000` = 5 min). Check
the banner (`Gateway timeout: ...`) or the started log to confirm.

## Configuration reference

| Variable | Default | Meaning |
| ---------- | --------- | --------- |
| `PORTKEY_API_KEY` | – (unset) | Optional fallback key for requests that send no API key; `AI_SANDBOX_KEY` also accepted |
| `PORTKEY_BASE_URL` | `https://api.portkey.ai/v1` | Upstream gateway base URL (`/v1` appended if missing) |
| `PROXY_HOST` / `PROXY_PORT` | `127.0.0.1` / `8090` | Where the proxy listens |
| `PORTKEY_TIMEOUT_MS` | – (unset) | Per-request upstream timeout sent to the Portkey gateway, in **milliseconds**. Leave unset so the gateway default applies. Do NOT use seconds (e.g. `600` = 600 ms) — that caused `408 Request exceeded the timeout` |
| `INCLUDE_STREAM_USAGE` | `1` | Send `stream_options.include_usage` to get token usage in streaming |
| `TOOLS_REASONING_EFFORT` | `strip` | For models that reject tools + reasoning (GPT-5.6 only): `strip`, `none`, or `keep` (the alternative is auto-tried on the upstream's tools+reasoning 400) |
| `GPT56_SOL_UPSTREAM_MODEL` / `GPT56_TERRA_UPSTREAM_MODEL` / `GPT56_LUNA_UPSTREAM_MODEL` / `GPT6_ASTRA_UPSTREAM_MODEL` / `GEMINI38_FLASH_UPSTREAM_MODEL` | alias name | Map an alias to a different upstream model |
| `GPT6_ASTRA_REASONING_EFFORTS` | `low,medium,high,xhigh` | Comma-separated `reasoning_effort` values the upstream accepts for `gpt-6-astra`; unsupported client values are rewritten to the first non-`none` entry |
| `GEMINI38_FLASH_REASONING_EFFORTS` | – (unset) | Optional comma-separated `reasoning_effort` list for `gemini-3.8-flash`; when unset, client values are forwarded untouched |
