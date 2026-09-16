# Copilot Instructions

## Project overview

A single-file local HTTP proxy (`proxy.py`) that lets GitHub Copilot talk to the
Princeton AI Sandbox (Portkey gateway). It exposes an OpenAI-compatible API
(`POST /v1/chat/completions`, `GET /v1/models`) and translates requests into
calls to the `portkey_ai` Python SDK, converting responses (including streaming
SSE) back into OpenAI JSON/SSE.

## Commands

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the proxy (serves on 127.0.0.1:8090)
./.venv/bin/python proxy.py

# Run the full integration test suite (108 checks; boots a mock upstream + the real proxy)
./.venv/bin/python test_proxy.py
```

- The test suite is a single monolithic script (`test_proxy.py`), **not** pytest —
  `pytest test_proxy.py` collects nothing. There is no per-test filter; to run a
  subset, temporarily comment out sections in `main()`.
- No linter or formatter is configured; there is no `pyproject.toml` or `Makefile`.

## Architecture

- `proxy.py` — everything: `.env` loading, model registry, reasoning handling,
  Portkey client cache, FastAPI app, error translation, `main()`.
- `usage.py` — usage & cost tracking (imported by `proxy.py` and `usage_report.py`).
  Reads its config at import time (after `.env` loading). `PricingCache` fetches
  `GET {PRICING_BASE_URL}/pricing/<provider>/<model>` (public Portkey API) with a
  file-backed TTL cache and negative caching; `record()` appends one JSONL line per
  call to `usage.jsonl` and emits budget-milestone warnings; `monthly_report()`
  aggregates by model for a `YYYY-MM` bucket. Cost formula:
  `tokens * price / 100` — the API returns the per-token USD price scaled x100.
- `usage_report.py` — CLI over the same JSONL log (works while the proxy is stopped).
- `mock_upstream.py` — mock OpenAI-compatible upstream used by the tests. Records
  every request to `_captured.json` and supports test-only "personality" modes via
  a `mock_upstream_mode` body field (popped before forwarding). Also serves
  `GET /pricing/<provider>/<model>` (fixed `openai`/`google` rates; `anthropic`
  404s to exercise the no-pricing path) and honors a `mock_cached_tokens` field to
  report cached input tokens in `usage`.
- `test_proxy.py` — end-to-end integration test: starts `mock_upstream.py` and
  `proxy.py` as subprocesses on free ports, drives them with `httpx`, asserts with
  `check()`. Section 9 boots a dedicated tracking proxy with `USAGE_TRACKING=1`
  pointed at isolated `_usage_test.jsonl` / `_pricing_cache_test.json` files.
- `portkey_text_example.py` / `faqs.txt` — reference material from Princeton
  (official SDK example and AI Sandbox FAQ); not part of the proxy.

### Request flow

1. Copilot sends an OpenAI chat-completions request to the proxy.
2. `chat_completions` validates the model against `MODEL_REGISTRY` (aliases only;
   unknown → 404 `model_not_found`).
3. The API key is resolved per request: `Authorization: Bearer` →
   `x-portkey-api-key` → optional `PORTKEY_API_KEY`/`AI_SANDBOX_KEY` fallback.
   No key → 401.
4. All other body params are forwarded untouched; the alias is rewritten to the
   registry's `upstream` model.
5. The `portkey_ai` SDK call is made with a cached client (one per API key,
   max 16).
6. Responses are converted back to OpenAI JSON/SSE; `model` is rewritten to the
   alias the client asked for; errors are translated to OpenAI-shaped error JSON.

## Key conventions

- **Routes are mounted twice**: every endpoint exists at both `/v1/...` and bare
  `/...` (e.g. `/v1/chat/completions` and `/chat/completions`). Keep both when
  adding endpoints.
- **Model aliases vs upstream models**: `MODEL_REGISTRY` maps the names Copilot
  requests (e.g. `gpt-5.6-sol`) to upstream deployment names. Each alias can be
  overridden with `*_UPSTREAM_MODEL` env vars. Add new models to `MODEL_REGISTRY`
  (and the README table). Each entry also declares a `provider` (`openai`/
  `anthropic`/`google`) used for the pricing lookup; if the deployment's pricing
  name differs from its `upstream`, set a `pricing_model` (e.g. `gemini-3.8-flash`
  → `gemini-3.8-flash-gt-128k`).
- **Reasoning is per-model, not global**: `REASONING_PROFILES` captures three
  upstream quirks:
  - `gpt-5.6` — upstream rejects function `tools` + reasoning together.
    `TOOLS_REASONING_EFFORT` (`strip`/`none`/`keep`) controls the first attempt;
    if the upstream still returns the tools+reasoning 400, the proxy auto-retries
    once with the alternative variant (both streaming and non-streaming).
  - `gpt-6` — accepts tools + reasoning but rejects `"none"`; unsupported effort
    values are rewritten to the model's fallback (first non-`none` configured
    value).
  - `flexible` — forwards effort values untouched unless an explicit
    `*_REASONING_EFFORTS` env list is configured.
  - When adding a model, pick a profile (or add one) and update
    `REASONING_PROFILES`/`MODEL_REGISTRY`; the tests in `test_proxy.py` cover all
    three behaviors.
- **Errors are OpenAI-shaped**: `ApiError` + `_exception_details` map SDK
  exception class names (e.g. `RateLimitError` → 429) to
  `{"error": {message, type, param, code}}`. Streaming errors are emitted as SSE
  error frames followed by `[DONE]`.
- **Streaming**: `_sse_stream` re-emits SDK chunks as SSE; `data["model"]` is
  always rewritten to the alias; `INCLUDE_STREAM_USAGE=1` adds
  `stream_options.include_usage` so the final chunk carries usage.
- **`.env` loading**: `_load_env_files` reads `.env` from the script dir and cwd
  with `override=False` (existing env vars win). Config is read at import time —
  changing `.env` requires a restart.
- **`PORTKEY_TIMEOUT_MS` is milliseconds** (sent as `x-portkey-request-timeout`).
  Never use seconds: `600` means 600 ms and caused upstream 408s. Leave unset by
  default.
- **Auth is per-request**: the key from the Copilot app is forwarded as
  `x-portkey-api-key` to Portkey; `PORTKEY_API_KEY` is only a fallback for
  keyless requests.
- **Test-only hooks**: `mock_upstream.py` reads a `mock_upstream_mode` field from
  the request body to simulate upstream personalities; keep it out of forwarded
  params.
- **Logging**: the access-log middleware logs every request; the proxy logs
  `chat request: ...` lines that are the primary debugging signal. Use
  `logging.info`/`warning` with the existing format.
- **Python 3.10+**, `from __future__ import annotations`, type hints on public
  functions. No type checker is configured.

## Testing notes

- The test suite requires no real API key — it points `PORTKEY_BASE_URL` at the
  mock upstream.
- `_captured.json` (written by the mock) is how tests assert what the proxy
  actually forwarded; it's gitignored and deleted at the end of the run.
- Logs go to `_mock.log`, `_proxy.log`, `_proxy2.log` (gitignored).
- After changing proxy behavior, run `./.venv/bin/python test_proxy.py` and
  confirm "ALL TESTS PASSED" (108 checks).
