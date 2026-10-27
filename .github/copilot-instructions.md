# Copilot Instructions

## Project overview

A single-file local HTTP proxy (`proxy.py`) that lets GitHub Copilot talk to the
Princeton AI Sandbox (Portkey gateway). It exposes an OpenAI-compatible API
(`POST /v1/chat/completions`, `POST /v1/responses`, `GET /v1/models`) and
translates requests into calls to the `portkey_ai` Python SDK, converting
responses (including streaming SSE) back into OpenAI JSON/SSE.

## Commands

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the proxy (serves on 127.0.0.1:8090)
./.venv/bin/python proxy.py

# Run the full integration test suite (222 checks; boots a mock upstream + the real proxy)
./.venv/bin/python test_proxy.py
```

- The test suite is a single monolithic script (`test_proxy.py`), **not** pytest —
  `pytest test_proxy.py` collects nothing. There is no per-test filter; to run a
  subset, temporarily comment out sections in `main()`.
- No linter or formatter is configured; there is no `pyproject.toml` or `Makefile`.

## Architecture

- `proxy.py` — everything: `.env` loading, model registry, reasoning handling,
  client cache (Portkey SDK + plain OpenAI client for NRP), FastAPI app (chat
  completions + Responses API), the Responses→chat bridge for `responses_via_chat`
  models, error translation, `main()`.
- `usage.py` — usage & cost tracking (imported by `proxy.py` and `usage_report.py`).
  Reads its config at import time (after `.env` loading). `PricingCache` fetches
  `GET {PRICING_BASE_URL}/pricing/<provider>/<model>` (public Portkey API) with a
  file-backed TTL cache and negative caching; `record()` appends one JSONL line per
  call to `usage.jsonl` and emits budget-milestone warnings; `monthly_report()`
  aggregates by model for a `YYYY-MM` bucket. Cost formula:
  `tokens * price / 100` — the API returns the per-token USD price scaled x100.
  Input is split three ways because `cached_tokens` and `cache_write_tokens` are
  both *subsets* of `prompt_tokens`: plain input, cache-write and cache-read are
  each billed at their own catalog rate (a missing rate falls back to plain input).
- `usage_report.py` — CLI over the same JSONL log (works while the proxy is stopped).
- `mock_upstream.py` — mock OpenAI-compatible upstream used by the tests. Records
  every request to `_captured.json` and supports test-only "personality" modes via
  a `mock_upstream_mode` body field (popped before forwarding). Serves both
  `/v1/chat/completions` and `/v1/responses` (the responses route deliberately
  accepts tools + reasoning, matching the real upstream on that endpoint, and
  rejects chat-only `stream_options` the way the real one does). Test-only
  `mock_tool_call` makes the responses route emit a `function_call` item (with
  streamed argument deltas) instead of a message; `mock_chat_tool_call` does the
  same on the chat route. `_captured.json` also records the request `path`. Also serves
  `GET /pricing/<provider>/<model>` (fixed `openai`/`google` rates; `anthropic`
  404s to exercise the no-pricing path) and honors a `mock_cached_tokens` field to
  report cached input tokens in `usage` and a `mock_cache_write_tokens` field to
  report cache-write input tokens.
- `test_proxy.py` — end-to-end integration test: starts `mock_upstream.py` and
  `proxy.py` as subprocesses on free ports, drives them with `httpx`, asserts with
  `check()`. Section 7 covers `/v1/responses`; section 8 covers the
  chat→Responses bridge (tool-call round trip, multi-turn tool history,
  `stream_options` dropping); section 9 boots a dedicated tracking
  proxy with `USAGE_TRACKING=1` pointed at isolated `_usage_test.jsonl` /
  `_pricing_cache_test.json` files; section 10 boots a proxy with
  `NRP_API_KEY`/`NRP_BASE_URL` pointed at the mock to cover the NRP Bearer auth
  and the Responses→chat bridge.
- `portkey_text_example.py` / `faqs.txt` — reference material from Princeton
  (official SDK example and AI Sandbox FAQ); not part of the proxy.

### Request flow

1. Copilot sends an OpenAI chat-completions (or Responses) request to the proxy.
2. `chat_completions` / `responses_create` validate the model against
   `MODEL_REGISTRY` via the shared `_validated_request` helper (aliases only;
   unknown → 404 `model_not_found`).
3. The API key is resolved per request: `Authorization: Bearer` →
   `x-portkey-api-key` → optional `PORTKEY_API_KEY`/`AI_SANDBOX_KEY` fallback.
   No key → 401.
4. All other body params are forwarded untouched; the alias is rewritten to the
   registry's `upstream` model. Exception: models flagged `sampling: False`
   (every `gpt-*` alias) get sampling knobs dropped / renamed first — see the
   sampling convention below.
5. The `portkey_ai` SDK call is made with a cached client (one per API key,
   max 16): `client.chat.completions.create` or `client.responses.create`.
6. Responses are converted back to OpenAI JSON/SSE; `model` is rewritten to the
   alias the client asked for; errors are translated to OpenAI-shaped error JSON.

## Key conventions

- **NRP is a second api_source, not Portkey**: registry entries with
  `api_source: "nrp"` (`qwen3`, `deepseek-v4-flash`, `glm-5`) are served by a
  plain `OpenAI` client (`get_nrp_client`, from `portkey_ai._vendor.openai`)
  with `Authorization: Bearer $NRP_API_KEY`; the Portkey SDK masks that header
  and sends `x-portkey-api-key`, which NRP rejects with 403. They are
  zero-cost: `track_usage: False` keeps them out of `usage.jsonl`. The vendored
  client's `chat.completions.create` has a keyword whitelist, so
  `_nrp_chat_kwargs` folds unknown body params into `extra_body` (Portkey
  does this itself).
- **Responses→chat bridge (`responses_via_chat: True`)**: NRP's `/v1/responses`
  is a broken passthrough (400s on Responses bodies), so `responses_create`
  converts the request via `_responses_to_chat_kwargs` (input items → messages,
  flat tools → nested, `function_call`/`function_call_output` → tool_calls/tool
  messages, `max_output_tokens` → `max_completion_tokens`,
  `reasoning.effort` → `reasoning_effort`, `text.format` → `response_format`),
  calls the chat endpoint, and converts the reply back via
  `_chat_to_responses_completion` / `_ChatToResponsesSSE` (typed events,
  `response.completed` terminator, no `[DONE]`). Log line:
  `responses request bridged via chat: ...`.
- **Routes are mounted twice**: every endpoint exists at both `/v1/...` and bare
  `/...` (e.g. `/v1/chat/completions` and `/chat/completions`, `/v1/responses`
  and `/responses`). Keep both when adding endpoints.
- **Two wire APIs, bridged for tools**: `/v1/chat/completions` and
  `/v1/responses` are both served and the *client* picks one. Never hand a chat
  client a `response` body (or vice versa) — the shapes are not interchangeable.
  The one sanctioned exception is `TOOLS_VIA_RESPONSES` (default `auto`): a chat
  request carrying function tools is *served* via `client.responses.create` and
  translated **back** into `chat.completion` / chat SSE by
  `_chat_to_responses_kwargs`, `_responses_to_chat_completion` and
  `_ResponsesToChatSSE`, so the client never knows. `auto` bridges only models
  with `tools_on_chat: False` (the `gpt-6` profile — `gpt-6-astra` and
  `gpt-6.1-sol` — which reject tools on chat in every shape; note `gpt-6-luna`
  is named like this family but uses the `gpt-5.6` profile and is not bridged);
  `always` also bridges GPT-5.6 (preserving its reasoning);
  `never` forwards the upstream 400. Requires `RESPONSES_ENABLED=1`. Log line:
  `chat request bridged via responses: ...`.
  Responses gotchas: function tools are **flat**
  (`{"type":"function","name",...,"parameters":...}`, not nested under
  `function`); reasoning is **nested only** (`reasoning.effort`, and
  `_responses_kwargs` normalises a chat-style flat `reasoning_effort` into it);
  usage reports `input_tokens`/`output_tokens` (normalised by
  `usage.extract_tokens` into the same JSONL schema as chat); the stream ends
  with a `response.completed` event and must **not** emit `data: [DONE]`; error
  events are flat `{"type":"error","message",...}`. `RESPONSES_ENABLED=0`
  restores the 404 stub. `previous_response_id` is forwarded untouched and is
  verified to work on the Sandbox.
- **Never leak chat-only params into a Responses call**: the upstream rejects
  them (`Unknown parameter: 'stream_options.include_usage'`). `stream_options`
  in particular is sent by chat clients *and* by `INCLUDE_STREAM_USAGE`, and is
  dropped by both `_responses_kwargs` and `_BRIDGE_DROPPED_PARAMS` — Responses
  reports usage on `response.completed` regardless. `mock_upstream.py` mirrors
  this rejection so the suite catches regressions.
- **Model aliases vs upstream models**: `MODEL_REGISTRY` maps the names Copilot
  requests (e.g. `gpt-5.6-sol`) to upstream deployment names. Each alias can be
  overridden with `*_UPSTREAM_MODEL` env vars. Add new models to `MODEL_REGISTRY`
  (and the README table). Each entry also declares a `provider` (`openai`/
  `anthropic`/`google`) used for the pricing lookup; if the deployment's pricing
  name differs from its `upstream`, set a `pricing_model` (e.g. `gemini-3.8-flash`
  -> `gemini-3.8-flash-gt-128k`, `gpt-6.1-sol` -> `gpt-6-sol`; both 404 under
  their deployment name and would otherwise log `cost_usd: null`).
- **Reasoning is per-model, not global**: `REASONING_PROFILES` captures three
  upstream quirks:
  - `gpt-5.6` — upstream rejects function `tools` + reasoning together **on
    /v1/chat/completions only** (it accepts them on /v1/responses).
    `TOOLS_REASONING_EFFORT` (`strip`/`none`/`keep`) controls the first attempt;
    if the upstream still returns the tools+reasoning 400, the proxy auto-retries
    once with the alternative variant (both streaming and non-streaming).
    `gpt-6-luna` uses this profile despite its name (verified live 2026-10-02:
    `"none"` is a valid effort and tools work on chat once reasoning is `"none"`,
    so no bridging is needed).
  - `gpt-6` — **`tools_on_chat: False`**: rejects function tools on chat
    completions in *every* shape (tools+reasoning 400s; `"none"` 400s because it
    is not a valid effort; omitting reasoning still 400s as the upstream applies
    an implicit default effort), so the strip/retry ladder cannot help and these
    requests are bridged via /v1/responses instead. Without tools, `"none"` is
    still rejected and unsupported effort values are rewritten to the model's
    fallback (first non-`none` configured value).
  - `flexible` — forwards effort values untouched unless an explicit
    `*_REASONING_EFFORTS` env list is configured.
  - When adding a model, pick a profile (or add one) and update
    `REASONING_PROFILES`/`MODEL_REGISTRY`; the tests in `test_proxy.py` cover all
    three behaviors.
- **Sampling is per-model (`sampling: False`)**: every `gpt-*` deployment
  rejects sampling knobs (verified live 2026-10-27): on chat completions
  `temperature` only accepts its default, `top_p`/`frequency_penalty`/
  `presence_penalty`/`stop` are refused, and `max_tokens` must be spelled
  `max_completion_tokens`; on `/v1/responses` only `temperature`/`top_p` are
  refused. `_normalize_chat_sampling` (called in `chat_completions` before
  kwargs are built, so it covers both the direct chat path and the
  tools-via-Responses bridge) and `_drop_unsupported_sampling` with
  `RESPONSES_SAMPLING_DROPS` (in `responses_create`) drop/rename them before
  the call, so clients that always send `temperature` (VSCode Copilot) keep
  working. Claude/Gemini/NRP forward values untouched.
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
  confirm "ALL TESTS PASSED" (222 checks).
