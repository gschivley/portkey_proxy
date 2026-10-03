# Portkey → OpenAI-compatible local proxy for GitHub Copilot

A small local HTTP proxy that lets the GitHub Copilot app talk to the
Princeton AI Sandbox (Portkey gateway). It exposes an OpenAI-compatible
API and translates requests into calls to the official `portkey_ai`
Python SDK, translating responses (including streaming SSE) back into
OpenAI JSON/SSE.

Optionally, `ROUTER_ENABLED=1` adds three virtual aliases (`auto-free`,
`auto-hard`, `auto-review`) that prefer the zero-cost NRP models and reserve the
paid Portkey models for hard work, with advisory budget gates — see
[Free-first model router](#free-first-model-router-opt-in).

Exposed models (by default):

| Alias (what Copilot requests) | Upstream model |
|-------------------------------|----------------|
| `gpt-5.6-sol`                 | `gpt-5.6-sol`  |
| `gpt-5.6-terra`               | `gpt-5.6-terra`|
| `gpt-6-astra`                 | `gpt-6-astra`  |
| `gpt-6.1-sol`                 | `gpt-6.1-sol`  |
| `gpt-6-luna`                  | `gpt-6-luna`   |
| `gemini-3.8-flash`            | `gemini-3.8-flash` |
| `claude-opus-5-5`             | `claude-opus-5-5` |
| `claude-sonnet-5-5`           | `claude-sonnet-5-5` |
| `claude-fable-5-1`            | `claude-fable-5-1` |
| `qwen3`                       | `qwen3` (NRP)  |
| `deepseek-v4-flash`           | `deepseek-v4-flash` (NRP) |
| `glm-5`                       | `glm-5` (NRP)  |

The Portkey aliases are passed through to Portkey as-is. If an alias is not a
real deployment name on the sandbox, map it to a real model by setting the
matching `*_UPSTREAM_MODEL` variable (e.g. `GPT56_SOL_UPSTREAM_MODEL`,
`GPT6_ASTRA_UPSTREAM_MODEL`, `GPT6_SOL_UPSTREAM_MODEL`, `GEMINI38_FLASH_UPSTREAM_MODEL`,
`CLAUDE_OPUS_55_UPSTREAM_MODEL`, `CLAUDE_SONNET_55_UPSTREAM_MODEL`,
`CLAUDE_FABLE_51_UPSTREAM_MODEL`; or editing
`MODEL_REGISTRY` in `proxy.py`) — e.g. `claude-sonnet-4-6`.

The `qwen3`, `deepseek-v4-flash`, and `glm-5` aliases are routed to the NRP
gateway (`https://ellm.nrp-nautilus.io/v1`) instead of Portkey. They use the
`NRP_API_KEY` from the environment (never the per-request key), and because
they are zero-cost they are excluded from the usage report (`track_usage=False`
in `MODEL_REGISTRY`). Their upstream names can be overridden with
`QWEN3_UPSTREAM_MODEL`, `DEEPSEEK_V4_FLASH_UPSTREAM_MODEL`, and
`GLM5_UPSTREAM_MODEL`.

Two NRP quirks are handled transparently:

- **Auth.** NRP is a vanilla OpenAI-compatible gateway expecting
  `Authorization: Bearer <key>`. The Portkey SDK instead masks that header and
  sends the key as `x-portkey-api-key`, which NRP rejects with 403 — so NRP
  requests go through a plain OpenAI client (`get_nrp_client`) instead of the
  SDK.
- **No working Responses API.** NRP exposes `/v1/responses` but does not
  implement it (its gateway forwards Responses bodies to chat-completions
  validation and 400s). For registry entries with `responses_via_chat: True`,
  the proxy converts the Responses request into chat-completions parameters
  (`input` items → `messages`, `input_text`/`output_text` → chat `text` parts
  for every role (including assistant history), refusal parts preserved,
  flat tools → nested, `function_call` /
  `function_call_output` → `tool_calls` / tool messages,
  `max_output_tokens` → `max_completion_tokens`, `reasoning.effort` →
  `reasoning_effort`), calls `/v1/chat/completions`, and translates the reply —
  JSON or SSE — back into Responses shapes. Copilot keeps talking Responses and
  never learns the difference.

## Endpoints

- `POST /v1/chat/completions` (also at `/chat/completions`) — OpenAI chat completions, streaming (`"stream": true`) and non-streaming.
- `POST /v1/responses` (also at `/responses`) — OpenAI Responses API, streaming and non-streaming. See [Completions vs Responses](#completions-vs-responses).
- `GET /v1/models` (also at `/models`) — lists the available models.
- `GET /v1/router` (also at `/router`) — live router policy: pools, gates, spend, exhausted scopes (404 when the router is off).
- `POST /v1/router/reset` (also at `/router/reset`) — clear verified quota-exhaustion markers for this process.
- `GET /v1/usage` (also at `/usage`) — monthly usage & cost report (`?month=YYYY-MM`).
- `GET /health`, `GET /` — liveness info.

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
| Wire API | **Completions** (recommended) |
| API key  | your Portkey / AI Sandbox key |

Keep **Completions** for the Copilot app: it is the wire API this proxy is
verified against, and Copilot's own tool-calling loop speaks chat completions.
The proxy also serves `POST /v1/responses` (see
[Completions vs Responses](#completions-vs-responses)) for clients that prefer
that API.

> The Princeton FAQ still states *"The Responses API from the newer SDK is not
> yet supported and attempts to use this may result in a 404 error."* That is
> out of date: `client.responses.create` was verified working against the
> Sandbox on 2026-09-29 for every registered model, streaming and
> non-streaming.

For GitHub Copilot CLI the same settings map to environment variables:

```bash
export COPILOT_PROVIDER_TYPE=openai
export COPILOT_PROVIDER_BASE_URL=http://127.0.0.1:8090/v1
export COPILOT_PROVIDER_API_KEY=<your Portkey / AI Sandbox key>
export COPILOT_MODEL=gpt-5.6-sol        # or gpt-5.6-terra / gpt-6-astra /
                                        # gpt-6.1-sol / gpt-6-luna /
                                        # gemini-3.8-flash / claude-opus-5-5 /
                                        # claude-sonnet-5-5 / claude-fable-5-1
copilot
```

Notes:

- The proxy serves its routes at both `/v1/chat/completions` and
  `/chat/completions` (and likewise `/v1/responses` and `/responses`), so a
  base URL with or without `/v1` both work.
- Copilot requires models to support **tool calling and streaming**.
  If the upstream model errors on `tools`, the error is forwarded verbatim.

## Completions vs Responses

Both wire APIs are served and the client picks one. The proxy does **not**
rewrite a chat request's response into a `response` object (or vice versa) — the
shapes are not interchangeable, and a chat client could not parse one. The one
exception is deliberate and transparent: a chat request carrying function tools
may be *served* through the Responses API and translated **back** into chat
form, so the client still sees an ordinary `chat.completion`. See
[Bridging tool calls via Responses](#bridging-tool-calls-via-responses).

| | `/v1/chat/completions` | `/v1/responses` |
| --- | --- | --- |
| Use for | GitHub Copilot (app + CLI); anything speaking chat completions | OpenAI Responses clients; multi-turn state; reasoning summaries |
| Input | `messages` | `input` (a plain `messages` array is converted when it maps cleanly) |
| Function tools | nested: `{"type":"function","function":{...}}` | **flat**: `{"type":"function","name":...,"parameters":...}` |
| Reasoning | `reasoning_effort` (flat) | `reasoning: {"effort": ...}` (nested) |
| Tools + reasoning on GPT-5.6 profile (`gpt-5.6-sol/terra`, `gpt-6-luna`) | **rejected upstream** — reasoning is stripped, with an auto-retry using `"none"` | **accepted**, reasoning forwarded untouched |
| Tools on the gpt-6 profile (`gpt-6-astra`, `gpt-6.1-sol`) | **rejected in every shape** — auto-bridged via Responses | **accepted** natively |
| NRP models (`qwen3`, `deepseek-v4-flash`, `glm-5`) | **accepted** natively | **broken upstream** — auto-bridged via chat completions (`responses_via_chat`) |
| Stream terminator | `data: [DONE]` | `response.completed` event (no `[DONE]`) |
| Usage fields | `prompt_tokens` / `completion_tokens` | `input_tokens` / `output_tokens` (normalised into the same `usage.jsonl` schema) |
| Conversation state | client resends history | `previous_response_id` — **verified working** on the Sandbox |

Reach for `/v1/responses` directly when you want the GPT-5.6 family to keep
reasoning *while* calling function tools, or when server-side conversation state
(`previous_response_id`) is useful. Otherwise stay on Completions.

Set `RESPONSES_ENABLED=0` to turn the endpoint off; it then answers the
original 404 stub, and bridging is disabled with it.

## Bridging tool calls via Responses

`TOOLS_VIA_RESPONSES` (default `auto`) lets a chat-completions request that
carries function tools be served through the Responses API under the hood:

```
client --chat.completions--> proxy --responses.create--> Sandbox
client <--chat.completion--- proxy <--response--------- Sandbox
```

The translation covers both directions:

- `messages` → Responses input items, including `tool_calls` → `function_call`
  items and `tool` results → `function_call_output` items, so the model still
  sees prior tool output in a multi-turn loop
- nested `function` tool definitions → the flat Responses shape, and back
- `max_tokens` → `max_output_tokens`; `response_format` → `text.format`
- `reasoning_effort` → `reasoning.effort`, **preserved** rather than stripped
- `stream_options` is dropped: it is chat-only and the upstream rejects it on
  this endpoint, while Responses reports usage on `response.completed` anyway
- replies come back as `chat.completion` (or `chat.completion.chunk` frames
  ending in `data: [DONE]`), with `resp_…` ids rewritten to `chatcmpl-…` and
  usage renamed to `prompt_tokens` / `completion_tokens`

| Value | Behaviour |
| ------- | ----------- |
| `auto` (default) | Bridge only models with no working chat shape for tools — the `gpt-6` family. Everything else behaves exactly as before. |
| `always` | Also bridge the GPT-5.6 family's tool calls, so they keep reasoning instead of having it stripped. |
| `never` | No bridging; the upstream's 400 is forwarded to the client unchanged. |

Bridged requests are logged as `chat request bridged via responses: ...`, so it
is always visible when this path is taken. Usage and cost tracking are
unaffected — records still key off the alias you requested.

## Free-first model router (opt-in)

Set `ROUTER_ENABLED=1` to accept three **virtual model aliases** in addition to
the named models above. The router prefers the zero-cost NRP models and saves
the paid Portkey models for hard work and reviews, so you can point Copilot at
one alias instead of choosing per conversation. Everything else about the proxy
keeps working exactly as before: named models are still served by that model,
and with the router off (the default) nothing changes at all.

| Alias | Pool used (in order) | For |
| ------- | ---------------------- | ----- |
| `auto-free` | free pool, then paid | everyday requests |
| `auto-hard` | hard pool (default: paid pool, Opus first) | difficult tasks |
| `auto-review` | review pool (default: same as hard) | code review / final checks |

- **free pool** — `qwen3`, `deepseek-v4-flash`, `glm-5` (zero cost). `qwen3` is
  the default pick; it declares no image input, so vision requests start on
  `deepseek-v4-flash` instead (and any free model that errors over falls back
  to the next one).
- **paid pool** — curated default: `gpt-6.1-sol` first (cost-conscious complex
  work), then `claude-opus-5-5`, then the remaining Portkey models in
  `MODEL_REGISTRY` order. `claude-fable-5-1` is **opt-in only**: it sits out of
  every default pool (2.5x Opus's rates) but still works when named explicitly
  or added to a pool via config.
- **hard / review pools** — the paid pool with `claude-opus-5-5` promoted to
  the front (Anthropic's recommended default for agentic coding and review).

Pools are reconfigured with `ROUTER_FREE_POOL` / `ROUTER_PAID_POOL` /
`ROUTER_HARD_POOL` / `ROUTER_REVIEW_POOL` (comma-separated aliases) or a
`ROUTER_CONFIG_PATH` JSON file. `GET /v1/router` shows the live configuration.
An explicit empty JSON pool (`"paid": []`, for example) stays empty; omitted
pools use defaults. A zero approval threshold requires approval from the first
paid request; a zero monthly budget disables the advisory dollar cap (not
verified upstream exhaustion). Invalid pool types, numeric limits, and
capability values fail startup instead of silently restoring defaults.

### Per-request hints

A client can steer one request without changing the alias it is configured
with, by adding a `proxy_routing` object (never forwarded upstream):

```json
{"model": "auto-free",
 "messages": [...],
 "proxy_routing": {"task": "hard"}}
```

| Key | Values | Effect |
| ----- | -------- | -------- |
| `task` | `ordinary`, `hard`, `review` | Overrides the alias's pool for this request |
| `difficulty` | `normal`, `hard` | Same escalation as `task: hard` |
| `allow_paid` | `false` | Hard "spend nothing" veto: only free models are eligible, even on `auto-hard` |
| `paid_approved` | `true` | Approves paid use for this request while the approval gate is closed (below) |

Unknown keys are rejected with a 400 (`code: invalid_routing`) rather than
ignored. `X-Proxy-Model` / `X-Proxy-Route` response headers report which
concrete model served a virtual alias.

### Capability filtering

A candidate is skipped when it cannot honour the request as sent — the proxy
never silently drops content or parameters to make a model fit:

- function tools on a model that has none
- image/audio input on a model declared without it (the registry declares
  `qwen3: {image: false}`; add or change extras in the config file's
  `capabilities`, which overrides the registry)
- Responses-only parameters (`previous_response_id`, `store`, `background`,
  `include`, `prompt`, `truncation`, `service_tier`, …), native tool types
  (`web_search`, `mcp`, `shell`, …), item references, files, unknown input/content
  types, JSON-schema formatting, and nested reasoning/text or tool-choice
  options the NRP chat bridge cannot preserve
- `store: false` / `background: false`, ordinary text, and supported function
  call/result history remain bridgeable; intrinsic fidelity exclusions cannot
  be disabled by `native_only_params` or capability overrides
- `RESPONSES_ENABLED=0` excludes every model from a Responses route

### Bounded fallback

Candidates are tried in order, at most `ROUTER_MAX_ATTEMPTS` (default 4)
dispatches per request and no new attempt after `ROUTER_DEADLINE_SECONDS`
(default 120). The default leaves a paid attempt after three free failures;
paid-to-free reverts share the same attempt count and deadline. A
404/408/425/429/5xx, an empty stream, or an upstream
"unsupported parameter" 400 moves to the next model; an authentication failure
does not. **Nothing is replayed once bytes have reached the client**: the first
upstream chunk is pulled before the response is committed (synthetic bridge
startup/completion frames do not count), and a stream that dies mid-body is
surfaced as an SSE error frame on the same response. Responses errors are typed
`error` events without a chat wrapper or `[DONE]` sentinel.

### Budget gates (advisory)

Spend comes from the same `usage.jsonl` the `/v1/usage` report uses, so no
second ledger exists. Budget gates degrade virtual routes to eligible free
models rather than refuse them; if no free model can honour the request, the
normal capability/policy error is returned. `paid_approved: true` bypasses the
approval threshold only, **never the monthly cap or verified exhaustion**.
Unreadable/corrupt logs, unpriced paid calls, and failed appends make total
spend unknown and require approval. The known subtotal is still exposed as
`known_spend_usd` and enforces the cap even when other costs are unknown.

| Recorded monthly spend | `auto-*` route | Explicit paid model |
| -------------------------- | ---------------- | --------------------- |
| below `PAID_APPROVAL_THRESHOLD_USD` (230) | paid models allowed | allowed |
| unknown (tracking off / unreadable log / missing prices / failed append) | free models unless `paid_approved: true` | 402 `paid_approval_required` unless `paid_approved: true` |
| at or above 230, below cap | free models unless `paid_approved: true` | 402 `paid_approval_required` unless `paid_approved: true` |
| at or above `MONTHLY_BUDGET_USD` (250) | free models only | 402 `budget_exhausted` |
| upstream proved the key's quota is spent (402 / `insufficient_quota`) | free models only | 402 `budget_exhausted` |

Verified quota exhaustion is remembered per Portkey key (reported as a hashed
`exhausted_scopes` entry, never the key itself) for the rest of the calendar
month, so a dead wallet stops costing round trips; a plain 429 or temporary
request/token `quota_exceeded` rate limit is *not* treated as exhaustion.
Explicit paid failures also update this state, without substituting another
model on that explicit request. Clear it with `POST /v1/router/reset`.
An in-flight virtual paid request can revert to the free pool in the same
request after a 402, subject to the remaining attempt/deadline limits and the
normal authentication/capability rules.

Because spend is read when a request is planned, several concurrent requests
can each be admitted before any of them is recorded: the caps are advisory, and
the last requests of a month may overshoot slightly.

### Observability

Every routed request logs one decision line, with no prompt content, no keys,
and no token counts:

```
router: alias=auto-free gate=hard hints=task:hard skipped=qwen3(no-tools)
```

`GET /v1/router` (JSON) and `POST /v1/router/reset` report and clear live
policy state. `/v1/models` and `/health` advertise the virtual aliases
alongside the named models.

## Quick sanity check

```bash
# Chat completions, non-streaming
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-5.6-sol","messages":[{"role":"user","content":"Say hi"}]}'

# Chat completions, streaming
curl -N http://127.0.0.1:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-5.6-terra","stream":true,"messages":[{"role":"user","content":"Say hi"}]}'

# Responses API, non-streaming (note the flat tool schema + nested reasoning)
curl -s http://127.0.0.1:8090/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-5.6-sol","input":"Say hi","reasoning":{"effort":"low"}}'

# Responses API, streaming
curl -N http://127.0.0.1:8090/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-5.6-sol","stream":true,"input":"Say hi"}'
```

## Usage tracking & budget

The AI Sandbox caps each user at **$250/month**, but Portkey offers no spend
report — so the proxy keeps its own. On every completed request (streaming or
not) it reads the response's `usage` block, computes the dollar cost from
Portkey's public pricing API, and appends one JSON line to `usage.jsonl`:

```json
{"ts":"2026-09-16T09:51:02","month":"2026-09","model":"gpt-5.6-sol","upstream":"gpt-5.6-sol",
 "provider":"openai","pricing_model":"gpt-5.6-sol","stream":true,"prompt_tokens":12,
 "cached_tokens":4,"cache_write_tokens":0,"completion_tokens":8,"reasoning_tokens":0,
 "total_tokens":20,
 "rates":{"input":0.0004,"output":0.002,"cache_read_input":4e-05,"cache_write_input":0.0005},
 "cost_usd":0.0001936}
```

- **Prices** come from `https://api.portkey.ai/model-configs/pricing/<provider>/<model>`
  (public, no key) and are cached in `.pricing_cache.json` for 24 h. The API's
  numbers are the per-token USD price x100, so
  `cost_usd = tokens x price / 100`. Each record stores the rates applied, so
  history is auditable and never rewritten when prices change. Models with no
  catalog entry (the Sandbox deployment names differ, e.g. `gemini-3.8-flash`
  is priced as `gemini-3.8-flash-gt-128k` and `gpt-6.1-sol` as `gpt-6-sol`) are
  recorded with `cost_usd: null` and flagged in reports — requests are never
  blocked over tracking issues.
- **Cost formula.** `cached_tokens` and `cache_write_tokens` are both subsets of
  `prompt_tokens`, so input is split three ways and each part is billed at its
  own catalog rate:

  ```
  plain        = prompt_tokens - cached_tokens - cache_write_tokens
  cost_usd     = (plain            x input
                + cache_write      x cache_write_input   # falls back to input
                + cached           x cache_read_input    # falls back to input
                + completion       x output) / 100
  ```

  Current catalog rates per 1M tokens (multiply the API value by 1e6/100):

  | Model (as requested) | Priced as | Input | Cached in | Cache write | Output |
  | ------------------------ | ----------- | -------: | -----------: | ------------: | -------: |
  | `gpt-5.6-sol` | `gpt-5.6-sol` | $4.00 | $0.40 | $5.00 | $20.00 |
  | `gpt-5.6-terra` | `gpt-5.6-terra` | $2.00 | $0.20 | $2.50 | $12.00 |
  | `gpt-6-astra` | `gpt-6-astra` | $10.00 | $1.00 | $12.50 | $50.00 |
  | `gpt-6.1-sol` | `gpt-6-sol` | $2.00 | $0.20 | $2.50 | $10.00 |
  | `gpt-6-luna` | `gpt-6-luna` | $0.01 | $0.001 | $0.0125 | $0.05 |
  | `gemini-3.8-flash` | `gemini-3.8-flash-gt-128k` | $0.75 | $0.075 | – | $3.75 |
  | `claude-opus-5-5` | `claude-opus-5-5` | $4.00 | $0.20 | $5.00 | $20.00 |
  | `claude-sonnet-5-5` | `claude-sonnet-5-5` | $2.00 | $0.20 | $2.50 | $10.00 |
  | `claude-fable-5-1` | `claude-fable-5-1` | $10.00 | $0.25 | $12.50 | $50.00 |

  A dash means the catalog has no rate for that cell, so it falls back to the
  plain input rate. Override a mapping with `<ALIAS>_PRICING_MODEL` (e.g.
  `GPT6_SOL_PRICING_MODEL`) when the catalog renames a deployment.
- **Budget warnings** are logged by the proxy when the running monthly total
  crosses $50 / $100 / $150 / $200 / $250, and when $25 or $10 remain.
  Tracking is observe-only: the proxy never rejects a request for budget —
  unless you turn on the [free-first router](#free-first-model-router-opt-in),
  which reuses this log as its spend source and applies the approval / cap
  gates described there.
- **Reports:**

```bash
curl -s http://127.0.0.1:8090/v1/usage              # current month (JSON)
curl -s "http://127.0.0.1:8090/v1/usage?month=2026-08"

./.venv/bin/python usage_report.py                  # pretty table, current month
./.venv/bin/python usage_report.py --month 2026-08  # ... or any past month
./.venv/bin/python usage_report.py --json           # raw report
```

`usage_report.py` reads `usage.jsonl` directly, so it works even while the
proxy is stopped. The log is append-only and gitignored — delete or rotate it
freely (a month's totals are always recomputed from the records present).

## Tests

Two suites, both against a mock upstream (no real key needed):

```bash
./.venv/bin/python test_routing.py   # 177 checks: router policy and spend reader
./.venv/bin/python test_proxy.py     # 345 checks: end-to-end + controlled stream regressions
```

`test_routing.py` exercises `routing.py` directly (pools, hints, capability
filtering, gates, exhaustion, config). Section 11 of `test_proxy.py` boots three
router proxies in different budget states and drives them over HTTP, including
the fallback ladder, mid-stream breaks, and the paid→free revert. Neither suite
uses pytest; each is a single script that prints `ALL TESTS PASSED` /
`ROUTING TESTS PASSED`.

## Troubleshooting

Reasoning (`reasoning_effort`) is handled **per model**, because the upstreams
disagree about it. Each model in `MODEL_REGISTRY` declares a profile from
`REASONING_PROFILES`:

| Profile | Models | Behaviour on `/v1/chat/completions` |
| --------- | -------- | ----------- |
| `gpt-5.6` | `gpt-5.6-sol/terra`, `gpt-6-luna` | Rejects function `tools` + reasoning together. The proxy strips reasoning when tools are present, and auto-retries with `reasoning_effort: "none"` if the upstream still complains. `"none"` **is** a valid value here. (On `/v1/responses` this profile has no conflict — reasoning is forwarded untouched.) Despite its name, `gpt-6-luna` behaves like this family, not the `gpt-6` one. |
| `gpt-6` | `gpt-6-astra`, `gpt-6.1-sol` | **No chat-completions shape accepts function tools at all** (see below), so tools-bearing requests are bridged via `/v1/responses` by default. Without tools, `"none"` is **not** a valid effort — supported values are `low`, `medium`, `high`, `xhigh`, and unsupported client values are rewritten to the fallback `low`. |
| `flexible` | `gemini-3.8-flash`, `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-fable-5-1` | Accepts tools + reasoning and its own effort vocabulary, so values are forwarded untouched unless you configure an explicit list. |

**"Function tools with reasoning_effort are not supported ... set reasoning_effort to 'none'"**

The GPT-5.6-family upstream rejects `/v1/chat/completions` requests that
combine function `tools` with reasoning, and Copilot sends both. On this
endpoint the proxy handles it automatically (only for the `gpt-5.6` profile):

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

If losing reasoning on a GPT-5.6 tool call matters, send that request to
`POST /v1/responses` instead: the same models accept tools + reasoning there,
and the proxy forwards reasoning untouched.

**"'reasoning_effort' does not support 'none' with this model"**

That's `gpt-6-astra` (or another model whose vocabulary excludes `"none"`). The
proxy rewrites unsupported values to the model's fallback automatically. If a
model's real vocabulary differs, set it explicitly, e.g.:

```bash
GPT6_ASTRA_REASONING_EFFORTS=low,medium,high,xhigh
GEMINI38_FLASH_REASONING_EFFORTS=minimal,low,medium,high
```

The first configured non-`none` value becomes the fallback for rejected inputs.

**`gpt-6-astra` rejects function tools on `/v1/chat/completions` entirely**

Verified against the live Sandbox (2026-09-29): for the `gpt-6` family
(`gpt-6-astra`, `gpt-6.1-sol`) there is **no** chat-completions request shape
that accepts function tools:

| Sent with `tools` | Upstream result |
| ------------------- | ----------------- |
| `reasoning_effort: "low"` | 400 — tools + reasoning not supported here |
| `reasoning_effort: "none"` | 400 — `'none'` is not a valid effort for this model |
| reasoning params stripped | 400 — the upstream applies an implicit default effort |

The two escape hatches the error message offers contradict each other, so no
chat-completions request shape can win. **v1.7.0 resolves this automatically**:
a chat request that carries function tools is transparently served through
`/v1/responses` and the answer is translated back into chat-completion shape,
so Copilot keeps working on these models and `reasoning` is preserved rather
than stripped. You still get the same `chat.completion` JSON (tool calls nested
under `function`) and the same SSE chunks ending in `data: [DONE]`.

The log line reads `chat request bridged via responses: model=...`. Set
`TOOLS_VIA_RESPONSES=never` to disable bridging and get the raw upstream 400
back, or `always` to also route the GPT-5.6 family's tool calls through
Responses, which keeps their reasoning intact instead of stripping it.

Calling `POST /v1/responses` directly works too and is the lower-magic option
when you control the client.

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
- If a bridged request fails, the log says so (`chat request bridged via
  responses: ...`, then `Bridged responses stream failed`). Re-running with
  `TOOLS_VIA_RESPONSES=never` isolates whether the bridge or the upstream is
  at fault.

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
| `NRP_API_KEY` | – (unset) | API key for the NRP gateway models (`qwen3`, `deepseek-v4-flash`, `glm-5`); required to use those models |
| `NRP_BASE_URL` | `https://ellm.nrp-nautilus.io/v1` | NRP gateway base URL (`/v1` appended if missing) |
| `PROXY_HOST` / `PROXY_PORT` | `127.0.0.1` / `8090` | Where the proxy listens |
| `PORTKEY_TIMEOUT_MS` | – (unset) | Per-request upstream timeout sent to the Portkey gateway, in **milliseconds**. Leave unset so the gateway default applies. Do NOT use seconds (e.g. `600` = 600 ms) — that caused `408 Request exceeded the timeout` |
| `INCLUDE_STREAM_USAGE` | `1` | Send `stream_options.include_usage` to get token usage in streaming |
| `RESPONSES_ENABLED` | `1` | Serve `POST /v1/responses`. Set `0` to disable it (the route then returns a 404 stub, and bridging is disabled too) |
| `TOOLS_VIA_RESPONSES` | `auto` | Serve chat requests with function tools through the Responses API and translate the reply back: `auto` (only models with no working chat shape for tools, i.e. the `gpt-6` family), `always` (also GPT-5.6, preserving its reasoning), `never` (forward the upstream 400) |
| `TOOLS_REASONING_EFFORT` | `strip` | For models that reject tools + reasoning (GPT-5.6 only): `strip`, `none`, or `keep` (the alternative is auto-tried on the upstream's tools+reasoning 400) |
| `GPT56_SOL_UPSTREAM_MODEL` / `GPT56_TERRA_UPSTREAM_MODEL` / `GPT6_ASTRA_UPSTREAM_MODEL` / `GPT6_SOL_UPSTREAM_MODEL` / `GPT6_LUNA_UPSTREAM_MODEL` / `GEMINI38_FLASH_UPSTREAM_MODEL` | alias name | Map an alias to a different upstream model |
| `QWEN3_UPSTREAM_MODEL` / `DEEPSEEK_V4_FLASH_UPSTREAM_MODEL` / `GLM5_UPSTREAM_MODEL` | alias name | Map an NRP alias to a different upstream model |
| `GPT6_ASTRA_REASONING_EFFORTS` | `low,medium,high,xhigh` | Comma-separated `reasoning_effort` values the upstream accepts for `gpt-6-astra`; unsupported client values are rewritten to the first non-`none` entry |
| `GEMINI38_FLASH_REASONING_EFFORTS` / `CLAUDE_OPUS_55_REASONING_EFFORTS` / `CLAUDE_SONNET_55_REASONING_EFFORTS` / `CLAUDE_FABLE_51_REASONING_EFFORTS` | – (unset) | Optional comma-separated `reasoning_effort` list for `gemini-3.8-flash` or a Claude model; when unset, client values are forwarded untouched |
| `USAGE_TRACKING` | `1` | Record token usage and dollar cost of every request to the JSONL usage log |
| `USAGE_LOG_PATH` | `usage.jsonl` (next to `proxy.py`) | Append-only usage log; also read directly by `usage_report.py` |
| `MONTHLY_BUDGET_USD` | `250` | Monthly budget used by `/v1/usage` reports and the proxy's budget warning logs |
| `PRICING_BASE_URL` | `https://api.portkey.ai/model-configs` | Pricing API base; the proxy fetches `/pricing/<provider>/<model>` (public, no key) |
| `PRICING_CACHE_PATH` | `.pricing_cache.json` | Cached pricing responses (refreshed after the TTL; stale cache is used if a refresh fails) |
| `PRICING_TTL_HOURS` | `24` | How long a cached price is considered fresh |
| `GEMINI38_FLASH_PRICING_MODEL` | `gemini-3.8-flash-gt-128k` | Pricing-catalog name used for `gemini-3.8-flash` (it is not listed under its request name) |
