---
name: check-portkey-models
description: 'Check which models the Portkey / Princeton AI Sandbox gateway actually serves and what each one costs. Use when asked about model availability, whether a deployment is live or retired, per-model pricing or rates, why a model logs cost_usd null, whether a pricing_model override is stale, or when adding a model to MODEL_REGISTRY and needing its catalog name and rates. Covers the public pricing catalog (GET /model-configs/pricing/<provider>/<model>), the absence of any list endpoint, and live availability probing.'
argument-hint: '[alias ...] [--no-live] [--json]'
---

# Check Portkey Model Availability & Pricing

Answers two questions per model alias in `MODEL_REGISTRY`:

1. **Availability** — does the gateway actually serve this deployment?
2. **Pricing** — what does the public catalog charge, and is the proxy looking
   it up under the right name?

## When to Use

- "Is `gpt-6.2-sol` still available?" / "did model X get retired?"
- "What does model X cost?" / "what are the rates per 1M tokens?"
- "Why does `usage.jsonl` show `cost_usd: null` for model X?"
- "Is the `pricing_model` override still correct?"
- Adding a model to `MODEL_REGISTRY` — you need its catalog name and rates.
- Auditing the README pricing table against the live catalog.

## Critical Facts

**There is no list endpoint.** Every discovery path returns
`400 Invalid pricing config path. Expected: /model-configs/pricing/<provider>/<model>`:

| Path tried | Result |
| --- | --- |
| `/model-configs/` | 400 Invalid path |
| `/model-configs/models` | 400 Invalid path |
| `/model-configs/providers` | 400 Invalid path |
| `/model-configs/pricing` | 400 Invalid pricing config path |
| `/model-configs/pricing/openai` | 400 Invalid pricing config path |

The upstream gateway has no model list either — `GET {PORTKEY_BASE_URL}/models`
returns `400 Invalid target name found in the query router: unknown-model`.
So **availability must be probed one model at a time**, and the only source of
truth for "what exists" is `MODEL_REGISTRY` plus a live call per entry.

**Price units.** The catalog returns the per-token USD price **scaled x100**.
Convert to dollars per 1M tokens with `price * 1e6 / 100` (i.e. `price * 10000`).
`usage.py` uses `PRICE_SCALE = 100.0` for the same reason.

**Price the deployment, not the alias.** The proxy looks up
`pricing_model or upstream` — mirror that exactly, or you will report rates for
a name the proxy never queries.

## Procedure

### 1. Run the bundled script

```bash
./.venv/bin/python .github/skills/check-portkey-models/scripts/check_models.py
```

Options:

| Flag | Effect |
| --- | --- |
| `--no-live` | Pricing only. No API calls, no key needed. |
| `--json` | Machine-readable output for further processing. |
| `--model ALIAS` | Restrict to one alias (repeatable). |
| `--base-url URL` | Point at a different catalog (default `PRICING_BASE_URL`). |

The script imports `proxy.py` to read `MODEL_REGISTRY`, `.env`, and the client
factories. **Never read `.env` directly** — importing `proxy` is the supported
way to see the live config.

### 2. Read the report

The table has one row per alias with the catalog rates per 1M tokens and two
status columns:

| Status | Meaning |
| --- | --- |
| `ok` | Catalog entry found with `pay_as_you_go` rates. |
| `missing` | 404 — the name is not in the catalog. |
| `empty` | 200 but no `pay_as_you_go` rates. |
| `error` | Network failure or non-200/404 HTTP status. |
| `zero-cost` | `track_usage: False` (NRP models) — not billed, not in the catalog. |
| `available` | Live probe returned a completion. |
| `unavailable` | Gateway returned `unknown-model` — the deployment is gone. |
| `skipped` | No key configured, or `--no-live`. |

### 3. Act on the warnings

The script prints a `## Warnings` section. Handle each case:

**`missing` with no override** — the deployment name is not a catalog name.
Find the real catalog name (see step 4) and set
`<ALIAS>_PRICING_MODEL` in `MODEL_REGISTRY`. Until then the proxy records
`cost_usd: null` and reports count the request under `requests_without_pricing`.

**`missing` with an override** — the override itself is stale. This is the
important case: a `pricing_model` that used to resolve can 404 later, silently
turning every request into `cost_usd: null`. Re-resolve the name.

**`unavailable`** — the deployment is retired. Remove the alias from
`MODEL_REGISTRY`, the README model table, `.env.example`, and the per-model
checks in `test_proxy.py`.

**`error`** — transient. Re-run before concluding anything.

### 4. Find a catalog name when the deployment name 404s

There is no search endpoint, so probe candidate names directly. Known patterns
in this repo:

| Alias | Catalog name | Why |
| --- | --- | --- |
| `gemini-3.8-flash` | `gemini-3.8-flash-gt-128k` | Google entries carry a context-window suffix |
| `gemini-4-argon` | `gemini-4-argon-gt-128k` | same |
| `gpt-6.1-sol` | `gpt-6.1-sol` | own name; the older `gpt-6-sol` also resolves but has a **2x cache-read rate** |
| `gpt-6.1-terra` | `gpt-6.1-terra` | own name; `gpt-6-terra` 404s |

Probe a candidate with a one-liner:

```bash
curl -sS -w '\nHTTP %{http_code}\n' \
  "https://api.portkey.ai/model-configs/pricing/google/gemini-3.8-flash-gt-128k"
```

A 200 body looks like:

```json
{"pay_as_you_go":{"request_token":{"price":0.0002},
 "response_token":{"price":0.001},
 "cache_write_input_token":{"price":0.00025},
 "cache_read_input_token":{"price":0.00001}}}
```

**Beware near-miss names.** `gpt-6-sol` and `gpt-6.1-sol` both resolve but
differ in cache-read rate ($0.20 vs $0.10 per 1M). Always compare the rates you
found against the README table rather than assuming a resolving name is the
right one.

### 5. Verify the proxy agrees

The script reads the catalog directly. To confirm the proxy's own cost path
produces the same numbers, check a recorded request:

```bash
./.venv/bin/python usage_report.py --json | head -40
```

Each `usage.jsonl` record stores the `rates` that were applied and the
`pricing_model` used, so a mismatch between the script's table and a record's
`rates` means the cache is stale or the override differs.

## Adding a Model

When the check is part of adding a new alias, the full checklist is:

1. Add the entry to `MODEL_REGISTRY` in `proxy.py` with `provider`, and
   `pricing_model` if the catalog name differs.
2. Run this skill to confirm the catalog name resolves and note the rates.
3. Add the row to the README model table **and** the pricing table.
4. Add the `*_UPSTREAM_MODEL` / `*_PRICING_MODEL` lines to `.env.example`.
5. Add the per-model checks to `test_proxy.py`.
6. Run `./.venv/bin/python test_proxy.py` and confirm `ALL TESTS PASSED`.

## Gotchas

- **Do not probe availability with a tiny token budget.** Reasoning models
  reject `max_completion_tokens=1` with
  `Could not finish the message because max_tokens or model output limit was
  reached` — which looks like a failure but is not. The script omits token and
  sampling params entirely; that is the only shape that works for every model
  (gpt-* deployments also reject sampling knobs).
- **Probe `upstream`, not the alias.** The alias is a proxy-local name; only
  `upstream` reaches the gateway.
- **NRP models are not in the catalog.** `qwen3`, `deepseek-v4-flash`, and
  `glm-5` are zero-cost and `track_usage: False`, so a 404 for them is expected,
  not a bug. They need `NRP_API_KEY` for a live probe.
- **A live probe costs money.** It is one short completion per model. Use
  `--no-live` when you only need rates.
- **The pricing cache is 24 h.** `.pricing_cache.json` can serve stale rates;
  the script bypasses it and hits the catalog directly, so a disagreement
  between the script and `usage.jsonl` is expected until the TTL expires.
- **Never paste multi-line shell loops.** The persistent zsh terminal wedges in
  a `for` continuation. Use the script, or a file.
