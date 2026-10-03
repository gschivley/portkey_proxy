#!/usr/bin/env python3
"""
Free-first request routing for the Portkey proxy (opt-in, see ROUTER_ENABLED).

The proxy already serves two upstreams: the paid Portkey / AI Sandbox
deployment and the zero-cost NRP gateway (qwen3, deepseek-v4-flash, glm-5).
This module turns that into a *policy*: ordinary traffic prefers the free
models and only reaches for the paid ones when the request actually needs a
capability the free pool cannot deliver, or when the caller asks for a harder
tier. Nothing here inspects prompt text: every decision comes from the
requested model, the request *structure* (wire API, tools, stateful params)
and the recorded monthly spend.

Design rules (see the project plan):

* Explicit model names stay authoritative. A request for `claude-opus-5-5`
  is never silently routed elsewhere; it is only subject to the paid gates
  below while the router is enabled.
* Virtual aliases are opt-in additions to the model list: `auto-free`
  (free-preferred, **not** free-only), `auto-hard`, `auto-review`.
* Budget is **advisory**, read from usage.py's recorded monthly total. There
  is no reservation ledger: concurrent in-flight calls can overshoot, and
  spend made outside this proxy is invisible to it.
  - spend >= PAID_APPROVAL_THRESHOLD_USD (default $230): every paid request,
    explicit or routed, needs request-scoped `proxy_routing.paid_approved`.
  - spend >= MONTHLY_BUDGET_USD (default $250), or a verified upstream
    quota-exhaustion error: virtual routes become free-only.
* Capability filtering never drops content. A candidate is excluded when the
  proxy could only serve the request *lossily* on it (a parameter the bridge
  discards, hosted tools, server-managed state), not by trimming the request.
* No credential material, prompts or tool arguments are ever logged or
  stored here. Paid-credential scopes are held as a truncated SHA-256 digest.

Configuration (environment / .env):
    ROUTER_ENABLED                 "1" to enable virtual aliases + gates
    ROUTER_CONFIG_PATH             optional JSON file (pools, capabilities)
    ROUTER_FREE_POOL               comma-separated free aliases
                                   (default qwen3,deepseek-v4-flash,glm-5)
    ROUTER_PAID_POOL               comma-separated paid aliases (default:
                                   curated order gpt-6.1-sol, claude-opus-5-5,
                                   then the rest in registry order; opt-in
                                   aliases like claude-fable-5-1 are excluded)
    ROUTER_HARD_POOL               pool for hard tasks (default: the paid
                                   pool with claude-opus-5-5 promoted first)
    ROUTER_REVIEW_POOL             pool for reviews (default: like hard)
    ROUTER_MAX_ATTEMPTS            upstream dispatches per request (default 4)
    ROUTER_DEADLINE_SECONDS        stop starting attempts after this (default 120)
    PAID_APPROVAL_THRESHOLD_USD    approval gate (default 230)

JSON config file (all keys optional):
    {"pools": {"free": [...], "paid": [...], "hard": [...], "review": [...]},
     "capabilities": {"qwen3": {"image": false}},
     "native_only_params": ["previous_response_id", "store", ...],
     "max_attempts": 4,
     "paid_approval_threshold_usd": 230,
     "monthly_budget_usd": 250}
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import usage as usage_mod

log = logging.getLogger("routing")

ROUTING_FIELD = "proxy_routing"
VIRTUAL_ALIASES = ("auto-free", "auto-hard", "auto-review")
TASKS = ("ordinary", "hard", "review")
DIFFICULTIES = ("normal", "hard")

DEFAULT_FREE_POOL = ("qwen3", "deepseek-v4-flash", "glm-5")

# Research-informed curated default orders (see README "Free-first model
# router").
# Everyday paid escalation starts on GPT-6.1 Sol (OpenAI's cost-conscious
# complex-work pick); hard/review work starts on Claude Opus 5.5 (Anthropic's
# recommended default for agentic coding, refactors and code review).
CURATED_PAID_LEAD = ("gpt-6.1-sol", "claude-opus-5-5")
CURATED_TASK_LEAD = ("claude-opus-5-5", "gpt-6.1-sol")
# 2.5x Opus's public rates: keep Fable out of every default pool so routed
# traffic never lands on it by accident. Add it to a pool (env or JSON
# config) to opt in; explicit model names always still reach it.
DEFAULT_OPT_IN_ALIASES = ("claude-fable-5-1",)

# Responses parameters the NRP bridge has no faithful chat equivalent for. A
# request that carries one of these is not served by an NRP model: the proxy
# would have to drop it, which is exactly what routing must never do silently.
DEFAULT_NATIVE_ONLY_PARAMS = (
    "previous_response_id",
    "store",
    "background",
    "include",
    "prompt",
    "prompt_cache_key",
    "safety_identifier",
    "service_tier",
    "truncation",
    "max_tool_output_tokens",
    "top_logprobs",
)

_TRUE = ("1", "true", "yes", "on")


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in _TRUE


def _first_set(*values: Any) -> Any:
    """First provided value; zero is explicit, blank environment strings are not."""
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return values[-1]


def is_enabled() -> bool:
    """Whether routing is on. Read *after* the proxy has loaded .env files."""
    return _flag("ROUTER_ENABLED")


def config_path() -> Optional[str]:
    return (os.environ.get("ROUTER_CONFIG_PATH") or "").strip() or None


def _pool_env(name: str, default: Tuple[str, ...]) -> List[str]:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


# Reasons a paid candidate was hidden by proxy policy rather than by the model
# itself. A routed request with only these blockers reverts to the free pool.
_PAID_GATE_REASONS = frozenset(
    {"over-budget", "exhausted", "approval-required", "no-portkey-key"}
)


class RoutingError(Exception):
    """A routing decision that must be reported to the client as an error."""

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


# --------------------------------------------------------------------------- #
# Request hints (the only routing input a caller can set)
# --------------------------------------------------------------------------- #

_ALLOWED_HINTS = ("task", "difficulty", "allow_paid", "paid_approved", "model")
_CONSENT_HINTS = ("allow_paid", "paid_approved")


def parse_hints(body: Dict[str, Any], *, enforce: bool) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split the router-only `proxy_routing` object out of a request body.

    Returns (hints, cleaned_body). The router field is **always** removed so it
    can never reach an upstream. When `enforce` is false (router disabled) the
    value is dropped without validation, which keeps legacy behaviour intact for
    clients that send the field by accident.
    """
    if ROUTING_FIELD not in body:
        return {}, body
    cleaned = {k: v for k, v in body.items() if k != ROUTING_FIELD}
    raw = body.get(ROUTING_FIELD)
    if not enforce:
        return {}, cleaned
    if raw is None:
        return {}, cleaned
    if not isinstance(raw, dict):
        raise RoutingError(
            400,
            f"`{ROUTING_FIELD}` must be a JSON object.",
            param=ROUTING_FIELD,
            code="invalid_routing",
        )
    unknown = sorted(set(raw) - set(_ALLOWED_HINTS))
    if unknown:
        raise RoutingError(
            400,
            f"Unsupported {ROUTING_FIELD} keys: {', '.join(unknown)}. "
            f"Supported keys: {', '.join(_ALLOWED_HINTS)}.",
            param=ROUTING_FIELD,
            code="invalid_routing",
        )
    hints: Dict[str, Any] = {}
    task = raw.get("task")
    if task is not None:
        if not isinstance(task, str) or task.strip().lower() not in TASKS:
            raise RoutingError(
                400,
                f"{ROUTING_FIELD}.task must be one of: {', '.join(TASKS)}.",
                param=f"{ROUTING_FIELD}.task",
                code="invalid_routing",
            )
        hints["task"] = task.strip().lower()
    difficulty = raw.get("difficulty")
    if difficulty is not None:
        if (
            not isinstance(difficulty, str)
            or difficulty.strip().lower() not in DIFFICULTIES
        ):
            raise RoutingError(
                400,
                f"{ROUTING_FIELD}.difficulty must be one of: "
                f"{', '.join(DIFFICULTIES)}.",
                param=f"{ROUTING_FIELD}.difficulty",
                code="invalid_routing",
            )
        hints["difficulty"] = difficulty.strip().lower()
    for key in _CONSENT_HINTS:
        value = raw.get(key)
        if value is not None:
            if not isinstance(value, bool):
                raise RoutingError(
                    400,
                    f"{ROUTING_FIELD}.{key} must be a boolean.",
                    param=f"{ROUTING_FIELD}.{key}",
                    code="invalid_routing",
                )
            hints[key] = value
    model = raw.get("model")
    if model is not None:
        if not isinstance(model, str) or not model.strip():
            raise RoutingError(
                400,
                f"{ROUTING_FIELD}.model must be a non-empty model name.",
                param=f"{ROUTING_FIELD}.model",
                code="invalid_routing",
            )
        hints["model"] = model.strip()
    return hints, cleaned


# --------------------------------------------------------------------------- #
# Requirements (derived from request structure, never from prompt text)
# --------------------------------------------------------------------------- #


def _has_tools(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    return any(isinstance(item, dict) for item in value)


def _native_tool_used(body: Dict[str, Any]) -> bool:
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            return True
    choice = body.get("tool_choice")
    return isinstance(choice, dict) and choice.get("type") != "function"


def _bridge_input_supported(inp: Any) -> bool:
    """Match the lossless subset of the Responses-to-chat input translator."""
    if inp is None or isinstance(inp, str):
        return True
    if not isinstance(inp, list):
        return False
    for item in inp:
        if isinstance(item, str):
            continue
        if not isinstance(item, dict):
            return False
        kind = item.get("type")
        if kind in (None, "message"):
            if set(item) - {"type", "role", "content", "id", "status"}:
                return False
            content = item.get("content")
            if isinstance(content, str):
                continue
            if not isinstance(content, list):
                return False
            allowed = {
                "input_text": {"type", "text"},
                "output_text": {"type", "text"},
                "text": {"type", "text"},
                "refusal": {"type", "refusal"},
                "input_image": {"type", "image_url", "image"},
                "image_url": {"type", "image_url"},
                "input_audio": {"type", "input_audio", "audio"},
                "audio": {"type", "audio"},
            }
            for part in content:
                if not isinstance(part, dict):
                    return False
                keys = allowed.get(part.get("type"))
                if keys is None or set(part) - keys:
                    return False
        elif kind == "function_call":
            if set(item) - {"type", "id", "status", "call_id", "name", "arguments"}:
                return False
        elif kind == "function_call_output":
            if (
                set(item) - {"type", "id", "status", "call_id", "name", "output"}
                or not isinstance(item.get("output"), str)
            ):
                return False
        else:
            return False
    return True


def _bridge_option_losses(body: Dict[str, Any]) -> List[str]:
    losses: List[str] = []
    if not _bridge_input_supported(body.get("input")):
        losses.append("input")
    if body.get("instructions") is not None and not isinstance(body["instructions"], str):
        losses.append("instructions")
    reasoning = body.get("reasoning")
    if reasoning is not None and (
        not isinstance(reasoning, dict)
        or set(reasoning) - {"effort"}
        or reasoning.get("effort") == "none"
    ):
        losses.append("reasoning")
    text = body.get("text")
    if text is not None:
        fmt = text.get("format", {}) if isinstance(text, dict) else None
        if (
            not isinstance(text, dict)
            or set(text) - {"format"}
            or not isinstance(fmt, dict)
            or fmt.get("type", "text") not in ("text", "json_object")
            or (fmt.get("type", "text") == "text" and set(fmt) - {"type"})
        ):
            losses.append("text")
    choice = body.get("tool_choice")
    if isinstance(choice, dict) and choice.get("type") == "function" and set(choice) - {"type", "name"}:
        losses.append("tool_choice")
    metadata = body.get("metadata")
    if metadata is not None and (
        not isinstance(metadata, dict)
        or any(not isinstance(value, str) for value in metadata.values())
    ):
        losses.append("metadata")
    return losses


def request_requirements(
    wire: str, body: Dict[str, Any], native_only_params: Tuple[str, ...]
) -> Dict[str, Any]:
    """What the request needs in order to be served faithfully."""
    reqs: Dict[str, Any] = {
        "wire": wire,
        "stream": bool(body.get("stream", False)),
        "tools": _has_tools(body.get("tools")),
        "native_only": [],
    }
    if wire == "responses":
        # Only a *set* value counts: `store: false` and `background: false` are
        # no-ops the bridge can honour, so they must not exclude free models.
        reqs["native_only"] = [
            key
            for key in native_only_params
            if key in body and _is_requested(body.get(key))
        ]
        reqs["native_only"].extend(_bridge_option_losses(body))
        if body.get("previous_response_id"):
            reqs["stateful"] = True
        if _native_tool_used(body):
            reqs["native_tools"] = True
        if _chat_has_image_part(body.get("input")):
            reqs["image"] = True
        if (body.get("reasoning") or {}).get("effort") if isinstance(body.get("reasoning"), dict) else None:
            reqs["reasoning"] = True
    else:
        if any(key in body for key in ("audio", "modalities")):
            reqs["audio"] = True
        if _chat_has_image_part(body.get("messages")):
            reqs["image"] = True
        if body.get("reasoning_effort") or isinstance(body.get("reasoning"), dict):
            reqs["reasoning"] = True
    return reqs


def _is_requested(value: Any) -> bool:
    """Whether a parameter value asks for behaviour (vs a falsy no-op)."""
    if value is None or value is False:
        return False
    if isinstance(value, (list, tuple, dict, str)):
        return len(value) > 0
    return True


def _chat_has_image_part(messages: Any) -> bool:
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and str(part.get("type")) in (
                "image_url",
                "input_image",
                "image",
            ):
                return True
    return False


# --------------------------------------------------------------------------- #
# Plans
# --------------------------------------------------------------------------- #


class Plan:
    """The router's decision for one request."""

    def __init__(
        self,
        requested: str,
        candidates: List[str],
        *,
        hints: Optional[Dict[str, Any]] = None,
        skipped: Optional[List[Tuple[str, str]]] = None,
        gate: str = "automatic",
        pinned: Optional[str] = None,
    ) -> None:
        self.requested = requested
        self.candidates = candidates
        self.hints = hints or {}
        self.skipped = skipped or []
        self.gate = gate
        self.pinned = pinned

    @property
    def virtual(self) -> bool:
        return self.requested in VIRTUAL_ALIASES

    def reason(self) -> str:
        parts = [f"alias={self.requested}"]
        if self.pinned:
            parts.append(f"pinned={self.pinned}")
        parts.append(f"gate={self.gate}")
        if self.hints:
            parts.append(
                "hints=" + ",".join(f"{k}:{v}" for k, v in sorted(self.hints.items()))
            )
        if self.skipped:
            parts.append(
                "skipped=" + ",".join(f"{a}({r})" for a, r in self.skipped)
            )
        return " ".join(parts)


class Router:
    """Deterministic candidate selection + advisory paid gates.

    Everything the Router needs is injected (registry, spend reader, clock) so
    it can be unit-tested without a running proxy or a real usage log.
    """

    def __init__(
        self,
        registry: Dict[str, Dict[str, Any]],
        *,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        responses_enabled: bool = True,
        approval_threshold_usd: Optional[float] = None,
        monthly_budget_usd: Optional[float] = None,
        max_attempts: Optional[int] = None,
        deadline_seconds: Optional[float] = None,
        spend_func: Optional[Callable[[], float]] = None,
        month_func: Optional[Callable[[], str]] = None,
    ) -> None:
        self.registry = registry
        self.responses_enabled = responses_enabled
        for name, value in (
            ("max_attempts", max_attempts), ("deadline_seconds", deadline_seconds),
            ("paid_approval_threshold_usd", approval_threshold_usd), ("monthly_budget_usd", monthly_budget_usd),
        ):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                raise ValueError(f"router {name} must be a number")
        self.config = self._load_config(config_path, config)
        for key in ("max_attempts", "deadline_seconds", "paid_approval_threshold_usd", "monthly_budget_usd"):
            if key in self.config and (
                isinstance(self.config[key], bool) or not isinstance(self.config[key], (int, float))
            ):
                raise ValueError(f"router {key} must be a number")
        pools = self.config.get("pools", {})
        if not isinstance(pools, dict) or set(pools) - {"free", "paid", "hard", "review"}:
            raise ValueError("router pools must be an object with free/paid/hard/review lists")

        def pool(name: str, default: Any) -> List[str]:
            value = pools[name] if name in pools else _pool_env(f"ROUTER_{name.upper()}_POOL", default)
            if not isinstance(value, list) or any(not isinstance(alias, str) or not alias.strip() for alias in value):
                raise ValueError(f"router pools.{name} must be a list of non-empty model names")
            return list(value)

        paid_pool = pool("paid", self._default_paid_pool())
        task_default = self._reorder_lead(paid_pool, CURATED_TASK_LEAD)
        self.pools: Dict[str, List[str]] = {
            "free": pool("free", DEFAULT_FREE_POOL),
            "paid": paid_pool,
            "hard": pool("hard", task_default),
            "review": pool("review", task_default),
        }
        self.capabilities = self.config.get("capabilities", {})
        if not isinstance(self.capabilities, dict) or any(
            not isinstance(caps, dict)
            or any(value is not None and not isinstance(value, bool) for value in caps.values())
            for caps in self.capabilities.values()
        ):
            raise ValueError("router capabilities must map model names to boolean flags")
        native = self.config.get("native_only_params", [])
        if not isinstance(native, list) or any(not isinstance(key, str) or not key for key in native):
            raise ValueError("native_only_params must be a list of parameter names")
        self.native_only_params = tuple(dict.fromkeys((*DEFAULT_NATIVE_ONLY_PARAMS, *native)))
        attempts = _first_set(
            max_attempts, self.config.get("max_attempts"),
            os.environ.get("ROUTER_MAX_ATTEMPTS"), 4,
        )
        if isinstance(attempts, bool) or not float(attempts).is_integer() or float(attempts) < 1:
            raise ValueError("ROUTER_MAX_ATTEMPTS must be an integer >= 1")
        self.max_attempts = int(attempts)
        self.deadline_seconds = float(
            _first_set(
                deadline_seconds,
                self.config.get("deadline_seconds"),
                os.environ.get("ROUTER_DEADLINE_SECONDS"),
                120,
            )
        )
        if not math.isfinite(self.deadline_seconds) or self.deadline_seconds <= 0:
            raise ValueError("ROUTER_DEADLINE_SECONDS must be finite and > 0")
        self.approval_threshold_usd = float(_first_set(
            approval_threshold_usd, self.config.get("paid_approval_threshold_usd"),
            os.environ.get("PAID_APPROVAL_THRESHOLD_USD"), 230.0,
        ))
        self.monthly_budget_usd = float(_first_set(
            monthly_budget_usd, self.config.get("monthly_budget_usd"), usage_mod.MONTHLY_BUDGET_USD,
        ))
        for name, value in (
            ("PAID_APPROVAL_THRESHOLD_USD", self.approval_threshold_usd),
            ("MONTHLY_BUDGET_USD", self.monthly_budget_usd),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0")
        if self.approval_threshold_usd >= self.monthly_budget_usd and self.monthly_budget_usd > 0:
            raise ValueError(
                "PAID_APPROVAL_THRESHOLD_USD "
                f"({self.approval_threshold_usd}) must be below the monthly "
                f"budget ({self.monthly_budget_usd}): at the budget the proxy "
                "is free-only, so an approval threshold above it could never "
                "be satisfied."
            )
        self._spend_explicit = spend_func is not None
        self._spend = spend_func or usage_mod.spent_this_month
        self._month = month_func or usage_mod.current_month
        self._lock = threading.Lock()
        self._exhausted: Dict[str, Dict[str, Any]] = {}
        self._pins: Dict[str, Dict[str, Any]] = {}
        self._warned_no_spend = False

    # -- configuration ------------------------------------------------------ #

    @staticmethod
    def _load_config(path: Optional[str], inline: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if inline is not None:
            if not isinstance(inline, dict):
                raise ValueError("router config must be a JSON object")
            return inline
        if not path:
            return {}
        file = Path(path)
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(f"cannot read ROUTER_CONFIG_PATH {file}: {exc}") from exc
        except ValueError as exc:
            raise ValueError(f"ROUTER_CONFIG_PATH {file} is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"ROUTER_CONFIG_PATH {file} must contain a JSON object")
        return data

    def _default_paid_pool(self) -> List[str]:
        """Curated paid order: Sol first (cost-sensitive complex work), Opus next."""
        known = [
            alias
            for alias, entry in self.registry.items()
            if entry.get("api_source") != "nrp"
            and alias not in DEFAULT_OPT_IN_ALIASES
        ]
        return self._reorder_lead(known, CURATED_PAID_LEAD)

    @staticmethod
    def _reorder_lead(pool: List[str], lead: Tuple[str, ...]) -> List[str]:
        """Move the curated lead aliases to the front of ``pool``, order preserved."""
        ordered = [alias for alias in lead if alias in pool]
        return ordered + [alias for alias in pool if alias not in ordered]

    def is_free(self, alias: str) -> bool:
        entry = self.registry.get(alias)
        if entry is None:
            return alias in self.pools["free"]
        return entry.get("api_source") == "nrp" or entry.get("track_usage", True) is False

    def source_of(self, alias: str) -> str:
        entry = self.registry.get(alias) or {}
        return str(entry.get("api_source") or "portkey")

    def _declared_caps(self, alias: str) -> Dict[str, Any]:
        """Capability flags for one alias: known-good, unknown is None."""
        entry = self.registry.get(alias) or {}
        nrp = self.source_of(alias) == "nrp"
        caps: Dict[str, Any] = {
            "chat": True,
            "responses_native": not nrp,
            "responses_bridged": bool(entry.get("responses_via_chat")),
            "stateful_responses": not nrp,
            "tools": True,
        }
        # Registry-declared modality defaults (e.g. qwen3 has no image input),
        # then router config on top so either can be overridden by operators.
        caps.update(entry.get("capabilities") or {})
        caps.update(self.capabilities.get(alias) or {})
        return caps

    # -- spend / budget gates ----------------------------------------------- #

    def spend(self) -> Optional[float]:
        """Recorded spend for the current month, or None if unknowable.

        An injected reader (tests, or a future alternate source) is trusted
        unconditionally; the default one needs USAGE_TRACKING to be on.
        """
        if self._spend_explicit:
            try:
                value = self._spend()
                return float(value) if value is not None and math.isfinite(value) and value >= 0 else None
            except Exception:  # noqa: BLE001 - spend is advisory, never fatal
                log.exception("router: could not read monthly spend")
                return None
        if not usage_mod.USAGE_TRACKING:
            if not self._warned_no_spend:
                self._warned_no_spend = True
                log.warning(
                    "router: USAGE_TRACKING is off, so monthly spend cannot be "
                    "read; paid requests will require explicit approval."
                )
            return None
        return usage_mod.spent_this_month()

    def spend_state(self) -> Dict[str, Any]:
        if not self._spend_explicit and usage_mod.USAGE_TRACKING:
            snapshot = usage_mod.spend_snapshot()
            spend = snapshot["spend_usd"]
            known = snapshot["known_spend_usd"]
        else:
            spend = self.spend()
            known = spend
        return {
            "spend_usd": spend,
            "known_spend_usd": known,
            "approval_threshold_usd": self.approval_threshold_usd,
            "monthly_budget_usd": self.monthly_budget_usd,
            "approval_required": spend is None or spend >= self.approval_threshold_usd,
            "budget_exhausted": (
                known is not None
                and self.monthly_budget_usd > 0
                and known >= self.monthly_budget_usd
            ),
            "month": self._month(),
        }

    # -- verified upstream exhaustion --------------------------------------- #

    @staticmethod
    def scope(key: Optional[str], source: str) -> str:
        """Stable, non-reversible identifier for one paid credential."""
        if source != "portkey":
            return source
        if not key:
            return "portkey:anonymous"
        return "portkey:" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]

    def mark_exhausted(self, scope_id: str, reason: str) -> None:
        with self._lock:
            self._exhausted[scope_id] = {
                "since": time.time(),
                "month": self._month(),
                "reason": reason[:200],
            }
        log.warning(
            "router: paid scope %s marked exhausted (%s); virtual routes are "
            "free-only until the month rolls over or POST /v1/router/reset",
            scope_id,
            reason[:200],
        )

    def exhausted_scopes(self) -> Dict[str, Dict[str, Any]]:
        month = self._month()
        with self._lock:
            stale = [k for k, v in self._exhausted.items() if v.get("month") != month]
            for key in stale:
                self._exhausted.pop(key, None)
                log.info("router: exhaustion marker for %s expired with the month", key)
            return dict(self._exhausted)

    def scope_exhausted(self, scope_id: Optional[str]) -> bool:
        """Whether one paid credential is known to be out of budget."""
        if not scope_id:
            return False
        return scope_id in self.exhausted_scopes()

    def reset_exhaustion(self) -> int:
        with self._lock:
            count = len(self._exhausted)
            self._exhausted.clear()
        log.info("router: cleared %s exhaustion marker(s)", count)
        return count

    # -- stateful Responses pinning ----------------------------------------- #

    def remember_response(self, response_id: Any, alias: str) -> None:
        if not isinstance(response_id, str) or not response_id:
            return
        with self._lock:
            self._pins[response_id] = {"alias": alias, "ts": time.time()}
            while len(self._pins) > 1000:
                self._pins.pop(next(iter(self._pins)))

    def pinned_alias(self, response_id: Any) -> Optional[str]:
        if not isinstance(response_id, str) or not response_id:
            return None
        with self._lock:
            hit = self._pins.get(response_id)
            return str(hit["alias"]) if hit else None

    # -- candidate selection ------------------------------------------------ #

    def _pool_for(self, requested: str, hints: Dict[str, Any]) -> Tuple[List[str], str]:
        if hints.get("allow_paid") is False:
            # A "spend nothing" veto outranks the alias: a routed request is
            # never failed just because the client refused to pay, so even
            # auto-hard/auto-review stay on the free pool. Only a request no
            # free model can serve (an unsupported native feature) still errors.
            return list(self.pools["free"]), "free-only"
        task = hints.get("task")
        if not task and requested in VIRTUAL_ALIASES:
            task = requested.split("-", 1)[1]
        difficulty = hints.get("difficulty")
        if task == "review":
            return self.pools["review"], "review"
        if task == "hard" or difficulty == "hard":
            return self.pools["hard"], "hard"
        free = list(self.pools["free"])
        return free + [a for a in self.pools["paid"] if a not in free], "free-first"

    def _filter(
        self,
        aliases: List[str],
        reqs: Dict[str, Any],
        state: Dict[str, Any],
        *,
        paid_allowed: bool,
        approved: bool,
        have_portkey_key: bool,
        have_nrp_key: bool,
        exhausted: Dict[str, Dict[str, Any]],
        portkey_scope: str,
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        out: List[str] = []
        skipped: List[Tuple[str, str]] = []
        for alias in aliases:
            if alias not in self.registry:
                skipped.append((alias, "unknown"))
                continue
            free = self.is_free(alias)
            reason = self._capability_block(alias, reqs)
            if reason:
                skipped.append((alias, reason))
                continue
            if free:
                if self.source_of(alias) == "nrp" and not have_nrp_key:
                    skipped.append((alias, "no-nrp-key"))
                    continue
                out.append(alias)
                continue
            if not paid_allowed:
                skipped.append((alias, "allow_paid=false"))
                continue
            if not have_portkey_key:
                skipped.append((alias, "no-portkey-key"))
                continue
            if portkey_scope and portkey_scope in exhausted:
                skipped.append((alias, "exhausted"))
                continue
            if state["budget_exhausted"]:
                skipped.append((alias, "over-budget"))
                continue
            if not approved and state["approval_required"]:
                skipped.append((alias, "approval-required"))
                continue
            out.append(alias)
        return out, skipped

    def _capability_block(self, alias: str, reqs: Dict[str, Any]) -> Optional[str]:
        caps = self._declared_caps(alias)
        if reqs["wire"] == "responses":
            if not self.responses_enabled:
                return "responses-disabled"
            if not caps["responses_native"] and not caps["responses_bridged"]:
                return "no-responses"
            if self.registry[alias].get("responses_via_chat") or not caps["responses_native"]:
                if reqs.get("native_only"):
                    return "native-param:" + ",".join(reqs["native_only"])
                if reqs.get("native_tools"):
                    return "native-tools"
                if reqs.get("stateful"):
                    return "stateful"
        if reqs.get("tools") and not caps["tools"]:
            return "no-tools"
        for feature in ("image", "audio"):
            if reqs.get(feature) and caps.get(feature) is False:
                return f"no-{feature}"
        return None

    def _paid_blocked_reason(
        self,
        state: Dict[str, Any],
        approved: bool,
        exhausted: Dict[str, Dict[str, Any]],
        portkey_scope: Optional[str],
    ) -> Optional[str]:
        """Why a paid candidate may not be dispatched right now (None = allowed)."""
        if state["budget_exhausted"]:
            return "over-budget"
        if portkey_scope and portkey_scope in exhausted:
            return "exhausted"
        if not approved and state["approval_required"]:
            return "approval-required"
        return None

    def _paid_blocked_error(
        self, reason: str, alias: str, state: Dict[str, Any]
    ) -> RoutingError:
        free = ", ".join(self.pools["free"])
        virtual = ", ".join(VIRTUAL_ALIASES)
        spend = state["spend_usd"]
        if reason in ("over-budget", "exhausted"):
            detail = (
                f"${state['known_spend_usd']:.2f} of the ${state['monthly_budget_usd']:.2f} monthly "
                "budget is already recorded"
                if reason == "over-budget"
                else "the upstream reported this API key's quota as exhausted"
            )
            return RoutingError(
                402,
                f"Paid model `{alias}` is unavailable: {detail}. Free models are "
                f"still available ({free}), and a virtual alias ({virtual}) now "
                "routes to them only. The gate lifts when the month rolls over "
                "or after POST /v1/router/reset.",
                err_type="insufficient_quota",
                code="budget_exhausted",
                param="model",
            )
        threshold = state["approval_threshold_usd"]
        detail = (
            "monthly spend cannot be read (usage tracking unavailable)"
            if spend is None
            else f"recorded monthly spend is ${spend:.2f}, at or above the "
            f"${threshold:.2f} approval threshold"
        )
        return RoutingError(
            402,
            f"Paid model `{alias}` needs explicit approval: {detail}. Resend "
            f'with "{ROUTING_FIELD}": {{"paid_approved": true}} to approve paid '
            f"use for this request only, or pick a free model ({free}) or a "
            f"virtual alias ({virtual}).",
            err_type="insufficient_quota",
            code="paid_approval_required",
            param="model",
        )

    def plan(
        self,
        requested: str,
        *,
        wire: str,
        body: Dict[str, Any],
        hints: Dict[str, Any],
        have_portkey_key: bool,
        have_nrp_key: bool,
        portkey_scope: Optional[str] = None,
    ) -> Plan:
        """Choose the ordered candidate list for one request."""
        reqs = request_requirements(wire, body, self.native_only_params)
        state = self.spend_state()
        approved = bool(hints.get("paid_approved"))
        paid_allowed = hints.get("allow_paid") is not False
        exhausted = self.exhausted_scopes()
        pinned = None
        if requested in VIRTUAL_ALIASES and reqs.get("stateful"):
            pinned = self.pinned_alias(body.get("previous_response_id"))

        if requested not in VIRTUAL_ALIASES:
            # An explicit model name stays authoritative: it is never replaced
            # by another model. Only the paid gates apply, and only while the
            # router is enabled.
            if not self.is_free(requested):
                if not paid_allowed:
                    raise RoutingError(
                        400,
                        f"Model `{requested}` is a paid model, but "
                        f"`{ROUTING_FIELD}.allow_paid` is false.",
                        param=f"{ROUTING_FIELD}.allow_paid",
                        code="invalid_routing",
                    )
                reason = self._paid_blocked_reason(
                    state, approved, exhausted, portkey_scope
                )
                if reason:
                    raise self._paid_blocked_error(reason, requested, state)
            return Plan(requested, [requested], hints=hints, gate="explicit")

        if pinned is not None:
            pool, gate = [pinned], "pinned"
        else:
            pool, gate = self._pool_for(requested, hints)
        candidates, skipped = self._filter(
            pool,
            reqs,
            state,
            paid_allowed=paid_allowed,
            approved=approved,
            have_portkey_key=have_portkey_key,
            have_nrp_key=have_nrp_key,
            exhausted=exhausted,
            portkey_scope=portkey_scope,
        )
        if (
            not candidates
            and gate in ("hard", "review")
            and paid_allowed
            and any(reason in _PAID_GATE_REASONS for _, reason in skipped)
        ):
            # "Revert to only free models": a routed request is never failed by
            # a budget gate. Only an *explicit* paid model name gets an error,
            # and an explicit allow_paid=false is honoured as a hard opt-out.
            free_candidates, free_skipped = self._filter(
                self.pools["free"],
                reqs,
                state,
                paid_allowed=True,
                approved=approved,
                have_portkey_key=have_portkey_key,
                have_nrp_key=have_nrp_key,
                exhausted=exhausted,
                portkey_scope=portkey_scope,
            )
            if free_candidates:
                gate = "paid-gated-to-free"
                candidates = free_candidates
            else:
                # Surface both blockers so the error explains every pool.
                skipped = skipped + free_skipped
        candidates = candidates[: self.max_attempts]
        if not candidates:
            raise self._no_candidate_error(reqs, skipped, state, hints)
        return Plan(
            requested,
            candidates,
            hints=hints,
            skipped=skipped,
            gate=gate,
            pinned=pinned,
        )

    def free_fallback(
        self,
        *,
        wire: str,
        body: Dict[str, Any],
        have_portkey_key: bool,
        have_nrp_key: bool,
        portkey_scope: Optional[str] = None,
    ) -> List[str]:
        """Free models that could still serve this request.

        Used mid-request: when a hard/review route's paid candidates answer
        with a budget or quota signal, the policy is to revert to free models
        rather than fail the request. Conversations pinned to a
        `previous_response_id` return nothing because switching models there
        would break the thread.
        """
        reqs = request_requirements(wire, body, self.native_only_params)
        if reqs.get("stateful"):
            return []
        candidates, _skipped = self._filter(
            self.pools["free"],
            reqs,
            self.spend_state(),
            paid_allowed=False,
            approved=False,
            have_portkey_key=have_portkey_key,
            have_nrp_key=have_nrp_key,
            exhausted={},
            portkey_scope=portkey_scope or "",
        )
        return candidates[: self.max_attempts]

    def _no_candidate_error(
        self,
        reqs: Dict[str, Any],
        skipped: List[Tuple[str, str]],
        state: Dict[str, Any],
        hints: Dict[str, Any],
    ) -> RoutingError:
        reasons = ", ".join(f"{a}: {r}" for a, r in skipped) or "no candidates configured"
        codes = {r for _, r in skipped}
        if "approval-required" in codes and not any(
            r in ("no-portkey-key", "no-nrp-key") for r in codes
        ):
            spend = state["spend_usd"]
            recorded = (
                "monthly spend cannot be read (usage tracking unavailable)"
                if spend is None
                else f"recorded monthly spend ${spend:.2f} is at or above the "
                f"${state['approval_threshold_usd']:.2f} threshold"
            )
            return RoutingError(
                402,
                "No free model can serve this request and paid models need "
                f"explicit approval ({recorded}). Resend with "
                f'"{ROUTING_FIELD}": {{"paid_approved": true}}.',
                err_type="insufficient_quota",
                code="paid_approval_required",
            )
        if state["budget_exhausted"] or "over-budget" in codes:
            return RoutingError(
                402,
                "Monthly budget is exhausted and no free model can serve this "
                f"request (excluded: {reasons}).",
                err_type="insufficient_quota",
                code="budget_exhausted",
            )
        if "no-portkey-key" in codes and "no-nrp-key" in codes:
            return RoutingError(
                401,
                "No model is reachable: the request carried no Portkey API key "
                "and NRP_API_KEY is not configured.",
                err_type="authentication_error",
                code="missing_api_key",
            )
        if "no-portkey-key" in codes and not self._any_free_eligible(reqs):
            return RoutingError(
                401,
                "This request needs a paid model and no API key was provided. "
                "Enter your Portkey / AI Sandbox key in the Copilot app.",
                err_type="authentication_error",
                code="missing_api_key",
            )
        if reqs["wire"] == "responses" and (
            reqs.get("native_only") or reqs.get("native_tools") or reqs.get("stateful")
        ):
            needed = ",".join(reqs.get("native_only") or []) or "native Responses"
            return RoutingError(
                400,
                f"No configured model can serve this Responses request without "
                f"dropping required parameters ({needed}). Excluded: {reasons}.",
                code="unsupported_native_feature",
            )
        return RoutingError(
            400,
            f"No configured model can serve this request. Excluded: {reasons}.",
            code="no_eligible_model",
        )

    def _any_free_eligible(self, reqs: Dict[str, Any]) -> bool:
        for alias in self.pools["free"]:
            if alias in self.registry and self._capability_block(alias, reqs) is None:
                return True
        return False

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "virtual_aliases": list(VIRTUAL_ALIASES),
            "pools": self.pools,
            "max_attempts": self.max_attempts,
            "deadline_seconds": self.deadline_seconds,
            "spend": self.spend_state(),
            "exhausted_scopes": sorted(self.exhausted_scopes()),
            "native_only_params": list(self.native_only_params),
        }


# --------------------------------------------------------------------------- #
# Verified upstream quota-exhaustion detection
# --------------------------------------------------------------------------- #

_EXHAUSTION_PATTERNS = (
    r"insufficient[_ ]quota",
    r"exceeded[_ ]your[_ ](current[_ ])?quota",
    r"budget[_ ](has[_ ]been[_ ])?exhausted",
    r"spending[_ ](limit|cap)",
    r"monthly[_ ](limit|cap)",
    r"credit[_ ]balance",
    r"payment[_ ]required",
    r"out[_ ]of[_ ](funds|budget|credits)",
)
_EXHAUSTION_RE = re.compile("|".join(_EXHAUSTION_PATTERNS), re.IGNORECASE)


def is_quota_exhaustion(
    status: int, message: str = "", code: Optional[str] = None
) -> bool:
    """True only for errors that *prove* the paid account is out of budget.

    Deliberately narrow: a plain 429 rate limit is transient and must keep its
    normal bounded-retry behaviour, and 401/403 are credential problems rather
    than evidence of exhausted spend. A 402, or a quota/budget phrasing that the
    upstream attaches to any status, counts as exhaustion.
    """
    if status in (401, 403):
        return False
    if status == 402:
        return True
    if code == "insufficient_quota":
        return True
    # Generic quota errors also describe short-lived token/request limits.
    if re.search(r"rate[_ ]limit|per (second|minute|hour)|retry[_ -]?after|requests? per|tokens? per", message or "", re.IGNORECASE):
        return False
    if code and _EXHAUSTION_RE.search(str(code)):
        return True
    return bool(_EXHAUSTION_RE.search(message or ""))


def month_totals(month: str) -> Dict[str, Any]:
    """Known spend + how many records for `month` lack a usable price."""
    records = usage_mod.iter_records(month)
    spend = 0.0
    unknown = 0
    for rec in records:
        cost = rec.get("cost_usd")
        if isinstance(cost, (int, float)):
            spend += float(cost)
        else:
            unknown += 1
    return {
        "month": month,
        "requests": len(records),
        "spent_usd": round(spend, 8),
        "requests_without_pricing": unknown,
    }
