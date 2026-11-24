"""Token pricing for the Responses provider, and the cost of one run.

The Responses API reports token counts but never money, so a run that is not
priced locally records $0 — and `total_cost_usd`, the per-round cost columns,
and the experiment report all silently read as free. This module supplies the
missing half.

`agents._summarize_session` prices the legacy Claude proxy path from its own
`_PRICE_*` constants; those rates do not apply here and are not shared.

Rates are USD per 1,000,000 tokens, keyed by the repo's model alias (the
provider-namespaced form, e.g. `openai/gpt-5.6-sol`, resolves to the same
entry). Update them from the provider's model listing when they change:

    curl -s https://openrouter.ai/api/v1/models | \
      jq '.data[] | select(.id=="openai/gpt-5.6-sol") | .pricing'

An unpriced model is reported as `priced=False` with `cost_usd=0.0` rather
than guessed, so "free" and "unknown" stay distinguishable downstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Verified against the OpenRouter model listing on 2026-09-08.
PRICING_SOURCE = "openrouter.ai/api/v1/models, 2026-09-08"


@dataclass(frozen=True)
class Rates:
    """USD per 1M tokens for one model."""

    prompt: float          # uncached input
    cache_read: float      # input served from the provider's prompt cache
    cache_write: float     # input written into that cache
    completion: float      # output, reasoning tokens included


RATES: dict[str, Rates] = {
    "gpt-5.6-sol": Rates(prompt=2.0, cache_read=0.2, cache_write=2.5, completion=10.0),
    # Same list price as gpt-5.6-sol on the OpenRouter listing, 2026-09-24.
    "gpt-6-sol": Rates(prompt=2.0, cache_read=0.2, cache_write=2.5, completion=10.0),
    # OpenRouter listing 2026-10-01 (openai/gpt-6.1-sol): same prompt/completion
    # as gpt-6-sol, cache read halved to $0.10.
    "gpt-6.1-sol": Rates(prompt=2.0, cache_read=0.1, cache_write=2.5, completion=10.0),
    # OpenRouter listing 2026-09-29: 5x gpt-6-sol on every rate.
    "gpt-6-astra": Rates(prompt=10.0, cache_read=1.0, cache_write=12.5, completion=50.0),
    # Anthropic list price 2026-09 (via the local gateway, same rates):
    # $4 / $20 per MTok, cache read $0.20, cache write 1.25x prompt.
    "claude-opus-5-5": Rates(prompt=4.0, cache_read=0.2, cache_write=5.0, completion=20.0),
    "gpt-5.6-luna": Rates(prompt=0.2, cache_read=0.02, cache_write=0.25, completion=1.2),
    "gpt-5.6-luna-pro": Rates(prompt=0.2, cache_read=0.02, cache_write=0.25, completion=1.2),
}


def rates_for(model: str) -> Rates | None:
    """Look up rates for a model alias, tolerating a provider namespace."""
    if not model:
        return None
    return RATES.get(model) or RATES.get(model.rsplit("/", 1)[-1])


def _int(value: Any) -> int:
    """Coerce a possibly-missing provider count to a non-negative int."""
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _detail(entry: Any, group: str, field: str) -> int:
    """Read `entry.<group>.<field>` whether it is an object or a plain dict."""
    details = (
        entry.get(group) if isinstance(entry, dict) else getattr(entry, group, None)
    )
    if details is None:
        return 0
    if isinstance(details, dict):
        return _int(details.get(field))
    return _int(getattr(details, field, 0))


def token_breakdown(usage: Any) -> dict[str, int]:
    """Split a Responses `Usage` into billable buckets.

    The aggregate `input_tokens` counts cached and uncached input together,
    and the pinned openai client's aggregate `InputTokensDetails` has no
    cache-write field at all, so the per-request entries are the only place
    the full breakdown survives. Fall back to the aggregate when a provider
    omits them — cache writes are then unknown and priced as plain input,
    which overstates rather than hides the cost.
    """
    entries = getattr(usage, "request_usage_entries", None) or []
    if entries:
        fresh = cache_read = cache_write = output = 0
        for entry in entries:
            total_in = _int(
                entry.get("input_tokens") if isinstance(entry, dict)
                else getattr(entry, "input_tokens", 0)
            )
            cached = _detail(entry, "input_tokens_details", "cached_tokens")
            cache_read += cached
            cache_write += _detail(entry, "input_tokens_details", "cache_write_tokens")
            fresh += max(0, total_in - cached)
            output += _int(
                entry.get("output_tokens") if isinstance(entry, dict)
                else getattr(entry, "output_tokens", 0)
            )
    else:
        cache_read = _detail(usage, "input_tokens_details", "cached_tokens")
        cache_write = 0
        fresh = max(0, _int(getattr(usage, "input_tokens", 0)) - cache_read)
        output = _int(getattr(usage, "output_tokens", 0))
    return {
        "fresh_input": fresh,
        "cache_read": cache_read,
        "cache_write": cache_write,
        "output": output,
        "reasoning": _detail(usage, "output_tokens_details", "reasoning_tokens"),
    }


def cost_usd(model: str, tokens: dict[str, int]) -> tuple[float, bool]:
    """Price a `token_breakdown` result. Returns (cost, priced)."""
    rates = rates_for(model)
    if rates is None:
        return 0.0, False
    total = (
        tokens.get("fresh_input", 0) * rates.prompt
        + tokens.get("cache_read", 0) * rates.cache_read
        + tokens.get("cache_write", 0) * rates.cache_write
        + tokens.get("output", 0) * rates.completion
    ) / 1_000_000
    return total, True


def summarize(usage: Any, model: str) -> dict[str, Any]:
    """Build the loop's usage-summary dict for one Responses run.

    Key names match `agents._summarize_session` so the two cost paths stay
    interchangeable for every consumer (result.json, the round tables, and
    the experiment report).
    """
    tokens = token_breakdown(usage)
    total, priced = cost_usd(model, tokens)
    return {
        "n_api_calls": _int(getattr(usage, "requests", 0)),
        "fresh_input": tokens["fresh_input"],
        "cache_read": tokens["cache_read"],
        # The provider does not distinguish 5m/1h cache TTLs the way the
        # Claude path does; report the single bucket it does report.
        "cache_write_5m": tokens["cache_write"],
        "cache_write_1h": 0,
        "output": tokens["output"],
        "reasoning_output": tokens["reasoning"],
        "cost_usd": round(total, 6),
        "cost_priced": priced,
        "cost_model": model,
        "cache_misses": 0,
    }


__all__ = [
    "RATES",
    "Rates",
    "PRICING_SOURCE",
    "rates_for",
    "token_breakdown",
    "cost_usd",
    "summarize",
]
