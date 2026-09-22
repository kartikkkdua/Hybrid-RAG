"""Anthropic client wrapper with token + cost accounting.

Every call returns a `Usage` (input/output tokens, USD cost, latency, model) so
the A/B harness and the API can report cost/latency per query. If no API key is
configured, `available` is False and callers fall back to extractive mode.

PRICE_TABLE is USD per 1M tokens. Update these to match current pricing — they
are the single source of truth for all cost numbers in the project.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .config import Settings
from .models import Usage

# USD per 1,000,000 tokens (input, output). Editable — see module docstring.
PRICE_TABLE: dict[str, tuple[float, float]] = {
    "claude-opus-5": (15.0, 75.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
    "claude-fable-5-1": (3.0, 15.0),
}
_DEFAULT_PRICE = (3.0, 15.0)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = PRICE_TABLE.get(model, _DEFAULT_PRICE)
    return round(input_tokens / 1e6 * pin + output_tokens / 1e6 * pout, 6)


@dataclass
class LLMResult:
    text: str
    usage: Usage


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = None
        self.available = bool(settings.anthropic_api_key)
        if self.available:
            try:
                from anthropic import Anthropic

                self._client = Anthropic(api_key=settings.anthropic_api_key)
            except Exception:
                self.available = False

    def complete(
        self,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResult:
        if not self.available:
            raise RuntimeError("No ANTHROPIC_API_KEY configured; LLM generation unavailable.")
        model = model or self.settings.gen_model
        t0 = time.perf_counter()
        resp = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        text = "".join(getattr(b, "text", "") for b in resp.content)
        in_tok = resp.usage.input_tokens
        out_tok = resp.usage.output_tokens
        usage = Usage(
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=estimate_cost(model, in_tok, out_tok),
            latency_ms=latency_ms,
            model=model,
        )
        return LLMResult(text=text, usage=usage)

    def stream(self, system: str, user: str, model: str | None = None, max_tokens: int = 1024):
        """Yield text deltas, then a final ('__usage__', Usage) tuple."""
        if not self.available:
            raise RuntimeError("No ANTHROPIC_API_KEY configured; streaming unavailable.")
        model = model or self.settings.gen_model
        t0 = time.perf_counter()
        with self._client.messages.stream(
            model=model, max_tokens=max_tokens, temperature=0.0,
            system=system, messages=[{"role": "user", "content": user}],
        ) as stream:
            for text in stream.text_stream:
                yield ("delta", text)
            final = stream.get_final_message()
        in_tok = final.usage.input_tokens
        out_tok = final.usage.output_tokens
        usage = Usage(
            input_tokens=in_tok, output_tokens=out_tok,
            cost_usd=estimate_cost(model, in_tok, out_tok),
            latency_ms=round((time.perf_counter() - t0) * 1000, 2), model=model,
        )
        yield ("usage", usage)
