"""LLM + embedding clients that meter every call (tokens, USD, seconds) for the Flat RAG vs GraphRAG benchmark.

Providers (pick with env vars, otherwise the first one in PROVIDER_ORDER that has an API key wins):

    LLM_PROVIDER        = openai | openrouter | gemini | anthropic    (chat)
    EMBEDDING_PROVIDER  = openai | openrouter | gemini                (Anthropic has no embedding API)
    <PROVIDER>_CHAT_MODEL / <PROVIDER>_EMBEDDING_MODEL override the default models below.

One run uses one provider for the whole benchmark — no mid-run failover, so cost/quality numbers stay comparable.
"""

from __future__ import annotations

import importlib
import os
import time
from dataclasses import dataclass, fields
from typing import Any

PROVIDERS = {
    "openai": {"key": "OPENAI_API_KEY", "base_url": None,
               "chat": "gpt-4o-mini", "embed": "text-embedding-3-small"},
    "openrouter": {"key": "OPENROUTER_API_KEY", "base_url": "https://openrouter.ai/api/v1",
                   "chat": "openai/gpt-4o-mini", "embed": "openai/text-embedding-3-small"},
    "gemini": {"key": "GEMINI_API_KEY", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "chat": "gemini-2.5-flash-lite", "embed": "gemini-embedding-001"},
    "anthropic": {"key": "ANTHROPIC_API_KEY", "base_url": None,
                  "chat": "claude-opus-5-5", "embed": None},
}
PROVIDER_ORDER = ["openai", "openrouter", "gemini", "anthropic"]

# USD per 1M tokens (input, output). Check each provider's pricing page before reporting real numbers.
PRICES_PER_M = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    # Gemini embedding pricing intentionally omitted: the current pricing page does not list gemini-embedding-001.
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    seconds: float = 0.0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(*(getattr(self, f.name) + getattr(other, f.name) for f in fields(self)))

    def __sub__(self, other: "Usage") -> "Usage":
        return Usage(*(getattr(self, f.name) - getattr(other, f.name) for f in fields(self)))

def price(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    per_in, per_out = PRICES_PER_M.get(model.split("/")[-1], (0.0, 0.0))   # "openai/gpt-4o-mini" -> "gpt-4o-mini"
    return (input_tokens * per_in + output_tokens * per_out) / 1_000_000

def pick_provider(env_var: str, need_embeddings: bool) -> str:
    """Explicit env choice, else the first provider (in PROVIDER_ORDER) whose API key is set."""
    usable = [p for p in PROVIDER_ORDER if not need_embeddings or PROVIDERS[p]["embed"]]
    chosen = os.getenv(env_var, "").strip().lower()
    if chosen:
        if chosen not in usable:
            raise RuntimeError(f"{env_var}={chosen} không hợp lệ; chọn một trong: {', '.join(usable)}")
        if not os.getenv(PROVIDERS[chosen]["key"]):
            raise RuntimeError(f"{env_var}={chosen} nhưng chưa có {PROVIDERS[chosen]['key']} trong .env")
        return chosen
    for provider in usable:
        if os.getenv(PROVIDERS[provider]["key"]):
            return provider
    keys = " / ".join(PROVIDERS[p]["key"] for p in usable)
    raise RuntimeError(f"Chưa có API key nào cho {'embedding' if need_embeddings else 'chat'}: cần một trong {keys}")

def _strip_fences(text: str) -> str:
    """Some providers wrap JSON in ```json fences even when asked for raw JSON."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()

def _openai_client(provider: str):
    from openai import OpenAI

    cfg = PROVIDERS[provider]
    # Free-tier keys hit per-minute limits (429) mid-benchmark; the SDK backs off and retries.
    return OpenAI(api_key=os.environ[cfg["key"]], base_url=cfg["base_url"], max_retries=12)

class MeteredLLM:
    """`chat` and `embed` are drop-in `llm_fn` / `embedding_fn`; `usage` accumulates across calls."""

    def __init__(self, chat_provider: str | None = None, embed_provider: str | None = None) -> None:
        self.chat_provider = chat_provider or pick_provider("LLM_PROVIDER", need_embeddings=False)
        self.embed_provider = embed_provider or pick_provider("EMBEDDING_PROVIDER", need_embeddings=True)
        self.chat_model_id = os.getenv(f"{self.chat_provider.upper()}_CHAT_MODEL", PROVIDERS[self.chat_provider]["chat"])
        self.embed_model_id = os.getenv(f"{self.embed_provider.upper()}_EMBEDDING_MODEL",
                                        PROVIDERS[self.embed_provider]["embed"])
        self.chat_model = f"{self.chat_provider}:{self.chat_model_id}"
        self.embedding_model = f"{self.embed_provider}:{self.embed_model_id}"
        self._backend_name = self.embedding_model
        self.usage = Usage()
        self._chat_client: Any
        self._embed_client: Any
        if self.chat_provider == "anthropic":
            anthropic = importlib.import_module("anthropic")
            self._chat_client = anthropic.Anthropic(api_key=os.environ[PROVIDERS["anthropic"]["key"]])
        else:
            self._chat_client = _openai_client(self.chat_provider)
        self._embed_client = (self._chat_client if self.embed_provider == self.chat_provider
                              else _openai_client(self.embed_provider))

    def chat(self, prompt: str, json_mode: bool = False) -> str:
        start = time.perf_counter()
        if self.chat_provider == "anthropic":
            text, model, tokens_in, tokens_out = self._chat_anthropic(prompt)
        else:
            if json_mode and self.chat_provider != "gemini":
                response = self._chat_client.chat.completions.create(
                    model=self.chat_model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
            else:
                response = self._chat_client.chat.completions.create(
                    model=self.chat_model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
            text, model = response.choices[0].message.content or "", self.chat_model_id
            usage = response.usage
            tokens_in = usage.prompt_tokens if usage else 0
            tokens_out = usage.completion_tokens if usage else 0
        self.usage += Usage(1, tokens_in, tokens_out, price(model, tokens_in, tokens_out), time.perf_counter() - start)
        return _strip_fences(text) if json_mode else text

    def _chat_anthropic(self, prompt: str) -> tuple[str, str, int, int]:
        # Claude Opus 5.5: thinking is always on and sampling params are removed; effort is the cost lever.
        # Server-side fallback re-runs a policy-declined request on another model inside the same call.
        response = self._chat_client.beta.messages.create(
            model=self.chat_model_id,
            max_tokens=16000,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            extra_body={"fallbacks": "default"},
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            text = ""
        else:
            text = "".join(block.text for block in response.content if block.type == "text")
        return text, response.model, response.usage.input_tokens, response.usage.output_tokens

    def embed(self, text: str) -> list[float]:
        start = time.perf_counter()
        response = self._embed_client.embeddings.create(model=self.embed_model_id, input=text)
        tokens = getattr(response.usage, "prompt_tokens", 0) or 0   # some OpenAI-compatible APIs omit usage
        self.usage += Usage(1, tokens, 0, price(self.embed_model_id, tokens), time.perf_counter() - start)
        return [float(value) for value in response.data[0].embedding]

    __call__ = embed
