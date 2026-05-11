"""
Dual-provider summarizer: calls OpenAI and Anthropic APIs through a
unified interface.

Design notes
------------
The two APIs are *similar enough* that you might be tempted to call them
directly from a single function with a few if-statements. Don't. The
Provider abstraction here is doing real work:

1.  It isolates SDK-specific quirks (Anthropic uses `system` as a separate
    parameter; OpenAI puts system in the messages array) into one place.
2.  It normalizes the response shape, so callers get the same
    `SummaryResult` regardless of which provider ran.
3.  It makes adding a new provider (Gemini, Mistral, a local model)
    a 30-line change instead of a project-wide refactor.

The Summarizer class is the main entry point. Use compare() to run both
providers on the same input — useful for evaluating prompts.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import Optional

from prompts import get_mode


# ----------------------------------------------------------------------
# Result type — what every provider returns
# ----------------------------------------------------------------------
@dataclass
class SummaryResult:
    """Normalized result from any provider."""
    text: str
    provider: str
    model: str
    mode: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------
# Abstract provider interface
# ----------------------------------------------------------------------
class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Concrete implementations only need to translate between the unified
    summarize() call and their SDK's specific request/response shape.
    """

    name: str = "unknown"
    model: str = "unknown"

    @abstractmethod
    def _call(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, int, int]:
        """
        Make the actual API call.

        Returns:
            (response_text, input_tokens, output_tokens)
        """

    def summarize(
        self,
        text: str,
        mode: str = "tldr",
        target_length: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> SummaryResult:
        """High-level summarize call. Catches errors so callers see them
        as fields on SummaryResult rather than uncaught exceptions —
        useful when running multiple providers in parallel."""
        template = get_mode(mode)
        system_prompt, user_prompt = template.render(text, target_length=target_length)

        start = time.monotonic()
        try:
            response_text, in_tokens, out_tokens = self._call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            elapsed = time.monotonic() - start
            return SummaryResult(
                text=response_text.strip(),
                provider=self.name,
                model=self.model,
                mode=mode,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                latency_seconds=round(elapsed, 2),
            )
        except Exception as e:
            elapsed = time.monotonic() - start
            return SummaryResult(
                text="",
                provider=self.name,
                model=self.model,
                mode=mode,
                latency_seconds=round(elapsed, 2),
                error=f"{type(e).__name__}: {e}",
            )


# ----------------------------------------------------------------------
# OpenAI provider
# ----------------------------------------------------------------------
class OpenAIProvider(LLMProvider):
    """Provider for OpenAI's Chat Completions API."""

    name = "openai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            ) from e

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY env var or "
                "pass api_key= to OpenAIProvider()."
            )
        self.client = OpenAI(api_key=key)
        self.model = model

    def _call(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, int, int]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = response.choices[0].message.content or ""
        # Defensive: usage can be None on some streaming/edge responses.
        usage = response.usage
        in_tokens = usage.prompt_tokens if usage else 0
        out_tokens = usage.completion_tokens if usage else 0
        return text, in_tokens, out_tokens


# ----------------------------------------------------------------------
# Anthropic provider
# ----------------------------------------------------------------------
class AnthropicProvider(LLMProvider):
    """Provider for Anthropic's Messages API."""

    name = "anthropic"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-haiku-4-5-20251001",
    ):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            ) from e

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY env var or "
                "pass api_key= to AnthropicProvider()."
            )
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model

    def _call(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, int, int]:
        # Anthropic's API takes `system` as a separate top-level parameter,
        # not as a message in the `messages` array. This is one of the
        # SDK differences the Provider abstraction hides from callers.
        response = self.client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        # Anthropic returns content as a list of content blocks. For text
        # responses there's typically one TextBlock; we concatenate any
        # text blocks defensively.
        text = "".join(
            block.text for block in response.content
            if getattr(block, "type", None) == "text"
        )
        return text, response.usage.input_tokens, response.usage.output_tokens


# ----------------------------------------------------------------------
# Comparison result — what compare() returns
# ----------------------------------------------------------------------
@dataclass
class ComparisonResult:
    """The result of running both providers on the same input."""
    openai: SummaryResult
    anthropic: SummaryResult
    mode: str
    input_chars: int

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "input_chars": self.input_chars,
            "openai": self.openai.to_dict(),
            "anthropic": self.anthropic.to_dict(),
        }


# ----------------------------------------------------------------------
# High-level Summarizer — the main interface for callers
# ----------------------------------------------------------------------
class Summarizer:
    """High-level summarizer that supports single-provider or dual-provider calls.

    Example:
        >>> s = Summarizer()
        >>> result = s.summarize("Long text here...", mode="executive", provider="anthropic")
        >>> print(result.text)

        >>> # Run both providers in parallel for comparison:
        >>> comparison = s.compare("Long text here...", mode="bullets")
        >>> print(comparison.openai.text)
        >>> print(comparison.anthropic.text)
    """

    def __init__(
        self,
        openai_model: str = "gpt-4o-mini",
        anthropic_model: str = "claude-haiku-4-5-20251001",
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
    ):
        self.openai_model = openai_model
        self.anthropic_model = anthropic_model
        self._openai_key = openai_api_key
        self._anthropic_key = anthropic_api_key
        # Providers are lazily instantiated so you can use just one
        # provider without needing both API keys present.
        self._openai: Optional[OpenAIProvider] = None
        self._anthropic: Optional[AnthropicProvider] = None

    @property
    def openai(self) -> OpenAIProvider:
        if self._openai is None:
            self._openai = OpenAIProvider(
                api_key=self._openai_key, model=self.openai_model
            )
        return self._openai

    @property
    def anthropic(self) -> AnthropicProvider:
        if self._anthropic is None:
            self._anthropic = AnthropicProvider(
                api_key=self._anthropic_key, model=self.anthropic_model
            )
        return self._anthropic

    def summarize(
        self,
        text: str,
        mode: str = "tldr",
        provider: str = "anthropic",
        target_length: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> SummaryResult:
        """Summarize using a single provider."""
        text = self._validate_input(text)
        provider_obj = self._resolve_provider(provider)
        return provider_obj.summarize(
            text=text,
            mode=mode,
            target_length=target_length,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def compare(
        self,
        text: str,
        mode: str = "tldr",
        target_length: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> ComparisonResult:
        """Run both providers on the same input, in parallel.

        Uses a thread pool because both calls are I/O-bound (network)
        and we want to overlap their latency, not stack it.
        """
        text = self._validate_input(text)

        def call(provider: LLMProvider) -> SummaryResult:
            return provider.summarize(
                text=text,
                mode=mode,
                target_length=target_length,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            future_openai = pool.submit(call, self.openai)
            future_anthropic = pool.submit(call, self.anthropic)
            openai_result = future_openai.result()
            anthropic_result = future_anthropic.result()

        return ComparisonResult(
            openai=openai_result,
            anthropic=anthropic_result,
            mode=mode,
            input_chars=len(text),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _resolve_provider(self, name: str) -> LLMProvider:
        name = name.lower().strip()
        if name == "openai":
            return self.openai
        if name == "anthropic":
            return self.anthropic
        raise ValueError(
            f"Unknown provider: '{name}'. Use 'openai' or 'anthropic'."
        )

    @staticmethod
    def _validate_input(text: str) -> str:
        if not isinstance(text, str):
            raise TypeError(f"Input must be a string, got {type(text).__name__}.")
        text = text.strip()
        if not text:
            raise ValueError("Input text is empty.")
        # A soft warning rather than a hard limit. Real apps would chunk
        # long inputs; for a portfolio tool, we let the API surface the
        # actual context-window error if it's truly too long.
        if len(text) > 200_000:
            import warnings
            warnings.warn(
                f"Input is very long ({len(text):,} chars). May exceed "
                "the model's context window."
            )
        return text
