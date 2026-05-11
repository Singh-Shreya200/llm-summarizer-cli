"""
Library usage examples for the Summarizer.

Run with:  python example.py

Requires either OPENAI_API_KEY or ANTHROPIC_API_KEY in your environment
(load from .env via python-dotenv).
"""

from __future__ import annotations

from dotenv import load_dotenv

from summarizer import Summarizer


SAMPLE_TEXT = """
In recent months, several major cloud providers have begun offering managed
inference for open-weight language models, shifting the economics of running
mid-sized models in production. Where a year ago teams either paid per-token
to a frontier-model API or stood up GPU infrastructure themselves, the new
offerings provide pay-per-second access to dedicated inference endpoints,
typically priced 40–70% below comparable frontier-model APIs. Early adopters
report that the savings are real but uneven: workloads with steady traffic
and tolerance for slightly higher latency benefit the most, while bursty,
latency-sensitive workloads still favor the established APIs.
""".strip()


def example_single_provider(s: Summarizer) -> None:
    print("=== Example 1: single provider, TL;DR mode ===")
    result = s.summarize(text=SAMPLE_TEXT, mode="tldr", provider="anthropic")
    if result.succeeded:
        print(result.text)
        print(f"  (latency: {result.latency_seconds}s, "
              f"tokens: {result.input_tokens}→{result.output_tokens})")
    else:
        print(f"FAILED: {result.error}")
    print()


def example_executive_mode(s: Summarizer) -> None:
    print("=== Example 2: executive summary with target length ===")
    result = s.summarize(
        text=SAMPLE_TEXT,
        mode="executive",
        provider="anthropic",
        target_length="120 words",
        temperature=0.2,
    )
    print(result.text if result.succeeded else f"FAILED: {result.error}")
    print()


def example_comparison(s: Summarizer) -> None:
    print("=== Example 3: side-by-side comparison (both providers in parallel) ===")
    comparison = s.compare(text=SAMPLE_TEXT, mode="bullets")
    print("--- OpenAI ---")
    print(comparison.openai.text if comparison.openai.succeeded
          else f"FAILED: {comparison.openai.error}")
    print()
    print("--- Anthropic ---")
    print(comparison.anthropic.text if comparison.anthropic.succeeded
          else f"FAILED: {comparison.anthropic.error}")
    print()


def example_error_handling(s: Summarizer) -> None:
    print("=== Example 4: errors come back as fields, not exceptions ===")
    result = s.summarize(text=SAMPLE_TEXT, mode="tldr", provider="openai")
    if result.succeeded:
        print(f"Got: {result.text[:80]}...")
    else:
        # No try/except needed — the result tells you what happened.
        print(f"Handled cleanly: {result.error}")
    print()


def main() -> None:
    load_dotenv()
    s = Summarizer()

    example_single_provider(s)
    example_executive_mode(s)
    example_comparison(s)
    example_error_handling(s)


if __name__ == "__main__":
    main()
