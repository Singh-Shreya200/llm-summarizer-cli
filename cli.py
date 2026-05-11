"""
Command-line interface for the summarizer.

Thin wrapper over the Summarizer library — handles argument parsing, input
sources (file/stdin/inline), output formatting (human/JSON), and the
side-by-side --compare mode.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from prompts import get_mode, list_modes
from summarizer import ComparisonResult, SummaryResult, Summarizer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="summarizer",
        description=(
            "Summarize text using OpenAI and/or Anthropic. "
            "Input can come from --file, --text, or stdin."
        ),
    )

    src = parser.add_argument_group("input source (pick one, or pipe via stdin)")
    src.add_argument("--file", "-f", type=Path, help="Path to a text file")
    src.add_argument("--text", "-t", type=str, help="Inline text to summarize")

    mode_group = parser.add_argument_group("summarization")
    mode_group.add_argument(
        "--mode", "-m", default="tldr",
        help="Summarization mode (default: tldr). Use --list-modes to see all.",
    )
    mode_group.add_argument(
        "--provider", "-p", default="anthropic", choices=["openai", "anthropic"],
        help="Which provider to use (default: anthropic).",
    )
    mode_group.add_argument(
        "--compare", "-c", action="store_true",
        help="Run both providers in parallel and show results side-by-side.",
    )

    models = parser.add_argument_group("model selection")
    models.add_argument("--openai-model", default="gpt-4o-mini")
    models.add_argument("--anthropic-model", default="claude-haiku-4-5-20251001")

    tuning = parser.add_argument_group("tuning")
    tuning.add_argument(
        "--target-length",
        help='Optional length hint, e.g. "100 words" or "3 paragraphs".',
    )
    tuning.add_argument("--max-tokens", type=int, default=1024)
    tuning.add_argument("--temperature", type=float, default=0.3)

    output = parser.add_argument_group("output")
    output.add_argument(
        "--json", action="store_true",
        help="Emit a JSON object instead of formatted text.",
    )
    output.add_argument(
        "--list-modes", action="store_true",
        help="List available summarization modes and exit.",
    )

    return parser


def _read_input(args: argparse.Namespace) -> str:
    """Resolve input from --file, --text, or stdin (in that order of priority)."""
    if args.file:
        if not args.file.is_file():
            sys.exit(f"Error: file not found: {args.file}")
        return args.file.read_text(encoding="utf-8")

    if args.text:
        return args.text

    if not sys.stdin.isatty():
        piped = sys.stdin.read()
        if piped.strip():
            return piped

    sys.exit(
        "Error: no input. Use --file, --text, or pipe text via stdin.\n"
        "       Run with --help for usage."
    )


def _format_result(r: SummaryResult) -> str:
    if not r.succeeded:
        return (
            f"[{r.provider} / {r.model}] FAILED after {r.latency_seconds}s\n"
            f"  → {r.error}"
        )
    return (
        f"[{r.provider} / {r.model}] mode={r.mode} "
        f"latency={r.latency_seconds}s "
        f"tokens={r.input_tokens}→{r.output_tokens}\n"
        f"{'-' * 60}\n"
        f"{r.text}"
    )


def _format_comparison(c: ComparisonResult) -> str:
    header = (
        f"=== Comparison (mode={c.mode}, input={c.input_chars} chars) ===\n"
    )
    return (
        header
        + "\n--- OpenAI ---\n" + _format_result(c.openai)
        + "\n\n--- Anthropic ---\n" + _format_result(c.anthropic)
    )


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_modes:
        print("Available modes:")
        for name, description in list_modes():
            print(f"  {name:<10}  {description}")
        return 0

    # Validate mode up front so a bad mode name doesn't get shadowed by
    # a later provider-init error (e.g., missing API key).
    try:
        get_mode(args.mode)
    except ValueError as e:
        sys.exit(f"Error: {e}")

    text = _read_input(args)

    summarizer = Summarizer(
        openai_model=args.openai_model,
        anthropic_model=args.anthropic_model,
    )

    try:
        if args.compare:
            result = summarizer.compare(
                text=text,
                mode=args.mode,
                target_length=args.target_length,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            if args.json:
                print(json.dumps(result.to_dict(), indent=2))
            else:
                print(_format_comparison(result))
            any_failed = not (result.openai.succeeded and result.anthropic.succeeded)
            return 1 if any_failed else 0

        result = summarizer.summarize(
            text=text,
            mode=args.mode,
            provider=args.provider,
            target_length=args.target_length,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(_format_result(result))
        return 0 if result.succeeded else 1
    except (ValueError, ImportError) as e:
        # Configuration errors (missing API key, bad mode, missing SDK) —
        # show cleanly instead of a traceback.
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    sys.exit(main())
