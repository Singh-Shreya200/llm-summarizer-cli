# Dual-Provider Summarizer

A Python tool that summarizes text using **OpenAI** and **Anthropic** APIs through a unified interface. Use either provider individually, or run both in parallel to compare outputs side-by-side.

Built as a portfolio piece demonstrating clean API integration, prompt engineering, and CLI design.

---

## Why this exists

Most "call an LLM API" examples are either toy snippets that don't survive contact with production, or sprawling frameworks where the core logic is buried under abstraction layers. This project sits in the middle: small enough to read end-to-end in 15 minutes, structured enough that adding a new provider (Gemini, Mistral, etc.) is a 30-line change.

It demonstrates three things:

1. **API integration** — proper SDK usage, error handling, lazy initialization, parallel I/O.
2. **Prompt engineering** — five distinct summarization modes, each with a system prompt, structured output requirements, and explicit anti-hallucination constraints. Prompts live in their own module, separated from the calling code.
3. **Tool design** — a usable CLI with stdin/file/inline input, JSON output, and side-by-side provider comparison.

---

## Features

- ✅ Supports **OpenAI** (GPT-4o family) and **Anthropic** (Claude family) out of the box
- ✅ **5 summarization modes** — TL;DR, Executive, Bullets, Technical, ELI5
- ✅ **Side-by-side comparison** — runs both providers in parallel (not serial), so total latency is `max(a, b)`, not `a + b`
- ✅ **Multiple input sources** — file, stdin (piped), or inline text
- ✅ **JSON output** for piping into other tools
- ✅ **Configurable** — model, temperature, max tokens, target length
- ✅ **Graceful error handling** — errors come back as fields on the result, not exceptions, so a single failing provider doesn't crash a comparison run
- ✅ **Library + CLI** — use it from Python code, or from the shell

---

## Installation

```bash
git clone <this-repo>
cd summarizer
pip install -r requirements.txt
```

Set up API keys:

```bash
cp .env.example .env
# Then edit .env and add your keys
```

You need at least one of:
- `OPENAI_API_KEY` — get from <https://platform.openai.com/api-keys>
- `ANTHROPIC_API_KEY` — get from <https://console.anthropic.com/>

Both keys are required if you want to use `--compare`.

---

## CLI Usage

### Quick examples

```bash
# Summarize a file with the default provider (Anthropic) and mode (TL;DR)
python cli.py --file sample_article.txt

# Pick a mode and provider
python cli.py --file sample_article.txt --mode executive --provider openai

# Pipe input from stdin
cat sample_article.txt | python cli.py --mode bullets

# Inline text
python cli.py --text "Long text here..." --mode eli5

# 🎯 Side-by-side comparison of both providers
python cli.py --file sample_article.txt --mode executive --compare

# JSON output for downstream tools
python cli.py --file sample_article.txt --mode tldr --json

# List available modes
python cli.py --list-modes

# Use specific models
python cli.py --file article.txt --provider openai --openai-model gpt-4o
python cli.py --file article.txt --provider anthropic --anthropic-model claude-opus-4-7
```

### All flags

```
--file, -f             Path to a text file
--text, -t             Inline text to summarize
--mode, -m             Summarization mode (default: tldr)
--provider, -p         openai or anthropic (default: anthropic)
--compare, -c          Run both providers in parallel
--openai-model         OpenAI model (default: gpt-4o-mini)
--anthropic-model      Anthropic model (default: claude-haiku-4-5-20251001)
--target-length        Optional length, e.g. "100 words" or "3 paragraphs"
--max-tokens           Max output tokens (default: 1024)
--temperature          Sampling temperature, 0.0–1.0 (default: 0.3)
--json                 JSON output instead of formatted text
--list-modes           Show available modes and exit
```

---

## Summarization Modes

Each mode is a carefully designed prompt with a specific system role, structured output format, and explicit constraints. See [`prompts.py`](./prompts.py) for the full templates.

| Mode | When to use | Output shape |
|------|------------|--------------|
| `tldr` | The "if I read nothing else" version | 1–2 sentences |
| `executive` | Briefing for a busy decision-maker | Bottom line + implications + key facts + what to watch |
| `bullets` | Scannable structured digest | Main point + key points + notable details |
| `technical` | Engineering or research audience | Subject + method + findings + caveats + open questions, with [established / claimed / speculative] tags |
| `eli5` | Plain-English explainer | The gist + an everyday-life analogy + practical implications |

All modes include explicit anti-hallucination constraints ("do not invent facts, if unclear leave out") and anti-padding constraints ("no preamble, no filler phrases").

---

## Library Usage

The CLI is just a thin wrapper. The library is the real thing:

```python
from summarizer import Summarizer

s = Summarizer()

# Single provider
result = s.summarize(
    text="...",
    mode="executive",
    provider="anthropic",
)
print(result.text)
print(f"Latency: {result.latency_seconds}s")
print(f"Tokens: {result.input_tokens} in / {result.output_tokens} out")

# Side-by-side comparison (both providers, in parallel)
comparison = s.compare(text="...", mode="bullets")
print(comparison.openai.text)
print(comparison.anthropic.text)

# With custom models and parameters
s = Summarizer(
    openai_model="gpt-4o",
    anthropic_model="claude-opus-4-7",
)
result = s.summarize(
    text="...",
    mode="technical",
    provider="anthropic",
    target_length="200 words",
    temperature=0.2,
    max_tokens=2048,
)

# Error handling is built in — failures come back as fields, not exceptions
result = s.summarize("...", mode="tldr", provider="openai")
if result.succeeded:
    print(result.text)
else:
    print(f"Failed: {result.error}")
```

Run [`example.py`](./example.py) to see all the patterns in action:

```bash
python example.py
```

---

## Project Structure

```
summarizer/
├── prompts.py              # Summarization prompt templates (the heart of the project)
├── summarizer.py           # Provider abstraction + Summarizer class
├── cli.py                  # Command-line interface
├── example.py              # Library usage examples
├── sample_article.txt      # Sample input for testing
├── requirements.txt        # Pinned dependencies
├── .env.example            # Template for API keys
├── .gitignore              # Protects .env from being committed
└── README.md               # This file
```

The split between `prompts.py` and `summarizer.py` is deliberate. Prompts are first-class artifacts — they get iterated on, tested, versioned. Keeping them in their own module means you can change a prompt without touching the API-calling code, and vice versa.

---

## Design Decisions

A few choices worth calling out:

**Provider abstraction over direct calls.** It would be 30% less code to put both API calls in one function with an `if provider == "openai":` branch. The abstraction earns its place because the two SDKs differ in non-trivial ways (Anthropic takes `system` as a separate parameter; OpenAI puts it in the messages array; Anthropic returns content blocks, OpenAI returns a string), and concentrating those differences in one place keeps the rest of the code clean.

**Lazy provider instantiation.** You shouldn't need both API keys to use one provider. The `Summarizer` class only initializes a provider the first time you actually use it, so the missing-key error only fires if you actually call that provider.

**Parallel comparison via `ThreadPoolExecutor`.** Both API calls are I/O-bound network requests — running them serially would double the latency for no reason. A thread pool is the right primitive here (over `asyncio`) because it works with the synchronous SDKs without rewriting the call sites.

**Errors as values, not exceptions.** When you're running both providers in parallel, one of them failing shouldn't kill the whole comparison. The provider's `summarize()` method catches exceptions and returns them as `SummaryResult.error` — callers can check `result.succeeded` instead of wrapping every call in try/except.

**Temperature defaults to 0.3.** For summarization, you want consistency more than creativity. 0.3 is low enough to reduce variance run-to-run, high enough to avoid the stiff feel of temp=0.

**Prompts include explicit "do not hallucinate" rules.** Every prompt has lines like *"do not invent facts"* and *"if unclear, leave it out."* These don't fully eliminate hallucination, but they meaningfully reduce the rate at which models fill in gaps with plausible-sounding fabrications.

---

## Extending

### Adding a new provider (e.g., Gemini)

Subclass `LLMProvider` and implement `_call()`:

```python
class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, api_key=None, model="gemini-pro"):
        import google.generativeai as genai
        genai.configure(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
        self.client = genai.GenerativeModel(model)
        self.model = model

    def _call(self, system_prompt, user_prompt, max_tokens, temperature):
        # Translate to Gemini's API
        # Return (text, input_tokens, output_tokens)
        ...
```

Then wire it into `Summarizer` the same way `openai` and `anthropic` are wired.

### Adding a new summarization mode

In `prompts.py`, add a new `PromptTemplate` and register it in the `MODES` dict. The CLI and library will pick it up automatically.

---

## What's intentionally not in this project

- **Streaming responses** — added complexity for a tool whose outputs are short by design.
- **Retries with exponential backoff** — both SDKs handle this internally by default.
- **Caching** — would be the right addition for production use, but adds storage/keying decisions that distract from the core demo.
- **Chunking long inputs** — relevant for very long documents (>100K chars). Worth adding as a v2 feature; out of scope here.
- **Cost estimation** — pricing tables change frequently and would date this code. The token counts in `SummaryResult` give you what you need to compute cost against current pricing.

These are deliberate omissions, not oversights. A portfolio piece is more valuable when it does a focused thing well than when it tries to do everything.

---

## License

MIT.
