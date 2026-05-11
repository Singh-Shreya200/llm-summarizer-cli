"""
Summarization prompt templates.

Prompts are first-class artifacts in this project — they live in their own
module so they can be iterated on without touching the API-calling code.

Each mode has:
- A system prompt (role + behavior + output structure + constraints)
- A user prompt template (injects the input text and any length hint)

All modes share two anti-failure rules:
- "Do not invent facts" — reduces hallucination
- "No preamble or filler" — reduces "Here is a summary of..." padding
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ----------------------------------------------------------------------
# Template type
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class PromptTemplate:
    """One summarization mode's prompts.

    `render()` returns (system_prompt, user_prompt) ready to send to a provider.
    """

    name: str
    description: str
    system_prompt: str
    user_prompt_template: str

    def render(
        self,
        text: str,
        target_length: Optional[str] = None,
    ) -> tuple[str, str]:
        length_clause = (
            f"\n\nTarget length: {target_length}." if target_length else ""
        )
        user_prompt = self.user_prompt_template.format(
            text=text,
            length_clause=length_clause,
        )
        return self.system_prompt, user_prompt


# ----------------------------------------------------------------------
# Shared constraints — repeated across modes
# ----------------------------------------------------------------------
_ANTI_HALLUCINATION = (
    "Do not invent facts, numbers, names, or details that are not in the "
    "source text. If something is unclear or missing, leave it out rather "
    "than guessing."
)

_NO_PREAMBLE = (
    "Output the summary directly. No preamble (\"Here is a summary...\"), "
    "no meta-commentary, no closing remarks."
)


# ----------------------------------------------------------------------
# Mode: TL;DR
# ----------------------------------------------------------------------
TLDR = PromptTemplate(
    name="tldr",
    description="The 'if I read nothing else' version — 1–2 sentences.",
    system_prompt=(
        "You are an expert at compressing text to its essential point.\n\n"
        "Your task: write a 1–2 sentence TL;DR that captures the single most "
        "important takeaway from the source text. Lead with the takeaway, not "
        "with what the text 'is about'.\n\n"
        f"{_ANTI_HALLUCINATION}\n\n"
        f"{_NO_PREAMBLE}"
    ),
    user_prompt_template=(
        "Summarize the following text as a TL;DR (1–2 sentences):"
        "{length_clause}\n\n"
        "---\n{text}\n---"
    ),
)


# ----------------------------------------------------------------------
# Mode: Executive
# ----------------------------------------------------------------------
EXECUTIVE = PromptTemplate(
    name="executive",
    description="Briefing for a busy decision-maker.",
    system_prompt=(
        "You are a senior analyst writing a briefing for a busy executive who "
        "has 60 seconds.\n\n"
        "Your task: produce a structured executive summary using exactly these "
        "four sections, in this order:\n\n"
        "**Bottom line:** One sentence — the single thing the executive needs "
        "to know.\n"
        "**Implications:** 2–3 bullets — what this means for decisions or "
        "strategy.\n"
        "**Key facts:** 2–4 bullets — the specific facts/figures that support "
        "the above.\n"
        "**What to watch:** 1–2 bullets — open questions or things that could "
        "change the picture.\n\n"
        f"{_ANTI_HALLUCINATION}\n\n"
        f"{_NO_PREAMBLE}"
    ),
    user_prompt_template=(
        "Write an executive briefing on the following text:"
        "{length_clause}\n\n"
        "---\n{text}\n---"
    ),
)


# ----------------------------------------------------------------------
# Mode: Bullets
# ----------------------------------------------------------------------
BULLETS = PromptTemplate(
    name="bullets",
    description="Scannable structured digest.",
    system_prompt=(
        "You produce scannable, structured digests of text.\n\n"
        "Your task: output a digest with exactly this structure:\n\n"
        "**Main point:** One sentence stating the central claim or topic.\n\n"
        "**Key points:**\n"
        "- 3–6 bullets covering the most important supporting points.\n"
        "- Each bullet should be one sentence, complete and self-contained.\n\n"
        "**Notable details:**\n"
        "- 1–3 bullets with specifics worth remembering (numbers, names, "
        "examples, caveats).\n\n"
        f"{_ANTI_HALLUCINATION}\n\n"
        f"{_NO_PREAMBLE}"
    ),
    user_prompt_template=(
        "Produce a bulleted digest of the following text:"
        "{length_clause}\n\n"
        "---\n{text}\n---"
    ),
)


# ----------------------------------------------------------------------
# Mode: Technical
# ----------------------------------------------------------------------
TECHNICAL = PromptTemplate(
    name="technical",
    description="Engineering- or research-grade summary with epistemic tags.",
    system_prompt=(
        "You summarize technical content for an audience of engineers and "
        "researchers who care about precision and provenance.\n\n"
        "Your task: produce a summary with exactly these five sections:\n\n"
        "**Subject:** What is being studied/built/proposed.\n"
        "**Method:** How — approach, technique, design, or experimental setup.\n"
        "**Findings:** Specific results or conclusions.\n"
        "**Caveats:** Limitations, scope conditions, or known weaknesses.\n"
        "**Open questions:** What remains unresolved or worth investigating.\n\n"
        "For each substantive claim, append an epistemic tag in brackets:\n"
        "- **[established]** — directly stated and well-supported in the text\n"
        "- **[claimed]** — asserted by the source but not independently verified\n"
        "- **[speculative]** — interpretation, projection, or implication\n\n"
        f"{_ANTI_HALLUCINATION}\n\n"
        f"{_NO_PREAMBLE}"
    ),
    user_prompt_template=(
        "Produce a technical summary of the following text:"
        "{length_clause}\n\n"
        "---\n{text}\n---"
    ),
)


# ----------------------------------------------------------------------
# Mode: ELI5
# ----------------------------------------------------------------------
ELI5 = PromptTemplate(
    name="eli5",
    description="Plain-English explainer with an everyday analogy.",
    system_prompt=(
        "You explain complex topics in plain English to someone with no "
        "background in the subject. Think bright, curious 12-year-old — not "
        "condescending, just clear.\n\n"
        "Your task: produce an explanation with exactly these three sections:\n\n"
        "**The gist:** 2–3 sentences in everyday words. No jargon. If a "
        "technical term is unavoidable, define it inline.\n\n"
        "**An analogy:** One concrete, everyday-life comparison that captures "
        "the core idea. Pick an analogy a non-expert can picture.\n\n"
        "**Why it matters:** 1–2 sentences on the practical implication — what "
        "changes, who's affected, or what to do with this knowledge.\n\n"
        f"{_ANTI_HALLUCINATION}\n\n"
        f"{_NO_PREAMBLE}"
    ),
    user_prompt_template=(
        "Explain the following text in plain English:"
        "{length_clause}\n\n"
        "---\n{text}\n---"
    ),
)


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------
MODES: dict[str, PromptTemplate] = {
    TLDR.name: TLDR,
    EXECUTIVE.name: EXECUTIVE,
    BULLETS.name: BULLETS,
    TECHNICAL.name: TECHNICAL,
    ELI5.name: ELI5,
}


def get_mode(name: str) -> PromptTemplate:
    """Look up a mode by name. Raises ValueError with a helpful message."""
    key = name.lower().strip()
    if key not in MODES:
        available = ", ".join(sorted(MODES.keys()))
        raise ValueError(
            f"Unknown mode: '{name}'. Available modes: {available}."
        )
    return MODES[key]


def list_modes() -> list[tuple[str, str]]:
    """Return (name, description) pairs for all registered modes."""
    return [(m.name, m.description) for m in MODES.values()]
