"""LLM-as-Judge prompt loader."""
from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from typing import Literal, Optional

PromptMode = Literal["image", "code", "both"]

_PROMPT_FILES = {
    "image": Path("prompt_llmasjudge_imageonly.txt"),
    "code": Path("prompt_llmasjudge_codeonly.txt"),
    "both": Path("prompt_llmasjudge_both.txt"),
}

@lru_cache(maxsize=None)
def load_prompt(mode: PromptMode) -> str:
    file_path = _PROMPT_FILES.get(mode)
    if file_path is None:
        raise ValueError(f"Unsupported prompt mode: {mode}")
    try:
        return file_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Prompt file not found for mode '{mode}': {file_path}") from exc


def available_modes() -> tuple[PromptMode, ...]:
    return tuple(_PROMPT_FILES.keys())  # type: ignore[return-value]


def build_prompt(
    mode: PromptMode,
    *,
    tikz_code: Optional[str] = None,
    include_image: bool = False,
) -> str:
    """Return a prompt string augmented with optional TikZ code or image hint."""

    base = load_prompt(mode).rstrip()
    extra_parts: list[str] = []

    if tikz_code:
        extra_parts.append("\nTikZ code:\n" + tikz_code.strip())

    if include_image:
        extra_parts.append("\nThe corresponding diagram image is attached.")

    if not extra_parts:
        return base + "\n"

    return base + "".join(extra_parts) + "\n"
