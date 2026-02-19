"""Backtranslation utilities for TikZ -> IR pipeline and diagram rendering."""

import asyncio
import json
import os
import pathlib
import shutil
import subprocess
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv
from openai import AsyncOpenAI

from utils.ir_schema import TikzIR
load_dotenv()

BACKTRANSLATION_PROMPT_PATH = pathlib.Path(__file__).resolve().parent / "utils" / "prompt_backtranslation.md"


@lru_cache(maxsize=1)
def load_backtranslation_prompt() -> str:
    try:
        return BACKTRANSLATION_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Backtranslation prompt file not found: {BACKTRANSLATION_PROMPT_PATH}"
        ) from exc

def _resolve_model_name(model_name: Optional[str] = None) -> str:
    resolved = model_name or os.environ.get("OPENAI_MODEL")
    if not resolved:
        raise RuntimeError(
            "OPENAI_MODEL must be set (or passed explicitly as model_name) to run tikz_to_ir."
        )
    if "/" in resolved:
        raise ValueError(
            f"Unsupported model '{resolved}'. This project now supports OpenAI model IDs only."
        )
    return resolved


async def tikz_to_ir(
    tikz_code: str,
    *,
    model_name: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    prompt_template = load_backtranslation_prompt()
    prompt = prompt_template.replace("{tikz_code}", tikz_code)

    model = _resolve_model_name(model_name)

    args = {
        "model": model,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "TikzIR",
                "strict": False,
                "schema": TikzIR.model_json_schema(),
            }
        },
        "input": [{"role": "user", "content": prompt}],
    }
    is_gpt5 = "gpt-5" in model.lower()
    if not is_gpt5:
        args["temperature"] = 0.0
        args["top_p"] = 1.0

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set.")
    client = AsyncOpenAI(api_key=api_key)
    if is_gpt5:
        args["reasoning"] = {"effort": "low"}
    resp = await client.responses.create(**args)

    usage = getattr(resp, "usage", None)
    token_usage = {
        "prompt_tokens": getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0)) if usage else 0,
        "completion_tokens": getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0)) if usage else 0,
        "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
    }

    return json.loads(resp.output_text), token_usage

def compile_tikz(code: str, output_path: pathlib.Path, output_format: str = "png"):
    """Compile TikZ to PNG or SVG using lualatex + ImageMagick/dvisvgm (--pdf)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".tex")
    temp_path.write_text(code, encoding='utf-8')
    temp_dir = output_path.parent

    for f in ["IMlongdivision.sty", "Tikz-IM.sty", "Tikz-IM-ES.sty", "IM.cls"]:
        src = pathlib.Path("styles") / f
        if src.exists():
            shutil.copy(src, temp_dir / f)

    latex_engine = os.environ.get("LATEX_ENGINE", "lualatex")
    imagemagick_bin = os.environ.get("IMAGEMAGICK_BIN", "magick")
    dvisvgm_bin = os.environ.get("DVISVGM_BIN", "dvisvgm")

    try:
        # compile once with lualatex to produce PDF
        subprocess.run(
            [latex_engine, "-interaction=nonstopmode", "-output-directory", str(temp_dir), str(temp_path)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        
        if output_format.lower() == "svg":
            pdf_path = temp_path.with_suffix(".pdf")
            subprocess.run(
                [dvisvgm_bin, "--pdf", str(pdf_path), "-o", str(output_path)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        else:  # png
            subprocess.run(
                [imagemagick_bin, "-density", "300", str(temp_path.with_suffix(".pdf")), str(output_path)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        return True
    except subprocess.CalledProcessError:
        return False
