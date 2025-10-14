from openai import AsyncOpenAI
import asyncio
import json
import os
import pathlib
import shutil
import subprocess
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv
from together import Together

from IR_model import TikzIR
load_dotenv()
MODEL = os.environ["OPENAI_MODEL"]

BACKTRANSLATION_PROMPT_SMALL = """You are a deterministic parser that extracts geometric entities from TikZ code into JSON that matches the provided schema.

### Rules
1. Extract ONLY entities explicitly present in the TikZ code.
2. Omit empty fields (do not include keys with empty lists).
3. Preserve exact numerical coordinates from the code. Resolve and compute if necessary (e.g. \\def or \\newcommand), but DO NOT infer extra ones that are not present in the code or complete partial shapes.
4. For shapes array, list vertices in the order they are drawn, set "cycle": true if the draw command ends with `-- cycle`.
5. For rectangle_primitives array, set "is_right_angle_symbol": true when the rectangle is drawn as a right-angle marker (e.g., tiny square sharing corners with two incident edges or comments mentioning a right angle). Otherwise set it to false. If a \draw explicitly passes the 'right angle symbol' option, do not add it as a rectangle_primitive.
6. For 3D parts (shapes w/ 3D coords) and Ucubes, use one integer id per physical solid (e.g., 1, 1, 1). Faces or unit cube entries that belong to the same block—especially those emitted inside the same scope/loop—must reuse that id; only assign a new id when you are describing a genuinely different solid.
7. When you encounter the helper macro `\\Ucube` (or any change of coordinates that draws the front/right/top faces of a unit cube), output a single shape object with `"type": "Ucube"` instead of three separate 3D-part entries. The cube should include its `id`, `size` (usually `[1,1,1]` unless the macro scales it), the scope transform (`scale`, `shift`, `xshift`, `yshift`), and `fill`. Do not emit the individual faces separately.
8. If a scope applies transformations (shift, scale, xshift, yshift, rotate), include them in the optional transform object. Do not numerically apply the transform.
9. Transform separation: 
   - `transform.shift` ← only TikZ's `shift={...}` argument.  
   - `transform.xshift` ← only TikZ's `xshift=...`.  
   - `transform.yshift` ← only TikZ's `yshift=...`.  
   These must NEVER be combined. If xshift/yshift values are omitted or folded into `shift`, the JSON is invalid.
10. For node options such as `rotate=...`, set the node's `node_rotate` field. Keep scope-level rotations in `transform` and do not duplicate them in `node_rotate`.
11. For tikzpicture_options, map x, y, and z to the corresponding options, or fill out 'scale' if the options include scale.
12. Loop expansion: Expand every `\\foreach` loop literally. Substitute each variable with its values and emit the corresponding repeated scopes and draw commands. Never summarize or replace with a generic cube — output must reflect exactly the iterations.
13. For custom commands, resolve the command using this table:
IM_MACROS = [
        "\\TFP": "4.875in",
        "\\TTP": "4.2in",
        "\\TwoThirdsPage": "4.2in",
        "\\HP": "3.25in",
        "\\HalfPage": "3.25in",
        "\\THP": "2.1in",
        "\\ThirdPage": "2.1in",
        "\\QP": "1.625in",
    "\\QuarterPage": "1.625in",
    ]
13. Resolve relative coordinate syntax precisely: when you encounter forms like `+(...)`, `++(...)`, or `($(P)!t!(Q)$)`, evaluate them to absolute coordinates using previously defined points. For `++`, remember it updates the current point before the next coordinate is processed.

### Example

TikZ:
```latex
\\foreach \\x in {0,1} {
  \\begin{scope}[scale=0.5, shift={(\\x,0,0)}, xshift=-2in, yshift=-3in]
    \\draw (0,0,0) -- (1,0,0) -- (1,1,0) -- (0,1,0) -- cycle;
  \\end{scope}
}
```

JSON:

```json
{
  "shapes": [
    {
      "type": "3D-part",
      "vertices": [[0,0,0],[1,0,0],[1,1,0],[0,1,0]],
      "cycle": true,
      "id": 1,
      "transform": {
        "scale": 0.5,
        "shift": [0,0,0],
        "xshift": "-2in",
        "yshift": "-3in"
      }
    },
    {
      "type": "3D-part",
      "vertices": [[0,0,0],[1,0,0],[1,1,0],[0,1,0]],
      "cycle": true,
      "id": 2,
      "transform": {
        "scale": 0.5,
        "shift": [1,0,0],
        "xshift": "-2in",
        "yshift": "-3in"
      }
    }
  ]
}
```

### TikZ
{tikz_code}

### Output
JSON only, no explanations.
"""

TOGETHER_MODEL_ALIASES = {
    "llama-4-maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "llama-4-mavrick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "llama-4-scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
}

async def tikz_to_ir(tikz_code: str) -> Tuple[Dict[str, Any], Dict[str, int]]:
    prompt_template = BACKTRANSLATION_PROMPT_SMALL.strip()
    prompt = prompt_template.replace("{tikz_code}", tikz_code)

    canonical_model = TOGETHER_MODEL_ALIASES.get(MODEL.lower(), MODEL)

    args = {
        "model": canonical_model,
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
    is_gpt5 = "gpt-5" in canonical_model.lower()
    if not is_gpt5:
        args["temperature"] = 0.0
        args["top_p"] = 1.0

    # Use OpenAI Responses API for GPT models
    if canonical_model.lower().startswith("gpt") and "/" not in canonical_model:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY must be set for GPT models")
        client = AsyncOpenAI(api_key=api_key)
        if is_gpt5:
            args["reasoning"] = {"effort": "low"}
        resp = await client.responses.create(**args)

        token_usage = {
            "prompt_tokens": getattr(resp.usage, "input_tokens", getattr(resp.usage, "prompt_tokens", 0)) if getattr(resp, "usage", None) else 0,
            "completion_tokens": getattr(resp.usage, "output_tokens", getattr(resp.usage, "completion_tokens", 0)) if getattr(resp, "usage", None) else 0,
            "total_tokens": getattr(resp.usage, "total_tokens", 0) if getattr(resp, "usage", None) else 0,
        }

        return json.loads(resp.output_text), token_usage

    # Otherwise treat as Together chat completion
    together_client = Together()
    messages = []
    for message in args.get("input", []):
        messages.append({
            "role": message.get("role", "user"),
            "content": message.get("content", ""),
        })

    payload: Dict[str, Any] = {
        "model": canonical_model,
        "messages": messages,
        "temperature": 0.0 if not is_gpt5 else None,
        "top_p": 1.0 if not is_gpt5 else None,
    }

    format_spec = args.get("text", {}).get("format") if args.get("text") else None
    if format_spec:
        fmt_type = format_spec.get("type")
        if fmt_type == "json_object":
            payload["response_format"] = {"type": "json_object"}
        elif fmt_type == "json_schema":
            schema_payload: Dict[str, Any] = {
                "type": "json_schema",
                "json_schema": {
                    "name": format_spec.get("name", "Schema"),
                    "schema": format_spec.get("schema", {}),
                },
            }
            if "strict" in format_spec:
                schema_payload["json_schema"]["strict"] = format_spec["strict"]
            payload["response_format"] = schema_payload

    if payload["temperature"] is None:
        payload.pop("temperature")
    if payload["top_p"] is None:
        payload.pop("top_p")

    raw_resp = await asyncio.to_thread(together_client.chat.completions.create, **payload)
    content = raw_resp.choices[0].message.content if raw_resp.choices else ""
    usage = getattr(raw_resp, "usage", None)
    token_usage = {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
        "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
    }

    return json.loads(content), token_usage

def compile_tikz(code: str, output_path: pathlib.Path, output_format: str = "png"):
    """Compile TikZ to PNG or SVG using lualatex + ImageMagick/dvisvgm (--pdf)."""
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
