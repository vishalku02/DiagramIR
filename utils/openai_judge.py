"""OpenAI Responses API helpers for the LLM-as-judge pipeline."""
from __future__ import annotations

import asyncio
import base64
from typing import Any, Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential


OPENAI_REASONING_MODELS = {"o3", "gpt-5", "gpt-5-mini"}
NO_TEMPERATURE_MODELS = {"o3", "gpt-5", "gpt-5-mini"}

_openai_sync: Optional[OpenAI] = None


def get_openai_sync() -> OpenAI:
    global _openai_sync
    if _openai_sync is None:
        _openai_sync = OpenAI()
    return _openai_sync


def _normalize_model_name(model_name: str) -> str:
    name = model_name.strip()
    if not name:
        raise ValueError("model_name must be non-empty.")
    if "/" in name:
        raise ValueError(f"Unsupported model '{name}'. This project supports OpenAI model IDs only.")
    return name


def _get(obj: Any, attr: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def _convert_response_format(response_format: Optional[dict]) -> Optional[dict]:
    if not response_format:
        return None
    if response_format.get("type") != "json_schema":
        return response_format

    json_schema_cfg = dict(response_format.get("json_schema") or {})
    fmt: dict[str, Any] = {
        "type": "json_schema",
        "name": json_schema_cfg.get("name", "Schema"),
        "schema": json_schema_cfg.get("schema", {}),
    }
    if "strict" in json_schema_cfg:
        fmt["strict"] = json_schema_cfg.get("strict")
    if "description" in json_schema_cfg and json_schema_cfg.get("description"):
        fmt["description"] = json_schema_cfg.get("description")
    return fmt


def supports_reasoning(model_name: str) -> bool:
    return _normalize_model_name(model_name) in OPENAI_REASONING_MODELS


def build_judge_input(
    prompt: str,
    *,
    tikz_code: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    image_media_type: str = "image/png",
) -> list[dict[str, Any]]:
    if not prompt:
        raise ValueError("Prompt text must be provided for LLM judgement")

    content: list[dict[str, str]] = [{"type": "input_text", "text": prompt}]

    if tikz_code:
        formatted = tikz_code if tikz_code.endswith("\n") else f"{tikz_code}\n"
        content.append({"type": "input_text", "text": f"TikZ code:\n{formatted}"})

    if image_bytes:
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:{image_media_type};base64,{b64_image}",
            }
        )

    return content


@retry(stop=stop_after_attempt(5), wait=wait_random_exponential(min=1, max=20))
async def call_judge(
    *,
    model_name: str,
    prompt: str,
    temperature: float,
    reasoning_effort: Optional[str] = None,
    max_tokens: int = 100000,
    tikz_code: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    image_media_type: str = "image/png",
    response_format: Optional[dict] = None,
    client: Optional[OpenAI] = None,
) -> Any:
    model = _normalize_model_name(model_name)
    resolved_client = client or get_openai_sync()
    content = build_judge_input(
        prompt,
        tikz_code=tikz_code,
        image_bytes=image_bytes,
        image_media_type=image_media_type,
    )

    def _call_openai_sync() -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": max_tokens,
        }

        if response_format:
            fmt = _convert_response_format(response_format)
            if fmt:
                kwargs["text"] = {"format": fmt}

        if model not in NO_TEMPERATURE_MODELS:
            kwargs["temperature"] = temperature
        else:
            effort_value = reasoning_effort if reasoning_effort else None
            if model in {"gpt-5", "gpt-5-mini"} and effort_value is None:
                effort_value = "low"
            if effort_value:
                kwargs["reasoning"] = {"effort": effort_value}

        return resolved_client.responses.create(**kwargs)

    return await asyncio.to_thread(_call_openai_sync)


def extract_response_text(response: Any) -> str:
    txt = getattr(response, "output_text", None)
    if isinstance(txt, str) and txt:
        return txt

    output = getattr(response, "output", None) or []
    for item in output:
        if getattr(item, "type", None) != "message":
            continue
        parts = getattr(item, "content", None) or []
        texts: list[str] = []
        for part in parts:
            value = getattr(part, "text", None)
            if value:
                texts.append(value)
        if texts:
            return "".join(texts)

    return ""


def extract_token_usage(response: Any) -> dict:
    usage = _get(response, "usage")
    input_tokens = _get(usage, "input_tokens")
    output_tokens = _get(usage, "output_tokens")
    input_details = _get(usage, "input_tokens_details")
    cached_tokens = _get(input_details, "cached_tokens") if input_details else None
    total_tokens = _get(usage, "total_tokens")
    if total_tokens is None and (input_tokens is not None and output_tokens is not None):
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "cached_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
