# code copied from interestingness project
from __future__ import annotations

import asyncio
import base64
import re
import tiktoken
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Callable, List, Dict
from openai import OpenAI
from together import Together

from tenacity import retry, stop_after_attempt, wait_random_exponential


_openai_sync: Optional[OpenAI] = None
_together_sync: Optional[Together] = None

def get_openai_sync() -> OpenAI:
    global _openai_sync
    if _openai_sync is None:
        _openai_sync = OpenAI()
    return _openai_sync

def get_together_sync() -> Together:
    global _together_sync
    if _together_sync is None:
        _together_sync = Together()
    return _together_sync

@dataclass(frozen=True)
class ModelInfo:
    name: str
    provider: str                 # "openai" or "together"
    supports_reasoning: bool      # whether we expose reasoning-only helpers


class BaseLLM(ABC):
    def __init__(self, info: ModelInfo,
                 get_client_sync: Callable[[], Any]):
        self.info = info
        self._get_client_sync = get_client_sync

    @staticmethod
    def _get(obj, attr, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr, default)
    
    @abstractmethod
    async def get_judgement_async(
        self,
        *,
        prompt: str,
        temperature: float,
        reasoning_effort: Optional[str] = None,
        max_tokens: int = 100000,
        tikz_code: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        image_media_type: str = "image/png",
        response_format: Optional[dict] = None,
    ) -> Any:
        """
        Return the raw SDK response object (OpenAI Responses or Together chat.completions).
        """
        pass

    @staticmethod
    def _assemble_user_content(
        prompt: str,
        tikz_code: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        image_media_type: str = "image/png",
    ) -> List[Dict[str, Any]]:
        if not prompt:
            raise ValueError("Prompt text must be provided for LLM judgement")

        content: List[Dict[str, str]] = [
            {"type": "input_text", "text": prompt}
        ]

        if tikz_code:
            formatted = tikz_code if tikz_code.endswith("\n") else f"{tikz_code}\n"
            content.append({
                "type": "input_text",
                "text": f"TikZ code:\n{formatted}"
            })

        if image_bytes:
            b64_image = base64.b64encode(image_bytes).decode("utf-8")
            data_uri = f"data:{image_media_type};base64,{b64_image}"
            content.append({
                "type": "input_image",
                "image_url": data_uri,
            })

        return content

    # TODO
    # extracts response text from Response object
    def get_response_text(self, response: Any) -> str:
        """
        Extract message text from either:
          - OpenAI Responses API object (gpt-4o, o3), or
          - Together OpenAI-compatible chat.completions object.
        """
        txt = getattr(response, "output_text", None)
        if isinstance(txt, str) and txt:
            return txt

        output = getattr(response, "output", None) or []
        for item in output:
            if getattr(item, "type", None) == "message":
                parts = getattr(item, "content", None) or []
                texts = []
                for p in parts:
                    t = getattr(p, "text", None)
                    if t:
                        texts.append(t)
                if texts:
                    return "".join(texts)

        # Fallback for Together (OpenAI-compatible)
        choices = getattr(response, "choices", None)
        if choices and len(choices) > 0:
            msg = getattr(choices[0], "message", None)
            if msg:
                content = getattr(msg, "content", None)
                if isinstance(content, str):
                    return content

        return ""
    
    # for reasoning models, depending on if it together or openai, returns tokens
    def count_reasoning_tokens(self, response: Any) -> int:
        """
        - OpenAI o3: use response.usage.output_tokens_details.reasoning_tokens
        - DeepSeek-R1 on Together: count tokens inside <think>...</think>
        Raises if this model is not marked as supports_reasoning.
        """
        if not self.info.supports_reasoning:
            raise ValueError(f"Model {self.info.name} does not expose reasoning tokens.")

        # OpenAI o3 branch
        if self.info.provider == "openai" and self.info.name == "o3":
            usage = getattr(response, "usage", None)
            if not usage:
                return 0
            details = getattr(usage, "output_tokens_details", None)
            if not details:
                return 0
            return int(getattr(details, "reasoning_tokens", 0))

        # DeepSeek-R1 on Together branch
        if self.info.provider == "together" and self.info.name in ["deepseek-ai/DeepSeek-R1", "Qwen/Qwen3-235B-A22B-Thinking-2507", "Qwen/QwQ-32B"]:
            text = self.get_response_text(response)
            if not text:
                return 0
            # Extract everything inside <think>...</think>
            matches = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
            if not matches:
                return 0
            reasoning_text = "\n".join(matches).strip()
            if not reasoning_text:
                return 0
            # Tokenize with tiktoken if available, else fallback to char count
            if tiktoken is not None:
                try:
                    enc = tiktoken.encoding_for_model("gpt-4o")  # reasonable default
                except Exception:
                    enc = tiktoken.get_encoding("cl100k_base")
                return len(enc.encode(reasoning_text))
            else:
                return len(reasoning_text.encode("utf-8"))

        # If you add more reasoning-capable Together/OpenAI models, extend here.
        return 0

    def get_token_usage(self, response) -> dict:
        """
        Returns a normalized dict:
        {
          "input_tokens": int|None,
          "cached_tokens": int|None,  # only for OpenAI (Responses) when present
          "output_tokens": int|None,
          "total_tokens": int|None
        }
        """
        u = self._get(response, "usage")

        if self.info.provider == "openai":
            it = self._get(u, "input_tokens")
            ot = self._get(u, "output_tokens")
            itd = self._get(u, "input_tokens_details")
            cached = self._get(itd, "cached_tokens") if itd else None
            total = self._get(u, "total_tokens")
            if total is None and (it is not None and ot is not None):
                total = it + ot
            return {
                "input_tokens": it,
                "cached_tokens": cached,
                "output_tokens": ot,
                "total_tokens": total,
            }

        # Together (chat.completions) usage fields
        pt = self._get(u, "prompt_tokens")
        ct = self._get(u, "completion_tokens")
        total = self._get(u, "total_tokens")
        return {
            "input_tokens": pt,
            "cached_tokens": None,    # Together doesn’t expose cached token count
            "output_tokens": ct,
            "total_tokens": total,
        }

class OpenAIModel(BaseLLM):
    """
    Wrapper for OpenAI models using the Responses API.
    Works for models like 'gpt-4o' and 'o3'.
    """

    @retry(stop=stop_after_attempt(5), wait=wait_random_exponential(min=1, max=20))
    async def get_judgement_async(
        self,
        *,
        prompt: str,
        temperature: float,
        reasoning_effort: Optional[str] = None,
        max_tokens: int = 100000,
        tikz_code: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        image_media_type: str = "image/png",
        response_format: Optional[dict] = None,
    ) -> Any:
        content = self._assemble_user_content(
            prompt=prompt,
            tikz_code=tikz_code,
            image_bytes=image_bytes,
            image_media_type=image_media_type,
        )

        def _convert_response_format_for_openai(rfmt: Optional[dict]) -> Optional[dict]:
            if not rfmt:
                return None
            if rfmt.get("type") != "json_schema":
                return rfmt

            json_schema_cfg = dict(rfmt.get("json_schema") or {})
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

        def _call_openai_sync() -> Any:
            client = self._get_client_sync()
            kwargs = dict(
                model=self.info.name,
                input=[{"role": "user", "content": content}],
                max_output_tokens=max_tokens,
            )
            if response_format:
                fmt = _convert_response_format_for_openai(response_format)
                if fmt:
                    kwargs["text"] = {"format": fmt}
            no_temp_models = {"o3", "gpt-5", "gpt-5-mini"}
            if self.info.name not in no_temp_models:
                kwargs["temperature"] = temperature
            else:
                effort_value = reasoning_effort if reasoning_effort else None
                if self.info.name in {"gpt-5", "gpt-5-mini"} and effort_value is None:
                    effort_value = "low"
                if effort_value:
                    kwargs["reasoning"] = {"effort": effort_value}
            return client.responses.create(**kwargs)

        return await asyncio.to_thread(_call_openai_sync)

class TogetherAIModel(BaseLLM):
    """
    Wrapper for Together models using their OpenAI-compatible chat.completions API.
    Example model: 'deepseek-ai/DeepSeek-R1'
    """

    @retry(stop=stop_after_attempt(5), wait=wait_random_exponential(min=1, max=20))
    async def get_judgement_async(
        self,
        *,
        prompt: str,
        temperature: float,
        reasoning_effort: Optional[str] = None,  # not used by Together
        max_tokens: int = 30000,  # TODO change
        tikz_code: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        image_media_type: str = "image/png",
        response_format: Optional[dict] = None,
    ) -> Any:
        content = self._assemble_user_content(
            prompt=prompt,
            tikz_code=tikz_code,
            image_bytes=image_bytes,
            image_media_type=image_media_type,
        )

        def _call_together_sync() -> Any:
            client = self._get_client_sync()

            # Convert Responses-style content into chat.completions content pieces.
            chat_parts: List[Dict[str, Any]] = []
            for item in content:
                if item.get("type") == "input_text":
                    chat_parts.append({"type": "text", "text": item.get("text", "")})
                elif item.get("type") == "input_image":
                    image_entry = item.get("image_url")
                    if isinstance(image_entry, str):
                        image_entry = {"url": image_entry}
                    chat_parts.append({"type": "image_url", "image_url": image_entry})

            if len(chat_parts) == 1 and chat_parts[0].get("type") == "text":
                message_content: Any = chat_parts[0]["text"]
            else:
                message_content = chat_parts

            kwargs = {
                "model": self.info.name,
                "messages": [{"role": "user", "content": message_content}],
                "max_tokens": max_tokens,
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            if response_format:
                kwargs["response_format"] = response_format

            return client.chat.completions.create(**kwargs)

        return await asyncio.to_thread(_call_together_sync)



class ModelFactory:
    """
    Construct the right wrapper from a model name.
    You provide how to build the sync client for each provider.
    """

    def __init__(self):
        self._openai = get_openai_sync
        self._together = get_together_sync

    def make(self, model_name: str) -> BaseLLM:
        # Declare which models support reasoning
        openai_reasoning = {"o3", "gpt-5", "gpt-5-mini"}
        together_reasoning = {"deepseek-ai/DeepSeek-R1", "Qwen/Qwen3-235B-A22B-Thinking-2507"}

        if model_name in {"gpt-4o", "o3", "gpt-5", "gpt-5-mini", "gpt-4.1", "gpt-4.1-mini"}:
            info = ModelInfo(
                name=model_name,
                provider="openai",
                supports_reasoning=(model_name in openai_reasoning),
            )
            return OpenAIModel(info, self._openai)

        # Default to Together for everything else passed here
        info = ModelInfo(
            name=model_name,
            provider="together",
            supports_reasoning=(model_name in together_reasoning),
        )
        return TogetherAIModel(info, self._together)
