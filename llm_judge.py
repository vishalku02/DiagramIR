# code from interestingness project

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from pydantic import ValidationError
from tqdm import tqdm

from llm_judge_schema import STRICT_RESPONSE_FORMAT, JudgeOutput
from models import ModelFactory
from prompts import PromptMode, build_prompt


RESULTS_ROOT = Path("results/llm_judge")


@dataclass
class JudgeSample:
    diagram_id: str
    tikz_code: Optional[str]
    image_png_path: Optional[Path]


def load_samples(csv_path: Path, mode: PromptMode) -> List[JudgeSample]:
    df = pd.read_csv(csv_path)
    samples: List[JudgeSample] = []

    requires_tikz = mode in ("code", "both")
    requires_image = mode in ("image", "both")

    for _, row in df.iterrows():
        diagram_id = str(row.get("diagram_id", "")) or str(len(samples))
        tikz_code = str(row.get("tikz", "")) if not pd.isna(row.get("tikz")) else ""
        tikz_code = tikz_code or None
        image_path_value = row.get("image_png_path")
        image_path = Path(image_path_value) if isinstance(image_path_value, str) and image_path_value else None

        if requires_tikz and not tikz_code:
            continue
        if requires_image and (image_path is None or not image_path.exists()):
            continue

        samples.append(
            JudgeSample(
                diagram_id=diagram_id,
                tikz_code=tikz_code,
                image_png_path=image_path,
            )
        )

    return samples
def temp_key(t: float) -> str:
    return f"{float(t):.2f}"

async def run_judge_evaluations(
    samples: Iterable[JudgeSample],
    *,
    mode: PromptMode,
    models: List[str],
    temperature: float = 0.0,
    reasoning_effort: str = "low",
    concurrency: int = 8,
    max_retries: int = 3,
    results_root: Path = RESULTS_ROOT,
) -> None:
    samples = list(samples)
    if not samples:
        print("No samples to evaluate.")
        return

    sem = asyncio.Semaphore(concurrency)
    factory = ModelFactory()
    results_root = results_root / mode
    results_root.mkdir(parents=True, exist_ok=True)

    async def one_judgement(sample: JudgeSample, model_name: str) -> None:
        safe_model = model_name.replace("/", "_")
        model_dir = results_root / safe_model
        model_dir.mkdir(parents=True, exist_ok=True)
        cache_path = model_dir / f"diagram_{sample.diagram_id}.json"
        if cache_path.exists():
            return

        async with sem:
            llm = factory.make(model_name)
            effective_effort = reasoning_effort
            if llm.info.name == "gpt-5" and not effective_effort:
                effective_effort = "low"

            include_image = mode in ("image", "both") and sample.image_png_path is not None
            prompt = build_prompt(
                mode,
                tikz_code=sample.tikz_code if mode in ("code", "both") else None,
                include_image=include_image,
            )
            image_bytes = None
            if include_image and sample.image_png_path and sample.image_png_path.exists():
                image_bytes = sample.image_png_path.read_bytes()

            attempt = 0
            while attempt < max_retries:
                start = time.perf_counter()
                resp = await llm.get_judgement_async(
                    prompt=prompt,
                    temperature=temperature,
                    reasoning_effort=effective_effort,
                    tikz_code=sample.tikz_code if mode in ("code", "both") else None,
                    image_bytes=image_bytes,
                    response_format=STRICT_RESPONSE_FORMAT,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                text = llm.get_response_text(resp)
                try:
                    structured = JudgeOutput.model_validate_json(text)
                    break
                except ValidationError as exc:
                    attempt += 1
                    if attempt >= max_retries:
                        print(f"Validation failed for diagram {sample.diagram_id} ({model_name}) after {max_retries} attempts: {exc}")
                        failure_record = {
                            "diagram_id": sample.diagram_id,
                            "model": model_name,
                            "reasoning_effort": effective_effort,
                            "mode": mode,
                            "error": str(exc),
                            "response_text": text,
                        }
                        cache_path.write_text(json.dumps(failure_record, indent=2))
                        return
                    continue

            usage = llm.get_token_usage(resp)
            record = {
                "diagram_id": sample.diagram_id,
                "model": model_name,
                "mode": mode,
                "temperature": temperature,
                "reasoning_effort": effective_effort if llm.info.supports_reasoning else None,
                "elapsed_ms": elapsed_ms,
                "tokens": usage,
                "rubric": structured.model_dump(),
                "tikz_code": sample.tikz_code,
                "image_png_path": str(sample.image_png_path) if sample.image_png_path else None,
            }
            cache_path.write_text(json.dumps(record, indent=2))

    tasks = []
    for sample in samples:
        for model_name in models:
            tasks.append(asyncio.create_task(one_judgement(sample, model_name)))

    if not tasks:
        print("No tasks scheduled (all evaluations may already be cached).")
        return

    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Collecting judgements"):
        await fut


def main():
    csv_path = Path("data/geometric_shapes_test_set.csv")
    mode: PromptMode = "both"
    samples = load_samples(csv_path, mode)
    # samples = load_samples(csv_path, mode)[:10] # for a dry run

    models = ["gpt-4.1", "gpt-4.1-mini", "gpt-5", "gpt-5-mini"]

    asyncio.run(
        run_judge_evaluations(
            samples,
            mode=mode,
            models=models,
            temperature=0.0,
            reasoning_effort="low",
            concurrency=4,
        )
    )

if __name__ == "__main__":
    main()
