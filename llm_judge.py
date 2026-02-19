"""Run structured LLM-as-judge evaluations for diagram datasets."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from pydantic import ValidationError
from tqdm import tqdm

from utils.llm_judge_schema import STRICT_RESPONSE_FORMAT, JudgeOutput
from utils.openai_judge import call_judge, extract_response_text, extract_token_usage, supports_reasoning
from utils.prompts import PromptMode, available_modes, build_prompt


RESULTS_ROOT = Path("results/llm_judge")
DEFAULT_DATASET_CSV = Path("data/geometric_shapes_test_set.csv")
models = ("gpt-4.1", "gpt-4.1-mini", "gpt-5", "gpt-5-mini")

@dataclass(frozen=True)
class JudgeSample:
    diagram_id: str
    tikz_code: Optional[str]
    image_png_path: Optional[Path]


def parse_models(raw: str) -> list[str]:
    models: list[str] = []
    for model in raw.split(","):
        candidate = model.strip()
        if candidate and candidate not in models:
            models.append(candidate)
    if not models:
        raise ValueError("At least one model must be provided.")
    return models


def load_samples(csv_path: Path, mode: PromptMode) -> list[JudgeSample]:
    df = pd.read_csv(csv_path)
    samples: list[JudgeSample] = []

    requires_tikz = mode in ("code", "both")
    requires_image = mode in ("image", "both")
    skipped_missing_tikz = 0
    skipped_missing_image = 0

    for idx, row in df.iterrows():
        diagram_id = str(row.get("diagram_id", "")).strip() or str(idx)
        tikz_code = str(row.get("tikz", "")) if not pd.isna(row.get("tikz")) else ""
        tikz_code = tikz_code or None

        image_path_value = row.get("image_png_path")
        image_path = Path(image_path_value) if isinstance(image_path_value, str) and image_path_value else None

        if requires_tikz and not tikz_code:
            skipped_missing_tikz += 1
            continue
        if requires_image and (image_path is None or not image_path.exists()):
            skipped_missing_image += 1
            continue

        samples.append(
            JudgeSample(
                diagram_id=diagram_id,
                tikz_code=tikz_code,
                image_png_path=image_path,
            )
        )

    print(
        f"Loaded {len(samples)}/{len(df)} rows "
        f"(skipped missing tikz={skipped_missing_tikz}, "
        f"missing image={skipped_missing_image})."
    )
    return samples


async def run_judge_evaluations(
    samples: Iterable[JudgeSample],
    *,
    mode: PromptMode,
    models: list[str],
    temperature: float = 0.0,
    reasoning_effort: Optional[str] = "low",
    concurrency: int = 8,
    max_retries: int = 3,
    results_root: Path = RESULTS_ROOT,
) -> None:
    sample_list = list(samples)
    if not sample_list:
        print("No samples to evaluate.")
        return

    sem = asyncio.Semaphore(concurrency)
    mode_results_root = results_root / mode
    mode_results_root.mkdir(parents=True, exist_ok=True)

    async def evaluate_one(sample: JudgeSample, model_name: str) -> None:
        safe_model = model_name.replace("/", "_")
        model_dir = mode_results_root / safe_model
        model_dir.mkdir(parents=True, exist_ok=True)
        cache_path = model_dir / f"diagram_{sample.diagram_id}.json"
        if cache_path.exists():
            return

        async with sem:
            effective_effort = reasoning_effort
            if model_name in {"gpt-5", "gpt-5-mini"} and not effective_effort:
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
                resp = await call_judge(
                    model_name=model_name,
                    prompt=prompt,
                    temperature=temperature,
                    reasoning_effort=effective_effort,
                    tikz_code=sample.tikz_code if mode in ("code", "both") else None,
                    image_bytes=image_bytes,
                    response_format=STRICT_RESPONSE_FORMAT,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                text = extract_response_text(resp)
                try:
                    structured = JudgeOutput.model_validate_json(text)
                    break
                except ValidationError as exc:
                    attempt += 1
                    if attempt >= max_retries:
                        print(
                            f"Validation failed for diagram {sample.diagram_id} ({model_name}) "
                            f"after {max_retries} attempts: {exc}"
                        )
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

            usage = extract_token_usage(resp)
            record = {
                "diagram_id": sample.diagram_id,
                "model": model_name,
                "mode": mode,
                "temperature": temperature,
                "reasoning_effort": effective_effort if supports_reasoning(model_name) else None,
                "elapsed_ms": elapsed_ms,
                "tokens": usage,
                "rubric": structured.model_dump(),
                "tikz_code": sample.tikz_code,
                "image_png_path": str(sample.image_png_path) if sample.image_png_path else None,
            }
            cache_path.write_text(json.dumps(record, indent=2))

    tasks: list[asyncio.Task[None]] = [
        asyncio.create_task(evaluate_one(sample, model_name))
        for sample in sample_list
        for model_name in models
    ]
    if not tasks:
        print("No tasks scheduled (all evaluations may already be cached).")
        return

    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Collecting judgements"):
        await fut

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATASET_CSV, help="Input CSV with diagram samples.")
    parser.add_argument(
        "--mode",
        choices=available_modes(),
        default="both",
        help="Prompt mode to evaluate.",
    )
    parser.add_argument(
        "--models",
        default=",".join(models),
        help="Comma-separated model names to evaluate.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument(
        "--reasoning-effort",
        default="low",
        help="Reasoning effort for reasoning-capable models. Use 'none' to disable.",
    )
    parser.add_argument("--concurrency", type=int, default=4, help="Max in-flight model calls.")
    parser.add_argument("--max-retries", type=int, default=3, help="Validation retry attempts.")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT, help="Directory for cached outputs.")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of samples (0 = all).")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    mode: PromptMode = args.mode
    selected_models = parse_models(args.models)

    if not args.csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.csv}")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.max_retries < 1:
        raise ValueError("--max-retries must be >= 1")

    samples = load_samples(args.csv, mode)
    if args.limit > 0:
        samples = samples[: args.limit]
    if args.limit > 0:
        print(f"Evaluating first {len(samples)} samples due to --limit={args.limit}.")

    reasoning_effort = None if str(args.reasoning_effort).strip().lower() == "none" else args.reasoning_effort

    asyncio.run(
        run_judge_evaluations(
            samples,
            mode=mode,
            models=selected_models,
            temperature=args.temperature,
            reasoning_effort=reasoning_effort,
            concurrency=args.concurrency,
            max_retries=args.max_retries,
            results_root=args.results_root,
        )
    )


if __name__ == "__main__":
    main()
