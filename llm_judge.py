"""Run structured LLM-as-judge evaluations for diagram datasets.

Usage:
    python llm_judge.py --csv data/geometric_shapes_test_set.csv --mode both --model gpt-4.1-mini --limit 10
    python llm_judge.py --mode image --model gpt-5 --concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
from pydantic import ValidationError
from tqdm import tqdm

from utils.schema_llm_judge import STRICT_RESPONSE_FORMAT, JudgeOutput
from utils.openai_judge import call_judge, extract_response_text, extract_token_usage, supports_reasoning
from utils.prompts import PromptMode, build_prompt


CACHE_ROOT = Path("results/llm_judge/cache")
RESULTS_ROOT = Path("results/llm_judge")
DEFAULT_DATASET_CSV = Path("data/geometric_shapes_test_set.csv")
PROMPT_MODES = ("image", "code", "both")

@dataclass(frozen=True)
class JudgeSample:
    diagram_id: str
    tikz_code: Optional[str]
    image_png_path: Optional[Path]


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
    model: str,
    temperature: float = 0.0,
    reasoning_effort: Optional[str] = "low",
    concurrency: int = 8,
    max_retries: int = 3,
    results_root: Path = CACHE_ROOT,
) -> None:
    sample_list = list(samples)
    if not sample_list:
        print("No samples to evaluate.")
        return

    sem = asyncio.Semaphore(concurrency)
    mode_results_root = results_root / mode
    mode_results_root.mkdir(parents=True, exist_ok=True)

    resolved_model = resolve_model_name(model)

    async def evaluate_one(sample: JudgeSample) -> None:
        model_dir = mode_results_root / safe_model_name(resolved_model)
        model_dir.mkdir(parents=True, exist_ok=True)
        cache_path = model_dir / f"diagram_{sample.diagram_id}.json"
        if cache_path.exists():
            return

        async with sem:
            effective_effort = reasoning_effort
            if resolved_model in {"gpt-5", "gpt-5-mini"} and not effective_effort:
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
                    model_name=resolved_model,
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
                            f"Validation failed for diagram {sample.diagram_id} ({resolved_model}) "
                            f"after {max_retries} attempts: {exc}"
                        )
                        failure_record = {
                            "diagram_id": sample.diagram_id,
                            "model": resolved_model,
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
                "model": resolved_model,
                "mode": mode,
                "temperature": temperature,
                "reasoning_effort": effective_effort if supports_reasoning(resolved_model) else None,
                "elapsed_ms": elapsed_ms,
                "tokens": usage,
                "rubric": structured.model_dump(),
                "tikz_code": sample.tikz_code,
                "image_png_path": str(sample.image_png_path) if sample.image_png_path else None,
            }
            cache_path.write_text(json.dumps(record, indent=2))

    tasks: list[asyncio.Task[None]] = [asyncio.create_task(evaluate_one(sample)) for sample in sample_list]
    if not tasks:
        print("No tasks scheduled (all evaluations may already be cached).")
        return

    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Collecting judgements"):
        await fut


def _flatten_rubric_fields(rubric: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(rubric, dict):
        return {}
    out: dict[str, Any] = {}
    for check_name, check_payload in rubric.items():
        if isinstance(check_payload, dict):
            out[f"{check_name}_value"] = check_payload.get("value")
            out[f"{check_name}_rationale"] = check_payload.get("rationale")
        else:
            out[f"{check_name}_value"] = None
            out[f"{check_name}_rationale"] = None
    return out


def export_mode_model_csvs(
    *,
    mode: PromptMode,
    model: str,
    cache_root: Path,
    output_root: Path,
) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    exported_paths: list[Path] = []

    resolved_model = resolve_model_name(model)
    safe_model = safe_model_name(resolved_model)
    model_dir = cache_root / mode / safe_model
    if not model_dir.exists():
        return exported_paths

    rows: list[dict[str, Any]] = []
    for path in sorted(model_dir.glob("diagram_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        tokens = payload.get("tokens", {}) if isinstance(payload, dict) else {}
        row: dict[str, Any] = {
            "diagram_id": payload.get("diagram_id"),
            "model": payload.get("model", resolved_model),
            "mode": payload.get("mode", mode),
            "temperature": payload.get("temperature"),
            "reasoning_effort": payload.get("reasoning_effort"),
            "elapsed_ms": payload.get("elapsed_ms"),
            "input_tokens": tokens.get("input_tokens") if isinstance(tokens, dict) else None,
            "cached_tokens": tokens.get("cached_tokens") if isinstance(tokens, dict) else None,
            "output_tokens": tokens.get("output_tokens") if isinstance(tokens, dict) else None,
            "total_tokens": tokens.get("total_tokens") if isinstance(tokens, dict) else None,
            "error": payload.get("error"),
            "tikz_code": payload.get("tikz_code"),
            "image_png_path": payload.get("image_png_path"),
        }
        row.update(_flatten_rubric_fields(payload.get("rubric")))
        rows.append(row)

    if not rows:
        return exported_paths

    df = pd.DataFrame(rows)
    csv_path = output_root / f"evaluation_results_{mode}_{safe_model}_{date_str}.csv"
    df.to_csv(csv_path, index=False)
    exported_paths.append(csv_path)

    return exported_paths


def resolve_model_name(model_name: Optional[str] = None) -> str:
    resolved = (model_name or os.environ.get("OPENAI_MODEL") or "").strip()
    if not resolved:
        raise RuntimeError("OPENAI_MODEL must be set (or passed explicitly as --model).")
    if "/" in resolved:
        raise ValueError(f"Unsupported model '{resolved}'. This project supports OpenAI model IDs only.")
    return resolved


def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(":", "_")


def require_positive(value: int, *, flag_name: str) -> None:
    if value < 1:
        raise ValueError(f"{flag_name} must be >= 1")


def require_nonnegative(value: int, *, flag_name: str) -> None:
    if value < 0:
        raise ValueError(f"{flag_name} must be >= 0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATASET_CSV)
    parser.add_argument("--mode", choices=PROMPT_MODES, default="both")
    parser.add_argument("--model", default=None, help="Model name (defaults to OPENAI_MODEL env var).")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", default="low", help="Use 'none' to disable.")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Cap on number of samples (0 = all).")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.csv}")
    require_positive(args.concurrency, flag_name="--concurrency")
    require_positive(args.max_retries, flag_name="--max-retries")
    require_nonnegative(args.limit, flag_name="--limit")
    resolved_model = resolve_model_name(args.model)
    cache_root = CACHE_ROOT

    mode: PromptMode = args.mode
    samples = load_samples(args.csv, mode)
    if args.limit > 0:
        samples = samples[: args.limit]
        print(f"Limited to first {len(samples)} samples.")

    reasoning_effort = None if str(args.reasoning_effort).strip().lower() == "none" else args.reasoning_effort

    asyncio.run(
        run_judge_evaluations(
            samples,
            mode=mode,
            model=resolved_model,
            temperature=args.temperature,
            reasoning_effort=reasoning_effort,
            concurrency=args.concurrency,
            max_retries=args.max_retries,
            results_root=cache_root,
        )
    )

    exported = export_mode_model_csvs(
        mode=mode,
        model=resolved_model,
        cache_root=cache_root,
        output_root=RESULTS_ROOT,
    )
    if exported:
        print("Exported CSV summaries:")
        for path in exported:
            print(f"  {path}")


if __name__ == "__main__":
    main()
