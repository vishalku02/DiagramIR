"""Backtranslation pipeline: TikZ -> IR extraction, rule-based evaluation, and diagram rendering.

Usage:
    python backtranslation.py --csv data/geometric_shapes_test_set.csv --limit 10
    python backtranslation.py --model gpt-4.1 --concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm.asyncio import tqdm

from evaluator import Evaluator
from utils.schema_ir import TikzIR

load_dotenv()


RESULTS_DIR = Path("results/backtranslation")
DEFAULT_DATASET_CSV = Path("data/geometric_shapes_test_set.csv")

BACKTRANSLATION_PROMPT_PATH = pathlib.Path(__file__).resolve().parent / "utils" / "prompts" / "prompt_backtranslation.md"


@lru_cache(maxsize=1)
def load_backtranslation_prompt() -> str:
    try:
        return BACKTRANSLATION_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Backtranslation prompt file not found: {BACKTRANSLATION_PROMPT_PATH}"
        ) from exc


@retry(stop=stop_after_attempt(5), wait=wait_random_exponential(min=1, max=20), reraise=True)
async def tikz_to_ir(
    tikz_code: str,
    *,
    model_name: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    prompt_template = load_backtranslation_prompt()
    prompt = prompt_template.replace("{tikz_code}", tikz_code)

    model = resolve_model_name(model_name)

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


async def run_backtranslation_pipeline(
    df: pd.DataFrame,
    *,
    model: Optional[str] = None,
    concurrency: int = 6,
    results_dir: Path = RESULTS_DIR,
) -> List[dict]:
    resolved_model = resolve_model_name(model)
    cache_dir = results_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    evaluator = Evaluator()
    sem = asyncio.Semaphore(concurrency)

    print(f"Processing {len(df)} diagrams with model={resolved_model}, concurrency={concurrency}")

    async def process_one(row: pd.Series) -> dict:
        diagram_id = row["diagram_id"]
        cache_path = cache_dir / f"diagram_{diagram_id}_{resolved_model}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        async with sem:
            record: dict[str, Any] = {
                "diagram_id": diagram_id,
                "model": resolved_model,
                "tikz_code": row["tikz"],
                "prompt": row.get("prompt", ""),
                "main_category": row.get("main_category", ""),
                "subcategory": row.get("subcategory", ""),
                "timestamp": datetime.now().isoformat(),
            }

            token_usage: dict[str, int] = {}
            raw_ir = None
            extraction_time_ms = 0.0

            try:
                start = time.perf_counter()
                raw_ir, token_usage = await tikz_to_ir(row["tikz"], model_name=resolved_model)
                extraction_time_ms = (time.perf_counter() - start) * 1000

                ir_validated = True
                validation_error = None
                ir_for_evaluation = raw_ir
                try:
                    ir_for_evaluation = TikzIR.model_validate(raw_ir)
                except Exception as exc:
                    ir_validated = False
                    validation_error = str(exc)

                eval_start = time.perf_counter()
                eval_results = evaluator.evaluate(ir_for_evaluation)
                evaluation_time_ms = (time.perf_counter() - eval_start) * 1000

                record["ir"] = {
                    "success": ir_validated,
                    "time_ms": extraction_time_ms,
                    "prompt_tokens": token_usage.get("prompt_tokens", 0),
                    "completion_tokens": token_usage.get("completion_tokens", 0),
                    "error": validation_error,
                    "generated_ir": raw_ir,
                }
                record["evaluation_results"] = {
                    check["check"]: {"passed": check["passed"], "message": check["message"]}
                    for check in eval_results["global"]
                }
                record["evaluation_time_ms"] = evaluation_time_ms
                record["overall_score"] = eval_results["score"]
                record["overall_pass"] = eval_results["overall_pass"]

            except Exception as exc:
                record["ir"] = {
                    "success": False,
                    "time_ms": extraction_time_ms,
                    "prompt_tokens": token_usage.get("prompt_tokens", 0),
                    "completion_tokens": token_usage.get("completion_tokens", 0),
                    "error": str(exc),
                    "generated_ir": raw_ir,
                }
                record["evaluation_results"] = {}
                record["evaluation_time_ms"] = 0
                record["overall_score"] = 0.0
                record["overall_pass"] = False

            cache_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            return record

    tasks = [process_one(row) for _, row in df.iterrows()]
    results = await tqdm.gather(*tasks, desc="Processing diagrams")
    return list(results)


def export_results_csv(results: List[dict], output_path: Path) -> pd.DataFrame:
    rows = []
    for result in results:
        row: dict[str, Any] = {
            "diagram_id": result["diagram_id"],
            "model": result.get("model", ""),
            "main_category": result.get("main_category", ""),
            "subcategory": result.get("subcategory", ""),
            "extraction_success": result.get("ir", {}).get("success", False),
            "extraction_time_ms": result.get("ir", {}).get("time_ms", 0),
            "prompt_tokens": result.get("ir", {}).get("prompt_tokens", 0),
            "completion_tokens": result.get("ir", {}).get("completion_tokens", 0),
            "evaluation_time_ms": result.get("evaluation_time_ms", 0),
            "overall_score": result.get("overall_score", 0.0),
            "overall_pass": result.get("overall_pass", False),
        }
        for check_name, check_result in result.get("evaluation_results", {}).items():
            row[f"{check_name}_passed"] = check_result.get("passed")
            row[f"{check_name}_message"] = check_result.get("message")
        rows.append(row)

    results_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"Results exported to {output_path}")
    return results_df


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
    parser.add_argument("--model", default=None, help="Model name (defaults to OPENAI_MODEL env var).")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0, help="Cap on number of samples (0 = all).")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.csv}")
    require_positive(args.concurrency, flag_name="--concurrency")
    require_nonnegative(args.limit, flag_name="--limit")

    df = pd.read_csv(args.csv)
    if args.limit > 0:
        df = df.head(args.limit)
        print(f"Limited to first {len(df)} samples.")

    results = asyncio.run(
        run_backtranslation_pipeline(
            df,
            model=args.model,
            concurrency=args.concurrency,
            results_dir=RESULTS_DIR,
        )
    )

    resolved_model = resolve_model_name(args.model)
    date_str = datetime.now().strftime("%Y%m%d")
    csv_path = RESULTS_DIR / f"evaluation_results_{safe_model_name(resolved_model)}_{date_str}.csv"
    export_results_csv(results, csv_path)

    success_count = sum(1 for r in results if r.get("ir", {}).get("success", False))
    print(f"Done. {success_count}/{len(results)} extractions succeeded.")


if __name__ == "__main__":
    main()
