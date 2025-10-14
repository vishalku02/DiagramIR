# DiagramIR

Experiment code for **DiagramIR: An Automatic Evaluation Pipeline for Educational Math Diagrams**.

## Requirements

- **ImageMagick** (`magick` CLI) for PDF→PNG conversion.
- **TeX distribution** with `lualatex` and `dvisvgm` (TeX Live / MacTeX recommended).
- Optional: [uv](https://docs.astral.sh/uv/) for environment management.

Verify the system tools are on your `PATH`:

```bash
magick -version
lualatex --version
dvisvgm --version
```

The TeX compilation step also expects the supporting styles in `styles/`
(`IM.cls`, `IMlongdivision.sty`, `Tikz-IM.sty`, `Tikz-IM-ES.sty`).

## Setup

### Python Environment

Create the virtual environment and install the project plus notebook extras:

```bash
uv venv
uv sync --extra notebooks
source .venv/bin/activate
```

### Environment Variables

Copy the example file and populate your API keys / configuration:

```bash
cp .env.example .env
```

Required entries:

- `OPENAI_API_KEY`, `OPENAI_MODEL`
- `TORCH_DEVICE`, `MAX_CONCURRENCY` 

## Repository Layout

- `backtranslation.py` – TikZ → IR extraction helpers (includes `compile_tikz`).
- `evaluator.py` – Rule-based checks applied to the extracted IR.
- `geometry_engine.py`, `IR_model.py` – Geometry helpers and Pydantic schema for the IR.
- `scripts/` – Utilities for cache cleanup, CSV normalization, and judge rendering.
- `notebooks/` – Backtranslation experiment and analysis notebooks.
- `data/` – Benchmark prompts and human annotation CSVs.
- `results/` – Cached model outputs and evaluation results.

## Usage

### Backtranslation Workflow

- Run `notebooks/backtranslation_evaluation.ipynb` to generate model IR extractions and populate `results/backtranslation/`.
- Run `notebooks/backtranslation_analysis.ipynb` to compute human agreement and cost and time metrics. 

### LLM-as-Judge Workflow

- Pre-render judge PNGs if you need image inputs:
  ```bash
  python scripts/precompute_judge_pngs.py
  ```
  Requires `magick`, `lualatex`, and `dvisvgm`.

- Ensure `data/geometric_shapes_test_set.csv` (or your chosen dataset) includes `diagram_id`, TikZ, and optional PNG paths.
- Adjust `llm_judge.py` to list the models, prompt mode (`code`, `image`, or `both`), and concurrency you want.
- Launch the evaluation:
  ```bash
  python llm_judge.py
  ```
  Results will appear under `results/llm_judge/<mode>/<model>/diagram_*.json`.
