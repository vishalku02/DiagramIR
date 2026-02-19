# DiagramIR

Experiment code for **DiagramIR: An Automatic Evaluation Pipeline for Educational Math Diagrams**.

DiagramIR is a scalable and reliable method for automatic evaluation of geometric figures. It uses a language model to translate diagram code into an *intermediate representation (IR)*, a standardized, structured format where rule-based checks can be applied against the key geometric and mathematical constraints of the figure.
This approach achieves higher agreement with human raters (Cohen’s K) and enables smaller models, such as GPT-4.1 Mini, to perform on par with larger frontier models like GPT-5—at nearly 10x lower inference cost.

## Notebook Workflow
Run the two evaluation notebooks first, then the analysis notebook to compare results:
1. `notebooks/backtranslation_evaluation.ipynb`
2. `notebooks/llm_judge_evaluation.ipynb`
3. `notebooks/backtranslation_analysis.ipynb`

## Requirements

- **ImageMagick** (`magick` CLI)
- **TeX distribution** with `lualatex` and `dvisvgm`
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

- Root entry modules:
  - `backtranslation.py` – TikZ → IR extraction helpers (includes `compile_tikz`).
  - `llm_judge.py` – LLM-as-Judge evaluation harness.
  - `evaluator.py` – Rule-based checks applied to extracted IR.
- `utils/` – Internal schemas, geometry utilities, model wrappers, and judge prompts.
- `scripts/` – Utilities for cache cleanup and judge rendering.
- `notebooks/` – Notebook-first execution and analysis surface.
- `data/` – Benchmark prompts and human annotation CSVs.
- `results/` – Cached model outputs and evaluation results.

## Usage

### Backtranslation Workflow

- Run `notebooks/backtranslation_evaluation.ipynb` to generate model IR extractions and populate `results/backtranslation/`.
- Run `notebooks/backtranslation_analysis.ipynb` to compare approaches and compute agreement, cost, and timing metrics.

### LLM-as-Judge Workflow

- Run `notebooks/llm_judge_evaluation.ipynb` for the notebook-first judge execution path.
- Optional: pre-render judge PNGs if you need image inputs:
  ```bash
  python scripts/precompute_judge_pngs.py
  ```
  Requires `magick`, `lualatex`, and `dvisvgm`.

- Optional CLI (secondary to notebooks):
  ```bash
  python llm_judge.py --csv data/geometric_shapes_test_set.csv --mode both --models gpt-4.1-mini --limit 10
  ```
  Results will appear under `results/llm_judge/<mode>/<model>/diagram_*.json`.
