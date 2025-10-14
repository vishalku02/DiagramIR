"""Precompute PNG renders for TikZ diagrams referenced in the judge dataset."""

import csv
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtranslation import compile_tikz

DATA_CSV = Path("data/geometric_shapes_test_set.csv")
PNG_DIR = Path("data/judge_pngs")

PNG_DIR.mkdir(parents=True, exist_ok=True)


def process_csv(
    input_csv: Path = DATA_CSV,
    *,
    output_csv: Optional[Path] = None,
    png_dir: Path = PNG_DIR,
    tikz_column: Optional[str] = None,
) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    rows = []
    with input_csv.open(newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if tikz_column is None:
            if "tikz" in fieldnames:
                tikz_column = "tikz"
            elif "tikz_code" in fieldnames:
                tikz_column = "tikz_code"
            else:
                raise ValueError("CSV must contain a 'tikz' or 'tikz_code' column")

        png_column = "image_png_path"
        if png_column not in fieldnames:
            fieldnames.append(png_column)

        for idx, row in enumerate(reader):
            tikz_code = row.get(tikz_column, "")
            if not tikz_code.strip():
                row[png_column] = ""
                rows.append(row)
                continue

            diagram_id = row.get("diagram_id") or str(idx)
            dest_png = png_dir / f"diagram_{diagram_id}.png"
            success = compile_tikz(tikz_code, dest_png, output_format="png")
            row[png_column] = str(dest_png) if success else ""
            rows.append(row)
            if success:
                print(f"Rendered diagram {diagram_id} → {dest_png}")

    target_csv = output_csv or input_csv
    tmp_path = target_csv.with_suffix(target_csv.suffix + ".tmp")
    with tmp_path.open("w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    tmp_path.replace(target_csv)


if __name__ == "__main__":
    process_csv()
