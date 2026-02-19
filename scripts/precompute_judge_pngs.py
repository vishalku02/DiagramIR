"""Precompute PNG renders for TikZ diagrams referenced in the judge dataset."""

import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_CSV = Path("data/geometric_shapes_test_set.csv")
PNG_DIR = Path("data/judge_pngs")

PNG_DIR.mkdir(parents=True, exist_ok=True)


def compile_tikz(code: str, output_path: Path, output_format: str = "png") -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".tex")
    temp_path.write_text(code, encoding="utf-8")
    temp_dir = output_path.parent

    for fname in ("IMlongdivision.sty", "Tikz-IM.sty", "Tikz-IM-ES.sty", "IM.cls"):
        src = ROOT / "styles" / fname
        if src.exists():
            shutil.copy(src, temp_dir / fname)

    latex_engine = os.environ.get("LATEX_ENGINE", "lualatex")
    imagemagick_bin = os.environ.get("IMAGEMAGICK_BIN", "magick")
    dvisvgm_bin = os.environ.get("DVISVGM_BIN", "dvisvgm")

    try:
        subprocess.run(
            [latex_engine, "-interaction=nonstopmode", "-output-directory", str(temp_dir), str(temp_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        pdf_path = temp_path.with_suffix(".pdf")

        if output_format.lower() == "svg":
            subprocess.run(
                [dvisvgm_bin, "--pdf", str(pdf_path), "-o", str(output_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.run(
                [imagemagick_bin, "-density", "300", str(pdf_path), str(output_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return True
    except subprocess.CalledProcessError:
        return False


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
