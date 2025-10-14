#!/usr/bin/env python3
"""Remove auxiliary judge PNG conversion artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path

TARGET_DIR = Path("data/judge_pngs")
ARTIFACT_EXTENSIONS = (".aux", ".log", ".pdf", ".tex")


def collect_targets(root: Path) -> list[Path]:
    targets: list[Path] = []
    for ext in ARTIFACT_EXTENSIONS:
        targets.extend(root.glob(f"diagram_*{ext}"))
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="List files without deleting them")
    parser.add_argument("--root", type=Path, default=TARGET_DIR, help="Directory containing judge PNG artifacts")
    args = parser.parse_args()

    root = args.root
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Directory not found: {root}")

    targets = collect_targets(root)
    if not targets:
        print("No artifacts found.")
        return

    for path in targets:
        if args.dry_run:
            print(f"KEEP {path}")
        else:
            path.unlink(missing_ok=True)
            print(f"REMOVED {path}")

    if not args.dry_run:
        print(f"Removed {len(targets)} files from {root}")


if __name__ == "__main__":
    main()
