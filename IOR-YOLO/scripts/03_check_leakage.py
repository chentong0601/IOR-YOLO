"""Read-only duplicate candidates; does not certify absence of fruit-level leakage."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ior_yolo.utils.io import image_size, sha256_file

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DERIVATIVE_TOKENS = re.compile(r"(?:^|[_\-.])(aug|augment|brightness|bright|noise|gauss|resiz(?:e|ed)|640|480|369|277|404|303)(?:$|[_\-.])", re.I)


def inspect(root: Path) -> dict:
    images = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    by_hash: dict[str, list[str]] = defaultdict(list)
    by_stem: dict[str, list[dict]] = defaultdict(list)
    size_errors = []
    for path in images:
        relative = str(path.relative_to(root))
        by_hash[sha256_file(path)].append(relative)
        stem = DERIVATIVE_TOKENS.sub("_", path.stem).strip("_.-").lower()
        try:
            dimensions = image_size(path)
        except (OSError, ValueError) as exc:
            dimensions = None
            size_errors.append({"path": relative, "error": str(exc)})
        by_stem[stem].append({"path": relative, "size": dimensions})
    return {"image_files": len(images),
            "exact_duplicate_groups": [v for v in by_hash.values() if len(v) > 1],
            "possible_source_groups_by_filename": [v for v in by_stem.values() if len(v) > 1],
            "unreadable_dimensions": size_errors,
            "resized_duplicate_status": "candidate groups only; pixel comparison pending",
            "offline_augmented_status": "candidate groups only; filename rules cannot prove derivation",
            "near_duplicate_status": "Unverified; perceptual hash/manual review only if needed",
            "fruit_tree_session_group_status": "Unverified unless package metadata provides IDs",
            "split_status": "not created or validated"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f"directory not found: {args.root}")
    print(json.dumps(inspect(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
