"""Future E01 prediction exporter; val by default, test only after final-test lock.

Do not execute during Stage 3B preparation. Writes real model predictions only.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e01_analysis_export", Path(__file__).with_name("17_analyze_e01_results.py"))
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def export(run_dir: Path, data_root: Path, *, split: str = "val", final_test: bool = False) -> dict:
    manifest = yaml.safe_load((run_dir / "run_manifest.yaml").read_text(encoding="utf-8"))
    analysis.checked_split(split, final_test, manifest)
    if not manifest.get("validation_metrics_mask"):
        raise RuntimeError("model/validation must be fixed before exporting predictions")
    analysis.provenance(manifest, run_dir)
    out = run_dir / "predictions"
    csv_path, mask_dir = out / f"predictions_{split}.csv", out / f"masks_{split}"
    if csv_path.exists() or mask_dir.exists():
        raise FileExistsError("prediction export already exists; do not overwrite")
    out.mkdir(parents=True, exist_ok=True)
    frozen = [row for row in analysis.csv_rows(ROOT / "data/manifests/d2_split_frozen.csv")
              if row["final_split"] == split]
    by_name = {row["filename"]: row for row in frozen}
    if len(by_name) != len(frozen):
        raise ValueError("frozen split contains ambiguous filenames")
    from ultralytics import YOLO

    best = run_dir / "weights/best.pt"
    if analysis.digest(best) != manifest.get("best_checkpoint_sha256"):
        raise RuntimeError("best checkpoint differs from run provenance")
    rows = []
    seen = set()
    with tempfile.TemporaryDirectory(prefix="e01-predictions-", dir=out) as temp:
        temp_dir = Path(temp) / f"masks_{split}"
        temp_dir.mkdir()
        model = YOLO(str(best))
        # conf=.001 follows the pinned 8.3.220 validation threshold; no
        # threshold sweep or test-guided changes are performed here.
        stream = model.predict(source=str(data_root / "images" / split), stream=True,
                               device=0, imgsz=640, conf=0.001, iou=0.7,
                               save=False, verbose=False)
        for result in stream:
            name = Path(result.path).name
            if name not in by_name or name in seen:
                raise ValueError(f"unknown/repeated prediction image: {name}")
            seen.add(name)
            entry = by_name[name]
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            if result.masks is None or len(result.masks.xy) != len(boxes):
                raise ValueError(f"segmentation masks missing: {name}")
            h, w = result.orig_shape
            for j, (box, polygon) in enumerate(zip(boxes, result.masks.xy)):
                cls = int(box.cls[0].item())
                if cls not in (0, 1, 2):
                    raise ValueError(f"unexpected predicted class {cls}")
                pred_id = f"p{j:04d}"
                mask_name = f"{Path(name).stem}__{pred_id}.json"
                normalized = [[float(x)/w, float(y)/h] for x, y in polygon]
                (temp_dir / mask_name).write_text(
                    json.dumps({"image_id": name, "pred_instance_id": pred_id,
                                "polygon_normalized": normalized}, ensure_ascii=False,
                               separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
                rows.append({"image_id": name, "pred_instance_id": pred_id,
                             "pred_class": cls, "confidence": f"{float(box.conf[0].item()):.10f}",
                             "box": json.dumps([float(v) for v in box.xyxy[0].tolist()], separators=(",", ":")),
                             "mask_reference": f"predictions/masks_{split}/{mask_name}",
                             "source_group_id": entry["source_group_id"],
                             "split_guard_cluster_id": entry["split_guard_cluster_id"], "split": split})
        if seen != set(by_name):
            raise ValueError(f"missing images from prediction export: {len(set(by_name)-seen)}")
        rows.sort(key=lambda r: (r["image_id"], r["pred_instance_id"]))
        analysis.csv_write(out / f".predictions_{split}.csv", analysis.PRED_FIELDS, rows)
        temp_dir.replace(mask_dir)
        (out / f".predictions_{split}.csv").replace(csv_path)
    return {"split": split, "images": len(seen), "predictions": len(rows), "csv": str(csv_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=analysis.DEFAULT_RUN)
    parser.add_argument("--data-root", type=Path, default=analysis.DEFAULT_DATA)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--final-test", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export(args.run_dir, args.data_root, split=args.split,
                            final_test=args.final_test), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
