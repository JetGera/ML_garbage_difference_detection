from __future__ import annotations

import csv
import json
from datetime import datetime
from time import perf_counter
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(METRICS_DIR))
sys.path.insert(0, str(ROOT))

from extract_method_metrics import (
    DEFAULT_DATASET_ROOT,
    METHODS,
    OUTPUT_DIR,
    _compute_quantity_metrics,
    _compute_segmentation_metrics,
    _extract_predicted_quantity_pixels,
    _json_safe,
    _metric_value_to_text,
    discover_pairs,
)

try:
    from launcher.methods import get_method_spec
    from launcher.runners import create_runner
except ImportError:
    from methods import get_method_spec
    from runners import create_runner


def _save_outputs(payload: dict[str, Any], json_path: Path, csv_path: Path) -> None:
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fieldnames = [
        "pair_id",
        "pair_dir",
        "method_id",
        "method_name",
        "status",
        "metric_key",
        "metric_value",
        "value_type",
        "env_name",
        "env_file",
        "error",
    ]
    rows: list[dict[str, Any]] = []
    for entry in payload.get("entries", []):
        metrics = entry.get("metrics", {}) or {}
        if metrics:
            for metric_key, metric_value in metrics.items():
                rows.append(
                    {
                        "pair_id": entry.get("pair_id", ""),
                        "pair_dir": entry.get("pair_dir", ""),
                        "method_id": entry["method_id"],
                        "method_name": entry["method_name"],
                        "status": entry["status"],
                        "metric_key": metric_key,
                        "metric_value": _metric_value_to_text(metric_value),
                        "value_type": type(metric_value).__name__,
                        "env_name": entry["env_name"],
                        "env_file": entry["env_file"],
                        "error": entry.get("error", ""),
                    }
                )
        else:
            rows.append(
                {
                    "pair_id": entry.get("pair_id", ""),
                    "pair_dir": entry.get("pair_dir", ""),
                    "method_id": entry["method_id"],
                    "method_name": entry["method_name"],
                    "status": entry["status"],
                    "metric_key": "",
                    "metric_value": "",
                    "value_type": "",
                    "env_name": entry["env_name"],
                    "env_file": entry["env_file"],
                    "error": entry.get("error", ""),
                }
            )

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    latest_json_path = OUTPUT_DIR / "method_metrics_latest.json"
    latest_csv_path = OUTPUT_DIR / "method_metrics_latest.csv"
    latest_json_path.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_csv_path.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")


def collect_metrics_incremental() -> dict[str, Any]:
    pairs, skipped_pairs = discover_pairs()
    if not pairs:
        raise RuntimeError(f"No valid before/after/mask pairs found under {DEFAULT_DATASET_ROOT}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / f"method_metrics_{timestamp}.json"
    csv_path = OUTPUT_DIR / f"method_metrics_{timestamp}.csv"

    entries: list[dict[str, Any]] = []
    all_metric_keys: set[str] = set()
    runners: dict[str, Any] = {}

    print(f"Found {len(pairs)} valid pairs; {len(skipped_pairs)} skipped folders", flush=True)

    for pair_index, pair in enumerate(pairs, start=1):
        pair_id = str(pair["pair_id"])
        pair_dir = Path(pair["pair_dir"])
        before_path = Path(pair["before_path"])
        after_path = Path(pair["after_path"])
        ground_truth_mask_path = Path(pair["ground_truth_mask_path"])

        ground_truth_mask = cv2.imread(str(ground_truth_mask_path), cv2.IMREAD_GRAYSCALE)
        if ground_truth_mask is None:
            skipped_pairs.append({"pair_id": pair_id, "reason": f"unreadable mask: {ground_truth_mask_path.name}"})
            print(f"[{pair_index}/{len(pairs)}] skipped {pair_id}: unreadable mask")
            continue

        print(f"[{pair_index}/{len(pairs)}] processing pair {pair_id}", flush=True)

        for method_index, method_id in enumerate(METHODS, start=1):
            spec = get_method_spec(method_id)
            print(
                f"[{pair_index}/{len(pairs)} | {method_index}/{len(METHODS)}] {pair_id} -> {method_id}",
                flush=True,
            )
            method_start = perf_counter()
            entry: dict[str, Any] = {
                "pair_id": pair_id,
                "pair_dir": str(pair_dir),
                "ground_truth_mask": str(ground_truth_mask_path),
                "method_id": method_id,
                "method_name": spec.label,
                "env_name": spec.env_name,
                "env_file": str(spec.env_file),
                "status": "ok",
            }

            try:
                runner = runners.get(method_id)
                if runner is None:
                    runner = create_runner(method_id, device="auto")
                    runners[method_id] = runner

                result = runner.analyze(before_path, after_path)
                metrics = {key: _json_safe(value) for key, value in result.metrics.items()}

                predicted_mask = None
                predicted_mask_path = result.artifacts.get("change_mask")
                if predicted_mask_path is not None and Path(predicted_mask_path).exists():
                    predicted_mask = cv2.imread(str(predicted_mask_path), cv2.IMREAD_GRAYSCALE)
                    if predicted_mask is not None and predicted_mask.shape != ground_truth_mask.shape:
                        predicted_mask = cv2.resize(
                            predicted_mask,
                            (ground_truth_mask.shape[1], ground_truth_mask.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    if predicted_mask is not None:
                        segm_metrics = _compute_segmentation_metrics(ground_truth_mask, predicted_mask)
                        metrics.update({f"segm_{k}": v for k, v in segm_metrics.items()})

                gt_pixels = float(np.count_nonzero(ground_truth_mask > 0))
                total_pixels = float(ground_truth_mask.shape[0] * ground_truth_mask.shape[1])
                pred_pixels = _extract_predicted_quantity_pixels(metrics, predicted_mask)
                if pred_pixels is not None:
                    qty_px = _compute_quantity_metrics(gt_pixels, float(pred_pixels))
                    metrics.update({f"qty_pixels_{k}": v for k, v in qty_px.items()})

                    gt_ratio = gt_pixels / max(total_pixels, 1.0)
                    pred_ratio = float(pred_pixels) / max(total_pixels, 1.0)
                    qty_ratio = _compute_quantity_metrics(gt_ratio, pred_ratio)
                    metrics.update({f"qty_ratio_{k}": v for k, v in qty_ratio.items()})

                entry.update(
                    {
                        "summary": result.summary,
                        "preview_text": result.preview_text,
                        "before_path": str(result.before_path),
                        "after_path": str(result.after_path),
                        "preview_image_path": str(result.preview_image_path) if result.preview_image_path is not None else None,
                        "artifacts": {name: str(path) for name, path in result.artifacts.items()},
                        "metrics": metrics,
                        "metric_keys": sorted(metrics),
                        "metric_count": len(metrics),
                    }
                )
                all_metric_keys.update(metrics)
            except Exception as exc:  # pragma: no cover - defensive capture for unavailable runners
                entry.update(
                    {
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "metrics": {},
                        "metric_keys": [],
                        "metric_count": 0,
                        "summary": None,
                        "preview_text": None,
                        "before_path": str(before_path),
                        "after_path": str(after_path),
                        "preview_image_path": None,
                        "artifacts": {},
                    }
                )

            entries.append(entry)
            method_elapsed_ms = (perf_counter() - method_start) * 1000.0
            print(
                f"[{pair_index}/{len(pairs)} | {method_index}/{len(METHODS)}] {pair_id} -> {method_id} done in {method_elapsed_ms:.1f} ms",
                flush=True,
            )

            payload = {
                "generated_at": timestamp,
                "dataset_root": str(DEFAULT_DATASET_ROOT),
                "pair_count": len(pairs),
                "processed_entries": len(entries),
                "skipped_pairs": skipped_pairs,
                "method_count": len(METHODS),
                "metric_key_union": sorted(all_metric_keys),
                "entries": entries,
            }
            _save_outputs(payload, json_path, csv_path)
            print(
                f"Saved progress: pair {pair_id}, method {method_id}, entries={len(entries)}, metrics={len(entry.get('metrics', {}))}",
                flush=True,
            )

        print(f"Saved progress after pair {pair_id}: {len(entries)} entries total", flush=True)

    payload = {
        "generated_at": timestamp,
        "dataset_root": str(DEFAULT_DATASET_ROOT),
        "pair_count": len(pairs),
        "processed_entries": len(entries),
        "skipped_pairs": skipped_pairs,
        "method_count": len(METHODS),
        "metric_key_union": sorted(all_metric_keys),
        "entries": entries,
    }
    _save_outputs(payload, json_path, csv_path)

    print(f"Saved JSON to {json_path}", flush=True)
    print(f"Saved CSV to {csv_path}", flush=True)
    print(f"Processed pair dirs: {len(pairs)}", flush=True)
    print(f"Skipped pair dirs: {len(skipped_pairs)}", flush=True)
    print(f"Metric key union: {len(all_metric_keys)} keys", flush=True)
    return payload


if __name__ == "__main__":
    collect_metrics_incremental()
