from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import traceback
from pathlib import Path

try:
    from .runners import create_runner
except ImportError:
    from runners import create_runner


def _serialize_result(result) -> dict:
    return {
        "method_id": result.method_id,
        "method_name": result.method_name,
        "summary": result.summary,
        "metrics": result.metrics,
        "before_path": str(result.before_path),
        "after_path": str(result.after_path),
        "preview_text": result.preview_text,
        "preview_image_path": str(result.preview_image_path) if result.preview_image_path else None,
        "artifacts": {key: str(value) for key, value in result.artifacts.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an analysis method inside its conda environment.")
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        runner = create_runner(args.method_id, device=args.device, force_cpu=args.force_cpu)
        result = runner.analyze(Path(args.before), Path(args.after))
        payload = {"ok": True, "result": _serialize_result(result)}
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        payload = {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()