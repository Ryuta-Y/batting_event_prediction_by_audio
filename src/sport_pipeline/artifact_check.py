"""Small artifact and JSON helpers for the audio release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from sport_pipeline.colab_paths import BASE_DIR_DEFAULT, expected_artifacts_for_stage


def check_artifacts(
    base_dir: str | Path = BASE_DIR_DEFAULT,
    artifacts: Iterable[str] = (),
) -> dict:
    """Return existence status for artifacts under the Drive artifact root."""

    root = Path(base_dir)
    entries = []
    for relative in artifacts:
        path = root / relative
        entries.append(
            {
                "artifact": relative,
                "path": str(path),
                "exists": path.exists(),
                "is_file": path.is_file(),
                "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
                "missing_hint_ja": None if path.exists() else missing_hint(relative),
            }
        )
    return {
        "base_dir": str(root),
        "all_present": all(entry["exists"] for entry in entries),
        "artifacts": entries,
    }


def check_stage_artifacts(
    stage: str,
    base_dir: str | Path = BASE_DIR_DEFAULT,
    run_id: str | None = None,
) -> dict:
    """Check artifacts for a configured audio-release stage."""

    return check_artifacts(base_dir=base_dir, artifacts=expected_artifacts_for_stage(stage, run_id=run_id))


def missing_hint(relative_path: str) -> str:
    """Return a short Japanese hint for a missing audio artifact."""

    if "bbe_events_v1.parquet" in relative_path:
        return "upstream pipeline が作成した Statcast BBE manifest を Drive に置いてください。"
    if "clips_v1.parquet" in relative_path:
        return "upstream pipeline が作成した clips_v1 と clip files を Drive に置いてください。"
    if "audio_presence" in relative_path or "audio_valid_clips" in relative_path:
        return "notebooks/39_cpu_audio_audit_baseline_compare.ipynb を実行してください。"
    if "audio_embedding" in relative_path or "audio_separated" in relative_path:
        return "notebooks/37_gpu_audio_separation_and_embeddings.ipynb を実行してください。"
    if "audio_impact" in relative_path or "audio_raw" in relative_path or "audio_enhanced" in relative_path:
        return "notebooks/36_cpu_audio_impact_baseline.ipynb を実行してください。"
    if "fusion_audio" in relative_path or "method_evaluation_with_audio" in relative_path:
        return "notebooks/38_cpu_audio_fusion_compare.ipynb を実行してください。"
    if "predictions_v1.parquet" in relative_path or "metrics_v1.json" in relative_path:
        return "必要な upstream baseline prediction run が Drive にあるか確認してください。"
    return "Drive artifact root と run profile の run_id / artifact_namespace を確認してください。"


def write_json(payload: dict, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check audio-release artifacts under BASE_DIR.")
    parser.add_argument("--base-dir", default=str(BASE_DIR_DEFAULT))
    parser.add_argument("--stage", default="audio_inputs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    result = check_stage_artifacts(args.stage, base_dir=args.base_dir, run_id=args.run_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json:
        write_json(result, args.output_json)


if __name__ == "__main__":
    main()
