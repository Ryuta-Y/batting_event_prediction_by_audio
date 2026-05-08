"""Optional audio separation experiments for batting-impact clips.

This module keeps source separation as an ablation branch, not as a mandatory
precondition for the raw impact baseline. In Colab it can call Demucs when
installed; otherwise the ``transient_enhance`` backend runs the same prediction
contract using deterministic transient emphasis so the comparison table still
has a separated/enhanced branch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from sport_pipeline.artifact_check import write_json
from sport_pipeline.audio.impact import run_audio_impact_baseline


DEFAULT_DEMUCS_MODEL = "htdemucs"


def _demucs_stem_path(output_dir: Path, model_name: str, media_path: Path, stem: str) -> Path:
    return output_dir / model_name / media_path.stem / f"{stem}.wav"


def _run_demucs_for_clip(
    media_path: Path,
    *,
    output_dir: Path,
    model_name: str,
    stem: str,
    device: str,
    overwrite: bool,
) -> tuple[Path | None, str]:
    expected = _demucs_stem_path(output_dir, model_name, media_path, stem)
    if expected.exists() and not overwrite:
        return expected, "cached"
    command = [
        sys.executable,
        "-m",
        "demucs.separate",
        "-n",
        model_name,
        "--two-stems",
        "vocals",
        "--out",
        str(output_dir),
        "--device",
        device,
        str(media_path),
    ]
    proc = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        return None, f"demucs_failed:{stderr[:240]}"
    if expected.exists():
        return expected, "demucs_complete"
    alternate = _demucs_stem_path(output_dir, model_name, media_path, "no_vocals")
    if alternate.exists():
        return alternate, "demucs_complete_alternate"
    return None, "demucs_output_missing"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    from sport_pipeline.io import read_table

    return read_table(path) if path.exists() else []


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    from sport_pipeline.io import write_table

    return write_table(path, rows)


def build_demucs_audio_manifest(
    base_dir: str | Path,
    *,
    clip_run_id: str,
    clips_path: str | Path | None = None,
    output_manifest: str | Path | None = None,
    max_clips: int | None = None,
    model_name: str = DEFAULT_DEMUCS_MODEL,
    stem: str = "no_vocals",
    device: str = "cuda",
    overwrite: bool = False,
    allow_model_download: bool = False,
    output_suffix: str = ".parquet",
) -> dict[str, Path]:
    """Run Demucs and return a clips-like manifest with ``audio_path`` rows."""

    if not allow_model_download:
        raise RuntimeError("Demucs may download weights. Re-run in Colab with --allow-model-download.")
    base = Path(base_dir)
    source_path = Path(clips_path) if clips_path else base / f"clips/{clip_run_id}/clips_v1.parquet"
    rows = _read_rows(source_path)
    if max_clips is not None:
        rows = rows[:max_clips]
    demucs_root = base / "features" / "audio_separation_demucs" / clip_run_id / "demucs_outputs"
    manifest_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    for row in rows:
        raw_path = row.get("clip_path")
        if not raw_path:
            status_rows.append({"clip_id": row.get("clip_id"), "event_id": row.get("event_id"), "status": "missing_clip_path"})
            continue
        media_path = Path(str(raw_path))
        if not media_path.is_absolute():
            media_path = base / media_path
        if not media_path.exists():
            status_rows.append({"clip_id": row.get("clip_id"), "event_id": row.get("event_id"), "status": "clip_file_missing", "path": str(media_path)})
            continue
        separated, status = _run_demucs_for_clip(
            media_path,
            output_dir=demucs_root,
            model_name=model_name,
            stem=stem,
            device=device,
            overwrite=overwrite,
        )
        status_rows.append({"clip_id": row.get("clip_id"), "event_id": row.get("event_id"), "status": status, "path": None if separated is None else str(separated)})
        if separated is None:
            continue
        enriched = dict(row)
        enriched["audio_path"] = str(separated)
        enriched["audio_separation_backend"] = "demucs"
        enriched["audio_separation_model"] = model_name
        enriched["audio_separation_stem"] = stem
        manifest_rows.append(enriched)
    output_path = Path(output_manifest) if output_manifest else base / f"features/audio_separation_demucs/{clip_run_id}/separated_clips_v1{output_suffix}"
    status_path = base / f"reports/preflight/audio_demucs_separation_{clip_run_id}.json"
    _write_rows(output_path, manifest_rows)
    write_json(
        {
            "schema_version": "audio_demucs_separation_summary_v1",
            "clip_run_id": clip_run_id,
            "source_clips": str(source_path),
            "output_manifest": str(output_path),
            "model_name": model_name,
            "stem": stem,
            "device": device,
            "input_rows": len(rows),
            "separated_rows": len(manifest_rows),
            "status_rows": status_rows[:500],
        },
        status_path,
    )
    return {"separated_clips": output_path, "summary": status_path}


def run_audio_separation_experiment(
    base_dir: str | Path,
    *,
    clip_run_id: str = "mlb_2024_2026_full_v2",
    prediction_run_id: str = "audio_separated_impact_mlb_2024_2026_v2",
    audio_feature_id: str = "audio_separated_impact_mlb_2024_2026_v2",
    bbe_events: str | Path | None = None,
    clips_path: str | Path | None = None,
    target_registry: str | Path | None = None,
    model_config: str | Path | None = None,
    separation_backend: str = "transient_enhance",
    demucs_model: str = DEFAULT_DEMUCS_MODEL,
    demucs_stem: str = "no_vocals",
    device: str = "cuda",
    allow_model_download: bool = False,
    max_clips: int | None = None,
    require_non_empty: bool = False,
    output_suffix: str = ".parquet",
    cache_dir: str | Path | None = None,
    cache_inputs: bool = False,
) -> dict[str, Path]:
    """Create a separated-audio comparison branch and event predictions."""

    resolved_clips_path = clips_path
    extra_outputs: dict[str, Path] = {}
    if separation_backend == "demucs":
        demucs_outputs = build_demucs_audio_manifest(
            base_dir,
            clip_run_id=clip_run_id,
            clips_path=clips_path,
            max_clips=max_clips,
            model_name=demucs_model,
            stem=demucs_stem,
            device=device,
            allow_model_download=allow_model_download,
            output_suffix=output_suffix,
        )
        resolved_clips_path = demucs_outputs["separated_clips"]
        extra_outputs.update(demucs_outputs)
    elif separation_backend != "transient_enhance":
        raise ValueError(f"unknown separation_backend: {separation_backend}")
    impact_outputs = run_audio_impact_baseline(
        base_dir,
        clip_run_id=clip_run_id,
        prediction_run_id=prediction_run_id,
        audio_feature_id=audio_feature_id,
        bbe_events=bbe_events,
        clips_path=resolved_clips_path,
        target_registry=target_registry or str(Path(__file__).resolve().parents[3] / "configs/targets/target_registry_v1.yaml"),
        model_config=model_config,
        max_clips=max_clips,
        preprocessing_mode="separated",
        require_non_empty=require_non_empty,
        output_suffix=output_suffix,
        cache_dir=cache_dir,
        cache_inputs=cache_inputs,
    )
    return {**extra_outputs, **impact_outputs}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run optional separated audio impact experiment.")
    parser.add_argument("--base-dir", default="/content/drive/MyDrive/baseball_vision")
    parser.add_argument("--clip-run-id", default="mlb_2024_2026_full_v2")
    parser.add_argument("--prediction-run-id", default="audio_separated_impact_mlb_2024_2026_v2")
    parser.add_argument("--audio-feature-id", default="audio_separated_impact_mlb_2024_2026_v2")
    parser.add_argument("--bbe-events", default=None)
    parser.add_argument("--clips", default=None)
    parser.add_argument("--target-registry", default=None)
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--separation-backend", choices=("transient_enhance", "demucs"), default="transient_enhance")
    parser.add_argument("--demucs-model", default=DEFAULT_DEMUCS_MODEL)
    parser.add_argument("--demucs-stem", default="no_vocals")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--require-non-empty", action="store_true")
    parser.add_argument("--output-format", choices=("parquet", "jsonl", "json", "csv"), default="parquet")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--cache-inputs", action="store_true")
    args = parser.parse_args(argv)
    outputs = run_audio_separation_experiment(
        args.base_dir,
        clip_run_id=args.clip_run_id,
        prediction_run_id=args.prediction_run_id,
        audio_feature_id=args.audio_feature_id,
        bbe_events=args.bbe_events,
        clips_path=args.clips,
        target_registry=args.target_registry,
        model_config=args.model_config,
        separation_backend=args.separation_backend,
        demucs_model=args.demucs_model,
        demucs_stem=args.demucs_stem,
        device=args.device,
        allow_model_download=args.allow_model_download,
        max_clips=args.max_clips,
        require_non_empty=args.require_non_empty,
        output_suffix="." + args.output_format,
        cache_dir=args.cache_dir,
        cache_inputs=args.cache_inputs,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
