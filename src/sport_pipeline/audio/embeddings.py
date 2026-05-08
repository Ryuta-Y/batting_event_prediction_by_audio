"""Hugging Face audio embedding runner for contact-centered batting sounds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sport_pipeline.artifact_check import write_json
from sport_pipeline.audio.impact import (
    AUDIO_SCOPES,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_TARGET_REGISTRY,
    AudioImpactConfig,
    _build_audio_sample,
    _config_from_payload,
    _enhance_transient,
    _event_label_fields,
    _load_audio_context,
    _load_json_config,
    _resolved_split,
    _select_audio_clips,
    _slice_samples,
    _train_predict_audio_heads,
)
from sport_pipeline.audio.research_report import write_audio_research_report
from sport_pipeline.evaluation import evaluate_predictions, load_target_registry, validate_prediction_rows
from sport_pipeline.io import read_table, write_table
from sport_pipeline.models.video.heads import build_event_head_specs, build_loss_masks
from sport_pipeline.models.video.predictions import build_visual_prediction_rows


DEFAULT_AUDIO_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"


def _require_download_flag(model_id: str, allow_model_download: bool) -> None:
    if Path(model_id).exists():
        return
    if not allow_model_download:
        raise RuntimeError(f"Model '{model_id}' may download weights. Re-run in Colab with --allow-model-download.")


def _auto_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _load_hf_audio_encoder(
    model_id: str,
    *,
    allow_model_download: bool,
    device: str,
    trust_remote_code: bool,
) -> tuple[Any, Any, Any, str]:
    _require_download_flag(model_id, allow_model_download)
    try:
        import torch  # type: ignore
        from transformers import AutoFeatureExtractor, AutoModel, AutoModelForAudioClassification, AutoProcessor  # type: ignore
    except ImportError as exc:  # pragma: no cover - Colab dependency path
        raise RuntimeError("HF audio embeddings require torch and transformers in Colab.") from exc
    processor_errors: list[str] = []
    processor = None
    for cls in (AutoProcessor, AutoFeatureExtractor):
        try:
            processor = cls.from_pretrained(model_id, trust_remote_code=trust_remote_code)
            break
        except Exception as exc:  # pragma: no cover - depends on remote model config
            processor_errors.append(f"{cls.__name__}: {exc}")
    if processor is None:
        raise RuntimeError(f"could not load processor '{model_id}': {' | '.join(processor_errors)}")
    model_errors: list[str] = []
    for cls in (AutoModel, AutoModelForAudioClassification):
        try:
            model = cls.from_pretrained(model_id, trust_remote_code=trust_remote_code).to(device).eval()
            return torch, processor, model, cls.__name__
        except Exception as exc:  # pragma: no cover - depends on remote model config
            model_errors.append(f"{cls.__name__}: {exc}")
    raise RuntimeError(f"could not load audio encoder '{model_id}': {' | '.join(model_errors)}")


def _tensor_to_vector(output: Any) -> list[float]:
    tensor = getattr(output, "pooler_output", None)
    if tensor is None:
        tensor = getattr(output, "last_hidden_state", None)
        if tensor is not None:
            tensor = tensor.mean(dim=1)
    if tensor is None:
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states:
            tensor = hidden_states[-1].mean(dim=1)
    if tensor is None:
        tensor = getattr(output, "logits", None)
    if tensor is None:
        raise RuntimeError("audio encoder output has no pooler, hidden state, or logits tensor")
    return [float(value) for value in tensor.detach().cpu()[0].float().tolist()]


def _extract_embedding(
    clip_row: dict[str, Any],
    *,
    config: AudioImpactConfig,
    preprocessing_mode: str,
    torch: Any,
    processor: Any,
    model: Any,
    device: str,
) -> dict[str, Any]:
    context_samples, sr, contact_offset_sec, source_window_start_sec = _load_audio_context(clip_row, config)
    start_sec = max(0.0, contact_offset_sec + config.window_start_sec)
    end_sec = max(start_sec + 0.001, contact_offset_sec + config.window_end_sec)
    impact = _slice_samples(context_samples, sr, start_sec, end_sec)
    impact = _enhance_transient(impact, sample_rate=sr, mode=preprocessing_mode)
    if len(impact) == 0:
        raise RuntimeError("empty_audio_window")
    if len(impact) < 400:
        import numpy as np  # type: ignore

        padded = np.zeros(400, dtype="float32")
        padded[: len(impact)] = impact
        impact = padded
    inputs = processor(impact, sampling_rate=sr, return_tensors="pt")
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.no_grad():
        output = model(**inputs)
    vector = _tensor_to_vector(output)
    sample_id = f"{clip_row.get('clip_id')}__hf_audio_embedding"
    return {
        "schema_version": "audio_embedding_v1",
        "sample_id": sample_id,
        "segment_id": f"{clip_row.get('clip_id')}__{preprocessing_mode}__impact",
        "clip_id": clip_row.get("clip_id"),
        "event_id": clip_row.get("event_id"),
        "same_event_group_id": clip_row.get("same_event_group_id", clip_row.get("event_id")),
        "batter_id": clip_row.get("batter_id"),
        "season": clip_row.get("season"),
        "batter_season_id": clip_row.get("batter_season_id"),
        "clip_path": clip_row.get("_resolved_clip_path"),
        "preprocessing_mode": preprocessing_mode,
        "sample_rate": sr,
        "source_window_start_sec": source_window_start_sec,
        "contact_offset_sec": contact_offset_sec,
        "encoder_name": "hf_audio_transformer",
        "embedding_values": vector,
        "embedding_dim": len(vector),
        "split": clip_row.get("split", "unknown"),
    }


def run_hf_audio_embedding_baseline(
    base_dir: str | Path,
    *,
    clip_run_id: str = "mlb_2024_2026_full_v2",
    prediction_run_id: str = "audio_embedding_impact_mlb_2024_2026_v2",
    audio_feature_id: str = "audio_embedding_impact_mlb_2024_2026_v2",
    bbe_events: str | Path | None = None,
    clips_path: str | Path | None = None,
    target_registry: str | Path = DEFAULT_TARGET_REGISTRY,
    model_config: str | Path | None = DEFAULT_MODEL_CONFIG,
    hf_model_id: str = DEFAULT_AUDIO_MODEL_ID,
    allow_model_download: bool = False,
    trust_remote_code: bool = False,
    device: str = "auto",
    max_clips: int | None = None,
    preprocessing_mode: str = "enhanced",
    model_family: str = "auto",
    require_non_empty: bool = False,
    output_suffix: str = ".parquet",
    random_state: int = 42,
    write_report: bool = True,
) -> dict[str, Path]:
    """Extract HF audio embeddings from impact windows and train event heads."""

    hf_model_id = hf_model_id or DEFAULT_AUDIO_MODEL_ID
    payload = _load_json_config(model_config)
    config = _config_from_payload(payload)
    resolved_device = _auto_device(device)
    torch, processor, model, model_class_name = _load_hf_audio_encoder(
        hf_model_id,
        allow_model_download=allow_model_download,
        device=resolved_device,
        trust_remote_code=trust_remote_code,
    )
    base = Path(base_dir)
    bbe_path = Path(bbe_events) if bbe_events else base / "manifests/bbe_events_v1.parquet"
    clips = Path(clips_path) if clips_path else base / f"clips/{clip_run_id}/clips_v1.parquet"
    bbe_rows = read_table(bbe_path)
    clip_rows = read_table(clips) if clips.exists() else []
    events = {str(row["event_id"]): row for row in bbe_rows}
    selected, skipped = _select_audio_clips(
        clip_rows,
        base,
        max_clips=max_clips,
        representative_clip_per_event=config.representative_clip_per_event,
    )

    embedding_rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for clip in selected:
        event = events.get(str(clip["event_id"]))
        if event is None:
            skipped.append({"clip_id": clip.get("clip_id"), "event_id": clip.get("event_id"), "reason": "event_not_found"})
            continue
        try:
            embedding = _extract_embedding(
                clip,
                config=config,
                preprocessing_mode=preprocessing_mode,
                torch=torch,
                processor=processor,
                model=model,
                device=resolved_device,
            )
        except Exception as exc:
            skipped.append({"clip_id": clip.get("clip_id"), "event_id": clip.get("event_id"), "reason": f"hf_audio_embedding_failed:{exc}"})
            continue
        embedding["hf_model_id"] = hf_model_id
        embedding["hf_model_class"] = model_class_name
        embedding_rows.append(embedding)
        sample = {
            "sample_id": embedding["sample_id"],
            "event_id": embedding["event_id"],
            "batter_season_id": embedding["batter_season_id"],
            "embedding_values": embedding["embedding_values"],
            "split": _resolved_split(embedding.get("split"), event.get("split", "unknown")),
            **_event_label_fields(event),
        }
        if not sample["target_xba_available"]:
            sample["xba_missing_reason"] = event.get("label_missing_reason") or "statcast_expected_outcome_missing"
        if not sample["target_xwoba_available"]:
            sample["xwoba_missing_reason"] = event.get("label_missing_reason") or "statcast_expected_outcome_missing"
        samples.append(sample)

    targets = load_target_registry(target_registry)
    head_specs = build_event_head_specs(targets)
    predictions: list[dict[str, Any]] = []
    if samples:
        raw_predictions = _train_predict_audio_heads(
            samples,
            head_specs,
            model_family=model_family,
            ridge_alpha=float((payload.get("head_model") or {}).get("ridge_alpha", 1.0)),
            random_state=random_state,
        )
        predictions = build_visual_prediction_rows(
            run_id=prediction_run_id,
            samples=samples,
            predictions=raw_predictions,
            head_specs=list(head_specs),
            model_family=f"hf_audio_embedding_head:{model_family}",
            aggregation_scope=AUDIO_SCOPES["embedding"],
            loss_masks=build_loss_masks(samples, head_specs),
        )
    validate_prediction_rows(predictions)
    metrics = evaluate_predictions(predictions, targets, run_id=prediction_run_id)

    outputs: dict[str, Path] = {
        "audio_embeddings": base / f"features/{audio_feature_id}/manifest{output_suffix}",
        "audio_samples": base / f"datasets/audio_feature_samples/{prediction_run_id}/manifest{output_suffix}",
        "predictions": base / f"predictions/{prediction_run_id}/predictions_v1{output_suffix}",
        "metrics": base / f"predictions/{prediction_run_id}/metrics_v1.json",
        "summary": base / f"reports/preflight/hf_audio_embedding_baseline_{prediction_run_id}.json",
    }
    summary = {
        "schema_version": "hf_audio_embedding_baseline_summary_v1",
        "clip_run_id": clip_run_id,
        "prediction_run_id": prediction_run_id,
        "audio_feature_id": audio_feature_id,
        "hf_model_id": hf_model_id,
        "model_class_name": model_class_name,
        "device": resolved_device,
        "preprocessing_mode": preprocessing_mode,
        "input_events": len(bbe_rows),
        "input_clips": len(clip_rows),
        "selected_audio_clips": len(selected),
        "embedding_rows": len(embedding_rows),
        "sample_rows": len(samples),
        "prediction_rows": len(predictions),
        "skipped": skipped[:200],
    }
    if require_non_empty and not samples:
        write_json(summary, outputs["summary"])
        raise RuntimeError(f"HF audio embedding baseline produced 0 samples. summary_path={outputs['summary']}")
    write_table(outputs["audio_embeddings"], embedding_rows)
    write_table(outputs["audio_samples"], samples)
    write_table(outputs["predictions"], predictions)
    write_json(metrics, outputs["metrics"])
    write_json(summary, outputs["summary"])
    if write_report:
        report_segments = [
            {
                "audio_status": "complete",
                "clip_id": row.get("clip_id"),
                "event_id": row.get("event_id"),
                "segment_id": row.get("segment_id"),
                "split": row.get("split", "unknown"),
            }
            for row in embedding_rows
        ]
        report_features = [
            {
                "sample_id": row["sample_id"],
                "feature_names": [f"embed_{index}" for index in range(min(32, len(row.get("embedding_values") or [])))],
                "feature_values": list(row.get("embedding_values") or [])[:32],
            }
            for row in embedding_rows
        ]
        outputs.update(
            write_audio_research_report(
                base_dir=base,
                run_id=prediction_run_id,
                preprocessing_mode="hf_embedding",
                model_family=f"hf_audio:{hf_model_id}",
                config={"sample_rate": config.sample_rate, "window_start_ms": config.window_start_ms, "window_end_ms": config.window_end_ms},
                selected_clips=selected,
                segments=report_segments,
                features=report_features,
                predictions=predictions,
                metrics=metrics,
                skipped=skipped,
            )
        )
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run HF audio embedding baseline for batting impact windows.")
    parser.add_argument("--base-dir", default="/content/drive/MyDrive/baseball_vision")
    parser.add_argument("--clip-run-id", default="mlb_2024_2026_full_v2")
    parser.add_argument("--prediction-run-id", default="audio_embedding_impact_mlb_2024_2026_v2")
    parser.add_argument("--audio-feature-id", default="audio_embedding_impact_mlb_2024_2026_v2")
    parser.add_argument("--bbe-events", default=None)
    parser.add_argument("--clips", default=None)
    parser.add_argument("--target-registry", default=str(DEFAULT_TARGET_REGISTRY))
    parser.add_argument("--model-config", default=str(DEFAULT_MODEL_CONFIG))
    parser.add_argument("--hf-model-id", default=DEFAULT_AUDIO_MODEL_ID)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--preprocessing-mode", choices=("raw", "enhanced", "separated"), default="enhanced")
    parser.add_argument("--model-family", choices=("auto", "sklearn_hgb", "ridge"), default="auto")
    parser.add_argument("--require-non-empty", action="store_true")
    parser.add_argument("--output-format", choices=("parquet", "jsonl", "json", "csv"), default="parquet")
    parser.add_argument("--skip-report", action="store_true")
    args = parser.parse_args(argv)
    outputs = run_hf_audio_embedding_baseline(
        args.base_dir,
        clip_run_id=args.clip_run_id,
        prediction_run_id=args.prediction_run_id,
        audio_feature_id=args.audio_feature_id,
        bbe_events=args.bbe_events,
        clips_path=args.clips,
        target_registry=args.target_registry,
        model_config=args.model_config,
        hf_model_id=args.hf_model_id,
        allow_model_download=args.allow_model_download,
        trust_remote_code=args.trust_remote_code,
        device=args.device,
        max_clips=args.max_clips,
        preprocessing_mode=args.preprocessing_mode,
        model_family=args.model_family,
        require_non_empty=args.require_non_empty,
        output_suffix="." + args.output_format,
        write_report=not args.skip_report,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
