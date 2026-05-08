"""Audio impact feature extraction and event-level baseline runner.

The default path is intentionally local-safe: it uses ffmpeg when clips are mp4,
the stdlib ``wave`` module for wav tests, numpy for deterministic impact
features, and scikit-learn only when available. Colab runs can scale this same
contract to the full Drive artifact tree and later compare enhanced/separated
audio variants without changing the downstream prediction schema.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import io
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable, Sequence
import wave

from sport_pipeline.artifact_check import write_json
from sport_pipeline.evaluation import evaluate_predictions, load_target_registry, validate_prediction_rows
from sport_pipeline.io import read_table, write_table
from sport_pipeline.io.runtime_cache import cache_file
from sport_pipeline.models.video.heads import BaselineHeadSpec, build_event_head_specs, build_loss_masks
from sport_pipeline.models.video.predictions import build_visual_prediction_rows
from sport_pipeline.audio.research_report import write_audio_research_report


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TARGET_REGISTRY = PROJECT_ROOT / "configs/targets/target_registry_v1.yaml"
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs/models/audio/audio_impact_baseline_v1.json"
AUDIO_SCOPES = {
    "raw": "audio_raw_impact",
    "enhanced": "audio_enhanced_impact",
    "separated": "audio_separated_impact",
    "embedding": "audio_embedding_impact",
}


@dataclass(frozen=True)
class AudioImpactConfig:
    """Config for contact-centered audio impact extraction."""

    sample_rate: int = 16000
    window_start_ms: float = -250.0
    window_end_ms: float = 150.0
    context_start_ms: float = -750.0
    context_end_ms: float = 500.0
    frame_ms: float = 10.0
    hop_ms: float = 5.0
    high_frequency_hz: float = 2500.0
    low_frequency_hz: float = 500.0
    onset_floor: float = 1e-8
    representative_clip_per_event: bool = True
    min_confidence_for_clean: float = 0.0

    @property
    def window_start_sec(self) -> float:
        return self.window_start_ms / 1000.0

    @property
    def window_end_sec(self) -> float:
        return self.window_end_ms / 1000.0

    @property
    def context_start_sec(self) -> float:
        return self.context_start_ms / 1000.0

    @property
    def context_end_sec(self) -> float:
        return self.context_end_ms / 1000.0


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _to_float(value: Any, default: float = 0.0) -> float:
    if _is_missing(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any, default: bool = False) -> bool:
    if _is_missing(value):
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _resolved_split(primary: Any, fallback: Any = "unknown") -> str:
    value = str(primary or "").strip()
    if value and value.lower() not in {"unknown", "none", "nan"}:
        return value
    fallback_value = str(fallback or "").strip()
    return fallback_value if fallback_value else "unknown"


def _load_json_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def _config_from_payload(payload: dict[str, Any] | None) -> AudioImpactConfig:
    raw = (payload or {}).get("audio_window", payload or {})
    fields = {field.name for field in AudioImpactConfig.__dataclass_fields__.values()}
    kwargs = {key: value for key, value in raw.items() if key in fields}
    return AudioImpactConfig(**kwargs)


def _resolve_clip_path(clip_row: dict[str, Any], base_dir: Path) -> Path | None:
    raw = clip_row.get("audio_path") or clip_row.get("clip_path")
    if _is_missing(raw) or not str(raw):
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = base_dir / path
    return path if path.exists() else None


def _clip_score(row: dict[str, Any]) -> tuple[float, float, float, str]:
    clean_bonus = 1.0 if row.get("clip_status") == "clean_clip" else 0.0
    quality_bonus = 1.0 if row.get("quality_tier") == "usable_primary" else 0.0
    confidence = (
        _to_float(row.get("contact_confidence"))
        + _to_float(row.get("view_confidence"))
        + _to_float(row.get("batter_visibility_score"))
        + _to_float(row.get("bat_visibility_score"))
        + _to_float(row.get("plate_visibility_score"))
    )
    return (clean_bonus, quality_bonus, confidence, str(row.get("clip_id", "")))


def _select_audio_clips(
    clip_rows: list[dict[str, Any]],
    base_dir: Path,
    *,
    max_clips: int | None,
    representative_clip_per_event: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped: list[dict[str, Any]] = []
    for row in clip_rows:
        if row.get("clip_status") == "excluded":
            skipped.append({"clip_id": row.get("clip_id"), "event_id": row.get("event_id"), "reason": "excluded_clip"})
            continue
        clip_path = _resolve_clip_path(row, base_dir)
        if clip_path is None:
            skipped.append({"clip_id": row.get("clip_id"), "event_id": row.get("event_id"), "reason": "missing_clip_path"})
            continue
        enriched = dict(row)
        enriched["_resolved_clip_path"] = str(clip_path)
        by_event[str(row["event_id"])].append(enriched)
    if representative_clip_per_event:
        selected = [sorted(rows, key=_clip_score, reverse=True)[0] for rows in by_event.values()]
    else:
        selected = [row for rows in by_event.values() for row in rows]
    selected = sorted(selected, key=lambda row: (str(row.get("event_id", "")), str(row.get("clip_id", ""))))
    if max_clips is not None:
        selected = selected[:max_clips]
    return selected, skipped


def _read_wav_bytes(payload: bytes) -> tuple["Any", int]:
    import numpy as np  # type: ignore

    with wave.open(io.BytesIO(payload), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if sample_width == 1:
        arr = np.frombuffer(frames, dtype=np.uint8).astype("float32")
        arr = (arr - 128.0) / 128.0
    elif sample_width == 2:
        arr = np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0
    elif sample_width == 4:
        arr = np.frombuffer(frames, dtype="<i4").astype("float32") / 2147483648.0
    else:
        raise RuntimeError(f"unsupported wav sample width: {sample_width}")
    if channels > 1:
        arr = arr.reshape(-1, channels).mean(axis=1)
    return arr, int(sample_rate)


def _slice_samples(samples: Any, sample_rate: int, start_sec: float, end_sec: float) -> Any:
    start = max(0, int(round(start_sec * sample_rate)))
    end = max(start, int(round(end_sec * sample_rate)))
    return samples[start:end]


def _read_wav_window(path: Path, *, start_sec: float, end_sec: float, sample_rate: int) -> tuple[Any, int]:
    import numpy as np  # type: ignore

    samples, source_rate = _read_wav_bytes(path.read_bytes())
    sliced = _slice_samples(samples, source_rate, start_sec, end_sec)
    if source_rate == sample_rate:
        return sliced.astype("float32"), sample_rate
    if len(sliced) == 0:
        return np.asarray([], dtype="float32"), sample_rate
    old_x = np.linspace(0.0, 1.0, num=len(sliced), endpoint=False)
    new_len = max(1, int(round(len(sliced) * sample_rate / source_rate)))
    new_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
    return np.interp(new_x, old_x, sliced).astype("float32"), sample_rate


def _ffmpeg_audio_window(path: Path, *, start_sec: float, duration_sec: float, sample_rate: int) -> tuple[Any, int]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, start_sec):.3f}",
        "-t",
        f"{max(duration_sec, 0.001):.3f}",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        "pipe:1",
    ]
    proc = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg audio extraction failed: {stderr[:240]}")
    return _read_wav_bytes(proc.stdout)


def _contact_time_sec(row: dict[str, Any]) -> float:
    for key in ("contact_time_sec", "contact_sec", "impact_time_sec"):
        if not _is_missing(row.get(key)):
            return max(0.0, _to_float(row.get(key)))
    frame = _to_float(row.get("contact_frame"), default=-1.0)
    fps = _to_float(row.get("fps") or row.get("clip_fps") or row.get("source_fps"), default=0.0)
    if frame >= 0 and fps > 0:
        return max(0.0, frame / fps)
    start = _to_float(row.get("clip_start_sec"), default=0.0)
    end = _to_float(row.get("clip_end_sec"), default=0.0)
    if end > start:
        return max(0.0, (end - start) / 2.0)
    return _to_float(row.get("duration_sec"), default=0.0) / 2.0


def _load_audio_context(clip_row: dict[str, Any], config: AudioImpactConfig) -> tuple[Any, int, float, float]:
    clip_path = Path(str(clip_row.get("_runtime_clip_path") or clip_row["_resolved_clip_path"]))
    contact_sec = _contact_time_sec(clip_row)
    start_sec = max(0.0, contact_sec + config.context_start_sec)
    end_sec = max(start_sec + 0.001, contact_sec + config.context_end_sec)
    if clip_path.suffix.lower() == ".wav":
        samples, sr = _read_wav_window(clip_path, start_sec=start_sec, end_sec=end_sec, sample_rate=config.sample_rate)
    else:
        samples, sr = _ffmpeg_audio_window(
            clip_path,
            start_sec=start_sec,
            duration_sec=end_sec - start_sec,
            sample_rate=config.sample_rate,
        )
    return samples, sr, contact_sec - start_sec, start_sec


def _preemphasis(samples: Any, coefficient: float = 0.97) -> Any:
    import numpy as np  # type: ignore

    if len(samples) == 0:
        return samples
    output = np.empty_like(samples, dtype="float32")
    output[0] = samples[0]
    output[1:] = samples[1:] - coefficient * samples[:-1]
    return output


def _moving_average_subtract(samples: Any, sample_rate: int, window_ms: float = 25.0) -> Any:
    import numpy as np  # type: ignore

    if len(samples) == 0:
        return samples
    window = max(3, int(round(sample_rate * window_ms / 1000.0)))
    kernel = np.ones(window, dtype="float32") / float(window)
    smooth = np.convolve(samples, kernel, mode="same")
    return (samples - smooth).astype("float32")


def _enhance_transient(samples: Any, sample_rate: int, mode: str) -> Any:
    import numpy as np  # type: ignore

    arr = np.asarray(samples, dtype="float32")
    if mode == "raw":
        return arr
    if mode in {"enhanced", "separated"}:
        enhanced = _moving_average_subtract(arr, sample_rate=sample_rate)
        enhanced = _preemphasis(enhanced)
        peak = float(np.max(np.abs(enhanced))) if len(enhanced) else 0.0
        return enhanced / peak if peak > 0 else enhanced
    return arr


def _frame_view(samples: Any, frame_length: int, hop_length: int) -> Any:
    import numpy as np  # type: ignore

    if len(samples) < frame_length:
        padded = np.zeros(frame_length, dtype="float32")
        padded[: len(samples)] = samples
        return padded[None, :]
    starts = np.arange(0, len(samples) - frame_length + 1, hop_length)
    return np.stack([samples[start : start + frame_length] for start in starts])


def _safe_stat(value: float) -> float:
    if math.isfinite(value):
        return float(value)
    return 0.0


def _spectral_features(samples: Any, sample_rate: int, config: AudioImpactConfig) -> dict[str, float]:
    import numpy as np  # type: ignore

    if len(samples) == 0:
        return {
            "spectral_centroid_hz": 0.0,
            "spectral_bandwidth_hz": 0.0,
            "spectral_rolloff_85_hz": 0.0,
            "high_frequency_energy_ratio": 0.0,
            "low_frequency_energy_ratio": 0.0,
            "spectral_flatness": 0.0,
            "dominant_frequency_hz": 0.0,
        }
    windowed = samples * np.hanning(len(samples))
    spectrum = np.abs(np.fft.rfft(windowed)).astype("float64")
    freqs = np.fft.rfftfreq(len(windowed), d=1.0 / sample_rate)
    energy = spectrum**2
    total = float(np.sum(energy) + 1e-12)
    centroid = float(np.sum(freqs * energy) / total)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * energy) / total))
    cumulative = np.cumsum(energy)
    rolloff_idx = int(np.searchsorted(cumulative, 0.85 * cumulative[-1])) if cumulative[-1] > 0 else 0
    high_ratio = float(np.sum(energy[freqs >= config.high_frequency_hz]) / total)
    low_ratio = float(np.sum(energy[freqs <= config.low_frequency_hz]) / total)
    flatness = float(np.exp(np.mean(np.log(spectrum + 1e-12))) / (np.mean(spectrum) + 1e-12))
    dominant = float(freqs[int(np.argmax(energy))]) if len(energy) else 0.0
    return {
        "spectral_centroid_hz": _safe_stat(centroid),
        "spectral_bandwidth_hz": _safe_stat(bandwidth),
        "spectral_rolloff_85_hz": _safe_stat(float(freqs[min(rolloff_idx, len(freqs) - 1)])),
        "high_frequency_energy_ratio": _safe_stat(high_ratio),
        "low_frequency_energy_ratio": _safe_stat(low_ratio),
        "spectral_flatness": _safe_stat(flatness),
        "dominant_frequency_hz": _safe_stat(dominant),
    }


def _impact_features(samples: Any, sample_rate: int, config: AudioImpactConfig) -> dict[str, float]:
    import numpy as np  # type: ignore

    arr = np.asarray(samples, dtype="float32")
    if len(arr) == 0:
        return {
            "audio_duration_ms": 0.0,
            "rms": 0.0,
            "peak_abs": 0.0,
            "crest_factor": 0.0,
            "zero_crossing_rate": 0.0,
            "frame_energy_peak": 0.0,
            "frame_energy_mean": 0.0,
            "frame_energy_std": 0.0,
            "impact_onset_strength": 0.0,
            "impact_peak_time_ms": 0.0,
            "post_pre_energy_ratio": 0.0,
            "decay_slope": 0.0,
            "impact_confidence": 0.0,
            **_spectral_features(arr, sample_rate, config),
        }
    rms = float(np.sqrt(np.mean(arr**2)))
    peak = float(np.max(np.abs(arr)))
    crest = peak / max(rms, 1e-12)
    zcr = float(np.mean(np.abs(np.diff(np.signbit(arr))).astype("float32"))) if len(arr) > 1 else 0.0
    frame_len = max(16, int(round(sample_rate * config.frame_ms / 1000.0)))
    hop_len = max(1, int(round(sample_rate * config.hop_ms / 1000.0)))
    frames = _frame_view(arr, frame_length=frame_len, hop_length=hop_len)
    energies = np.mean(frames**2, axis=1)
    energy_peak = float(np.max(energies)) if len(energies) else 0.0
    energy_mean = float(np.mean(energies)) if len(energies) else 0.0
    energy_std = float(np.std(energies)) if len(energies) else 0.0
    deltas = np.diff(energies, prepend=energies[0]) if len(energies) else np.asarray([0.0])
    onset = float(np.max(deltas) / max(energy_mean, config.onset_floor))
    peak_idx = int(np.argmax(energies)) if len(energies) else 0
    peak_time_ms = float((peak_idx * hop_len + frame_len / 2) / sample_rate * 1000.0)
    midpoint = max(1, len(arr) // 2)
    pre_energy = float(np.mean(arr[:midpoint] ** 2)) if midpoint > 0 else 0.0
    post_energy = float(np.mean(arr[midpoint:] ** 2)) if len(arr) > midpoint else 0.0
    post_pre = post_energy / max(pre_energy, 1e-12)
    if len(energies) >= 3:
        x = np.arange(len(energies), dtype="float32")
        y = np.log(energies + 1e-12)
        decay_slope = float(np.polyfit(x, y, 1)[0])
    else:
        decay_slope = 0.0
    impact_confidence = float(
        min(
            1.0,
            0.45 * math.tanh(onset / 10.0)
            + 0.35 * math.tanh(crest / 8.0)
            + 0.20 * math.tanh(peak * 8.0),
        )
    )
    return {
        "audio_duration_ms": float(len(arr) / sample_rate * 1000.0),
        "rms": _safe_stat(rms),
        "peak_abs": _safe_stat(peak),
        "crest_factor": _safe_stat(crest),
        "zero_crossing_rate": _safe_stat(zcr),
        "frame_energy_peak": _safe_stat(energy_peak),
        "frame_energy_mean": _safe_stat(energy_mean),
        "frame_energy_std": _safe_stat(energy_std),
        "impact_onset_strength": _safe_stat(onset),
        "impact_peak_time_ms": _safe_stat(peak_time_ms),
        "post_pre_energy_ratio": _safe_stat(post_pre),
        "decay_slope": _safe_stat(decay_slope),
        "impact_confidence": _safe_stat(impact_confidence),
        **_spectral_features(arr, sample_rate, config),
    }


def extract_audio_impact_features(
    clip_row: dict[str, Any],
    *,
    config: AudioImpactConfig | None = None,
    preprocessing_mode: str = "raw",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract contact-window audio features for one clip row.

    Returns ``(segment_row, feature_row)``. The segment row is useful for QA and
    leakage review; the feature row is used by the supervised heads.
    """

    import numpy as np  # type: ignore

    resolved_config = config or AudioImpactConfig()
    context_samples, sr, contact_offset_sec, source_window_start_sec = _load_audio_context(clip_row, resolved_config)
    start_sec = max(0.0, contact_offset_sec + resolved_config.window_start_sec)
    end_sec = max(start_sec + 0.001, contact_offset_sec + resolved_config.window_end_sec)
    impact = _slice_samples(context_samples, sr, start_sec, end_sec)
    impact = _enhance_transient(impact, sample_rate=sr, mode=preprocessing_mode)
    features = _impact_features(impact, sample_rate=sr, config=resolved_config)
    features["signal_abs_mean"] = float(np.mean(np.abs(impact))) if len(impact) else 0.0
    clip_path = str(clip_row.get("_resolved_clip_path"))
    audio_status = "complete" if len(impact) > 0 else "empty_audio_window"
    segment_id = f"{clip_row.get('clip_id')}__{preprocessing_mode}__impact"
    segment = {
        "schema_version": "audio_segments_v1",
        "segment_id": segment_id,
        "clip_id": clip_row.get("clip_id"),
        "event_id": clip_row.get("event_id"),
        "same_event_group_id": clip_row.get("same_event_group_id", clip_row.get("event_id")),
        "batter_id": clip_row.get("batter_id"),
        "season": clip_row.get("season"),
        "batter_season_id": clip_row.get("batter_season_id"),
        "clip_path": clip_path,
        "preprocessing_mode": preprocessing_mode,
        "sample_rate": sr,
        "source_window_start_sec": source_window_start_sec,
        "contact_offset_sec": contact_offset_sec,
        "impact_window_start_ms": resolved_config.window_start_ms,
        "impact_window_end_ms": resolved_config.window_end_ms,
        "num_samples": int(len(impact)),
        "audio_status": audio_status,
        "impact_peak_time_ms": features["impact_peak_time_ms"],
        "impact_confidence": features["impact_confidence"],
        "contact_confidence": _to_float(clip_row.get("contact_confidence"), 0.0),
        "view_confidence": _to_float(clip_row.get("view_confidence"), 0.0),
        "split": clip_row.get("split", "unknown"),
    }
    feature_values = [float(features[key]) for key in AUDIO_FEATURE_KEYS]
    feature_row = {
        "schema_version": "audio_features_v1",
        "sample_id": segment_id,
        "segment_id": segment_id,
        "clip_id": clip_row.get("clip_id"),
        "event_id": clip_row.get("event_id"),
        "same_event_group_id": clip_row.get("same_event_group_id", clip_row.get("event_id")),
        "batter_id": clip_row.get("batter_id"),
        "season": clip_row.get("season"),
        "batter_season_id": clip_row.get("batter_season_id"),
        "clip_path": clip_path,
        "preprocessing_mode": preprocessing_mode,
        "extractor_name": "numpy_contact_impact_features",
        "extractor_version": "audio_impact_v1",
        "audio_status": audio_status,
        "feature_names": list(AUDIO_FEATURE_KEYS),
        "feature_values": feature_values,
        "feature_dim": len(feature_values),
        "split": clip_row.get("split", "unknown"),
        **{key: float(value) for key, value in features.items()},
    }
    return segment, feature_row


AUDIO_FEATURE_KEYS = (
    "audio_duration_ms",
    "rms",
    "peak_abs",
    "crest_factor",
    "zero_crossing_rate",
    "frame_energy_peak",
    "frame_energy_mean",
    "frame_energy_std",
    "impact_onset_strength",
    "impact_peak_time_ms",
    "post_pre_energy_ratio",
    "decay_slope",
    "impact_confidence",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "spectral_rolloff_85_hz",
    "high_frequency_energy_ratio",
    "low_frequency_energy_ratio",
    "spectral_flatness",
    "dominant_frequency_hz",
    "signal_abs_mean",
)


def _event_label_fields(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "launch_speed": event.get("launch_speed"),
        "launch_angle": event.get("launch_angle"),
        "target_hard_hit": event.get("target_hard_hit"),
        "target_barrel": event.get("target_barrel"),
        "estimated_ba_using_speedangle": event.get("estimated_ba_using_speedangle"),
        "estimated_woba_using_speedangle": event.get("estimated_woba_using_speedangle"),
        "target_ev_available": _to_bool(event.get("target_ev_available"), event.get("launch_speed") is not None),
        "target_la_available": _to_bool(event.get("target_la_available"), event.get("launch_angle") is not None),
        "target_hard_hit_available": _to_bool(event.get("target_hard_hit_available"), event.get("target_hard_hit") is not None),
        "target_barrel_available": _to_bool(event.get("target_barrel_available"), event.get("target_barrel") is not None),
        "target_xba_available": _to_bool(event.get("target_xba_available"), event.get("estimated_ba_using_speedangle") is not None),
        "target_xwoba_available": _to_bool(event.get("target_xwoba_available"), event.get("estimated_woba_using_speedangle") is not None),
    }


def _build_audio_sample(feature_row: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    sample = {
        "sample_id": str(feature_row["sample_id"]),
        "clip_id": feature_row.get("clip_id"),
        "event_id": feature_row["event_id"],
        "same_event_group_id": feature_row.get("same_event_group_id"),
        "batter_id": feature_row.get("batter_id"),
        "season": feature_row.get("season"),
        "batter_season_id": feature_row["batter_season_id"],
        "embedding_values": feature_row["feature_values"],
        "preprocessing_mode": feature_row.get("preprocessing_mode"),
        "audio_status": feature_row.get("audio_status"),
        "impact_confidence": feature_row.get("impact_confidence"),
        "split": _resolved_split(feature_row.get("split"), event.get("split", "unknown")),
        **_event_label_fields(event),
    }
    if not sample["target_xba_available"]:
        sample["xba_missing_reason"] = event.get("label_missing_reason") or "statcast_expected_outcome_missing"
    if not sample["target_xwoba_available"]:
        sample["xwoba_missing_reason"] = event.get("label_missing_reason") or "statcast_expected_outcome_missing"
    return sample


def _training_indices(samples: list[dict[str, Any]], spec: BaselineHeadSpec) -> list[int]:
    availability_key = f"target_{spec.name}_available"
    indices = []
    for index, sample in enumerate(samples):
        split = str(sample.get("split", "")).lower()
        if split not in {"train", "training"}:
            continue
        if availability_key in sample and not bool(sample[availability_key]):
            continue
        if sample.get(spec.column) is None:
            continue
        indices.append(index)
    if len(indices) >= 2:
        return indices
    return [
        index
        for index, sample in enumerate(samples)
        if (availability_key not in sample or bool(sample[availability_key])) and sample.get(spec.column) is not None
    ]


def _ridge_predict(features: list[list[float]], targets: list[float], all_features: list[list[float]], *, alpha: float, kind: str) -> list[float]:
    import numpy as np  # type: ignore

    x = np.asarray(features, dtype="float64")
    y = np.asarray(targets, dtype="float64")
    x_all = np.asarray(all_features, dtype="float64")
    if x.ndim != 2 or x.shape[0] == 0:
        value = float(np.mean(y)) if len(y) else 0.0
        preds = np.full((len(all_features),), value, dtype="float64")
    else:
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std == 0] = 1.0
        x_scaled = (x - mean) / std
        x_design = np.c_[np.ones(len(x_scaled)), x_scaled]
        penalty = np.eye(x_design.shape[1])
        penalty[0, 0] = 0.0
        try:
            weights = np.linalg.solve(x_design.T @ x_design + alpha * penalty, x_design.T @ y)
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(x_design.T @ x_design + alpha * penalty) @ x_design.T @ y
        all_scaled = (x_all - mean) / std if len(x_all) else x_all
        preds = np.c_[np.ones(len(all_scaled)), all_scaled] @ weights if len(all_scaled) else np.asarray([])
    if kind in {"binary", "probability"}:
        return [float(min(1.0, max(0.0, value))) for value in preds]
    return [float(value) for value in preds]


def _sklearn_predict(
    features: list[list[float]],
    targets: list[float],
    all_features: list[list[float]],
    *,
    spec: BaselineHeadSpec,
    random_state: int,
) -> list[float] | None:
    if len(features) < 5:
        return None
    try:
        import numpy as np  # type: ignore
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor  # type: ignore
        from sklearn.pipeline import make_pipeline  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
    except ImportError:
        return None
    x = np.asarray(features, dtype="float32")
    y = np.asarray(targets)
    x_all = np.asarray(all_features, dtype="float32")
    if spec.kind in {"binary", "probability"}:
        if len(set(int(value >= 0.5) for value in y.tolist())) < 2:
            return None
        model = make_pipeline(
            StandardScaler(),
            HistGradientBoostingClassifier(max_iter=120, learning_rate=0.05, max_leaf_nodes=15, random_state=random_state),
        )
        model.fit(x, (y >= 0.5).astype("int32"))
        probs = model.predict_proba(x_all)[:, 1]
        return [float(value) for value in probs]
    model = make_pipeline(
        StandardScaler(),
        HistGradientBoostingRegressor(max_iter=160, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=0.05, random_state=random_state),
    )
    model.fit(x, y.astype("float32"))
    preds = model.predict(x_all)
    return [float(value) for value in preds]


def _train_predict_audio_heads(
    samples: list[dict[str, Any]],
    head_specs: Sequence[BaselineHeadSpec],
    *,
    model_family: str,
    ridge_alpha: float,
    random_state: int,
) -> dict[str, list[float]]:
    all_features = [sample["embedding_values"] for sample in samples]
    predictions: dict[str, list[float]] = {}
    for spec in head_specs:
        indices = _training_indices(samples, spec)
        y = [_to_float(samples[index].get(spec.column)) for index in indices]
        x = [all_features[index] for index in indices]
        if model_family in {"auto", "sklearn_hgb"}:
            fitted = _sklearn_predict(x, y, all_features, spec=spec, random_state=random_state)
            if fitted is not None:
                predictions[spec.name] = fitted
                continue
        predictions[spec.name] = _ridge_predict(x, y, all_features, alpha=ridge_alpha, kind=spec.kind)
    return predictions


def _cache_selected_clips(
    selected: list[dict[str, Any]],
    *,
    cache_dir: str | Path | None,
    namespace: str,
    enabled: bool,
    num_workers: int,
    max_file_mb: float | None,
    min_free_disk_gb: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stats: dict[str, Any] = {"enabled": bool(enabled and cache_dir is not None), "used": 0, "reasons": {}}
    if not enabled or cache_dir is None or not selected:
        return selected, stats
    max_file_bytes = None if max_file_mb is None else int(float(max_file_mb) * 1024**2)
    min_free_bytes = int(float(min_free_disk_gb) * 1024**3)

    def stage(index: int, clip: dict[str, Any]) -> tuple[int, dict[str, Any], str, bool]:
        result = cache_file(
            clip["_resolved_clip_path"],
            cache_dir=cache_dir,
            namespace=namespace,
            key=str(clip.get("clip_id") or index),
            enabled=True,
            max_file_bytes=max_file_bytes,
            min_free_disk_bytes=min_free_bytes,
        )
        staged = dict(clip)
        staged["_runtime_clip_path"] = str(result.path)
        return index, staged, result.reason, result.used_cache

    max_workers = max(1, int(num_workers or 1))
    if max_workers == 1:
        results = [stage(index, clip) for index, clip in enumerate(selected)]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(stage, index, clip) for index, clip in enumerate(selected)]
            for future in as_completed(futures):
                results.append(future.result())
    staged_by_index: dict[int, dict[str, Any]] = {}
    for index, staged, reason, used in results:
        staged_by_index[index] = staged
        stats["reasons"][reason] = int(stats["reasons"].get(reason, 0)) + 1
        if used:
            stats["used"] += 1
    return [staged_by_index.get(index, clip) for index, clip in enumerate(selected)], stats


def _write_audio_report(
    *,
    outputs: dict[str, Path],
    base: Path,
    run_id: str,
    preprocessing_mode: str,
    model_family: str,
    config: AudioImpactConfig,
    selected: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    features: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    metrics: dict[str, Any],
    skipped: list[dict[str, Any]],
) -> None:
    report_outputs = write_audio_research_report(
        base_dir=base,
        run_id=run_id,
        preprocessing_mode=preprocessing_mode,
        model_family=model_family,
        config={
            "sample_rate": config.sample_rate,
            "window_start_ms": config.window_start_ms,
            "window_end_ms": config.window_end_ms,
            "context_start_ms": config.context_start_ms,
            "context_end_ms": config.context_end_ms,
        },
        selected_clips=selected,
        segments=segments,
        features=features,
        predictions=predictions,
        metrics=metrics,
        skipped=skipped,
    )
    outputs.update(report_outputs)


def run_audio_impact_baseline(
    base_dir: str | Path,
    *,
    clip_run_id: str = "mlb_2024_2026_full_v2",
    prediction_run_id: str = "audio_raw_impact_mlb_2024_2026_v2",
    audio_feature_id: str = "audio_impact_mlb_2024_2026_v2",
    bbe_events: str | Path | None = None,
    clips_path: str | Path | None = None,
    target_registry: str | Path = DEFAULT_TARGET_REGISTRY,
    model_config: str | Path | None = DEFAULT_MODEL_CONFIG,
    max_clips: int | None = None,
    preprocessing_mode: str = "raw",
    model_family: str = "auto",
    ridge_alpha: float = 1.0,
    require_non_empty: bool = False,
    output_suffix: str = ".parquet",
    cache_dir: str | Path | None = None,
    cache_inputs: bool = False,
    cache_num_workers: int = 4,
    cache_min_free_disk_gb: float = 20.0,
    cache_max_file_mb: float | None = None,
    valid_audio_only: bool = False,
    random_state: int = 42,
    write_report: bool = True,
) -> dict[str, Path]:
    """Run contact-window audio feature extraction and supervised heads."""

    payload = _load_json_config(model_config)
    config = _config_from_payload(payload)
    resolved_model = payload.get("head_model", {}) if isinstance(payload.get("head_model"), dict) else {}
    if model_family == "auto":
        model_family = str(resolved_model.get("model_family", model_family))
    if ridge_alpha == 1.0:
        ridge_alpha = float(resolved_model.get("ridge_alpha", ridge_alpha))

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
    selected, cache_stats = _cache_selected_clips(
        selected,
        cache_dir=cache_dir,
        namespace=f"runtime_io/audio_impact/{prediction_run_id}/clips",
        enabled=cache_inputs,
        num_workers=cache_num_workers,
        max_file_mb=cache_max_file_mb,
        min_free_disk_gb=cache_min_free_disk_gb,
    )

    segments: list[dict[str, Any]] = []
    features: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for clip in selected:
        event = events.get(str(clip["event_id"]))
        if event is None:
            skipped.append({"clip_id": clip.get("clip_id"), "event_id": clip.get("event_id"), "reason": "event_not_found"})
            continue
        try:
            segment, feature = extract_audio_impact_features(clip, config=config, preprocessing_mode=preprocessing_mode)
        except Exception as exc:
            skipped.append({"clip_id": clip.get("clip_id"), "event_id": clip.get("event_id"), "reason": f"audio_feature_extraction_failed:{exc}"})
            continue
        segments.append(segment)
        if valid_audio_only and segment.get("audio_status") != "complete":
            skipped.append(
                {
                    "clip_id": clip.get("clip_id"),
                    "event_id": clip.get("event_id"),
                    "reason": f"filtered_{segment.get('audio_status', 'invalid_audio')}",
                }
            )
            continue
        features.append(feature)
        samples.append(_build_audio_sample(feature, event))

    targets = load_target_registry(target_registry)
    head_specs = build_event_head_specs(targets)
    predictions: list[dict[str, Any]] = []
    if samples:
        raw_predictions = _train_predict_audio_heads(
            samples,
            head_specs,
            model_family=model_family,
            ridge_alpha=ridge_alpha,
            random_state=random_state,
        )
        predictions = build_visual_prediction_rows(
            run_id=prediction_run_id,
            samples=samples,
            predictions=raw_predictions,
            head_specs=list(head_specs),
            model_family=model_family if model_family != "auto" else "audio_hgb_or_ridge",
            aggregation_scope=AUDIO_SCOPES.get(preprocessing_mode, f"audio_{preprocessing_mode}_impact"),
            loss_masks=build_loss_masks(samples, head_specs),
        )
    validate_prediction_rows(predictions)
    metrics = evaluate_predictions(predictions, targets, run_id=prediction_run_id)

    outputs: dict[str, Path] = {
        "audio_segments": base / f"features/{audio_feature_id}/segments_v1{output_suffix}",
        "audio_features": base / f"features/{audio_feature_id}/manifest{output_suffix}",
        "audio_samples": base / f"datasets/audio_feature_samples/{prediction_run_id}/manifest{output_suffix}",
        "predictions": base / f"predictions/{prediction_run_id}/predictions_v1{output_suffix}",
        "metrics": base / f"predictions/{prediction_run_id}/metrics_v1.json",
        "summary": base / f"reports/preflight/audio_impact_baseline_{prediction_run_id}.json",
    }
    summary_payload = {
        "schema_version": "audio_impact_baseline_summary_v1",
        "clip_run_id": clip_run_id,
        "prediction_run_id": prediction_run_id,
        "audio_feature_id": audio_feature_id,
        "preprocessing_mode": preprocessing_mode,
        "model_family": model_family,
        "ridge_alpha": ridge_alpha,
        "input_events": len(bbe_rows),
        "input_clips": len(clip_rows),
        "selected_audio_clips": len(selected),
        "segment_rows": len(segments),
        "feature_rows": len(features),
        "sample_rows": len(samples),
        "prediction_rows": len(predictions),
        "target_names": [spec.name for spec in head_specs],
        "audio_window": {
            "sample_rate": config.sample_rate,
            "window_start_ms": config.window_start_ms,
            "window_end_ms": config.window_end_ms,
            "context_start_ms": config.context_start_ms,
            "context_end_ms": config.context_end_ms,
        },
        "cache_dir": None if cache_dir is None else str(cache_dir),
        "cache_inputs": cache_inputs,
        "cache_stats": cache_stats,
        "valid_audio_only": valid_audio_only,
        "invalid_audio_rows_filtered": sum(1 for row in segments if row.get("audio_status") != "complete") if valid_audio_only else 0,
        "skipped": skipped[:200],
        "output_suffix": output_suffix,
    }
    if require_non_empty and not samples:
        write_json(summary_payload, outputs["summary"])
        raise RuntimeError(
            "audio impact baseline produced 0 samples; check clips_v1 clip_path audio streams and ffmpeg availability. "
            f"summary_path={outputs['summary']}"
        )
    write_table(outputs["audio_segments"], segments)
    write_table(outputs["audio_features"], features)
    write_table(outputs["audio_samples"], samples)
    write_table(outputs["predictions"], predictions)
    write_json(metrics, outputs["metrics"])
    write_json(summary_payload, outputs["summary"])
    if write_report:
        _write_audio_report(
            outputs=outputs,
            base=base,
            run_id=prediction_run_id,
            preprocessing_mode=preprocessing_mode,
            model_family=model_family,
            config=config,
            selected=selected,
            segments=segments,
            features=features,
            predictions=predictions,
            metrics=metrics,
            skipped=skipped,
        )
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run contact-window audio impact baseline artifacts.")
    parser.add_argument("--base-dir", default="/content/drive/MyDrive/baseball_vision")
    parser.add_argument("--clip-run-id", default="mlb_2024_2026_full_v2")
    parser.add_argument("--prediction-run-id", default="audio_raw_impact_mlb_2024_2026_v2")
    parser.add_argument("--audio-feature-id", default="audio_impact_mlb_2024_2026_v2")
    parser.add_argument("--bbe-events", default=None)
    parser.add_argument("--clips", default=None)
    parser.add_argument("--target-registry", default=str(DEFAULT_TARGET_REGISTRY))
    parser.add_argument("--model-config", default=str(DEFAULT_MODEL_CONFIG))
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--preprocessing-mode", choices=("raw", "enhanced", "separated", "embedding"), default="raw")
    parser.add_argument("--model-family", choices=("auto", "sklearn_hgb", "ridge"), default="auto")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--require-non-empty", action="store_true")
    parser.add_argument("--output-format", choices=("parquet", "jsonl", "json", "csv"), default="parquet")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--cache-inputs", action="store_true")
    parser.add_argument("--cache-num-workers", type=int, default=4)
    parser.add_argument("--cache-min-free-disk-gb", type=float, default=20.0)
    parser.add_argument("--cache-max-file-mb", type=float, default=None)
    parser.add_argument("--valid-audio-only", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-report", action="store_true")
    args = parser.parse_args(argv)
    outputs = run_audio_impact_baseline(
        args.base_dir,
        clip_run_id=args.clip_run_id,
        prediction_run_id=args.prediction_run_id,
        audio_feature_id=args.audio_feature_id,
        bbe_events=args.bbe_events,
        clips_path=args.clips,
        target_registry=args.target_registry,
        model_config=args.model_config,
        max_clips=args.max_clips,
        preprocessing_mode=args.preprocessing_mode,
        model_family=args.model_family,
        ridge_alpha=args.ridge_alpha,
        require_non_empty=args.require_non_empty,
        output_suffix="." + args.output_format,
        cache_dir=args.cache_dir,
        cache_inputs=args.cache_inputs,
        cache_num_workers=args.cache_num_workers,
        cache_min_free_disk_gb=args.cache_min_free_disk_gb,
        cache_max_file_mb=args.cache_max_file_mb,
        valid_audio_only=args.valid_audio_only,
        random_state=args.random_state,
        write_report=not args.skip_report,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
