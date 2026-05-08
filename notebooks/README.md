# notebooks

This clean GitHub candidate keeps only the audio branch Colab entrypoints.

| order | notebook | runtime | creates |
|---:|---|---|---|
| 36 | `36_cpu_audio_impact_baseline.ipynb` | CPU / high RAM | `features/audio_impact_*`, `features/audio_enhanced_impact_*`, `predictions/audio_raw_impact_*`, `predictions/audio_enhanced_impact_*`, `reports/audio_impact/*` |
| 37 | `37_gpu_audio_separation_and_embeddings.ipynb` | Colab GPU | `features/audio_separated_impact_*`, `features/audio_embedding_impact_*`, `predictions/audio_separated_impact_*`, `predictions/audio_embedding_impact_*`, `reports/audio_impact/*` |
| 38 | `38_cpu_audio_fusion_compare.ipynb` | CPU | audio player-season projections, `predictions/fusion_audio_*`, `reports/method_evaluation/*with_audio*` |
| 39 | `39_cpu_audio_audit_baseline_compare.ipynb` | CPU | `features/audio_presence_*/audio_valid_clips_v1.parquet`, valid-audio raw/enhanced predictions, `reports/audio_baseline_compare/*` |

## Required Inputs

All notebooks expect the code folder to be available at:

```text
/content/drive/MyDrive/codex/batting_codex_handoff_with_audio
```

and Drive artifacts at:

```text
/content/drive/MyDrive/baseball_vision
```

Minimum required artifacts:

```text
manifests/bbe_events_v1.parquet
clips/{full_run_id}/clips_v1.parquet
```

For `38` and the comparison part of `39`, existing baseline prediction runs are also expected, especially:

```text
predictions/context_catboost_mlb_2024_2026_v2/predictions_v1.parquet
predictions/video_lightweight_cv2_mlb_2024_2026_v2/predictions_v1.parquet
predictions/fusion_mlb_2024_2026_v2/predictions_v1.parquet
```

## Practical Run Order

Use `mlb_2024_2026_real_colab_v2.json` unless you deliberately create a new profile.

1. Run `39` once if you need an audio-presence audit or `audio_valid_clips_v1.parquet`.
2. Run `36` to create raw/enhanced audio impact baselines.
3. Run `37` if you want separated audio and HF audio-transformer embeddings.
4. Run `38` to connect audio runs to player-season projection, fusion, and method evaluation.
5. Run `39` again when you want valid-audio-only raw/enhanced baselines and pairwise audio-vs-baseline comparison.

No reusable logic should be added directly to notebooks. Put code in `src/sport_pipeline/`.
