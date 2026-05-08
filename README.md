# batting_codex_handoff_with_audio

MLB batting clips から Statcast-style batting outcomes を予測する研究パイプラインのうち、打球音 audio branch に絞った GitHub release candidate です。

このフォルダは Google Colab + Google Drive で動かす前提です。大きな動画、Drive artifact、モデル重み、生成済み prediction/report は含めません。

## What This Contains

中心 notebook は次の 4 本です。

| order | notebook | purpose |
|---:|---|---|
| 36 | `notebooks/36_cpu_audio_impact_baseline.ipynb` | contact-centered audio window から raw/enhanced audio-only predictions と report を作る |
| 37 | `notebooks/37_gpu_audio_separation_and_embeddings.ipynb` | separated audio branch と Hugging Face audio embedding branch を作る |
| 38 | `notebooks/38_cpu_audio_fusion_compare.ipynb` | audio predictions を player-season projection / audio-aware fusion / method evaluation に接続する |
| 39 | `notebooks/39_cpu_audio_audit_baseline_compare.ipynb` | audio stream と contact-window を監査し、valid-audio baseline と pairwise baseline compare を作る |

`39` は診断用ですが重要です。音声が無い clip や空の impact window が多い場合は、`39` で `audio_valid_clips_v1.parquet` を作ってから `36` を回すと、空音声に引っ張られにくくなります。

## Required Drive Artifacts

この release candidate は upstream の full batting pipeline を再生成するものではありません。Colab の Drive root に、少なくとも次が存在している必要があります。

```text
/content/drive/MyDrive/baseball_vision/
  manifests/bbe_events_v1.parquet
  clips/mlb_2024_2026_full_v2/clips_v1.parquet
  predictions/context_catboost_mlb_2024_2026_v2/predictions_v1.parquet
  predictions/video_lightweight_cv2_mlb_2024_2026_v2/predictions_v1.parquet
  predictions/fusion_mlb_2024_2026_v2/predictions_v1.parquet
```

`37` と `38` は、追加で audio branch の prediction artifacts を読みます。足りない artifact は notebook 冒頭で `MISSING` と表示されます。

## Fixed Colab Paths

```text
Colab code root: /content/drive/MyDrive/codex/batting_codex_handoff_with_audio
Drive artifact root: /content/drive/MyDrive/baseball_vision
Colab cache: /content/cache/baseball_vision
Run profile: configs/runs/mlb_2024_2026_real_colab_v2.json
Python package: src/sport_pipeline/
```

別の場所に置く場合は、notebook の最初のセルより前に次を設定してください。

```python
%env BATTING_CODE_ROOT=/content/drive/MyDrive/codex/batting_codex_handoff_with_audio
%env BASEBALL_VISION_RUN_PROFILE=mlb_2024_2026_real_colab_v2.json
```

## Repository Layout

```text
configs/       audio/fusion/run profile/target registry configs
contracts/     audio, prediction, target registry, metrics contracts
docs/          audio branch design and selected project constraints
notebooks/     Colab entrypoints 36-39 only
src/           reusable sport_pipeline Python modules
tests/         lightweight local tests for audio/eval/fusion/projection behavior
```

## Local Checks

Small unit checks can run locally without large videos or model downloads:

```bash
PYTHONPATH=src python -m unittest tests.unit.test_audio_impact_pipeline
PYTHONPATH=src python -m unittest tests.unit.test_evaluation_contracts
PYTHONPATH=src python -m unittest tests.unit.test_event_player_season_projection
PYTHONPATH=src python -m unittest tests.unit.test_fusion_contracts
```

Do not run heavy training, large video processing, or large model downloads locally. Those belong in Colab.
