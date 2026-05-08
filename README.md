# batting_codex_handoff_with_audio

MLB batting clips から Statcast-style batting outcomes を予測するパイプラインのうち、打球音 audio branch に絞ったものです

## What This Contains

中心 notebook は次の 4 本です。

| order | notebook | purpose |
|---:|---|---|
| 36 | `notebooks/36_cpu_audio_impact_baseline.ipynb` | contact-centered audio window から raw/enhanced audio-only predictions と report を作る |
| 37 | `notebooks/37_gpu_audio_separation_and_embeddings.ipynb` | separated audio branch と Hugging Face audio embedding branch を作る |
| 38 | `notebooks/38_cpu_audio_fusion_compare.ipynb` | audio predictions を player-season projection / audio-aware fusion / method evaluation に接続する |
| 39 | `notebooks/39_cpu_audio_audit_baseline_compare.ipynb` | audio stream と contact-window を監査し、valid-audio baseline と pairwise baseline compare を作る |

`39` は診断用ですが重要です。音声が無い clip や空の impact window が多い場合は、`39` で `audio_valid_clips_v1.parquet` を作ってから `36` を回すと、空音声に引っ張られにくくなります。

## Repository Layout

```text
configs/       audio/fusion/run profile/target registry configs
contracts/     audio, prediction, target registry, metrics contracts
docs/          audio branch design and selected project constraints
notebooks/     Colab entrypoints 36-39 only
src/           reusable sport_pipeline Python modules
tests/         lightweight local tests for audio/eval/fusion/projection behavior
```
