# configs

This clean folder keeps the configuration needed by notebooks 36-39.

Retained groups:

- `runs/`: Colab run profiles, with `mlb_2024_2026_real_colab_v2.json` as the default for audio work.
- `targets/`: `target_registry_v1.yaml` for EV / LA / hard-hit / barrel / optional xBA / optional xwOBA / player-season OPS-style targets.
- `models/audio/`: audio impact and HF audio embedding settings.
- `models/fusion/`: late-fusion scope weights.
- `audio/`: audio contract-level configuration.

Do not store secrets, Drive artifacts, videos, model weights, generated predictions, or reports in this directory.
