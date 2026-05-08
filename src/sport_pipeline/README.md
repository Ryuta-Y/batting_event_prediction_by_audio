# sport_pipeline

Reusable Python code for the audio-focused batting Statcast prediction release.

Kept modules are intentionally narrow:

- `audio/`: audio audit, impact feature extraction, separated-audio ablation, HF audio embeddings, audio reports
- `evaluation/`: target registry, prediction validation, metrics
- `io/`: small table/json helpers and Colab cache helpers
- `models/context/`: lightweight context baseline interface used by retained contract tests
- `models/fusion/`: prediction late-fusion helpers and runner
- `models/player_season/`: event prediction to player-season projection
- `models/video/`: minimal shared target-head and prediction-row helpers reused by audio code
- `pipeline/run_profile.py`: run profile helpers for v2 Colab profiles
- `reports/`: audio baseline comparison, method evaluation, and small HTML helpers
- `schemas/`: compact manifest schema validation helpers

Large CV, video, VLM, Statcast download, and full-pipeline runner modules were removed from this clean copy. They remain outside the audio release scope.
