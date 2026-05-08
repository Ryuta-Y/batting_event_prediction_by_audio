# contracts

This clean folder keeps only the contracts needed by the audio branch and downstream comparison.

Retained contracts:

- `audio/`: audio segment, feature, and audio prediction mapping contracts.
- `prediction/predictions_v1.md`: shared long-format prediction rows.
- `eval/target_registry_v1.md`: target registry semantics.
- `eval/metrics_v1.md`: metric output shape.

Important rules:

- Audio predictions use the same `predictions_v1` contract as context/video/fusion runs.
- `prediction_level=event` is valid for EV, LA, hard-hit, barrel, xBA, and xwOBA.
- OPS/OBP/SLG rows must be `prediction_level=player_season` or another aggregate level, never BBE-only event-level rows.
- Missing xBA/xwOBA labels must remain missing; do not fill them with zero.
