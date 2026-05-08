# tests

Lightweight local tests retained for the clean audio branch:

- `tests/unit/test_audio_impact_pipeline.py`
- `tests/unit/test_evaluation_contracts.py`
- `tests/unit/test_event_player_season_projection.py`
- `tests/unit/test_fusion_contracts.py`

Run examples:

```bash
PYTHONPATH=src python -m unittest tests.unit.test_audio_impact_pipeline
PYTHONPATH=src python -m unittest tests.unit.test_evaluation_contracts
PYTHONPATH=src python -m unittest tests.unit.test_event_player_season_projection
PYTHONPATH=src python -m unittest tests.unit.test_fusion_contracts
```

Heavy Colab workflows, large video processing, model downloads, and full Drive artifacts are intentionally outside local tests.

`test_audio_impact_pipeline.py` needs `numpy` for waveform feature extraction. In minimal local environments without `numpy`, that test class is skipped; install `requirements_audio_colab.txt` to run it fully.
