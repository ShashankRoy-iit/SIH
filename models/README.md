# Model weights live here, and are not in git

Nothing in this directory is committed except this file. Weights are large,
licensed under terms that are not ours to redistribute, and reproducible:

```bash
python3 scripts/fetch_models.py --list          # what the registry expects
python3 scripts/fetch_models.py --model yolo11n-rgb-coco
python3 scripts/fetch_models.py --synthetic     # tiny CI plumbing model
python3 scripts/fetch_models.py --verify-all    # checksums
```

`sar/ai/registry.py` is the source of truth for file names, input sizes, class
maps and expected checksums. `registry.json` (if present here) adds or
overrides entries at runtime, so a model that post-dates the code can be flown
without a code change.

**An empty directory is a supported configuration.** With no weights the
audited heuristic detector runs, every test still passes, and the sortie report
records which stack produced its numbers. See `docs/06_AI_MODELS_AND_DATASETS.md`.
