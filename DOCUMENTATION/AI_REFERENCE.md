# AI Models & Datasets Reference

## Model: YOLOv8n Person Detection

- **Architecture**: YOLOv8n (nano) — 3.2M parameters.
- **Training Data**: Synthetic thermal + RGB pairs; references: HERIDAL, SARD, TinyPerson.
- **Format**: ONNX (included as `.onnx`) and exportable to TFLite.
- **Input Size**: 320×320 (can be scaled to 640 with export).
- **Performance**: <30 ms per frame on modern laptop (CPU).
- **Calibration**: Recall of resolvable persons = 1.00 at 35m, 50m, 70m (see `base/artifacts/`).

## Datasets Referenced

| Dataset | Images | Notes |
|---|---|---|
| HERIDAL | 68,750 RGB | Best published mAP 95.11% |
| SARD | — | SAR-specific |
| TinyPerson | — | Small aerial persons |
| AFO | — | Thermal + RGB pairs |
| xBD | 850,736 | Building annotations for hazard detection |

## Cross-Modal Fusion

The AI pipeline uses a physics-gated veto rather than pure learned fusion:
- RGB confirms thermal only when radiometry allows it.
- Night preset suppresses the veto conditionally.
- Asymmetric penalties: thermal-only vs RGB-only have different failure modes.
