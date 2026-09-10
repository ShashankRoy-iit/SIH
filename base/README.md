# Base — Ready-to-Fly AI + Drone Stack

This folder contains the **base version**: a complete, ready-to-use AI drone stack
that only requires connecting your drone to work.

## What's Inside

- `sar/perception/yolo_detector.py` — Trained YOLOv8n AI model (ONNX) ready for thermal + RGB.
- `models/yolov8n_person_thermal.onnx` — Pre-trained model file (ready, no training needed).
- `sar/perception/thermal_simulation.py` — Thermal camera simulation / real feed handler.
- `sar/perception/pipeline.py` — Full perception pipeline: AI + thermal + geo-tagging.
- `scripts/run_drone_ai.py` — Main script: connect drone, run AI, serve dashboard.
- `rescue_dashboard/app.py` — Rescue command centre dashboard (zero internet).
- `configs/ardupilot_base_params.parm` — PX4 / ArduPilot parameter set.
- `deploy/` — Installation scripts, systemd service, rules.

## Quick Start

```bash
# Ensure model exists
ls models/yolov8n_person_thermal.onnx

# Run AI (simulated thermal feed for testing)
python3 scripts/run_drone_ai.py --model models/yolov8n_person_thermal.onnx --live --dashboard

# With real thermal camera (update source in script)
# python3 scripts/run_drone_ai.py --model models/yolov8n_person_thermal.onnx --live

# Access dashboard
open http://localhost:8088
```

## Connect Your Drone

1. Flash PX4 / ArduPilot with `configs/ardupilot_base_params.parm`.
2. Connect MAVLink via USB or telemetry radio.
3. Run the onboard script on the companion computer:
   ```bash
   sudo ./deploy/install_companion.sh
   sudo systemctl start sar-onboard.service
   ```
4. The AI will start identifying people from thermal imagery and geo-tag them.

## Dependencies

See root `requirements-base.txt` (or `requirements.txt`).
Key packages: `numpy`, `opencv-python`, `onnxruntime`, `flask`, `pymavlink`.
