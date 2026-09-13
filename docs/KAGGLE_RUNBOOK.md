# Kaggle runbook — run the whole SAHYOG project (incl. GPU training)

Kaggle Notebooks is the free-GPU complement to Codespaces. It runs **every
script in this repo**, and it is the place the full YOLO11n-P2 fine-tune on
real SAR data actually completes. This file is the exact sequence.

**Verdict up front**

| Capability | Kaggle? | Note |
|---|---|---|
| `doctor.py`, `pytest`, mission/rescue/eval/bench scripts | ✅ | identical to any Linux box |
| CPU training (sandbox path, `datasets/train_all.py`) | ✅ | works; but use GPU instead |
| **GPU fine-tune** (`--train --p2`, 640×512, 120 epochs) | ✅ | the whole point — free P100/T4 |
| ONNX + TFLite-INT8 + QNN export, registry, verification | ✅ | all script-driven |
| Phone RGB bridge bench (`phone_onboard.py --mode bench`) | ✅ | runs headless |
| Interactive dashboards (`dashboard.py`, `serve_replay.py`) | ⚠️ | run & curl-able, but **not viewable** in a browser (no port forward) |
| ArduPilot SITL / Gazebo with a window | ⚠️ | headless only; MiniSITL (sockets) works |

**Current limits (2026, free tier):** P100 **or** T4 (16 GB VRAM), sometimes
2×T4; ~**30 GPU-hours/week**; **12 h/session** (9 h TPU); ~**20 GB** working
disk (wiped when the session ends — download what you need); **internet is OFF
by default**; auto-saves every 5 min; no credit card.

---

## 1. Create the notebook + switch the two settings on

1. kaggle.com → **Create** → **New Notebook**.
2. **Settings (right panel):**
   - **Accelerator → GPU T4 x2** (or GPU P100). ← this is the whole reason we're here
   - **Internet → On**. ← required for `pip` and `git clone`
3. (Optional) Set **Notebook options → Persistence → "Output files only"**
   so your trained weights land in `/kaggle/working` and can be downloaded.

## 2. Clone the repo + install (first cell)

```python
import os, sys
os.chdir("/kaggle/working")
!git clone https://github.com/ShashankRoy-iit/SIH.git
os.chdir("/kaggle/working/SIH")
!pip install -q -e '.[sim,dev]' onnx onnxruntime
!pip install -q ultralytics        # torch/torchvision come with the Kaggle image

import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
```

Expect `cuda True` and `NVIDIA T4 / P100`.

## 3. Run the whole project (one cell each — nothing is GUI-dependent)

```bash
!python scripts/doctor.py                                       # env check: all PASS
!python -m pytest tests/ -q                                    # full suite: 158 passed, 3 skipped
!python scripts/run_rescue_simulation.py --scenario flood --duration 460
!python scripts/run_mission.py --scenario flood --duration 460 --live
!python scripts/eval_detector.py --mode both --json artifacts/detector_eval_neural.json
!python mobile/phone_onboard.py --mode bench --duration 15      # phone-as-RGB bridge
```

### The dashboard caveat (the one thing that is not the same)
`dashboard.py` (FastAPI) and `serve_replay.py` bind a port and expect a browser.
Kaggle does not forward ports, so you cannot *see* them there. You can still
prove they work, in-notebook, with:

```python
import threading, time, urllib.request
from sar.gcs.dashboard import app            # the ASGI app, no server needed
# or: run uvicorn in a thread and curl it:
import uvicorn
t = threading.Thread(target=lambda: uvicorn.run(app, host="127.0.0.1", port=8000,
                                               log_level="error"), daemon=True)
t.start(); time.sleep(2)
print(urllib.request.urlopen("http://127.0.0.1:8000/health").read().decode())
```

For the real interactive view, download `artifacts/rescue_mission_flood.json`
and run `python scripts/serve_replay.py --artifact <file>` on your laptop.

## 4. The GPU fine-tune (the part Codespaces cannot do)

### 4a. Get the data — Kaggle hosts it, use "Add Input"

**Right panel → Data → Add Input → search**, then attach:

- **HIT-UAV** (2,898 aerial LWIR frames, person/vehicle) — there are public
  Kaggle mirrors ("HIT-UAV Infrared Thermal Dataset").
- **AIResQ** (9,788 high-res thermal frames, 17,550 person boxes) — best
  published aerial-thermal source; on **Zenodo** (records/17405323, request) —
  if no Kaggle mirror, `!wget` it from Zenodo (internet is on) or upload the
  zip as a private dataset once.
- **SARD** (search-and-rescue postures, RGB) — IEEE DataPort (needs a free
  account) → upload as a private Kaggle dataset.

Attached datasets appear read-only under `/kaggle/input/<dataset>/`.

### 4b. Convert to YOLO layout (HIT-UAV example)

HIT-UAV ships YOLO-style `class xc yc w h` (normalised) txt labels. Map its ids
(`0 Person, 1 Car, 2 Bicycle, 3 OtherVehicle, 4 DontCare`) onto this repo's
`CLASSES = person, person_group, vehicle, animal, fire, structure`, dropping
DontCare:

```python
import shutil, glob, os
from pathlib import Path
SRC = "/kaggle/input/hit-uav"          # adjust to the attached dataset path
DST = Path("datasets/hit-uav")
MAP = {0: 0, 1: 2, 2: 2, 3: 2}         # person->person(0); car/bicycle/other->vehicle(2); 4 dropped
for split in ("train", "test", "val"):
    (DST/"images"/split).mkdir(parents=True, exist_ok=True)
    (DST/"labels"/split).mkdir(parents=True, exist_ok=True)
    for im in glob.glob(f"{SRC}/images/{split}/*.jpg"):
        shutil.copy(im, DST/"images"/split)
        lab = Path(im.replace("/images/", "/labels/")).with_suffix(".txt")
        out = []
        for line in Path(lab).read_text().splitlines():
            p = line.split()
            if not p or int(p[0]) not in MAP:
                continue
            out.append(f"{MAP[int(p[0])]} " + " ".join(p[1:]))
        (DST/"labels"/split/(Path(im).stem+".txt")).write_text("\n".join(out))
```

Then point the repo's dataset helper at it (or write `data.yaml` by hand with
`nc: 6`, `names: [person, person_group, vehicle, animal, fire, structure]`):

```bash
!python scripts/train_detector.py --dataset-yaml hit-uav --out datasets/
```

### 4c. Train (GPU) — thermal first, then RGB

```bash
# Thermal (LWIR) — the deployment detector. --p2 resolves configs/models/yolo11n-p2.yaml
!python scripts/train_detector.py --train --data datasets/hit-uav/data.yaml \
    --model yolo11n.yaml --p2 --channels 1 --imgsz 640 --epochs 120 --batch 16 \
    --project /kaggle/working/runs --name thermal --device 0

# RGB (phone camera)
!python scripts/train_detector.py --train --data datasets/sim-rgb/data.yaml \
    --model yolo11n.yaml --imgsz 640 --epochs 120 --batch 16 \
    --project /kaggle/working/runs --name rgb --device 0
```

### 4d. Export + register + verify

```bash
!python scripts/export_model.py --weights /kaggle/working/runs/thermal/weights/best.pt \
    --out /kaggle/working/models/yolo11n-thermal-sar.onnx --imgsz 640 512 --channels 3
!python scripts/export_phone_model.py --recipe          # phone TFLite-INT8 (Hexagon DSP)
!python scripts/fetch_models.py --list                  # confirm registered
!SAR_DETECTOR=auto python scripts/eval_detector.py --mode both --json artifacts/detector_eval_neural.json
```

## 5. Keep the results (Kaggle wipes `/kaggle/working` at session end)

1. **Right panel → Output** — tick the files you want (best.pt, .onnx, JSONs),
   then **Download** (or "Save & Run All" to regenerate them).
2. Or pin them as a **Dataset version** ("Create Dataset → New Version") so the
   next session mounts them read-only from `/kaggle/input/`.
3. `git push` the results JSONs back to the repo (the token works in Kaggle too).

## 6. Weekly-quota tips

- 120 epochs on HIT-UAV @ T4 ≈ **1–3 h** — well inside one 12 h session and the
  30 h/week allowance.
- Long runs: `--patience 30` early-stops; `ultralytics` resumes from
  `last.pt` if a session dies (`resume=True`).
- Idle notebooks keep burning GPU time — **stop the session** when done.

---

**TL;DR:** one notebook, two settings (GPU + Internet on), clone, run the
cells. The GPU training that Codespaces can't do completes here; the only real
loss vs. a local machine is *viewing* the dashboards, which you download-and-
replay on your laptop instead.
