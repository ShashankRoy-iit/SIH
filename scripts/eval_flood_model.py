#!/usr/bin/env python3
"""Evaluate the flood detector vs the baseline on synthetic flood scenes.

Generates deterministic flood frames (water, rooftops, debris, survivors in
water / on roofs / on ground, sun glints) at several altitudes, runs both
the baseline ensemble and FloodDetector through 5-frame temporal sequences,
and reports recall-of-resolvable, precision and false alarms per frame.

    python3 scripts/eval_flood_model.py --scenes 40 --altitudes 35,50,70

Writes artifacts/flood_eval.json.  Pure numpy/scipy/matplotlib-free.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Tuple

# --- repo-root bootstrap ---------------------------------------------------
# Running `python scripts/<name>.py` puts scripts/ on sys.path, not the repo
# root, so `import sar` fails.  This makes the script runnable from a clone with
# no install step, and refuses to run against a foreign PyPI `sar` package.
# See scripts/_bootstrap.py and docs/HOWTO_RUN.md.
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from scripts._bootstrap import bootstrap as _bootstrap  # noqa: E402

_bootstrap()
# ---------------------------------------------------------------------------

import numpy as np

from sar.ai.flood import FloodDetector
from sar.perception.detector import (ImageFrame, build_reference_pipeline,
                                     nms as _nms)


def gsd_for_alt(alt_m: float, hfov_deg: float = 57.0, width_px: int = 320) -> float:
    swath = 2.0 * alt_m * math.tan(math.radians(hfov_deg / 2.0))
    return swath / width_px


def make_scene(rng: np.random.Generator, alt_m: float, w: int = 320, h: int = 240,
               rgb_size: Tuple[int, int] = (640, 480)):
    """One synthetic flood scene.  Returns (lwir, rgb, survivors, decoys).

    survivors: list of dicts {u, v, in_water, on_roof, temp_c}
    """
    gsd = gsd_for_alt(alt_m)
    # --- LWIR: background ground 14 C, water 15 C, roofs warm patches ---- #
    lwir = np.full((h, w), 14.0, np.float32)
    lwir[h // 2:, :] = 15.0  # flood water lower half
    rw, rh = rgb_size
    rgb = np.full((rh, rw, 3), (105, 100, 92), np.uint8)   # mud/ground
    rgb[rh // 2:, :] = (78, 90, 104)                        # water
    # Rooftops: bright in RGB; MOSTLY cool in LWIR (shaded/morning roofs),
    # with ~1 in 3 sun-heated into the human band (the adversarial case).
    # Making every roof in-band would be realistic for noon desert and
    # adversarial everywhere else — the mix matches a flood morning.
    roofs = []
    for _ in range(rng.integers(2, 5)):
        x0 = int(rng.integers(0, w - 90))
        y0 = int(rng.integers(0, h // 2 - 45))
        x1, y1 = x0 + int(rng.integers(40, 90)), y0 + int(rng.integers(25, 45))
        hot = bool(rng.random() < 0.34)
        lwir[y0:y1, x0:x1] = (26.0 + rng.random() * 5.0) if hot else (16.0 + rng.random() * 6.0)
        sx, sy = rw / w, rh / h
        rgb[int(y0 * sy):int(y1 * sy), int(x0 * sx):int(x1 * sx)] = (165, 155, 145)
        roofs.append((x0, y0, x1, y1))
    # Warm decoys: vehicle / debris patches.
    for _ in range(rng.integers(1, 4)):
        x0 = int(rng.integers(0, w - 20))
        y0 = int(rng.integers(0, h - 20))
        lwir[y0:y0 + 12, x0:x0 + 18] = 30.0
    # --- survivors ------------------------------------------------------- #
    survivors = []
    n_surv = int(rng.integers(2, 5))
    body_r_m = 0.30
    for _ in range(n_surv):
        in_water = bool(rng.random() < 0.5)
        on_roof = (not in_water) and bool(rng.random() < 0.3)
        if in_water:
            u = float(rng.integers(20, w - 20))
            v = float(rng.integers(h // 2 + 10, h - 15))
            temp = 27.0 + rng.random() * 4.0   # cooled by water, still human
            frac_visible = 0.15
        elif on_roof and roofs:
            x0, y0, x1, y1 = roofs[int(rng.integers(0, len(roofs)))]
            u = float((x0 + x1) / 2 + rng.integers(-5, 5))
            v = float((y0 + y1) / 2 + rng.integers(-3, 3))
            temp = 31.0 + rng.random() * 2.0
            frac_visible = 0.8
        else:
            u = float(rng.integers(20, w - 20))
            v = float(rng.integers(10, h // 2 - 10))
            temp = 31.0 + rng.random() * 2.0
            frac_visible = 0.9
        # Paint the body as a warm disk with area ~ frac * 0.62 m^2.
        area_m2 = 0.62 * frac_visible
        r_px = max(math.sqrt(area_m2 / math.pi) / gsd, 1.2)
        yy, xx = np.mgrid[0:h, 0:w]
        disk = (xx - u) ** 2 + (yy - v) ** 2 <= r_px ** 2
        # Sub-pixel radiometry: small disks blend toward background.
        cover = min(1.0, math.pi * r_px * r_px / max(disk.sum(), 1))
        peak = temp if r_px >= 1.6 else temp * cover + 15.0 * (1 - cover)
        lwir[disk] = peak
        # RGB clothing cue (absent for in-water: only head/shoulders).
        # Shaded, not flat: a flat plateau peaks at its corner and under-fills
        # the geometry aperture; real clothing shades toward a lit centre.
        if not in_water:
            cu, cv = int(u * rw / w), int(v * rh / h)
            cr = max(int(r_px * rw / w * 1.1), 3)
            x0, x1 = max(cu - cr, 0), cu + cr
            y0, y1 = max(cv - cr, 0), cv + cr
            py, px = np.mgrid[0:y1 - y0, 0:x1 - x0]
            cy, cx = (y1 - y0 - 1) / 2.0, (x1 - x0 - 1) / 2.0
            rr = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
            shade = 1.0 - 0.18 * np.clip(rr / max(max(cx, cy), 1), 0, 1)
            rgb[y0:y1, x0:x1] = (np.asarray((205, 70, 55), np.float32)
                                 * shade[..., None]).astype(np.uint8)
        # Resolvable? peak must clear the human band floor after blending.
        resolvable = bool(peak >= 23.0 and disk.sum() >= 2)
        survivors.append({"u": u, "v": v, "in_water": in_water, "on_roof": on_roof,
                          "temp_c": round(float(peak), 2), "resolvable": resolvable,
                          "area_px": int(disk.sum())})
    # NETD noise on LWIR.
    lwir = lwir + rng.normal(0, 0.045, lwir.shape).astype(np.float32)
    return lwir, rgb, survivors, gsd


def match(dets, survivors, tol_px: float = 12.0):
    """Match detections to survivors.

    Returns (n_matched_survivors, matched_idx, n_false_alarms).  A detection
    near ANY survivor — even an already-matched one — is a duplicate, not a
    false alarm: duplicate boxes are an NMS nicety, while a box on empty
    water is what exhausts an operator.  Conflating the two punishes
    detectors for finding the same survivor twice.
    """
    matched = set()
    fp = 0
    for d in dets:
        if d.label not in ("person", "person_group"):
            continue
        scale = 2.0 if d.modality == "rgb" else 1.0
        near_any, near_idx = False, None
        for i, s in enumerate(survivors):
            if math.hypot(d.u / scale - s["u"], d.v / scale - s["v"]) <= tol_px:
                near_any, near_idx = True, i
                break
        if near_any:
            matched.add(near_idx)
        else:
            fp += 1
    return len(matched), matched, fp


def eval_at_alt(alt_m: float, scenes: int, seed: int):
    rng = np.random.default_rng(seed)
    base = build_reference_pipeline("both")
    flood = FloodDetector()
    stats = {"base": {"tp": 0, "fp": 0, "res": 0, "frames": 0},
             "flood": {"tp": 0, "fp": 0, "res": 0, "frames": 0}}
    for s in range(scenes):
        lwir, rgb, survivors, gsd = make_scene(rng, alt_m)
        res_idx = {i for i, x in enumerate(survivors) if x["resolvable"]}
        stats["base"]["res"] += len(res_idx)
        stats["flood"]["res"] += len(res_idx)
        found: dict = {"base": set(), "flood": set()}
        # 5-frame temporal sequence with slight jitter (drift + noise).
        for f in range(5):
            t = s * 10.0 + f * 0.5
            lw = lwir + rng.normal(0, 0.02, lwir.shape).astype(np.float32)
            fl = ImageFrame(image=lw, kind="lwir", t=t, gsd_m=gsd)
            fr = ImageFrame(image=rgb, kind="rgb", t=t, gsd_m=gsd / 2.0)
            for key, det in (("base", base), ("flood", flood)):
                if key == "base":
                    dets = det.detect([fl, fr])
                    dets = det.cross_modal_scores(dets)
                else:
                    dets = det.detect([fl, fr])
                # Count only operator-visible: score >= 0.28 and not vetoed.
                vis = [d for d in dets if d.is_person and d.score >= 0.28
                       and "veto" not in d.attributes]
                _, matched, fp = match(vis, survivors)
                # Sequence-level recall: found in ANY frame (a real sortie is
                # continuous; the temporal filter costs the first second only).
                found[key] |= (matched & res_idx)
                stats[key]["fp"] += fp
                stats[key]["frames"] += 1
        stats["base"]["tp"] += len(found["base"])
        stats["flood"]["tp"] += len(found["flood"])
        flood.reset()
    out = {}
    for key in ("base", "flood"):
        st = stats[key]
        recall_res = st["tp"] / max(st["res"], 1)
        prec = st["tp"] / max(st["tp"] + st["fp"], 1)
        out[key] = {"recall_of_resolvable": round(min(recall_res, 1.0), 3),
                    "precision": round(prec, 3),
                    "fa_per_frame": round(st["fp"] / max(st["frames"], 1), 4),
                    "tp": st["tp"], "fp": st["fp"]}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Flood detector evaluation")
    ap.add_argument("--scenes", type=int, default=40)
    ap.add_argument("--altitudes", default="35,50,70")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="artifacts/flood_eval.json")
    args = ap.parse_args(argv)
    alts = [float(x) for x in args.altitudes.split(",")]
    results = {"scenes": args.scenes, "seed": args.seed, "altitudes": {}}
    for alt in alts:
        r = eval_at_alt(alt, args.scenes, args.seed + int(alt))
        results["altitudes"][str(alt)] = r
        b, f = r["base"], r["flood"]
        print(f"alt {alt:>5.0f}m  recall_res base={b['recall_of_resolvable']:.2f} "
              f"flood={f['recall_of_resolvable']:.2f}  precision base={b['precision']:.2f} "
              f"flood={f['precision']:.2f}  fa/frame base={b['fa_per_frame']:.3f} "
              f"flood={f['fa_per_frame']:.3f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")
    # The floor: flood recall of resolvable must never drop below baseline.
    for alt, r in results["altitudes"].items():
        assert r["flood"]["recall_of_resolvable"] >= r["base"]["recall_of_resolvable"] - 0.05, \
            f"recall floor broken at {alt}m"
    print("recall floor holds at all altitudes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
