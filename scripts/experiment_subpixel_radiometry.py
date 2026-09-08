#!/usr/bin/env python3
"""Why confirmation must be flown low: sub-pixel radiometry experiment.

A radiometric LWIR camera does not report a target's temperature.  It reports
the *area-weighted average* of everything inside a pixel, convolved with the
sensor point-spread function.  When a target is smaller than a pixel that
average is dominated by the background, so the apparent temperature collapses
toward the ground temperature - and the collapse is what destroys the single
most powerful discriminator we have, the human thermal band (23-40.5 C).

This script measures the collapse directly.  It renders the same set of targets
- survivors in five categories, plus decoys whose whole purpose is to look warm
- at a ladder of altitudes, and reports for each:

* the true apparent temperature the world model assigned;
* the peak temperature the detector actually measures in its aperture;
* the measured core area and the TTP probability it implies;
* whether the temperature band still separates the target from its decoy.

The result is the quantitative justification for the two-pass search strategy
(``docs/05_SEARCH_THEORY.md``): a broad high-altitude pass is allowed to produce
candidates because it cannot produce confirmations, and a low narrow-FOV pass
does the confirming.

Usage
-----
    python scripts/experiment_subpixel_radiometry.py
    python scripts/experiment_subpixel_radiometry.py --json artifacts/subpixel.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sar.perception.detector import (          # noqa: E402
    ThermalAnomalyDetector, aperture_radius_px, local_background,
    measure_aperture, ttp_probability)
from sar.sim.renderer import CameraRenderer    # noqa: E402
from sar.sim.scenario import build_reference_scenario  # noqa: E402
from scripts.eval_detector import state_over   # noqa: E402


def run(args) -> Dict[str, Any]:
    world, lwir_spec, rgb_spec = build_reference_scenario(
        args.scenario, seed=args.seed, smoke=args.smoke)
    cam = CameraRenderer(world, lwir_spec)
    rows: List[Dict[str, Any]] = []

    targets: List[Dict[str, Any]] = []
    for v in world.victims:
        targets.append({"oid": v.vid, "kind": f"victim/{v.category}",
                        "north": v.north, "east": v.east,
                        "true_temp_c": v.body_temp_c, "is_person": True})
    for d in world.distractors:
        targets.append({"oid": d.did, "kind": f"decoy/{d.kind}",
                        "north": d.north, "east": d.east,
                        "true_temp_c": d.temp_c,
                        "is_person": d.kind in ("animal", "mannequin")})

    for agl in args.altitudes:
        for tg in targets:
            st = state_over(world, tg["north"], tg["east"], agl)
            f = cam.render(st, agl / 100.0)
            tr = next((t for t in f.truth if t.oid == tg["oid"]), None)
            if tr is None:
                continue
            img = np.asarray(f.image, dtype=np.float32)
            bg = local_background(img, 21, 4)
            diff = img - bg
            radius = aperture_radius_px(f.gsd_m)
            m = measure_aperture(img, diff, tr.pixel[0], tr.pixel[1], radius)
            lo, hi = ThermalAnomalyDetector.HUMAN_BAND
            in_band = lo <= m["peak"] <= hi
            rows.append({
                "agl_m": agl, "gsd_m_px": round(float(f.gsd_m), 4),
                "oid": tg["oid"], "kind": tg["kind"], "is_person": tg["is_person"],
                "source_temp_c": round(float(tg["true_temp_c"]), 2),
                "true_apparent_temp_c": round(float(tr.apparent_temp_c or 0.0), 2),
                "measured_peak_c": round(float(m["peak"]), 2),
                "measured_mean_c": round(float(m["mean"]), 2),
                "background_c": round(float(m["background"]), 2),
                "radiometric_error_k": round(float(m["peak"] - (tr.apparent_temp_c or 0.0)), 2),
                "core_area_px": round(float(m["area"]), 1),
                "expected_body_area_px": round(math.pi * radius * radius, 1),
                "area_fill": round(float(m["area"]) / max(math.pi * radius * radius, 1e-6), 3),
                "ttp": round(ttp_probability(math.sqrt(max(m["area"], 0.0))), 3),
                "ttp_truth": round(tr.ttp_detection(), 3),
                "detectability_truth": round(tr.detectability, 3),
                "in_human_band": bool(in_band),
                "extent_ratio": round(float(m.get("extent_ratio", 0.0)), 2),
            })

    # ---- summary: at what altitude does the band stop separating? -----------
    summary: Dict[str, Any] = {"altitudes": args.altitudes, "by_altitude": {}}
    for agl in args.altitudes:
        sub = [r for r in rows if r["agl_m"] == agl]
        people = [r for r in sub if r["is_person"]]
        hot_decoys = [r for r in sub if not r["is_person"] and r["source_temp_c"] > 40.0]
        summary["by_altitude"][f"{agl:.0f}m"] = {
            "gsd_m_px": sub[0]["gsd_m_px"] if sub else None,
            "survivors_in_band": sum(r["in_human_band"] for r in people),
            "survivors": len(people),
            "hot_decoys_in_band": sum(r["in_human_band"] for r in hot_decoys),
            "hot_decoys": len(hot_decoys),
            "mean_radiometric_error_k_survivors": (
                round(float(np.mean([r["radiometric_error_k"] for r in people])), 2)
                if people else None),
            "mean_ttp_survivors": (round(float(np.mean([r["ttp"] for r in people])), 3)
                                   if people else None),
            "mean_extent_ratio_hot_decoys": (
                round(float(np.mean([r["extent_ratio"] for r in hot_decoys])), 2)
                if hot_decoys else None),
        }
    return {"scenario": args.scenario, "seed": args.seed,
            "human_band_c": list(ThermalAnomalyDetector.HUMAN_BAND),
            "summary": summary, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="flood")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--altitudes", type=float, nargs="*",
                    default=[25.0, 35.0, 50.0, 70.0, 90.0, 110.0, 140.0])
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    rep = run(args)
    print(f"\n=== sub-pixel radiometry, {rep['scenario']} "
          f"(human band {rep['human_band_c'][0]}-{rep['human_band_c'][1]} C) ===\n")
    hdr = (f"{'AGL':>5}{'GSD':>8}{'survivors in band':>20}"
           f"{'hot decoys in band':>21}{'dT err':>9}{'TTP':>7}{'extent':>8}")
    print(hdr)
    print("-" * len(hdr))
    for alt, s in rep["summary"]["by_altitude"].items():
        print(f"{alt:>5}{s['gsd_m_px']:>8.3f}"
              f"{str(s['survivors_in_band']) + '/' + str(s['survivors']):>20}"
              f"{str(s['hot_decoys_in_band']) + '/' + str(s['hot_decoys']):>21}"
              f"{str(s['mean_radiometric_error_k_survivors']):>9}"
              f"{str(s['mean_ttp_survivors']):>7}"
              f"{str(s['mean_extent_ratio_hot_decoys']):>8}")

    print("\nPer-target apparent temperature vs altitude (C):")
    kinds = sorted({(r["kind"], r["oid"]) for r in rep["rows"]})
    alts = rep["summary"]["altitudes"]
    print(f"{'target':<24}" + "".join(f"{a:>8.0f}m" for a in alts) + "   true")
    for kind, oid in kinds:
        line = f"{kind + '/' + oid:<24}"
        true_t = None
        for a in alts:
            r = next((x for x in rep["rows"] if x["oid"] == oid and x["agl_m"] == a), None)
            if r is None:
                line += f"{'-':>9}"
            else:
                line += f"{r['measured_peak_c']:>9.1f}"
                true_t = r["true_apparent_temp_c"]
        print(line + f"   {true_t if true_t is not None else '-'}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=2, default=str))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
