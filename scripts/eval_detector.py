#!/usr/bin/env python3
"""Detector calibration and evaluation harness.

Two measurement modes, because they answer different questions.

**Aimed mode** (``--mode aimed``) flies the camera so each survivor and each
decoy lands at a controlled pixel offset, at each altitude.  This isolates the
detector from the coverage planner: every target gets a fair, repeatable look, so
recall is measured *per survivor category* (water_edge, rooftop, dry_refuge,
structure, road) and the decoy false-alarm rate is measured per decoy kind.
Coverage-driven surveys would give a recall number confounded with "did the
aircraft happen to fly over them".

**Survey mode** (``--mode survey``) flies boustrophedon passes and measures what
only a survey can: false alarms *per square kilometre* of background clutter, and
sustained throughput.  A detector with 0.1 FP/frame at 6 Hz over 2 km^2 is a
different system from one with 0.1 FP/frame on aimed targets.

Both modes report LWIR-only and the cross-modal ensemble side by side, because
that difference is the measured value of fusion.  Both also report the
**TTP physics ceiling** - the probability an ideal sensor/observer could detect
each target at that geometry - so "the detector failed" can be told apart from
"the target was never resolvable".

Usage
-----
    python scripts/eval_detector.py --quick
    python scripts/eval_detector.py --mode survey --scenario earthquake
    python scripts/eval_detector.py --json artifacts/detector_eval.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sar.perception.fusion import CrossModalFuser  # noqa: E402
from sar.perception.detector import (           # noqa: E402
    Detection, build_reference_pipeline, rgb_frame_quality)
from sar.sim.renderer import CameraRenderer      # noqa: E402
from sar.sim.scenario import build_reference_scenario  # noqa: E402
from sar.vehicle.dynamics import PlantState      # noqa: E402


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def state_over(world, north: float, east: float, agl: float) -> PlantState:
    gz = float(np.nan_to_num(world.terrain_z(north, east)))
    return PlantState(pos=np.array([float(north), float(east), -(gz + agl)]),
                      vel=np.array([7.0, 0.0, 0.0]),
                      euler=np.zeros(3), omega=np.zeros(3), on_ground=False)


def survey_states(world, agl: float, spacing_m: float,
                  n_cross: int = 5) -> List[PlantState]:
    """Boustrophedon sample points covering the world at one AGL."""
    margin = 12.0
    norths = np.arange(margin, world.north_m - margin + 1e-6, spacing_m)
    easts = np.linspace(margin, world.east_m - margin, n_cross)
    out: List[PlantState] = []
    for i, north in enumerate(norths):
        for east in (easts if i % 2 == 0 else easts[::-1]):
            out.append(state_over(world, north, east, agl))
    return out


def match_radius(tr, gsd: float) -> float:
    """Association gate in pixels: generous for tiny targets, bounded for big."""
    return float(max(2.8, 0.62 * max(tr.width_px, tr.height_px) + 2.0))


def offset_position(world, tr_north: float, tr_east: float, agl: float,
                    frac: float, spec) -> Tuple[float, float]:
    """Aircraft position that puts the target at ``frac`` of the half-swath.

    ``frac=0`` centres it, ``frac=0.8`` puts it near the frame edge where GSD is
    worst and the pixel footprint smallest - the honest test of a real survey,
    in which almost nothing is ever centred.
    """
    half_swath = spec.swath_width_m(agl) / 2.0
    # Nadir point sits frac*half_swath south-west of the target.
    return tr_north - frac * half_swath * 0.707, tr_east - frac * half_swath * 0.707


# --------------------------------------------------------------------------- #
# Result accumulation
# --------------------------------------------------------------------------- #
class Stats:
    """One (mode, altitude) cell of results."""

    def __init__(self, agl: float, mode: str) -> None:
        self.agl = agl
        self.mode = mode
        self.frames = 0
        self.tp_scores: List[float] = []
        self.fp_scores: List[float] = []
        self.decoy_scores: List[float] = []
        self.bg_fp = 0
        self.decoy_fp = 0
        self.targets_seen: set = set()
        self.targets_resolvable: set = set()
        self.by_category: Counter = Counter()
        self.by_category_total: Counter = Counter()
        self.by_decoy: Counter = Counter()
        self.by_decoy_total: Counter = Counter()
        self.area_km2 = 0.0
        self.t_total = 0.0
        self.t_lwir = 0.0
        self.t_rgb = 0.0
        self.rgb_quality: List[float] = []
        self.label_hist: Counter = Counter()

    # ------------------------------------------------------------------ #
    def add_target(self, oid: str, category: str, det_found: bool,
                   detectability: float, score: Optional[float],
                   resolvable: bool = True) -> None:
        self.by_category_total[category] += 1
        if resolvable:
            self.targets_resolvable.add(oid)
        if det_found:
            self.targets_seen.add(oid)
            self.by_category[category] += 1
            if score is not None:
                self.tp_scores.append(score)

    def summary(self) -> Dict[str, Any]:
        n_cat = sum(self.by_category_total.values())
        resolvable = len(self.targets_resolvable)
        return {
            "agl_m": self.agl, "mode": self.mode, "frames": self.frames,
            "gsd_m_px": None,
            "targets": n_cat,
            "found": len(self.targets_seen),
            "recall": round(len(self.targets_seen) / n_cat, 3) if n_cat else None,
            "resolvable": resolvable,
            "recall_of_resolvable": (round(len(self.targets_seen & self.targets_resolvable)
                                           / resolvable, 3) if resolvable else None),
            "recall_by_category": dict(self.by_category),
            "targets_by_category": dict(self.by_category_total),
            "decoy_false_alarms": self.decoy_fp,
            "decoys_triggered": dict(self.by_decoy),
            "decoys_presented": dict(self.by_decoy_total),
            "background_false_alarms": self.bg_fp,
            "bg_fp_per_frame": round(self.bg_fp / max(self.frames, 1), 3),
            "bg_fp_per_km2": (round(self.bg_fp / self.area_km2, 1)
                              if self.area_km2 > 0 else None),
            "score_tp_mean": round(float(np.mean(self.tp_scores)), 3) if self.tp_scores else None,
            "score_tp_min": round(float(np.min(self.tp_scores)), 3) if self.tp_scores else None,
            "score_fp_mean": round(float(np.mean(self.fp_scores)), 3) if self.fp_scores else None,
            "score_fp_max": round(float(np.max(self.fp_scores)), 3) if self.fp_scores else None,
            "score_decoy_mean": (round(float(np.mean(self.decoy_scores)), 3)
                                 if self.decoy_scores else None),
            "ms_per_frame": round(1000 * self.t_total / max(self.frames, 1), 1),
            "ms_lwir": round(1000 * self.t_lwir / max(self.frames, 1), 1),
            "ms_rgb": round(1000 * self.t_rgb / max(self.frames, 1), 1),
            "rgb_quality_mean": (round(float(np.mean(self.rgb_quality)), 3)
                                 if self.rgb_quality else None),
            "label_histogram": dict(self.label_hist.most_common(12)),
        }


# --------------------------------------------------------------------------- #
# Detection run
# --------------------------------------------------------------------------- #
#: One fuser for the whole evaluation.  The boresight offset is a property of the
#: rig, not of the altitude, so letting the estimator accumulate samples across
#: every frame of the run is both physically right and the harder test: it has to
#: converge from a mix of geometries rather than from one convenient pass.
_FUSER = CrossModalFuser()


def run_detectors(lwir_only, ens, f_lwir, f_rgb, q: float, stats: Stats,
                  lwir_stats: Stats,
                  fuser: Optional[CrossModalFuser] = None,
                  ) -> Tuple[List[Detection], List[Detection]]:
    """Run LWIR-only and the fused cross-modal stack on one frame pair.

    The ensemble returns *per-modality* detections; ``CrossModalFuser`` does the
    pairing and emits the single fused observation list the rest of the system
    consumes.  Keeping those in separate layers is what lets the boresight
    estimator see its training pairs - an earlier version scored agreement inside
    the ensemble, which left the fuser with nothing to pair and the ensemble
    column numerically identical to LWIR-only.
    """
    fuser = fuser or _FUSER
    t0 = time.perf_counter()
    dets_lo = lwir_only.detect([f_lwir], rgb_quality=0.0)
    dt_lwir = time.perf_counter() - t0

    t0 = time.perf_counter()
    raw_en = ens.detect([f_lwir, f_rgb], rgb_quality=q)
    dets_en = [o.to_detection() for o in fuser.fuse(raw_en, f_lwir.t, rgb_quality=q)]
    dt_all = time.perf_counter() - t0

    stats.frames += 1
    stats.t_total += dt_all
    stats.t_lwir += dt_lwir
    stats.t_rgb += max(dt_all - dt_lwir, 0.0)
    stats.rgb_quality.append(q)
    for d in dets_en:
        stats.label_hist[d.label] += 1

    lwir_stats.frames += 1
    lwir_stats.t_total += dt_lwir
    lwir_stats.t_lwir += dt_lwir
    lwir_stats.rgb_quality.append(0.0)
    for d in dets_lo:
        lwir_stats.label_hist[d.label] += 1
    return dets_lo, dets_en


def associate(dets: List[Detection], truths, oid: str, gsd: float
              ) -> Tuple[bool, Optional[Detection]]:
    """Did any person detection land on target ``oid``?"""
    best: Optional[Detection] = None
    for tr in truths:
        if tr.oid != oid or tr.category != "victim" or not tr.visible:
            continue
        r = match_radius(tr, gsd)
        for d in dets:
            if not d.is_person:
                continue
            if math.hypot(tr.pixel[0] - d.u, tr.pixel[1] - d.v) <= r:
                if best is None or d.score > best.score:
                    best = d
    return best is not None, best


def count_other_person_dets(dets: List[Detection], truths, gsd: float
                            ) -> Tuple[int, int, float]:
    """(background FPs, decoy FPs, max decoy score) among person detections."""
    bg = 0
    decoy = 0
    best_decoy = 0.0
    for d in dets:
        if not d.is_person:
            continue
        hit_victim = hit_decoy = False
        for tr in truths:
            if not tr.visible:
                continue
            if math.hypot(tr.pixel[0] - d.u, tr.pixel[1] - d.v) > match_radius(tr, gsd):
                continue
            if tr.category == "victim":
                hit_victim = True
            elif tr.category == "distractor":
                hit_decoy = True
        if hit_victim:
            continue
        if hit_decoy:
            decoy += 1
            best_decoy = max(best_decoy, d.score)
        else:
            bg += 1
    return bg, decoy, best_decoy


# --------------------------------------------------------------------------- #
# Aimed evaluation
# --------------------------------------------------------------------------- #
def eval_aimed(args, world, lwir_spec, rgb_spec, lwir_cam, rgb_cam,
               lwir_only, ens) -> Dict[str, Any]:
    out: Dict[str, Any] = {"lwir_only": {}, "ensemble": {}}
    offsets = [0.0, 0.45, 0.78] if not args.quick else [0.0, 0.62]

    for agl in args.altitudes:
        gsd = lwir_spec.gsd_at(agl)
        s_lo = Stats(agl, "lwir_only")
        s_en = Stats(agl, "ensemble")

        targets: List[Tuple[str, str, float, float, float, str]] = []
        for v in world.victims:
            targets.append((v.vid, "victim", v.north, v.east, v.elevation, v.category))
        for d in world.distractors:
            targets.append((d.did, "distractor", d.north, d.east, d.elevation, d.kind))

        for (oid, kind, north, east, elev, cat) in targets:
            for frac in offsets:
                an, ae = offset_position(world, north, east, agl, frac, lwir_spec)
                if not (0 <= an <= world.north_m and 0 <= ae <= world.east_m):
                    continue
                gz = float(np.nan_to_num(world.terrain_z(an, ae)))
                # Hold AGL above the *target*, so the target range is agl - elev.
                st = PlantState(pos=np.array([an, ae, -(gz + agl)]),
                                vel=np.array([7.0, 0.0, 0.0]),
                                euler=np.zeros(3), omega=np.zeros(3), on_ground=False)
                t = s_en.frames / 8.0
                f_lwir = lwir_cam.render(st, t)
                f_rgb = rgb_cam.render(st, t)
                q = rgb_frame_quality(f_rgb)
                truths = f_lwir.truth

                tr_self = next((tr for tr in truths if tr.oid == oid), None)
                detectability = tr_self.detectability if tr_self else 0.0
                resolvable = detectability >= args.resolvable_threshold

                dets_lo, dets_en = run_detectors(lwir_only, ens, f_lwir, f_rgb, q,
                                                 s_en, s_lo)

                if kind == "victim":
                    hit_lo, d_lo = associate(dets_lo, truths, oid, gsd)
                    hit_en, d_en = associate(dets_en, truths, oid, gsd)
                    s_lo.add_target(oid, cat, hit_lo, detectability,
                                    d_lo.score if d_lo else None, resolvable)
                    s_en.add_target(oid, cat, hit_en, detectability,
                                    d_en.score if d_en else None, resolvable)
                    for st_, dets in ((s_lo, dets_lo), (s_en, dets_en)):
                        bg, dec, bdec = count_other_person_dets(dets, truths, gsd)
                        st_.bg_fp += bg
                        st_.decoy_fp += dec
                        st_.fp_scores.extend([bdec] if dec else [])
                else:
                    s_lo.by_decoy_total[cat] += 1
                    s_en.by_decoy_total[cat] += 1
                    for st_, dets in ((s_lo, dets_lo), (s_en, dets_en)):
                        bg, dec, bdec = count_other_person_dets(dets, truths, gsd)
                        st_.bg_fp += bg
                        if dec:
                            st_.decoy_fp += 1
                            st_.by_decoy[cat] += 1
                            st_.decoy_scores.append(bdec)
                            st_.fp_scores.append(bdec)
                        else:
                            st_.fp_scores.extend([])

        for key, s in (("lwir_only", s_lo), ("ensemble", s_en)):
            d = s.summary()
            d["gsd_m_px"] = round(gsd, 4)
            d["offsets"] = offsets
            out[key][f"{agl:.0f}m"] = d
    return out


# --------------------------------------------------------------------------- #
# Survey evaluation
# --------------------------------------------------------------------------- #
def eval_survey(args, world, lwir_spec, rgb_spec, lwir_cam, rgb_cam,
                lwir_only, ens) -> Dict[str, Any]:
    out: Dict[str, Any] = {"lwir_only": {}, "ensemble": {}}
    vseen_lo: Dict[float, set] = defaultdict(set)
    vseen_en: Dict[float, set] = defaultdict(set)
    vdet: Dict[str, List[float]] = defaultdict(list)

    for agl in args.altitudes:
        gsd = lwir_spec.gsd_at(agl)
        swath = lwir_spec.swath_width_m(agl)
        spacing = swath * (0.7 if not args.quick else 1.5)
        states = survey_states(world, agl, spacing, n_cross=4 if args.quick else 6)
        if args.max_frames:
            states = states[: args.max_frames]

        s_lo = Stats(agl, "lwir_only")
        s_en = Stats(agl, "ensemble")
        # Surveyed area: passes x spacing x swath (with the 0.7/1.5 overlap).
        s_lo.area_km2 = s_en.area_km2 = (len(states) * spacing * swath) / 1e6

        for st in states:
            t = s_en.frames / 8.0
            f_lwir = lwir_cam.render(st, t)
            f_rgb = rgb_cam.render(st, t)
            q = rgb_frame_quality(f_rgb)
            truths = f_lwir.truth
            dets_lo, dets_en = run_detectors(lwir_only, ens, f_lwir, f_rgb, q,
                                             s_en, s_lo)
            for tr in truths:
                if tr.category != "victim" or not tr.visible:
                    continue
                vdet[tr.oid].append(tr.detectability)
            for label, dets, seen in (("lo", dets_lo, vseen_lo[agl]),
                                      ("en", dets_en, vseen_en[agl])):
                bg, dec, bdec = count_other_person_dets(dets, truths, gsd)
                st_ = s_lo if label == "lo" else s_en
                st_.bg_fp += bg
                st_.decoy_fp += dec
                st_.fp_scores.extend([d.score for d in dets if d.is_person])
                for d in dets:
                    if not d.is_person:
                        continue
                    for tr in truths:
                        if (tr.category == "victim" and tr.visible
                                and math.hypot(tr.pixel[0] - d.u, tr.pixel[1] - d.v)
                                <= match_radius(tr, gsd)):
                            seen.add(tr.oid)
                            st_.tp_scores.append(d.score)
                            st_.by_category[world_victim_cat(world, tr.oid)] += 1
                    if dec:
                        st_.by_decoy["survey"] += dec

        for st_, seen in ((s_lo, vseen_lo[agl]), (s_en, vseen_en[agl])):
            st_.targets_seen = seen
            for v in world.victims:
                st_.by_category_total[v.category] += 1
                if max(vdet.get(v.vid, [0.0])) >= args.resolvable_threshold:
                    st_.targets_resolvable.add(v.vid)
        for key, s in (("lwir_only", s_lo), ("ensemble", s_en)):
            d = s.summary()
            d["gsd_m_px"] = round(gsd, 4)
            d["swath_m"] = round(swath, 1)
            d["line_spacing_m"] = round(spacing, 1)
            d["area_km2"] = round(s.area_km2, 4)
            out[key][f"{agl:.0f}m"] = d
    return {"per_altitude": out,
            "survivor_best_detectability": {k: round(max(v), 3) for k, v in vdet.items()}}


def world_victim_cat(world, oid: str) -> str:
    for v in world.victims:
        if v.vid == oid:
            return v.category
    return "?"


# --------------------------------------------------------------------------- #
def evaluate(args) -> Dict[str, Any]:
    world, lwir_spec, rgb_spec = build_reference_scenario(
        args.scenario, seed=args.seed, smoke=args.smoke,
        hfov_deg=args.hfov, scale=args.scale)
    lwir_cam = CameraRenderer(world, lwir_spec)
    rgb_cam = CameraRenderer(world, rgb_spec)
    ens = build_reference_pipeline("both")
    lwir_only = build_reference_pipeline("lwir")

    report: Dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scenario": args.scenario, "seed": args.seed, "smoke": args.smoke,
        "mode": args.mode, "hfov_deg": lwir_spec.hfov_deg,
        "lwir_resolution": [lwir_spec.width, lwir_spec.height],
        "resolution_scale": round(lwir_spec.resolution_scale, 3),
        "native_lwir_resolution": [lwir_spec.native_width, lwir_spec.native_height],
        "resolvable_threshold": args.resolvable_threshold,
        "world": {"extent_m": [world.north_m, world.east_m],
                  "victims": len(world.victims),
                  "distractors": len(world.distractors),
                  "buildings": len(world.buildings),
                  "time_of_day_h": world.spec.time_of_day_h,
                  "cloud_cover": world.spec.cloud_cover,
                  "rain": world.spec.rain,
                  "smoke_density": world.spec.smoke_density,
                  "flood_fraction": world.spec.flood_fraction},
        "survivors": [{"vid": v.vid, "category": v.category,
                       "posture": v.posture.value, "in_water": v.in_water,
                       "on_rooftop": v.on_rooftop,
                       "occlusion": round(v.occlusion, 2),
                       "body_temp_c": round(v.body_temp_c, 1),
                       "visual_salience": round(v.visual_salience, 2)}
                      for v in world.victims],
        "decoys": [{"did": d.did, "kind": d.kind, "temp_c": round(d.temp_c, 1),
                    "size_m": d.size_m} for d in world.distractors],
    }
    if args.mode in ("aimed", "both"):
        report["aimed"] = eval_aimed(args, world, lwir_spec, rgb_spec, lwir_cam,
                                     rgb_cam, lwir_only, ens)
    if args.mode in ("survey", "both"):
        report["survey"] = eval_survey(args, world, lwir_spec, rgb_spec, lwir_cam,
                                       rgb_cam, lwir_only, ens)
    return report


# --------------------------------------------------------------------------- #
def print_table(report: Dict[str, Any], section: str) -> None:
    data = report[section]
    if section == "survey":
        per_alt = data["per_altitude"]
    else:
        per_alt = data
    n_vic = report["world"]["victims"]
    print(f"\n--- {section} ---")
    hdr = (f"{'mode':<10}{'AGL':>4}{'GSD':>7}{'frm':>5}{'recall':>9}"
           f"{'/resolv':>9}{'decFP':>7}{'bgFP':>7}{'bg/f':>7}"
           f"{'tp':>6}{'fp':>6}{'ms':>7}")
    print(hdr)
    print("-" * len(hdr))
    for mode in ("lwir_only", "ensemble"):
        for alt, b in per_alt[mode].items():
            rec = f"{b['found']}/{b['targets']}" if b["targets"] else "-"
            rr = (f"{b['recall_of_resolvable']:.2f}"
                  if b["recall_of_resolvable"] is not None else "-")
            ms = b["ms_per_frame"]
            print(f"{mode:<10}{b['agl_m']:>4.0f}{b['gsd_m_px'] or 0:>7.3f}"
                  f"{b['frames']:>5}{rec:>9}{rr:>9}{b['decoy_false_alarms']:>7}"
                  f"{b['background_false_alarms']:>7}{b['bg_fp_per_frame']:>7}"
                  f"{str(b['score_tp_mean']):>6}{str(b['score_fp_mean']):>6}{ms:>7}")
    print()
    for mode in ("lwir_only", "ensemble"):
        for alt, b in per_alt[mode].items():
            cats = ", ".join(f"{k}:{b['recall_by_category'].get(k,0)}/{v}"
                             for k, v in sorted(b["targets_by_category"].items()))
            dec = ", ".join(f"{k}:{v}" for k, v in sorted(b["decoys_triggered"].items()))
            print(f"  {mode:<9}@{alt:<5} {cats}")
            if dec:
                print(f"  {'':<15} decoys triggered -> {dec}")
            if b.get("bg_fp_per_km2"):
                print(f"  {'':<15} background FP density {b['bg_fp_per_km2']}/km^2 "
                      f"over {b.get('area_km2','-')} km^2")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="flood")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mode", choices=("aimed", "survey", "both"), default="both")
    ap.add_argument("--altitudes", type=float, nargs="*", default=[35.0, 50.0, 70.0])
    ap.add_argument("--hfov", type=float, default=None)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="render resolution multiplier (2.0 = real 640x512 core)")
    ap.add_argument("--max-frames", type=int, default=0, help="survey: cap frames/altitude")
    ap.add_argument("--resolvable-threshold", type=float, default=0.50,
                    help="TTP detectability above which a target counts as resolvable")
    ap.add_argument("--smoke", action="store_true", help="small fast world (tests)")
    ap.add_argument("--quick", action="store_true", help="fewer offsets / coarser passes")
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    t0 = time.perf_counter()
    rep = evaluate(args)
    rep["wall_clock_s"] = round(time.perf_counter() - t0, 1)

    print(f"\n=== {rep['scenario']} (seed {rep['seed']}) "
          f"{rep['world']['extent_m'][0]:.0f}x{rep['world']['extent_m'][1]:.0f} m | "
          f"{rep['world']['victims']} survivors, {rep['world']['distractors']} decoys | "
          f"solar hour {rep['world']['time_of_day_h']}, cloud "
          f"{rep['world']['cloud_cover']}, smoke {rep['world']['smoke_density']} | "
          f"LWIR {rep['lwir_resolution'][0]}x{rep['lwir_resolution'][1]} "
          f"(native {rep['native_lwir_resolution'][0]}x{rep['native_lwir_resolution'][1]}) "
          f"HFOV {rep['hfov_deg']:.0f}deg ===")
    for section in ("aimed", "survey"):
        if section in rep:
            print_table(rep, section)
    print(f"wall clock {rep['wall_clock_s']} s")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=2, default=str))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
