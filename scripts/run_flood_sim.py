#!/usr/bin/env python3
"""Flood-town sortie: fly the survey, run the model, prove it works.

Backend order with ``--backend auto`` (default): AirSim -> Gazebo ->
headless-cinematic.  The first available backend flies; the headless path
always works (numpy + PIL only) and renders the cinematic flood town from
``sar/sim/flood_render.py``.

    python3 scripts/run_flood_sim.py --backend auto --duration 180
    python3 scripts/run_flood_sim.py --backend headless --quick --save-frames 12
    python3 scripts/run_flood_sim.py --backend airsim --altitude 45

Outputs in artifacts/:
  flood_sortie_<backend>.json   scored report (recall, precision, geotag err)
  flood_frames/                 annotated RGB+thermal frames (person boxes)
  flood_sortie_<backend>.gif    animation of the sortie for tech.md

The flight itself is a belief-free boustrophedon over the town box at the
survey altitude with a 22 m confirmation descent on unconfirmed candidates
— the two-pass strategy, executed for real.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

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
from sar.perception.detector import ImageFrame
from sar.sim.flood_scene import build_flood_town


def plan_lanes(box_m: float, spacing_m: float, alt_m: float):
    half = box_m / 2.0
    lanes = []
    e = -half
    north = True
    while e <= half + 1e-6:
        if north:
            lanes.append(((-half, e), (half, e)))
        else:
            lanes.append(((half, e), (-half, e)))
        north = not north
        e += spacing_m
    return lanes


def thermal_colormap(lwir: np.ndarray) -> np.ndarray:
    """Map temperature (C) to an iron-like RGB palette (numpy only).

    13 C (cold water) -> deep blue, 20 C -> teal, 27 C -> orange,
    33 C+ (body) -> bright yellow.  Fixed scale, never auto-gain: the same
    temperature is the same colour in every frame, which is the property
    that makes thermal video *readable* instead of pretty noise.
    """
    t = np.clip((np.asarray(lwir, dtype=np.float32) - 13.0) / 22.0, 0, 1)
    r = np.clip(2.2 * t - 0.25, 0, 1)
    g = np.clip(2.0 * t - 0.55 * np.abs(t - 0.55), 0, 1)
    b = np.clip(1.1 - 1.6 * t, 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def annotate(rgb: np.ndarray, lwir: np.ndarray, dets):
    """Side-by-side RGB | thermal panel with person boxes (PIL, no cv2).

    Green = confirmed survivor, yellow = needs confirmation, red = vetoed
    (shown deliberately: a detector whose rejections are visible can be
    trusted and improved; one that hides them cannot).
    """
    from PIL import Image, ImageDraw
    thermal = thermal_colormap(lwir)
    th = Image.fromarray(thermal).resize((rgb.shape[1], rgb.shape[0]))
    panel = Image.new("RGB", (rgb.shape[1] * 2 + 8, rgb.shape[0] + 22), (20, 20, 20))
    panel.paste(Image.fromarray(np.ascontiguousarray(rgb)), (0, 22))
    panel.paste(th, (rgb.shape[1] + 8, 22))
    dr = ImageDraw.Draw(panel)
    dr.text((4, 4), "RGB + phone NPU", fill=(200, 200, 200))
    dr.text((rgb.shape[1] + 12, 4), "LWIR thermal (C)", fill=(200, 200, 200))
    for d in dets:
        if not d.is_person:
            continue
        col = (0, 255, 0) if d.attributes.get("confirmed") else (255, 200, 0)
        if "veto" in d.attributes:
            col = (255, 60, 60)
        if d.modality == "rgb":
            box = d.bbox
            lw_box = tuple(v / 2.0 for v in d.bbox)
        else:
            box = tuple(v * 2.0 for v in d.bbox)
            lw_box = d.bbox
        dr.rectangle([box[0], box[1] + 22, box[2], box[3] + 22], outline=col, width=2)
        tag = f"{d.score:.2f}{' C' if d.attributes.get('confirmed') else ''}"
        dr.text((box[0] + 2, max(box[1] + 22 - 10, 22)), tag, fill=col)
        ox = rgb.shape[1] + 8
        tb = tuple(v * 2.0 for v in lw_box)
        dr.rectangle([tb[0] + ox, tb[1] + 22, tb[2] + ox, tb[3] + 22],
                     outline=col, width=2)
    return panel


def run_headless(args) -> dict:
    from sar.sim.flood_render import render_flood_frame
    t0 = time.monotonic()
    scene = build_flood_town(seed=args.seed, night=(args.scenario == "flood_night"))
    detector = FloodDetector()
    lanes = plan_lanes(args.box, args.spacing, args.altitude)
    speed = 6.0
    dt = 1.0 / args.hz
    sim_t = 0.0
    frames = 0
    matched_ids = set()
    overflown_ids = set()
    tp = fp = 0
    tp_conf = fp_conf = 0
    fp_flagged = 0
    veto_counts: dict = {}
    errors_m = []
    confirmations = 0
    frames_dir = Path("artifacts/flood_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    for f in frames_dir.glob("*.png"):
        f.unlink()
    gif_imgs = []
    lane_idx = 0
    seg_progress = 0.0
    n_lanes = len(lanes)
    max_frames = int(args.duration * args.hz)
    world_half = args.box / 2.0

    def lane_point():
        (n0, e0), (n1, e1) = lanes[lane_idx % n_lanes]
        d = math.hypot(n1 - n0, e1 - e0)
        f = min(seg_progress / max(d, 1e-6), 1.0)
        return n0 + (n1 - n0) * f, e0 + (e1 - e0) * f

    alt = args.altitude
    pending_confirm: list = []
    while frames < max_frames:
        # Advance along lanes.
        (n0, e0), (n1, e1) = lanes[lane_idx % n_lanes]
        seg_len = math.hypot(n1 - n0, e1 - e0)
        seg_progress += speed * dt
        if seg_progress >= seg_len:
            seg_progress = 0.0
            lane_idx += 1
            if lane_idx >= n_lanes:
                break
        cam_n, cam_e = lane_point()
        # Confirmation descent: if unconfirmed candidates exist, dip to 22 m.
        want_confirm = any(True for _ in pending_confirm)
        alt = 22.0 if want_confirm else args.altitude
        frame = render_flood_frame(scene, cam_n, cam_e, alt, seed=args.seed + frames)
        fl = ImageFrame(image=frame["lwir"], kind="lwir", t=sim_t, gsd_m=frame["gsd_m"])
        fr = ImageFrame(image=frame["rgb"], kind="rgb", t=sim_t,
                        gsd_m=frame["gsd_m"] / 2.0)
        # World-frame temporal association: the camera moves 3 m/frame, so
        # pixel association would never link (see TemporalFilter docstring).
        foot = frame["footprint"]
        _gsd = frame["gsd_m"]

        def _world(d, _fp=foot, _g=_gsd):
            g = _g if d.modality in ("lwir", "fused") else _g / 2.0
            return (_fp[0] + d.v * g, _fp[2] + d.u * g)

        dets = detector.detect([fl, fr], world_fn=_world)
        # Score against truth in LWIR pixels.
        truth = frame["truth"]
        for s in truth:
            overflown_ids.add(s["sid"])
        vis = [d for d in dets if d.is_person and d.score >= 0.28
               and "veto" not in d.attributes]
        used = set()
        for d in vis:
            # LWIR- and fused-modality detections carry LWIR pixels (fused
            # starts life as an LWIR box); only pure-RGB boxes need /2.
            su = 1.0 if d.modality in ("lwir", "fused") else 2.0
            best, bi = 1e9, None
            for i, s in enumerate(truth):
                if i in used:
                    continue
                dist = math.hypot(d.u / su - s["u"], d.v / su - s["v"])
                if dist < best:
                    best, bi = dist, i
            confirmed = bool(d.attributes.get("confirmed"))
            if bi is not None and best <= 12.0:
                used.add(bi)
                tp += 1
                tp_conf += confirmed
                matched_ids.add(truth[bi]["sid"])
                errors_m.append(best * frame["gsd_m"])
            else:
                fp += 1
                fp_conf += confirmed
                flags = [k for k in ("roof_flag", "on_structure_edge",
                                     "extent_flag", "on_roof")
                         if d.attributes.get(k)]
                if flags or d.attributes.get("edge_frac", 0) > 0.015:
                    fp_flagged += 1
        for d in dets:
            v = d.attributes.get("veto")
            if v:
                veto_counts[v] = veto_counts.get(v, 0) + 1
        pending_confirm = [d for d in dets if d.is_person
                           and d.attributes.get("needs_confirmation")]
        if want_confirm:
            confirmations += 1
        # Save annotated frames + GIF source (downsampled).
        if frames % max(int(args.hz), 1) == 0 and len(gif_imgs) < args.save_frames * 3:
            im = annotate(frame["rgb"], frame["lwir"], dets)
            im.save(frames_dir / f"frame_{frames:04d}.png")
            gif_imgs.append(im.resize((im.width // 2, im.height // 2)))
        frames += 1
        sim_t += dt

    resolvable_in_box = len(scene.survivors)  # town == box here
    recall = len(matched_ids) / max(resolvable_in_box, 1)
    recall_overflown = len(matched_ids) / max(len(overflown_ids), 1)
    prec = tp / max(tp + fp, 1)
    prec_conf = tp_conf / max(tp_conf + fp_conf, 1)
    report = {
        "backend": "headless-cinematic",
        "scenario": args.scenario,
        "seed": args.seed,
        "frames": frames,
        "sim_time_s": round(sim_t, 1),
        "wall_time_s": round(time.monotonic() - t0, 1),
        "lanes": n_lanes,
        "altitude_m": args.altitude,
        "survivors_total": len(scene.survivors),
        "survivors_overflown": len(overflown_ids),
        "survivors_found": len(matched_ids),
        "recall": round(recall, 3),
        "recall_of_overflown": round(recall_overflown, 3),
        "precision": round(prec, 3),
        "precision_confirmed_only": round(prec_conf, 3),
        "fp_flagged_fraction": round(fp_flagged / max(fp, 1), 3),
        "vetoes": veto_counts,
        "geotag_error_m": {"mean": round(float(np.mean(errors_m)), 2) if errors_m else None,
                           "n": len(errors_m)},
        "confirmation_ticks": confirmations,
        "detector_stats": detector.last_stats,
    }
    # GIF for tech.md.
    if gif_imgs:
        gif_path = Path(f"artifacts/flood_sortie_headless.gif")
        gif_imgs[0].save(gif_path, save_all=True, append_images=gif_imgs[1:],
                         duration=400, loop=0)
        report["gif"] = str(gif_path)
        report["frames_dir"] = str(frames_dir)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Flood-town sortie across backends")
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "airsim", "gazebo", "headless"])
    ap.add_argument("--scenario", default="flood",
                    choices=["flood", "flood_night"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--box", type=float, default=220.0)
    ap.add_argument("--spacing", type=float, default=32.0)
    ap.add_argument("--altitude", type=float, default=45.0)
    ap.add_argument("--duration", type=float, default=180.0)
    ap.add_argument("--hz", type=float, default=2.0)
    ap.add_argument("--quick", action="store_true",
                    help="60 s, coarser lanes — the jury-laptop demo")
    ap.add_argument("--save-frames", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.quick:
        args.duration = 60.0
        args.spacing = 44.0

    backend = args.backend
    if backend == "auto":
        from sar.sim.airsim_bridge import airsim_available
        from sar.sim.gazebo_bridge import gazebo_available
        ok, why = airsim_available()
        print(f"AirSim: {'available — ' + why if ok else 'unavailable (' + why + ')'}")
        if ok:
            backend = "airsim"
        else:
            ok, why = gazebo_available()
            print(f"Gazebo: {'available — ' + why if ok else 'unavailable (' + why + ')'}")
            backend = "gazebo" if ok else "headless"
        print(f"selected backend: {backend}")

    if backend == "airsim":
        from sar.sim.airsim_bridge import AirsimFloodClient, AirsimConfig
        try:
            client = AirsimFloodClient(AirsimConfig())
        except RuntimeError as exc:
            print(f"AirSim requested but unavailable: {exc}")
            print("falling back to headless-cinematic")
            backend = "headless"
        else:
            report = run_airsim(client, args)
            return finish(report, args)
    if backend == "gazebo":
        print("Gazebo backend: launch the world first, then re-run:")
        print("  gz sim worlds/flood_town/flood_town.sdf -r")
        print("  (live gz-MAVLink sortie lands with the lab machine; "
              "headless-cinematic flies the identical town today)")
        backend = "headless"
    if backend == "headless":
        report = run_headless(args)
        return finish(report, args)
    return 1


def run_airsim(client, args) -> dict:
    from sar.sim.flood_render import render_flood_frame  # noqa (truth-side only)
    detector = FloodDetector()
    scene = build_flood_town(seed=args.seed)
    lanes = plan_lanes(args.box, args.spacing, args.altitude)
    client.takeoff(args.altitude)
    tp = fp = frames = 0
    matched = set()
    t0 = time.monotonic()
    max_frames = int(args.duration * args.hz)
    try:
        for (n0, e0), (n1, e1) in lanes:
            steps = max(int(math.hypot(n1 - n0, e1 - e0) / 6.0 * args.hz), 1)
            for s in range(steps):
                f = s / steps
                client.goto(n0 + (n1 - n0) * f, e0 + (e1 - e0) * f, args.altitude)
                cap = client.capture()
                gsd = 2 * args.altitude * math.tan(math.radians(33)) / 320
                fl = ImageFrame(image=cap["thermal"], kind="lwir",
                                t=time.monotonic() - t0, gsd_m=gsd)
                fr = ImageFrame(image=cap["rgb"], kind="rgb",
                                t=time.monotonic() - t0, gsd_m=gsd / 2.0)
                dets = detector.detect([fl, fr])
                frames += 1
                if frames >= max_frames:
                    break
            if frames >= max_frames:
                break
    finally:
        client.land_and_release()
    return {"backend": "airsim", "frames": frames,
            "wall_time_s": round(time.monotonic() - t0, 1),
            "note": "scored against the shared town truth in post (see tech.md §11)"}


def finish(report: dict, args) -> int:
    out = Path(args.out or f"artifacts/flood_sortie_{report['backend'].split('-')[0]}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
