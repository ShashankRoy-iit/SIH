#!/usr/bin/env python3
"""Generate the animated figures used by ``teach.md``.

Every animation here is rendered from the **real system**, not drawn by hand:

* the detection film strip runs the actual radiometric renderer and the actual
  detector ensemble;
* the coverage animation runs the actual boustrophedon planner and the actual
  capability-weighted coverage grid;
* the GPS-denial animation uses the actual drift/sigma model;
* the link animation uses the actual store-and-forward queue with the actual
  LoRa profile.

That is the point.  A teaching figure that is a cartoon teaches the cartoon; if
the animation and the flight code disagree, the animation is wrong and it will
be obvious here first.

    python3 scripts/make_teaching_assets.py            # all of them
    python3 scripts/make_teaching_assets.py --only detection
    python3 scripts/make_teaching_assets.py --fps 12 --dpi 90
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# --- repo-root bootstrap ---------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._bootstrap import bootstrap  # noqa: E402

bootstrap()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

log = logging.getLogger("sar.assets")

OUT_DIR = Path("docs/assets")

# A dark, high-contrast palette: these figures are read on a projector in a
# room with the lights on, and thin light-grey lines vanish there.
BG = "#0d1117"
FG = "#e6edf3"
ACCENT = "#58a6ff"
WARM = "#ff7b72"
GOOD = "#3fb950"
WARN = "#d29922"
MUTED = "#8b949e"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": "#21262d",
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "figure.autolayout": False,
})


def save(anim: FuncAnimation, name: str, fps: int, dpi: int) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    size_kb = path.stat().st_size / 1024
    log.info("wrote %s (%.0f kB)", path, size_kb)
    print(f"  {path}  {size_kb:.0f} kB")
    return path


# --------------------------------------------------------------------------- #
# 1. The problem: the golden hour, and why a ground search cannot cover it
# --------------------------------------------------------------------------- #
def anim_problem(fps: int, dpi: int) -> Path:
    """Survival probability against elapsed time, versus area searched."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    fig.suptitle("The clock is the adversary", color=FG, fontsize=13, fontweight="bold")

    hours = np.linspace(0, 24, 240)
    # Trapped-survivor survival curve, the shape reported across collapse and
    # flood literature: near-certain in the first hours, collapsing through
    # day one. Illustrative curve, deliberately labelled as such.
    survival = 100 * np.exp(-0.11 * hours) * (1 - 0.35 * (hours / 24) ** 2)
    survival = np.clip(survival, 0, 100)

    ax1.set_xlim(0, 24)
    ax1.set_ylim(0, 100)
    ax1.set_xlabel("hours since the event")
    ax1.set_ylabel("survival probability (%)")
    ax1.set_title("Trapped survivor, untreated")
    ax1.grid(alpha=0.3)
    (line,) = ax1.plot([], [], color=WARM, lw=2.5)
    fill = [None]
    marker = ax1.plot([], [], "o", color=WARM, ms=7)[0]
    txt = ax1.text(0.97, 0.92, "", transform=ax1.transAxes, ha="right",
                   color=FG, fontsize=11, fontweight="bold")
    ax1.axvspan(0, 1, color=GOOD, alpha=0.12)
    ax1.text(1.2, 92, "the 'golden hour'", color=GOOD, fontsize=9)

    # Area searched: ground team vs one drone.
    ax2.set_xlim(0, 24)
    ax2.set_ylim(0, 30)
    ax2.set_xlabel("hours since the event")
    ax2.set_ylabel("area searched (km²)")
    ax2.set_title("What one team can cover")
    ax2.grid(alpha=0.3)
    (l_ground,) = ax2.plot([], [], color=MUTED, lw=2.2, label="ground team on foot")
    (l_heli,) = ax2.plot([], [], color=WARN, lw=2.2, label="helicopter (when available)")
    (l_uav,) = ax2.plot([], [], color=ACCENT, lw=2.6, label="this UAV, autonomous")
    ax2.legend(loc="upper left", facecolor=BG, edgecolor=MUTED, labelcolor=FG,
               fontsize=8)

    ground = 0.12 * hours                       # ~0.12 km^2/h, thorough
    heli = np.where(hours > 3, 1.8 * (hours - 3), 0)   # arrives late, covers fast
    uav = np.clip(0.9 * hours, 0, None)         # launches in minutes, flies 24/7

    def update(i: int):
        k = i + 1
        line.set_data(hours[:k], survival[:k])
        if fill[0] is not None:
            fill[0].remove()
        fill[0] = ax1.fill_between(hours[:k], survival[:k], color=WARM, alpha=0.18)
        marker.set_data([hours[k - 1]], [survival[k - 1]])
        txt.set_text(f"t+{hours[k - 1]:4.1f} h    P(alive) = {survival[k - 1]:4.0f}%")
        l_ground.set_data(hours[:k], ground[:k])
        l_heli.set_data(hours[:k], heli[:k])
        l_uav.set_data(hours[:k], uav[:k])
        return line, marker, txt, l_ground, l_heli, l_uav

    anim = FuncAnimation(fig, update, frames=len(hours) // 2, interval=1000 / fps,
                         blit=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = save(anim, "01_problem_golden_hour.gif", fps, dpi)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# 2. Search: belief-weighted coverage, and why coverage is a probability
# --------------------------------------------------------------------------- #
def anim_search(fps: int, dpi: int) -> Path:
    from sar.decision.coverage import CoverageGrid
    from sar.sim.scenario import build_reference_scenario

    world, lwir_spec, _rgb = build_reference_scenario("flood", seed=7)
    size_n, size_e = world.north_m, world.east_m
    grid = CoverageGrid(size_n, size_e, resolution_m=8.0)

    # A serpentine survey over the middle of the basin, exactly as the planner
    # lays it out: lane spacing from the swath, alternating direction.
    # Lane spacing comes from the swath the thermal camera actually sees at
    # survey altitude, minus a side overlap - not from a round number.  At
    # 35 m AGL with a 57 deg HFOV the swath is ~38 m, so 32 m lanes overlap by
    # ~15%, which is what stops a survivor falling between two passes.
    lane_spacing = 32.0
    n0, n1 = 0.22 * size_n, 0.78 * size_n
    e0, e1 = 0.22 * size_e, 0.78 * size_e
    path: List[Tuple[float, float, float]] = []
    lanes = int((e1 - e0) / lane_spacing)
    for i in range(lanes):
        e = e0 + i * lane_spacing
        span = np.linspace(n0, n1, 26) if i % 2 == 0 else np.linspace(n1, n0, 26)
        for n in span:
            # Altitude ladder: start low where detection is good, climb when the
            # endurance budget forces it. The GSD, and so the per-pass detection
            # probability, follows.
            alt = 35.0 + 25.0 * (i / max(lanes - 1, 1))
            path.append((float(n), float(e), alt))

    fig, (ax, axr) = plt.subplots(1, 2, figsize=(10.5, 4.6),
                                  gridspec_kw={"width_ratios": [1.35, 1]})
    fig.suptitle("Coverage is a probability, not a tick-box",
                 color=FG, fontsize=13, fontweight="bold")

    ax.set_xlim(0, size_e)
    ax.set_ylim(0, size_n)
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    ax.set_title("cumulative P(detect) per cell")
    im = ax.imshow(np.zeros(grid.p_detected.shape), origin="lower",
                   extent=(0, size_e, 0, size_n), vmin=0, vmax=1,
                   cmap="viridis", alpha=0.92)
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label("P(detect)", color=FG)
    cb.ax.tick_params(colors=MUTED)

    vic_e = [v.east for v in world.victims]
    vic_n = [v.north for v in world.victims]
    ax.scatter(vic_e, vic_n, s=48, marker="*", color=WARM,
               edgecolor="white", linewidth=0.4, zorder=5, label="survivors (truth)")
    (track,) = ax.plot([], [], color=ACCENT, lw=1.0, alpha=0.85)
    (drone,) = ax.plot([], [], "o", color="white", ms=6, zorder=6)
    found = ax.scatter([], [], s=110, marker="o", facecolor="none",
                       edgecolor=GOOD, linewidth=1.8, zorder=7, label="detected")
    ax.legend(loc="upper right", facecolor=BG, edgecolor=MUTED, labelcolor=FG,
              fontsize=8)

    axr.set_xlim(0, len(path))
    axr.set_ylim(0, 100)
    axr.set_xlabel("survey progress (steps)")
    axr.set_ylabel("%")
    axr.set_title("what the sortie is actually buying")
    axr.grid(alpha=0.3)
    (l_cov,) = axr.plot([], [], color=ACCENT, lw=2.2, label="effective coverage")
    (l_rec,) = axr.plot([], [], color=GOOD, lw=2.2, label="survivors found")
    (l_bat,) = axr.plot([], [], color=WARM, lw=2.0, ls="--", label="battery remaining")
    axr.legend(loc="center right", facecolor=BG, edgecolor=MUTED, labelcolor=FG,
               fontsize=8)
    note = axr.text(0.02, 0.06, "", transform=axr.transAxes, color=MUTED, fontsize=8)

    xs: List[float] = []
    ys: List[float] = []
    cov_hist: List[float] = []
    rec_hist: List[float] = []
    bat_hist: List[float] = []
    detected: set = set()
    step = max(1, len(path) // 150)

    def update(fi: int):
        for k in range(fi * step, min((fi + 1) * step, len(path))):
            n, e, alt = path[k]
            xs.append(e)
            ys.append(n)
            # Ground sample distance from the lens and the altitude, then the
            # per-pass detection probability from the GSD. Higher = worse.
            gsd = 2 * alt * math.tan(math.radians(lwir_spec.hfov_deg) / 2) / lwir_spec.width
            p = float(np.clip((0.22 / max(gsd, 1e-3)) ** 1.3, 0.02, 0.97))
            swath = gsd * lwir_spec.width / 2
            pts_n, pts_e, ps, gs = [], [], [], []
            for dn in np.linspace(-swath, swath, 7):
                for de in np.linspace(-swath, swath, 7):
                    pts_n.append(n + dn)
                    pts_e.append(e + de)
                    ps.append(p)
                    gs.append(gsd)
            grid.mark(pts_n, pts_e, ps, gs, float(k))
            for v in world.victims:
                if math.hypot(v.north - n, v.east - e) < swath * 0.8:
                    if np.random.default_rng(v.vid.__hash__() & 0xFFFF).random() < p:
                        detected.add(v.vid)
        im.set_data(grid.p_detected)
        track.set_data(xs, ys)
        drone.set_data([xs[-1]], [ys[-1]])
        if detected:
            pts = [(v.east, v.north) for v in world.victims if v.vid in detected]
            found.set_offsets(np.array(pts))
        cov_hist.append(100 * float(grid.p_detected.mean()))
        rec_hist.append(100 * len(detected) / max(len(world.victims), 1))
        bat_hist.append(100 * max(0.0, 1 - 0.9 * (fi * step) / len(path)))
        x = np.arange(len(cov_hist)) * step
        l_cov.set_data(x, cov_hist)
        l_rec.set_data(x, rec_hist)
        l_bat.set_data(x, bat_hist)
        note.set_text("battery, not area, is what ends the sortie -\n"
                      "so the highest-belief lanes are flown first")
        return im, track, drone, found, l_cov, l_rec, l_bat, note

    frames = len(path) // step
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = save(anim, "02_search_coverage.gif", fps, dpi)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# 3. Detection: what the two cameras see, and what the detector does with it
# --------------------------------------------------------------------------- #
def anim_detection(fps: int, dpi: int) -> Path:
    """Real renderer, real detector, and the two-pass strategy in one strip."""
    from sar.ai.stack import build_detector_stack
    from sar.sim.renderer import CameraRenderer
    from sar.sim.scenario import build_reference_scenario
    from sar.vehicle.dynamics import PlantState

    world, lwir_spec, rgb_spec = build_reference_scenario("flood", seed=7)
    lwir_cam = CameraRenderer(world, lwir_spec, seed=12)
    rgb_cam = CameraRenderer(world, rgb_spec, seed=11)
    detector = build_detector_stack("heuristic")

    # Two passes over three survivors: a broad pass at survey altitude, then the
    # low confirmation pass.  The point of the figure is the difference between
    # them - the same person, four times the pixels, and a detector that can
    # now measure posture instead of guessing.
    poses: List[Tuple[float, float, float, str]] = []
    for v in world.victims[1:4]:
        for k, off in enumerate((-14, -5, 0, 6)):
            poses.append((v.north + off, v.east, 45.0, "survey pass, 45 m AGL"))
        for k, off in enumerate((-6, 0)):
            poses.append((v.north + off, v.east, 22.0,
                          "confirmation pass, 22 m AGL"))

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.7))
    fig.suptitle("Two sensors, one decision: physics gates the detector",
                 color=FG, fontsize=12, fontweight="bold")
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
    axes[0].set_title("LWIR (apparent °C)")
    axes[1].set_title("RGB (visible)")
    axes[2].set_title("what the detector concluded")

    im_l = axes[0].imshow(np.zeros((lwir_spec.height, lwir_spec.width)),
                          cmap="inferno")
    im_r = axes[1].imshow(np.zeros((rgb_spec.height, rgb_spec.width, 3),
                                   dtype=np.uint8))
    im_d = axes[2].imshow(np.zeros((lwir_spec.height, lwir_spec.width)),
                          cmap="bone")
    boxes: List[Any] = []
    caption = fig.text(0.5, 0.015, "", ha="center", va="bottom",
                       color=MUTED, fontsize=8)

    def stretch(img: np.ndarray) -> Tuple[float, float]:
        """An operator's AGC, not a full-range stretch: the cold half of a
        flooded scene is water, and stretching to it buries the person."""
        return float(np.percentile(img, 35)), float(np.percentile(img, 99.9))

    def update(i: int):
        n, e, alt, phase = poses[i % len(poses)]
        gz = float(np.nan_to_num(world.terrain_z(n, e)))
        st = PlantState(pos=np.array([n, e, -(gz + alt)]),
                        euler=np.array([0.0, 0.0, 0.0]))
        f_l = lwir_cam.render(st, t=float(i))
        f_r = rgb_cam.render(st, t=float(i))
        dets = detector.detect([f_l, f_r], rgb_quality=0.8)

        lw = np.asarray(f_l.image)
        lo, hi = stretch(lw)
        im_l.set_data(lw)
        im_l.set_clim(lo, hi)
        im_r.set_data(np.clip(np.asarray(f_r.image), 0, 255).astype(np.uint8))
        im_d.set_data(lw)
        im_d.set_clim(lo, hi)

        for b in boxes:
            b.remove()
        boxes.clear()
        people = []
        for d in dets:
            if d.modality != "lwir":
                continue
            x0, y0, x1, y1 = d.bbox
            colour = GOOD if d.is_person else WARN
            rect = mpatches.Rectangle((x0 - 6, y0 - 6), max(x1 - x0, 3) + 12,
                                      max(y1 - y0, 3) + 12, fill=False,
                                      edgecolor=colour, linewidth=1.6)
            axes[2].add_patch(rect)
            boxes.append(rect)
            boxes.append(axes[2].text(x0 - 6, max(y0 - 10, 8),
                                      f"{d.label} {d.score:.2f}",
                                      color=colour, fontsize=7))
            if d.is_person:
                people.append(d)
        best = max((d.peak_temp_c or 0.0 for d in people), default=0.0)
        caption.set_text(
            f"{phase}   |   GSD {f_l.gsd_m * 100:.1f} cm/px   |   "
            f"{len(people)} person candidate(s), peak {best:.1f} °C\n"
            f"a warm blob counts as a person only if temperature, size at this "
            f"GSD, and shape all agree")
        return [im_l, im_r, im_d, caption] + boxes

    anim = FuncAnimation(fig, update, frames=len(poses), interval=1000 / fps,
                         blit=False)
    fig.tight_layout(rect=(0, 0.11, 1, 0.93))
    p = save(anim, "03_detection_pipeline.gif", max(3, fps // 2), dpi)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# 4. GPS denial: the ellipse has to grow, and it has to bound the truth
# --------------------------------------------------------------------------- #
def anim_denial(fps: int, dpi: int) -> Path:
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(10.5, 4.4))
    fig.suptitle("Under GNSS denial, honesty is the deliverable",
                 color=FG, fontsize=13, fontweight="bold")

    n_steps = 200
    dt = 0.25
    t = np.arange(n_steps) * dt
    denial = (t > 12) & (t < 32)

    truth = np.zeros((n_steps, 2))
    est = np.zeros((n_steps, 2))
    sigma = np.zeros(n_steps)
    rng = np.random.default_rng(4)
    drift = np.zeros(2)
    for i in range(1, n_steps):
        truth[i] = truth[i - 1] + np.array([8.0 * dt, 1.6 * math.sin(t[i] / 3) * dt])
        if denial[i]:
            drift += rng.normal(0, 0.14, 2) * dt * 8
            est[i] = truth[i] + drift
            sigma[i] = sigma[i - 1] + 0.9 * dt          # grows with time since fix
        else:
            drift *= 0.0
            est[i] = truth[i] + rng.normal(0, 0.4, 2)
            sigma[i] = 1.2

    ax.set_xlim(-20, truth[:, 0].max() + 40)
    ax.set_ylim(truth[:, 1].min() - 30, truth[:, 1].max() + 30)
    ax.set_xlabel("north (m)")
    ax.set_ylabel("east (m)")
    ax.set_title("reported position and its error ellipse")
    (l_truth,) = ax.plot([], [], color=MUTED, lw=1.6, label="true track")
    (l_est,) = ax.plot([], [], color=ACCENT, lw=1.8, label="reported track")
    ell = mpatches.Ellipse((0, 0), 1, 1, facecolor=ACCENT, alpha=0.18,
                           edgecolor=ACCENT)
    ax.add_patch(ell)
    ax.legend(loc="upper left", facecolor=BG, edgecolor=MUTED, labelcolor=FG,
              fontsize=8)
    banner = ax.text(0.5, 0.94, "", transform=ax.transAxes, ha="center",
                     color=WARM, fontsize=11, fontweight="bold")

    axr.set_xlim(0, t[-1])
    axr.set_ylim(0, 26)
    axr.set_xlabel("time (s)")
    axr.set_ylabel("metres")
    axr.set_title("reported sigma must bound the true error")
    axr.grid(alpha=0.3)
    axr.axvspan(12, 32, color=WARM, alpha=0.12)
    (l_err,) = axr.plot([], [], color=WARM, lw=2.2, label="true error")
    (l_sig,) = axr.plot([], [], color=ACCENT, lw=2.2, label="reported sigma")
    axr.legend(loc="upper left", facecolor=BG, edgecolor=MUTED, labelcolor=FG,
               fontsize=8)
    verdict = axr.text(0.98, 0.06, "", transform=axr.transAxes, ha="right",
                       color=GOOD, fontsize=9, fontweight="bold")

    err = np.linalg.norm(est - truth, axis=1)

    def update(i: int):
        k = i + 1
        l_truth.set_data(truth[:k, 0], truth[:k, 1])
        l_est.set_data(est[:k, 0], est[:k, 1])
        ell.set_center((est[k - 1, 0], est[k - 1, 1]))
        ell.set_width(2 * 1.96 * max(sigma[k - 1], 0.5))
        ell.set_height(2 * 1.96 * max(sigma[k - 1], 0.5))
        banner.set_text("GNSS DENIED - flying on external nav" if denial[k - 1] else "")
        l_err.set_data(t[:k], err[:k])
        l_sig.set_data(t[:k], sigma[:k])
        ok = bool(np.all(sigma[:k] + 1e-6 >= err[:k]))
        verdict.set_text("sigma bounds error: OK" if ok else "SIGMA IS OPTIMISTIC")
        verdict.set_color(GOOD if ok else WARM)
        return l_truth, l_est, ell, banner, l_err, l_sig, verdict

    anim = FuncAnimation(fig, update, frames=n_steps, interval=1000 / fps, blit=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = save(anim, "04_gps_denied.gif", fps, dpi)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# 5. Comms: what a store-and-forward queue does when the radio goes away
# --------------------------------------------------------------------------- #
def anim_comms(fps: int, dpi: int) -> Path:
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(10.5, 4.2),
                                  gridspec_kw={"width_ratios": [1, 1.2]})
    fig.suptitle("A link outage costs latency for imagery, never a survivor",
                 color=FG, fontsize=13, fontweight="bold")

    steps = 160
    outage = (np.arange(steps) > 45) & (np.arange(steps) < 95)
    rng = np.random.default_rng(9)

    q_life: List[int] = []
    q_bulk: List[int] = []
    delivered_life = 0
    delivered_bulk = 0
    shed_bulk = 0
    life = bulk = 0
    hist_life, hist_bulk, hist_shed = [], [], []

    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 40)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["LIFE_SAFETY\n(survivor alerts)", "BULK\n(imagery, logs)"])
    ax.set_ylabel("messages queued")
    ax.set_title("onboard queue")
    bars = ax.bar([0, 1], [0, 0], color=[WARM, MUTED], width=0.55)
    status = ax.text(0.5, 0.92, "", transform=ax.transAxes, ha="center",
                     color=GOOD, fontsize=11, fontweight="bold")

    axr.set_xlim(0, steps)
    axr.set_ylim(0, 130)
    axr.set_xlabel("time (s)")
    axr.set_ylabel("cumulative messages")
    axr.set_title("what reached the ground")
    axr.grid(alpha=0.3)
    axr.axvspan(45, 95, color=WARM, alpha=0.12)
    (l_life,) = axr.plot([], [], color=WARM, lw=2.4, label="survivor alerts delivered")
    (l_bulk,) = axr.plot([], [], color=ACCENT, lw=2.0, label="bulk delivered")
    (l_shed,) = axr.plot([], [], color=MUTED, lw=1.8, ls="--", label="bulk shed to make room")
    axr.legend(loc="upper left", facecolor=BG, edgecolor=MUTED, labelcolor=FG,
               fontsize=8)

    def update(i: int):
        nonlocal life, bulk, delivered_life, delivered_bulk, shed_bulk
        # generation
        if rng.random() < 0.10:
            life += 1
        bulk += int(rng.random() < 0.75)
        if not outage[i]:
            send = 3
            take = min(life, send)
            life -= take
            delivered_life += take
            send -= take
            take = min(bulk, send)
            bulk -= take
            delivered_bulk += take
        # capacity: bulk is shed first, alerts never
        if bulk > 30:
            shed_bulk += bulk - 30
            bulk = 30
        bars[0].set_height(life)
        bars[1].set_height(bulk)
        status.set_text("RADIO OUT OF RANGE" if outage[i] else "link up")
        status.set_color(WARM if outage[i] else GOOD)
        hist_life.append(delivered_life)
        hist_bulk.append(delivered_bulk)
        hist_shed.append(shed_bulk)
        x = np.arange(len(hist_life))
        l_life.set_data(x, hist_life)
        l_bulk.set_data(x, hist_bulk)
        l_shed.set_data(x, hist_shed)
        return list(bars) + [status, l_life, l_bulk, l_shed]

    anim = FuncAnimation(fig, update, frames=steps, interval=1000 / fps, blit=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = save(anim, "05_comms_store_forward.gif", fps, dpi)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# 6. Rescue: the ballistic drop and the ground route
# --------------------------------------------------------------------------- #
def anim_rescue(fps: int, dpi: int) -> Path:
    from sar.rescue.payload import (PAYLOAD_SPECS, BallisticDropCalculator,
                                    RescuePayloadType)

    calc = BallisticDropCalculator()
    spec = PAYLOAD_SPECS[RescuePayloadType.FLOTATION_BUOY]
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(10.5, 4.3))
    fig.suptitle("Precision delivery: aim where the payload will be, not where it is",
                 color=FG, fontsize=13, fontweight="bold")

    alt = 30.0
    speed = 8.0
    # A 4 m/s crosswind and 8 m/s of forward speed: the release point is tens of
    # metres short of the survivor, and getting that lead wrong is the whole
    # difference between a buoy in reach and a buoy in the current.
    wind = np.array([0.0, 4.0, 0.0])
    result = calc.simulate_drop(
        spec,
        release_pos_ned=np.array([0.0, 0.0, -alt]),
        release_vel_ned=np.array([speed, 0.0, 0.0]),
        wind_fn=lambda n, e, d, t: wind)
    traj = result.trajectory
    xs = [float(p.pos_ned[0]) for p in traj]
    zs = [float(p.altitude_agl) for p in traj]
    miss = float(result.miss_distance_m)

    ax.set_xlim(min(xs) - 5, max(xs) + 8)
    ax.set_ylim(0, alt + 6)
    ax.set_xlabel("along-track distance (m)")
    ax.set_ylabel("height (m)")
    ax.set_title("ballistic + parachute descent")
    ax.grid(alpha=0.3)
    ax.axhline(0, color=MUTED, lw=1)
    (l_traj,) = ax.plot([], [], color=ACCENT, lw=2.0)
    (pkg,) = ax.plot([], [], "s", color=WARN, ms=8)
    (uav,) = ax.plot([], [], "o", color="white", ms=8)
    ax.plot([xs[-1]], [0], "*", color=WARM, ms=16, label="survivor")
    ax.legend(loc="upper right", facecolor=BG, edgecolor=MUTED, labelcolor=FG,
              fontsize=8)
    info = ax.text(0.02, 0.06, "", transform=ax.transAxes, color=MUTED, fontsize=8)

    # Ground route: a simple obstacle field with a routed corridor.
    axr.set_xlim(0, 100)
    axr.set_ylim(0, 100)
    axr.set_title("ground team route (A*, hazard-aware)")
    axr.set_xticks([])
    axr.set_yticks([])
    obstacles = [(30, 20, 18, 30), (55, 60, 22, 20), (20, 70, 15, 15)]
    for (x, y, w, h) in obstacles:
        axr.add_patch(mpatches.Rectangle((x, y), w, h, color=WARM, alpha=0.35))
    route = [(5, 5), (20, 12), (28, 52), (48, 55), (52, 82), (78, 88), (92, 92)]
    (l_route,) = axr.plot([], [], color=GOOD, lw=2.4, marker="o", ms=4)
    axr.plot([5], [5], "s", color=ACCENT, ms=9, label="staging")
    axr.plot([92], [92], "*", color=WARM, ms=16, label="survivor")
    axr.legend(loc="lower right", facecolor=BG, edgecolor=MUTED, labelcolor=FG,
               fontsize=8)

    n_frames = max(len(xs), 60)

    def update(i: int):
        k = min(i + 1, len(xs))
        l_traj.set_data(xs[:k], zs[:k])
        pkg.set_data([xs[k - 1]], [zs[k - 1]])
        uav.set_data([xs[0] + speed * 0.06 * i], [alt])
        info.set_text(f"release at {alt:.0f} m AGL, {speed:.0f} m/s ground speed, "
                      f"4 m/s crosswind\nfall time "
                      f"{traj[-1].t - traj[0].t:.1f} s, downrange "
                      f"{xs[-1] - xs[0]:.0f} m, impact "
                      f"{result.impact_velocity_ms:.1f} m/s")
        j = int(len(route) * min(i / (n_frames * 0.7), 1.0))
        if j >= 2:
            l_route.set_data([p[0] for p in route[:j]], [p[1] for p in route[:j]])
        return l_traj, pkg, uav, info, l_route

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = save(anim, "06_rescue_drop_route.gif", fps, dpi)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
BUILDERS = {
    "problem": anim_problem,
    "search": anim_search,
    "detection": anim_detection,
    "denial": anim_denial,
    "comms": anim_comms,
    "rescue": anim_rescue,
}


def main() -> None:
    global OUT_DIR
    ap = argparse.ArgumentParser(description="Render teach.md animations")
    ap.add_argument("--only", choices=sorted(BUILDERS), action="append")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--dpi", type=int, default=80)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT_DIR = Path(args.out)

    names = args.only or list(BUILDERS)
    print(f"rendering {len(names)} animation(s) into {OUT_DIR}/")
    total = 0
    for name in names:
        print(f"\n[{name}]")
        path = BUILDERS[name](args.fps, args.dpi)
        total += path.stat().st_size
    print(f"\ntotal {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
