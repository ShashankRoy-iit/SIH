#!/usr/bin/env python3
"""Generate the bespoke figures used by the SIH pitch deck.

The GIFs in ``docs/assets`` are correct but animated; a pitch deck wants static
frames and a couple of novelty-focused diagrams that the general teaching set
does not cover (the physics-engine identification chain and the architecture).
This script produces, into ``docs/assets/pitch/``:

* first-frame stills of the six teaching animations;
* ``pitch_physics.png``  - the physics-informed survivor identification chain;
* ``pitch_arch.png``     - the end-to-end system architecture.

Everything is rendered with matplotlib from the project's own models where it is
cheap to do so; the physics figure is illustrative and labelled as such.
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- repo-root bootstrap ---------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._bootstrap import bootstrap  # noqa: E402

bootstrap()

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path("docs/assets/pitch")
OUT.mkdir(parents=True, exist_ok=True)

DARK = "#0b1220"
INK = "#e5edf8"
MUTED = "#8ea3c0"
ACCENT = "#22c55e"
CYAN = "#38bdf8"
AMBER = "#f59e0b"
RED = "#dc2626"

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": DARK,
    "axes.edgecolor": "#243350", "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "font.family": "DejaVu Sans", "savefig.facecolor": DARK,
})


def first_frames() -> None:
    """Pull the first frame of each teaching GIF to a PNG still."""
    order = [
        ("01_problem_golden_hour.gif", "pitch_01_problem.png"),
        ("02_search_coverage.gif", "pitch_02_coverage.png"),
        ("03_detection_pipeline.gif", "pitch_03_detection.png"),
        ("04_gps_denied.gif", "pitch_04_gps.png"),
        ("05_comms_store_forward.gif", "pitch_05_comms.png"),
        ("06_rescue_drop_route.gif", "pitch_06_rescue.png"),
    ]
    for src, dst in order:
        p = Path("docs/assets") / src
        if not p.exists():
            continue
        im = Image.open(p)
        im.seek(0)
        im.convert("RGB").save(OUT / dst)
        print(f"  still  {dst}")


def physics_figure() -> None:
    """The physics-informed identification chain, in four labelled stages."""
    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.6), constrained_layout=True)
    fig.suptitle("Physics-engine survivor identification — every decision is a physical model, not a learned guess",
                 fontsize=11, color=INK, x=0.5, y=1.02)

    # 1. Radiometric signature (LWIR apparent temperature over time)
    ax = axes[0]
    t = np.linspace(0, 30, 200)
    ax.plot(t, 36.5 - 0.35 * np.sqrt(t), color=RED, lw=2.2,
            label="immersed survivor")
    ax.plot(t, 36.6 + 0.0 * t, color=AMBER, lw=2.0, ls="--",
            label="dry survivor")
    ax.plot(t, 27.0 + 0.0 * t, color=CYAN, lw=1.8, ls=":",
            label="warm roof (decoy)")
    ax.axhline(35.0, color=MUTED, lw=0.8)
    ax.text(31, 35.0, "hypothermia onset 35°C", color=MUTED, fontsize=6.5, va="center")
    ax.set_xlabel("minutes observed"); ax.set_ylabel("apparent temp °C")
    ax.set_title("1 · thermal physiology", fontsize=9, color=CYAN)
    ax.legend(fontsize=6, frameon=False, loc="lower left")
    ax.set_xlim(0, 32); ax.set_ylim(24, 38)

    # 2. Posture from shape + context
    ax = axes[1]
    water = np.linspace(0, 1.2, 50)
    elong = np.linspace(1.0, 2.0, 50)
    W, E = np.meshgrid(water, elong)
    regions = np.zeros_like(W)
    regions[(W > 0.5)] = 1                                  # in water
    regions[(W <= 0.5) & (E > 1.6)] = 2                     # prone/buried
    regions[(W <= 0.5) & (E <= 1.6) & (E > 1.35)] = 3       # lying
    regions[(W <= 0.5) & (E <= 1.35)] = 4                   # standing
    cmap = plt.matplotlib.colors.ListedColormap(
        ["#0b1220", "#1d4ed8", "#7c3aed", "#0891b2", "#16a34a"])
    ax.pcolormesh(water, elong, regions, cmap=cmap, vmin=0, vmax=4, shading="auto")
    labels = ["in-water clinging", "prone / partial burial", "lying",
              "standing / walking"]
    for i, (wd, el, lab) in enumerate([(0.85, 1.1, labels[0]),
                                        (0.2, 1.8, labels[1]),
                                        (0.2, 1.5, labels[2]),
                                        (0.2, 1.15, labels[3])]):
        ax.text(wd, el, lab, fontsize=6.5, ha="center", va="center", color="white")
    ax.set_xlabel("water depth m"); ax.set_ylabel("elongation")
    ax.set_title("2 · posture from physics", fontsize=9, color=CYAN)
    ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([1.0, 1.5, 2.0])

    # 3. Ballistic release (drag + wind)
    ax = axes[2]
    release = np.array([0.0, 0.0])
    impact_no_wind = np.array([-6.0, 0.0])
    impact_wind = np.array([-9.0, 3.2])
    xs = np.linspace(0, 1, 40)
    curve = np.array([xs * -6.0, (1 - (xs - 0.5) ** 2) * 2.0])
    ax.plot(curve[0], curve[1], color=ACCENT, lw=1.6)
    ax.scatter(*release, color=CYAN, s=40, zorder=5)
    ax.scatter(*impact_wind, color=RED, s=40, zorder=5, marker="x")
    ax.annotate("release", release + [0.5, 0.2], color=CYAN, fontsize=7)
    ax.annotate("impact (wind-corrected)", impact_wind + [0.5, 0.2],
                color=RED, fontsize=7)
    ax.add_patch(FancyArrowPatch((impact_no_wind[0], 1.0), (impact_wind[0], 1.3),
                                 arrowstyle="->", color=AMBER, lw=1.2))
    ax.text(-8.2, 1.05, "wind drift", color=AMBER, fontsize=7)
    ax.set_xlim(-11, 2); ax.set_ylim(-0.8, 2.2)
    ax.axis("off")
    ax.set_title("3 · ballistic drop (drag + wind)", fontsize=9, color=CYAN)

    # 4. A* rescue routing
    ax = axes[3]
    rng = np.random.default_rng(3)
    ax.set_facecolor("#0e1626")
    for _ in range(26):
        x, y = rng.uniform(0, 10, 2)
        ax.add_patch(FancyBboxPatch((x, y), 0.7, 0.7,
                                    boxstyle="round,pad=0.05",
                                    fc="#1d2b44", ec="none"))
    way = np.array([[0.5, 0.5], [2.1, 1.4], [3.0, 3.1], [5.0, 4.2],
                    [6.4, 5.6], [8.2, 7.0], [9.4, 9.4]])
    ax.plot(way[:, 0], way[:, 1], color=CYAN, lw=2.0, ls="--")
    ax.scatter(0.5, 0.5, color=ACCENT, s=60, zorder=5)
    ax.scatter(9.4, 9.4, color=RED, s=60, zorder=5)
    ax.annotate("GCS", (0.5, 0.2), color=ACCENT, fontsize=7)
    ax.annotate("survivor", (8.4, 9.1), color=RED, fontsize=7)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("4 · A* safe route to victim", fontsize=9, color=CYAN)

    fig.savefig(OUT / "pitch_physics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  figure pitch_physics.png")


def arch_figure() -> None:
    """The end-to-end system architecture as a layered block diagram."""
    fig, ax = plt.subplots(figsize=(12.8, 7.0))
    ax.axis("off")
    ax.set_xlim(0, 12.8); ax.set_ylim(0, 7.0)

    def box(x, y, w, h, title, sub="", fc="#111a2b", ec="#243350", tcol=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h - 0.34, title, ha="center", va="center",
                fontsize=9.5, color=tcol, weight="bold")
        if sub:
            ax.text(x + w / 2, y + h - 0.82, sub, ha="center", va="center",
                    fontsize=6.8, color=MUTED)
        return (x + w / 2, y + h / 2)

    def arrow(p0, p1, color=ACCENT, style="-|>"):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=14,
                                     color=color, lw=1.4, shrinkA=2, shrinkB=2))

    # Layer 1 - sensing
    c1 = box(0.5, 5.7, 2.6, 1.0, "RGB camera", "EO · saliency, shape", fc="#12203a")
    c2 = box(3.4, 5.7, 2.6, 1.0, "LWIR thermal", "radiometry · vitals", fc="#12203a")
    c3 = box(6.3, 5.7, 2.6, 1.0, "IMU / baro / GPS", "H743-Wing + ArduPilot", fc="#12203a")
    c4 = box(9.2, 5.7, 2.6, 1.0, "Range / LIDAR", "terrain, altitude", fc="#12203a")

    # Layer 2 - edge AI
    e1 = box(0.5, 3.9, 2.6, 1.1, "YOLO detect", "RGB + thermal backends", fc="#14213d", ec=CYAN)
    e2 = box(3.4, 3.9, 2.6, 1.1, "Physics engine", "posture · triage · decoy veto", fc="#14213d", ec=AMBER)
    e3 = box(6.3, 3.9, 2.6, 1.1, "Geo-tagging", "error ellipse, honest sigma", fc="#14213d", ec=CYAN)
    e4 = box(9.2, 3.9, 2.6, 1.1, "Hazard map", "flood · fire · collapse", fc="#14213d", ec=CYAN)

    # Layer 3 - navigation
    n1 = box(0.5, 2.1, 2.6, 1.1, "GNSS nav", "EKF source set 1", fc="#0f1d2e")
    n2 = box(3.4, 2.1, 2.6, 1.1, "GNSS-denied", "VIO + EKF3 ext-nav", fc="#0f1d2e", ec=AMBER)
    n3 = box(6.3, 2.1, 2.6, 1.1, "Belief search", "boustrophedon, POC×POD", fc="#0f1d2e")
    n4 = box(9.2, 2.1, 2.6, 1.1, "Rescue act", "drop + A* route", fc="#0f1d2e", ec=RED)

    # Layer 4 - comms + GCS
    g1 = box(0.5, 0.35, 3.4, 1.2, "ExpressLRS + LoRa", "store-and-forward, priority tiers", fc="#0d1730")
    g2 = box(4.2, 0.35, 3.4, 1.2, "MAVLink telemetry", "alerts not video", fc="#0d1730")
    g3 = box(7.9, 0.35, 3.9, 1.2, "GCS dashboard", "live + offline replay", fc="#0d1730", ec=ACCENT)

    # Vertical arrows
    for c in (c1, c2, c3, c4):
        arrow((c[0], 5.65), (c[0], 5.05))
    for i, e in enumerate((e1, e2, e3, e4)):
        arrow((e[0], 3.85), (e[0], 3.25))
        arrow((e[0], 3.85), (e[0], 3.25), color=CYAN)
    arrow((c1[0], 5.65), (e1[0], 5.05))
    # layer joins
    arrow((e1[0], 3.85), (n1[0], 3.25))
    arrow((e2[0], 3.85), (n2[0], 3.25))
    arrow((e3[0], 3.85), (n3[0], 3.25))
    arrow((e4[0], 3.85), (n4[0], 3.25))
    arrow((n1[0], 2.05), (g1[0] + 0.7, 1.6), color=CYAN)
    arrow((n2[0], 2.05), (g2[0] + 0.7, 1.6), color=CYAN)
    arrow((n3[0], 2.05), (g2[0] + 0.7, 1.6), color=CYAN)
    arrow((n4[0], 2.05), (g3[0] - 0.7, 1.6), color=RED)

    fig.savefig(OUT / "pitch_arch.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  figure pitch_arch.png")


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate stills and novelty figures for the SIH pitch deck.")
    parser.add_argument("--only", choices=["stills", "physics", "arch"],
                        help="build a single asset group")
    args = parser.parse_args(argv)
    if args.only in (None, "stills"):
        first_frames()
    if args.only in (None, "physics"):
        physics_figure()
    if args.only in (None, "arch"):
        arch_figure()
    print(f"pitch assets -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
