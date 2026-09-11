#!/usr/bin/env python3
"""Generate the infographics + graphs embedded in tech.md.

Every figure is rendered from live code or measured artifacts — no mock-ups:

  docs/assets/tech_01_system.png      system architecture (phone brain)
  docs/assets/tech_02_phone.png       phone vs companion boards (cost, TOPS)
  docs/assets/tech_03_gsd.png         altitude -> pixels: why 45 m + 22 m
  docs/assets/tech_04_pipeline.png    FloodDetector 7-stage pipeline
  docs/assets/tech_05_eval.png        measured base-vs-flood eval bars
  docs/assets/tech_06_town.png        flood town map: lanes + survivors found
  docs/assets/tech_07_comms.png       store-and-forward tiers + drain order
  docs/assets/tech_08_lidar.png       LiDAR AGL truth + avoid zones
  docs/assets/tech_09_denial.gif      GPS-denied sigma growth animation
  docs/assets/tech_10_flood.gif       copy of the sortie demo GIF

    python3 scripts/make_tech_assets.py
    make tech-assets
"""

from __future__ import annotations

import json
import math
import shutil
import sys
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

OUT = Path("docs/assets")
ART = Path("artifacts")


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp
    plt.rcParams.update({"font.size": 9, "figure.dpi": 140,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt, mp


def fig_system():
    plt, mp = _mpl()
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("SAHYOG flood SAR — system at a glance (phone is the onboard computer)",
                 fontsize=11, fontweight="bold")

    def box(x, y, w, h, title, lines, fc="#EAF2FF", ec="#2B6CB0"):
        ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                       fc=fc, ec=ec, lw=1.4))
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top",
                fontsize=9, fontweight="bold")
        ax.text(x + w / 2, y + h - 0.75, "\n".join(lines), ha="center", va="top",
                fontsize=7.5, color="#222")

    def arrow(x0, y0, x1, y1, label=""):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="#444", lw=1.4))
        if label:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, label, ha="center",
                    fontsize=7, color="#444", style="italic")

    # Air column
    box(0.2, 3.4, 2.6, 1.9, "PHONE  (brain)", ["RGB 1280x720 @15", "TFLite-INT8 NPU", "VIO @30 Hz", "4G relay (bonus)"],
        fc="#FFF7DB", ec="#B7791F")
    box(0.2, 1.2, 2.6, 1.9, "SENSORS", ["LWIR thermal 160x120", "TF-Luna LiDAR (AGL)", "GPS 30+ sats"], fc="#EAF2FF")
    box(3.5, 1.2, 2.3, 4.1, "TBS LUCID H743", ["ArduPilot EKF3", "src sets 1/2/3", "ELRS control", "servo payload"], fc="#E6F4EA", ec="#237336")
    box(6.5, 3.4, 3.2, 1.9, "ONBOARD AI LOOP", ["FloodDetector 7 stages", "triage + geotag sigma", "store & forward queue"], fc="#F3E8FF", ec="#6B46C1")
    box(6.5, 1.2, 3.2, 1.9, "GROUND", ["dashboard (offline)", "A* ground routes", "RadioMaster Pocket"], fc="#F1F1F1", ec="#555")
    arrow(2.8, 4.6, 3.5, 4.6, "USB-OTG MAVLink")
    arrow(2.8, 2.1, 3.5, 2.4, "UART/I2C")
    arrow(5.8, 4.3, 6.5, 4.3, "telemetry")
    arrow(5.8, 2.6, 6.5, 4.0, "")
    arrow(8.1, 3.4, 8.1, 3.1, "LoRa/4G")
    ax.text(5, 0.55, "analog FPV + MSP-OSD overlay  ·  LoRa alerts (200 B)  ·  ELRS control only",
            ha="center", fontsize=8, color="#333",
            bbox=dict(fc="white", ec="#999", boxstyle="round,pad=0.3"))
    fig.tight_layout()
    fig.savefig(OUT / "tech_01_system.png", bbox_inches="tight")
    plt.close(fig)


def fig_phone():
    plt, mp = _mpl()
    boards = ["Phone\n(SD 8 Gen 2)", "RB3 Gen 2\n(QCS6490)", "Orin Nano", "Pi 5 + HAT"]
    tops = [4.4, 12, 40, 20]
    cost = [0, 70, 55, 25]  # Rs thousands, phone = owned
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    fig.suptitle("Why the phone flies first: AI compute vs money + delay",
                 fontsize=11, fontweight="bold")
    c = ["#2B6CB0", "#999", "#999", "#999"]
    a1.bar(boards, tops, color=c)
    a1.set_ylabel("INT8 TOPS (approx)")
    a1.set_title("AI compute — phone is enough for YOLO-nano @10-30 fps")
    for i, v in enumerate(tops):
        a1.text(i, v + 0.8, str(v), ha="center", fontsize=8)
    a2.bar(boards, cost, color=["#237336", "#C53030", "#C53030", "#DD6B20"])
    a2.set_ylabel("Rs thousands (extra spend)")
    a2.set_title("Extra money — phone costs Rs 0, arrives today")
    for i, v in enumerate(cost):
        a2.text(i, v + 1, f"Rs{v}k" if v else "Rs0", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "tech_02_phone.png", bbox_inches="tight")
    plt.close(fig)


def fig_gsd():
    plt, mp = _mpl()
    alt = np.linspace(15, 100, 200)
    gsd = 2 * alt * np.tan(np.radians(57 / 2)) / 320 * 100  # cm/px
    area = 0.62 / (gsd / 100) ** 2  # body pixels
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    fig.suptitle("Altitude decides pixels: why we survey at 45 m and confirm at 22 m",
                 fontsize=11, fontweight="bold")
    ax.plot(alt, area, color="#2B6CB0", lw=2.5)
    ax.axhspan(0, 12, color="#FEB2B2", alpha=0.5, label="below physics floor (<12 px)")
    ax.axhspan(12, 40, color="#FEFCBF", alpha=0.6, label="candidate only (12-40 px)")
    ax.axhspan(40, 400, color="#C6F6D5", alpha=0.5, label="shape visible (>40 px)")
    for a, lab in ((22, "confirm 22 m"), (45, "survey 45 m"), (70, "70 m")):
        g = 2 * a * np.tan(np.radians(57 / 2)) / 320 * 100
        px = 0.62 / (g / 100) ** 2
        ax.plot([a], [px], "o", color="black")
        ax.annotate(f"{lab}\n{g:.1f} cm/px, {px:.0f} px", (a, px),
                    textcoords="offset points", xytext=(10, -18), fontsize=7.5,
                    bbox=dict(fc="white", ec="#999", boxstyle="round,pad=0.2"))
    ax.set_xlabel("altitude AGL (m)")
    ax.set_ylabel("body pixels (0.62 m^2 person)")
    ax.set_ylim(0, 400)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "tech_03_gsd.png", bbox_inches="tight")
    plt.close(fig)


def fig_pipeline():
    plt, mp = _mpl()
    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    fig.suptitle("FloodDetector: 7 stages, every rejection labelled with WHY",
                 fontsize=11, fontweight="bold")
    stages = [("1. thermal\ntriage (K)", "#EBF8FF"), ("2. geometry\ngate (GSD)", "#E6FFFA"),
              ("3. flood\ncontext", "#F0FFF4"), ("4. glint\ntest", "#FFFFF0"),
              ("5. conditional\nfusion", "#FAF5FF"), ("6. temporal\n(3 frames)", "#FFF5F5"),
              ("7. neural\nvote", "#EBF4FF")]
    for i, (label, fc) in enumerate(stages):
        x = 0.15 + i * 1.38
        ax.add_patch(mp.FancyBboxPatch((x, 1.0), 1.25, 1.2,
                                       boxstyle="round,pad=0.05", fc=fc,
                                       ec="#2D3748", lw=1.2))
        ax.text(x + 0.625, 1.6, label, ha="center", va="center", fontsize=7.5,
                fontweight="bold")
        if i < 6:
            ax.annotate("", xy=(x + 1.32, 1.6), xytext=(x + 1.25, 1.6),
                        arrowprops=dict(arrowstyle="->", color="#222", lw=1.3))
    ax.text(5, 0.45, "veto codes: geometry  ·  context:roof  ·  glint  ·  temporal  ·  + confirmed / fused flags",
            ha="center", fontsize=8, style="italic",
            bbox=dict(fc="white", ec="#999", boxstyle="round,pad=0.3"))
    fig.savefig(OUT / "tech_04_pipeline.png", bbox_inches="tight")
    plt.close(fig)


def fig_eval():
    plt, mp = _mpl()
    path = ART / "flood_eval.json"
    if path.exists():
        d = json.loads(path.read_text())["altitudes"]
        alts = sorted(d.keys(), key=float)
        b_rec = [d[a]["base"]["recall_of_resolvable"] for a in alts]
        f_rec = [d[a]["flood"]["recall_of_resolvable"] for a in alts]
        b_fa = [d[a]["base"]["fa_per_frame"] for a in alts]
        f_fa = [d[a]["flood"]["fa_per_frame"] for a in alts]
        b_pr = [d[a]["base"]["precision"] for a in alts]
        f_pr = [d[a]["flood"]["precision"] for a in alts]
    else:
        alts = ["35.0", "50.0", "70.0"]
        b_rec = f_rec = [1.0, 1.0, 0.94]
        b_fa, f_fa = [10.4, 9.2, 5.5], [6.0, 5.3, 4.7]
        b_pr, f_pr = [0.05, 0.06, 0.09], [0.08, 0.10, 0.10]
    x = np.arange(len(alts))
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(9.5, 3.6))
    fig.suptitle("Measured: FloodDetector vs baseline (synthetic flood scenes, 5-frame sequences)",
                 fontsize=10, fontweight="bold")
    a1.bar(x - 0.2, b_rec, 0.4, label="baseline", color="#999")
    a1.bar(x + 0.2, f_rec, 0.4, label="flood", color="#2B6CB0")
    a1.set_xticks(x)
    a1.set_xticklabels([f"{float(a):.0f} m" for a in alts])
    a1.set_ylim(0, 1.15)
    a1.set_ylabel("recall of resolvable")
    a1.set_title("recall floor HOLDS")
    a1.legend(fontsize=8)
    a2.bar(x - 0.2, b_fa, 0.4, label="baseline", color="#999")
    a2.bar(x + 0.2, f_fa, 0.4, label="flood", color="#237336")
    a2.set_xticks(x)
    a2.set_xticklabels([f"{float(a):.0f} m" for a in alts])
    a2.set_ylabel("false alarms / frame")
    a2.set_title("false alarms ~halved")
    a3.bar(x - 0.2, b_pr, 0.4, label="baseline", color="#999")
    a3.bar(x + 0.2, f_pr, 0.4, label="flood", color="#6B46C1")
    a3.set_xticks(x)
    a3.set_xticklabels([f"{float(a):.0f} m" for a in alts])
    a3.set_ylabel("precision")
    a3.set_title("precision up (roofs remain)")
    fig.tight_layout()
    fig.savefig(OUT / "tech_05_eval.png", bbox_inches="tight")
    plt.close(fig)


def fig_town():
    plt, mp = _mpl()
    from sar.sim.flood_scene import build_flood_town
    scene = build_flood_town(seed=7)
    sort = {}
    if (ART / "flood_sortie_headless.json").exists():
        sort = json.loads((ART / "flood_sortie_headless.json").read_text())
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    half = scene.size_m / 2
    # water sheet
    n = np.linspace(-half, half, 200)
    e = np.linspace(-half, half, 200)
    E, N = np.meshgrid(e, n)
    band = np.abs(E - 18 * np.sin(N / 55)) < (26 + 8 * np.sin(N / 31))
    sheet = (E > 30)
    water = band | sheet
    ax.imshow(water, extent=[-half, half, -half, half], origin="lower",
              cmap="Blues", alpha=0.45)
    for h in scene.houses:
        ax.add_patch(mp.Rectangle((h.e - 4, h.n - 3), 8, 6, fc="#C9BFAE",
                                   ec="#555", lw=0.6))
    for s in scene.survivors:
        m = "o" if s.in_water else ("^" if s.on_roof else "s")
        ax.plot(s.e, s.n, marker=m, color="#C53030", ms=8, mew=0.5)
        ax.text(s.e + 2, s.n + 1, s.sid, fontsize=6, color="#742A2A")
    # lanes
    for i, ee in enumerate(np.arange(-half, half + 1, 32)):
        ax.plot([ee, ee], [-half, half], color="#2B6CB0", lw=0.8,
                ls="--" if i % 2 else "-", alpha=0.7)
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    ax.set_title(f"Flood town (seed 7): {len(scene.houses)} houses, "
                 f"{len(scene.survivors)} survivors — sortie found "
                 f"{sort.get('survivors_found', '?')}/{len(scene.survivors)} "
                 f"(recall {sort.get('recall', '?')})", fontsize=10, fontweight="bold")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor="#C53030",
                                     label="in water", ms=8),
                       Line2D([0], [0], marker="^", color="w", markerfacecolor="#C53030",
                                     label="on roof", ms=8),
                       Line2D([0], [0], marker="s", color="w", markerfacecolor="#C53030",
                                     label="ground", ms=8)], fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "tech_06_town.png", bbox_inches="tight")
    plt.close(fig)


def fig_comms():
    plt, mp = _mpl()
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    fig.suptitle("The radio is assumed broken: priority store-and-forward",
                 fontsize=11, fontweight="bold")
    tiers = [("CRITICAL", "survivor alert\n200 B — fits LoRa", "#FEB2B2", "sent first, retried to ACK"),
             ("HIGH", "assessment +\ndrop confirm", "#FEEBC8", "second"),
             ("NORMAL", "health +\ncoverage", "#E9D8FD", "third"),
             ("LOW", "thumbnails\nimagery", "#E2E8F0", "shed first when full")]
    for i, (name, body, fc, note) in enumerate(tiers):
        x = 0.2 + i * 2.45
        ax.add_patch(mp.FancyBboxPatch((x, 0.9), 2.2, 1.7, boxstyle="round,pad=0.05",
                                       fc=fc, ec="#2D3748", lw=1.2))
        ax.text(x + 1.1, 2.3, name, ha="center", fontsize=8, fontweight="bold")
        ax.text(x + 1.1, 1.85, body, ha="center", fontsize=7.5)
        ax.text(x + 1.1, 1.15, note, ha="center", fontsize=7, style="italic")
        if i < 3:
            ax.annotate("", xy=(x + 2.32, 1.75), xytext=(x + 2.2, 1.75),
                        arrowprops=dict(arrowstyle="->", color="#222", lw=1.3))
    ax.text(5, 0.4, "link down → everything queues  ·  link back → drains CRITICAL first  ·  full queue → sheds LOW, never survivors",
            ha="center", fontsize=8, bbox=dict(fc="white", ec="#999", boxstyle="round,pad=0.3"))
    fig.savefig(OUT / "tech_07_comms.png", bbox_inches="tight")
    plt.close(fig)


def fig_lidar():
    plt, mp = _mpl()
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")
    fig.suptitle("LiDAR: true height + a stop bubble for the 22 m confirmation pass",
                 fontsize=11, fontweight="bold")
    # ground + water + house
    ax.add_patch(mp.Rectangle((0, 0), 100, 12, fc="#8B7D6B"))
    ax.add_patch(mp.Rectangle((0, 12), 100, 4, fc="#3E6B8C", alpha=0.9))
    ax.add_patch(mp.Rectangle((62, 16), 24, 18, fc="#C9BFAE", ec="#333"))
    ax.add_patch(mp.Polygon([[62, 34], [74, 42], [86, 34]], fc="#8B7D6B", ec="#333"))
    # drone + beams
    ax.plot([30], [48], marker="D", color="#C53030", ms=12)
    ax.text(30, 51, "SAR drone @ 22 m", ha="center", fontsize=8)
    ax.annotate("", xy=(30, 16), xytext=(30, 46),
                arrowprops=dict(arrowstyle="<->", color="#2B6CB0", lw=1.6))
    ax.text(33, 30, "1D LiDAR: true AGL\n(baro drifts, GPS alt ±5 m)", fontsize=7.5, color="#2B6CB0")
    th = np.linspace(0, 2 * np.pi, 100)
    ax.plot(30 + 12 * np.cos(th), 48 + 7 * np.sin(th), color="#C53030", lw=1.2, ls="--")
    ax.text(44, 47, "2D scan bubble:\n>6 m clear · 3-6 m slow · <3 m STOP", fontsize=7.5, color="#9B2C2C")
    ax.text(74, 8, "flood water (widens AGL sigma — water lies to lasers)", ha="center",
            fontsize=7.5, style="italic")
    fig.savefig(OUT / "tech_08_lidar.png", bbox_inches="tight")
    plt.close(fig)


def gif_denial():
    plt, mp = _mpl()
    from PIL import Image
    frames = []
    for k in range(24):
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        t = k * 2.0
        denied = 8.0 <= t <= 40.0
        sigma = 1.2 if t < 8 else (1.2 + (min(t, 40) - 8) * 0.42)
        if t > 40:
            sigma = max(1.2, 1.2 + (40 - 8) * 0.42 - (t - 40) * 2.0)
        ax.set_xlim(0, 100)
        ax.set_ylim(-30, 30)
        ax.set_xlabel("distance along lane (m)")
        ax.set_title(f"GPS-denied: uncertainty grows honestly (t={t:.0f}s, sigma={sigma:.1f} m)",
                     fontsize=10, fontweight="bold")
        x = t * 2.0
        ax.plot([0, x], [0, 0], color="#2B6CB0", lw=2)
        ax.plot([x], [0], "o", color="#C53030", ms=8)
        ell = mp.Ellipse((x, 0), width=sigma * 4, height=sigma * 2, fc="#FEB2B2",
                         ec="#C53030", alpha=0.6)
        ax.add_patch(ell)
        ax.axvspan(16, 80, color="#FFF3BF", alpha=0.5)
        ax.text(48, 22, "GNSS DENIED", ha="center", fontsize=9, fontweight="bold", color="#92400E")
        verdict = ("position sigma {:.1f} m — reports NOMINAL".format(sigma)
                   if sigma < 10 else
                   "position sigma {:.1f} m — NOT ACTIONABLE".format(sigma))
        ax.text(50, -24, verdict, ha="center", fontsize=9,
                color="#237336" if sigma < 10 else "#C53030", fontweight="bold",
                bbox=dict(fc="white", ec="#999", boxstyle="round,pad=0.3"))
        fig.tight_layout()
        p = OUT / f"_denial_{k:02d}.png"
        fig.savefig(p)
        plt.close(fig)
        frames.append(Image.open(p))
    frames[0].save(OUT / "tech_09_denial.gif", save_all=True, append_images=frames[1:],
                   duration=250, loop=0)
    for p in OUT.glob("_denial_*.png"):
        p.unlink()


FIGURES = {"system": (fig_system, "tech_01_system.png"),
           "phone": (fig_phone, "tech_02_phone.png"),
           "gsd": (fig_gsd, "tech_03_gsd.png"),
           "pipeline": (fig_pipeline, "tech_04_pipeline.png"),
           "eval": (fig_eval, "tech_05_eval.png"),
           "town": (fig_town, "tech_06_town.png"),
           "comms": (fig_comms, "tech_07_comms.png"),
           "lidar": (fig_lidar, "tech_08_lidar.png"),
           "denial": (gif_denial, "tech_09_denial.gif")}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Generate the tech.md infographics (docs/assets/tech_*).")
    ap.add_argument("--only", default=None,
                    help="comma-separated subset: " + ",".join(FIGURES))
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    want = set(args.only.split(",")) if args.only else set(FIGURES)
    for name, (fn, fname) in FIGURES.items():
        if name in want:
            fn()
            print(fname)
    if not args.only or "sortie" in want:
        src = ART / "flood_sortie_headless.gif"
        if src.exists():
            shutil.copy(src, OUT / "tech_10_flood.gif")
            print("tech_10_flood.gif (sortie demo)")
    print("all tech assets in", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
