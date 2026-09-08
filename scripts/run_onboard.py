#!/usr/bin/env python3
"""Run the autonomy stack on the aircraft (or rehearse it without hardware).

    # laptop, no hardware, full flight code path
    python3 scripts/run_onboard.py --dry-run --duration 60

    # bench: real autopilot over USB, simulated cameras, props OFF
    python3 scripts/run_onboard.py --mode hitl --target /dev/ttyACM0:921600

    # aircraft, on the ground, sensors live - the sortie rehearsal
    python3 scripts/run_onboard.py --mode flight --config configs/onboard.yaml

    # preflight only, then exit with 0/1 - what the systemd unit calls first
    python3 scripts/run_onboard.py --mode flight --preflight-only

What it does *not* do is arm and fly itself.  That is gated behind
``docs/FIELD_TEST_CHECKLIST.md`` and WP 7.8, and the code says so rather than
quietly offering a button nobody has flight-tested.  Everything up to that
line - perception on live cameras, geo-tagging from live EKF telemetry, triage,
store-and-forward reporting, the safety supervisor and the payload servo path -
runs here, which is exactly the part that needs hours of ground time before a
first flight.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# --- repo-root bootstrap ---------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._bootstrap import bootstrap  # noqa: E402

bootstrap()

from sar.hardware.onboard import OnboardAutonomy, load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="SAHYOG onboard autonomy")
    ap.add_argument("--config", default=None, help="configs/onboard.yaml")
    ap.add_argument("--mode", choices=("dry-run", "hitl", "flight"), default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="shorthand for --mode dry-run")
    ap.add_argument("--target", help="MAVLink target, e.g. /dev/ttyACM0:921600 "
                                     "or udpin:0.0.0.0:14550")
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--detector", choices=("auto", "hybrid", "heuristic", "neural"))
    ap.add_argument("--perception-hz", type=float, default=None)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--allow-non-radiometric", action="store_true",
                    help="fly with an 8-bit AGC thermal camera; triage physiology "
                         "is then disabled rather than fabricated")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S")

    cfg = load_config(args.config)
    if args.dry_run:
        cfg["mode"] = "dry-run"
    if args.mode:
        cfg["mode"] = args.mode
    if args.target:
        cfg["mavlink"]["target"] = args.target
    if args.detector:
        cfg["detector"] = args.detector
    if args.perception_hz:
        cfg["perception_hz"] = args.perception_hz
    cfg["artifacts_dir"] = args.artifacts

    node = OnboardAutonomy(cfg)
    node.connect()
    node.build_perception()
    node.build_uplink()

    ok, failures = node.preflight(
        require_radiometric=not args.allow_non_radiometric)
    print("\nPREFLIGHT")
    print(json.dumps(node.report.preflight, indent=2, default=str))
    if args.preflight_only:
        raise SystemExit(0 if ok else 1)
    if not ok and cfg["mode"] == "flight":
        print("\nRefusing to run the sortie loop with failed pre-flight checks.\n"
              "Fix them, or run with --mode hitl to exercise the parts that do work.")
        raise SystemExit(1)

    report = node.run(duration_s=args.duration, arm_and_fly=False)

    print("\nSORTIE SUMMARY")
    print(f"  mode        {report.mode}")
    print(f"  duration    {report.duration_s:.0f} s")
    print(f"  perception  {report.perception.get('cycles', 0)} cycles, "
          f"{report.perception.get('mean_ms', 0)} ms mean, "
          f"{report.perception.get('rate_hz', 0)} Hz")
    cam = report.cameras.get("sync", {}) if report.cameras else {}
    print(f"  cameras     {cam.get('pairs', 0)} pairs, "
          f"{cam.get('mean_skew_ms', 0)} ms mean skew, "
          f"{cam.get('rejected_skew', 0)} rejected")
    print(f"  survivors   {len(report.survivors)}")
    print(f"  safety      {report.safety.get('state')} "
          f"{report.safety.get('triggered_rules')}")
    if report.faults:
        print(f"  faults      {len(report.faults)}")
        for f in report.faults[:5]:
            print(f"    - {f}")
    print(f"\n  report -> {Path(args.artifacts) / 'onboard_report.json'}\n")


if __name__ == "__main__":
    main()
