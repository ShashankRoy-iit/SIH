#!/usr/bin/env python3
"""Diagnose the environment before a sortie - and explain `No module named 'sar'`.

Run this first when anything refuses to start::

    python3 scripts/doctor.py
    python3 scripts/doctor.py --json     # machine-readable, for CI

It checks, in the order that things actually break:

1. Python version.
2. Whether ``import sar`` resolves to **this repository** or to the unrelated
   ``sar`` package on PyPI (the trap that turns
   ``ModuleNotFoundError: No module named 'sar'`` into
   ``ModuleNotFoundError: No module named 'sar.core'`` after ``pip install sar``).
3. Whether the project is installed editable, or is running off the path shim.
4. Core dependencies, then optional ones grouped by the capability they unlock.
5. Whether every ``sar`` subpackage imports cleanly - which catches a partial
   checkout or a syntax error long before a 10-minute mission run does.
6. Whether an ArduPilot SITL binary is on PATH (optional: MiniSITL covers it).
7. Writability of ``artifacts/``.

Exit code is 0 if a simulated sortie can run, 1 otherwise.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# --- repo-root bootstrap ---------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._bootstrap import (  # noqa: E402
    OPTIONAL_REQUIREMENTS, REPO_ROOT, _foreign_sar_installed, ensure_repo_on_path)

GREEN, RED, YELLOW, BLUE, DIM, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[34m", "\033[2m", "\033[0m")
if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    GREEN = RED = YELLOW = BLUE = DIM = RESET = ""

OK, WARN, FAIL = "ok", "warn", "fail"
MARK = {OK: f"{GREEN}PASS{RESET}", WARN: f"{YELLOW}WARN{RESET}", FAIL: f"{RED}FAIL{RESET}"}

SUBPACKAGES = [
    "sar.core.geo", "sar.core.frames", "sar.vehicle.dynamics", "sar.sim.world",
    "sar.sim.sitl", "sar.sim.scenario", "sar.perception.detector",
    "sar.perception.pipeline", "sar.perception.human_id", "sar.nav.external_nav",
    "sar.decision.coverage", "sar.comms.link", "sar.mission.runner",
    "sar.rescue.coordinator", "sar.ai.registry", "sar.ai.runtime",
    "sar.hardware.safety",
]


class Report:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def add(self, name: str, status: str, detail: str, fix: str = "") -> None:
        self.rows.append({"check": name, "status": status, "detail": detail, "fix": fix})

    @property
    def failed(self) -> List[Dict[str, Any]]:
        return [r for r in self.rows if r["status"] == FAIL]

    @property
    def warned(self) -> List[Dict[str, Any]]:
        return [r for r in self.rows if r["status"] == WARN]


def check_python(rep: Report) -> None:
    v = sys.version_info
    detail = f"{platform.python_version()} ({sys.executable})"
    if v < (3, 9):
        rep.add("python version", FAIL, detail, "Python 3.9 or newer is required.")
    else:
        rep.add("python version", OK, detail)


def check_venv(rep: Report) -> None:
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if in_venv:
        rep.add("virtualenv", OK, f"active: {sys.prefix}")
    else:
        rep.add("virtualenv", WARN, "not in a virtualenv",
                "python3 -m venv .venv && source .venv/bin/activate")


def check_sar_import(rep: Report) -> None:
    """The check this whole script exists for."""
    foreign = _foreign_sar_installed()
    ensure_repo_on_path()
    spec = importlib.util.find_spec("sar")
    if spec is None or not spec.origin:
        rep.add("import sar", FAIL, "not importable at all",
                "Run from the repository root, or `pip install -e .`")
        return
    origin = Path(spec.origin).resolve()
    is_local = str(origin).startswith(str(REPO_ROOT))
    if not is_local:
        rep.add("import sar", FAIL, f"resolves to {origin}",
                "pip uninstall -y sar   # unrelated PyPI package\n"
                "     pip install -e .       # then install THIS repo")
        return
    rep.add("import sar", OK, str(origin))
    if foreign:
        rep.add("PyPI `sar` shadow", WARN,
                f"an unrelated `sar` is installed at {foreign}",
                "pip uninstall -y sar    (the repo copy wins today, but an IDE "
                "run configuration or notebook will pick the wrong one)")
    else:
        rep.add("PyPI `sar` shadow", OK, "no conflicting distribution installed")


def check_editable_install(rep: Report) -> None:
    try:
        from importlib.metadata import distribution
        dist = distribution("sahyog-sar")
        rep.add("editable install", OK, f"sahyog-sar {dist.version}")
    except Exception:
        rep.add("editable install", WARN,
                "sahyog-sar is not installed; scripts run via the path shim",
                "pip install -e '.[sim]'   (needed for the `sar-*` commands)")


def check_dependencies(rep: Report) -> None:
    for mod in ("numpy", "scipy"):
        if importlib.util.find_spec(mod) is None:
            rep.add(f"dep: {mod}", FAIL, "missing", f"pip install {mod}")
        else:
            m = importlib.import_module(mod)
            rep.add(f"dep: {mod}", OK, getattr(m, "__version__", "?"))
    for mod, what in OPTIONAL_REQUIREMENTS.items():
        if importlib.util.find_spec(mod) is None:
            rep.add(f"optional: {mod}", WARN, f"missing - {what}",
                    "pip install -e '.[sim,ai]'")
        else:
            rep.add(f"optional: {mod}", OK, what.split("(")[0].strip())


def check_subpackages(rep: Report) -> None:
    broken: List[Tuple[str, str]] = []
    for name in SUBPACKAGES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - we want every failure listed
            broken.append((name, f"{type(exc).__name__}: {exc}"))
    if broken:
        rep.add("sar subpackages", FAIL,
                "; ".join(f"{n} -> {e}" for n, e in broken),
                "A dependency is missing, or the checkout is incomplete "
                "(`git status`, then `pip install -e '.[sim]'`).")
    else:
        rep.add("sar subpackages", OK, f"{len(SUBPACKAGES)} modules import cleanly")


def check_ardupilot(rep: Report) -> None:
    binary = shutil.which("arducopter") or shutil.which("ArduCopter.elf")
    if binary:
        rep.add("ArduPilot SITL", OK, binary)
        return
    sim_vehicle = shutil.which("sim_vehicle.py")
    if sim_vehicle:
        rep.add("ArduPilot SITL", OK, f"sim_vehicle.py at {sim_vehicle}")
        return
    rep.add("ArduPilot SITL", WARN,
            "no arducopter binary on PATH - MiniSITL will be used instead",
            "Optional. See docs/HOWTO_RUN.md section 'Running against real "
            "ArduPilot SITL' to build it.")


def check_artifacts(rep: Report) -> None:
    art = REPO_ROOT / "artifacts"
    try:
        art.mkdir(parents=True, exist_ok=True)
        probe = art / ".doctor_write_probe"
        probe.write_text("ok")
        probe.unlink()
        rep.add("artifacts/ writable", OK, str(art))
    except Exception as exc:  # pragma: no cover
        rep.add("artifacts/ writable", FAIL, repr(exc), f"chmod u+w {art}")


def check_ai_models(rep: Report) -> None:
    try:
        from sar.ai.registry import ModelRegistry
    except Exception as exc:  # pragma: no cover
        rep.add("AI model registry", FAIL, repr(exc), "pip install -e '.[ai]'")
        return
    reg = ModelRegistry()
    present = [m.name for m in reg.models() if reg.is_available(m.name)]
    if present:
        rep.add("AI weights", OK, f"available locally: {', '.join(present)}")
    else:
        rep.add("AI weights", WARN,
                "no neural weights in models/ - the audited heuristic detector "
                "is used (this is a supported, tested configuration)",
                "python scripts/fetch_models.py --model yolo11n-visdrone")


def check_port(rep: Report, port: int = 5760) -> None:
    import socket
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port))
        rep.add(f"tcp:{port} free", OK, "MiniSITL can bind")
    except OSError:
        rep.add(f"tcp:{port} free", WARN, "port in use",
                f"A previous run may still be alive: `pkill -f run_mission` "
                f"or pass --port {port + 10}")
    finally:
        s.close()


def check_git(rep: Report) -> None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=REPO_ROOT, capture_output=True, text=True, timeout=5)
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                cwd=REPO_ROOT, capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            rep.add("git checkout", OK,
                    f"{branch.stdout.strip()} @ {out.stdout.strip()}")
    except Exception:  # pragma: no cover
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose the SAHYOG environment")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    rep = Report()
    check_python(rep)
    check_venv(rep)
    check_sar_import(rep)
    check_editable_install(rep)
    check_dependencies(rep)
    check_subpackages(rep)
    check_ai_models(rep)
    check_ardupilot(rep)
    check_artifacts(rep)
    check_port(rep)
    check_git(rep)

    if args.json:
        print(json.dumps({"rows": rep.rows,
                          "ok": not rep.failed}, indent=2))
        raise SystemExit(1 if rep.failed else 0)

    print(f"\n{BLUE}SAHYOG environment check{RESET}  ({REPO_ROOT})\n")
    width = max(len(r["check"]) for r in rep.rows)
    for r in rep.rows:
        print(f"  {MARK[r['status']]}  {r['check']:<{width}}  {r['detail']}")
    if rep.warned or rep.failed:
        print(f"\n{BLUE}What to do{RESET}")
        for r in rep.failed + rep.warned:
            if r["fix"]:
                print(f"  {MARK[r['status']]}  {r['check']}:\n     {r['fix']}")
    if rep.failed:
        print(f"\n{RED}Not ready.{RESET} Fix the FAIL rows above, then re-run "
              f"`python3 scripts/doctor.py`.\n")
        raise SystemExit(1)
    print(f"\n{GREEN}Ready.{RESET} Try:\n"
          "  python3 scripts/run_rescue_simulation.py --scenario flood "
          "--duration 60 --speedup 2.0\n")


if __name__ == "__main__":
    main()
