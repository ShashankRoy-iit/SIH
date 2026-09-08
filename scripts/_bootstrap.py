"""Make ``python scripts/<anything>.py`` work from a fresh clone.

Why this file exists
--------------------
Running ``python3 scripts/run_rescue_simulation.py`` puts **the script's own
directory** (``scripts/``) at the front of ``sys.path`` - not the repository
root.  The ``sar`` package lives at the repository root, so the import fails::

    ModuleNotFoundError: No module named 'sar'

The usual reflex is ``pip install sar``.  That is actively harmful here: there
is an unrelated project on PyPI called ``sar`` (a synthetic-aperture-radar
helper), and installing it makes the error *change* rather than go away::

    ModuleNotFoundError: No module named 'sar.core'

...because ``import sar`` now resolves to the PyPI package, which has no
``core`` submodule.  This module fixes both halves of that trap:

1. :func:`ensure_repo_on_path` puts the repository root first on ``sys.path``,
   so the sortie runs straight out of a clone with no install step at all;
2. :func:`assert_local_sar` checks that the ``sar`` that actually got imported
   is *this* repository's package, and if it is not, it prints the exact
   command to undo the mistake.

Nothing here is needed once the project is installed with
``pip install -e .`` - it is idempotent and costs a few microseconds, and it
means a teammate cloning the repo at 2 a.m. before a review can just run the
script.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

#: Repository root - the directory that contains ``sar/``, ``scripts/`` and
#: ``pyproject.toml``.
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Third-party modules the simulation cannot run without, mapped to the pip
#: name that provides them (import name != distribution name for Pillow etc.).
CORE_REQUIREMENTS = {
    "numpy": "numpy",
    "scipy": "scipy",
}

#: Modules that only some entry points need.  Missing ones are reported as a
#: hint rather than an error.
OPTIONAL_REQUIREMENTS = {
    "pymavlink": "pymavlink        (real ArduPilot SITL / hardware MAVLink)",
    "matplotlib": "matplotlib       (GIF/PNG figure export)",
    "fastapi": "fastapi uvicorn  (live command-centre dashboard, --live)",
    "onnxruntime": "onnxruntime      (neural detector backend, sar.ai)",
    "cv2": "opencv-python    (real camera capture, sar.hardware)",
}

_BANNER = "=" * 78


def ensure_repo_on_path() -> Path:
    """Put the repository root at the front of ``sys.path``.  Idempotent.

    Front, not back, deliberately: if a stale ``sar`` distribution is installed
    in the active environment, the repository copy must win, otherwise the code
    under test is not the code in the working tree.
    """
    root = str(REPO_ROOT)
    while root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    return REPO_ROOT


def _foreign_sar_installed() -> str | None:
    """Return the path of an installed, non-repository ``sar`` package, if any."""
    for entry in sys.path:
        if not entry:
            continue
        try:
            entry_path = Path(entry).resolve()
        except (TypeError, ValueError, OSError):  # pragma: no cover
            continue
        if entry_path == REPO_ROOT:
            continue
        candidate = entry_path / "sar"
        if (candidate / "__init__.py").is_file():
            return str(candidate)
        flat = entry_path / "sar.py"
        if flat.is_file():
            return str(flat)
    return None


def assert_local_sar() -> None:
    """Fail loudly and usefully if ``import sar`` would hit the wrong package."""
    spec = importlib.util.find_spec("sar")
    if spec is None or not spec.origin:
        return  # ensure_repo_on_path() has not run yet, or a namespace package
    origin = Path(spec.origin).resolve()
    try:
        origin.relative_to(REPO_ROOT)
    except ValueError:
        foreign = _foreign_sar_installed() or str(origin)
        sys.stderr.write(
            f"\n{_BANNER}\n"
            "WRONG `sar` PACKAGE ON THE PATH\n"
            f"{_BANNER}\n"
            f"  `import sar` resolves to : {foreign}\n"
            f"  it should resolve to     : {REPO_ROOT / 'sar'}\n\n"
            "There is an unrelated project called `sar` on PyPI (synthetic\n"
            "aperture radar).  If you ran `pip install sar` to fix a\n"
            "ModuleNotFoundError, undo it:\n\n"
            "    pip uninstall -y sar\n"
            "    pip install -e .        # installs THIS repo as `sar`\n\n"
            f"{_BANNER}\n\n")
        raise SystemExit(2)


def check_requirements(extra: Sequence[str] = (), *, strict: bool = True) -> List[str]:
    """Report missing dependencies with an actionable message.

    Returns the list of missing *core* requirements.  With ``strict`` the
    process exits rather than dying later with a bare ``ImportError`` half way
    through a mission.
    """
    missing: List[str] = []
    for module, pip_name in {**CORE_REQUIREMENTS,
                             **{m: m for m in extra}}.items():
        if importlib.util.find_spec(module) is None:
            missing.append(pip_name)
    if missing and strict:
        sys.stderr.write(
            f"\n{_BANNER}\n"
            "MISSING DEPENDENCIES\n"
            f"{_BANNER}\n"
            f"  cannot import: {', '.join(missing)}\n\n"
            "  python3 -m venv .venv && source .venv/bin/activate\n"
            "  pip install -e '.[sim]'\n\n"
            f"{_BANNER}\n\n")
        raise SystemExit(2)
    return missing


def optional_status() -> dict:
    """Which optional capabilities are available in this interpreter."""
    return {module: importlib.util.find_spec(module) is not None
            for module in OPTIONAL_REQUIREMENTS}


def warn_if_foreign_sar_installed() -> None:
    """Non-fatal notice: a PyPI ``sar`` is installed but the repo copy wins.

    The repo copy winning is exactly what we want, but leaving the foreign
    distribution installed will bite the next person who runs a tool that does
    not go through this bootstrap (a notebook, an IDE run configuration), so it
    is worth one line on stderr.
    """
    if os.environ.get("SAR_SUPPRESS_SHADOW_WARNING"):
        return
    foreign = _foreign_sar_installed()
    if foreign:
        sys.stderr.write(
            f"[sar] note: an unrelated `sar` package is installed at {foreign}; "
            "the repository copy is being used. Run `pip uninstall -y sar` to "
            "remove the ambiguity (see docs/HOWTO_RUN.md).\n")


def bootstrap(extra_requirements: Iterable[str] = (), *, strict: bool = True) -> Path:
    """The one call every script makes before importing :mod:`sar`."""
    root = ensure_repo_on_path()
    assert_local_sar()
    warn_if_foreign_sar_installed()
    check_requirements(tuple(extra_requirements), strict=strict)
    # Deterministic, headless, and never blocked on a display server: a sortie
    # that opens a window on a CI runner is a sortie that hangs.
    os.environ.setdefault("MPLBACKEND", "Agg")
    return root


__all__ = [
    "REPO_ROOT",
    "assert_local_sar",
    "bootstrap",
    "check_requirements",
    "ensure_repo_on_path",
    "optional_status",
]
