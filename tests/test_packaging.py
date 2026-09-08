"""Packaging and entry points: the failure the user actually hit.

``ModuleNotFoundError: No module named 'sar'`` when running
``python3 scripts/run_rescue_simulation.py`` from the repository root, followed
by ``pip install sar`` making it worse (that PyPI package is unrelated, and the
error then becomes ``No module named 'sar.core'``).

These tests assert that the fix stays fixed:

* every script carries the bootstrap and can be executed directly from a clone;
* the bootstrap puts the repository first on ``sys.path``, so a stale installed
  ``sar`` cannot shadow the working tree;
* the distribution is *not* named ``sar``, so nobody can install the wrong one
  on top of us by accident.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = sorted(p for p in (REPO / "scripts").glob("*.py")
                 if p.name not in ("__init__.py", "_bootstrap.py"))


def test_there_are_scripts_to_check():
    assert SCRIPTS, "no scripts found - has the layout changed?"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_every_script_bootstraps_before_importing_sar(script: Path):
    """A script that imports `sar` before the shim will fail from a clone."""
    tree = ast.parse(script.read_text())
    bootstrap_line = sar_import_line = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("scripts._bootstrap") and bootstrap_line is None:
                bootstrap_line = node.lineno
            if node.module.split(".")[0] == "sar" and sar_import_line is None:
                sar_import_line = node.lineno
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "sar" and sar_import_line is None:
                    sar_import_line = node.lineno
    if sar_import_line is None:
        pytest.skip(f"{script.name} does not import sar")
    assert bootstrap_line is not None, (
        f"{script.name} imports sar without the repo-root bootstrap; running it "
        f"as `python3 scripts/{script.name}` from a clone would raise "
        f"ModuleNotFoundError")
    assert bootstrap_line < sar_import_line


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_every_script_runs_its_help_from_the_repo_root(script: Path):
    """The end-to-end version of the check above: actually execute it."""
    proc = subprocess.run([sys.executable, str(script.relative_to(REPO)), "--help"],
                          cwd=REPO, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, (
        f"{script.name} --help failed:\n{proc.stdout}\n{proc.stderr}")
    assert "usage" in (proc.stdout + proc.stderr).lower()


def test_bootstrap_puts_the_repository_first_on_the_path():
    from scripts._bootstrap import REPO_ROOT, ensure_repo_on_path
    ensure_repo_on_path()
    assert sys.path[0] == str(REPO_ROOT)


def test_imported_sar_is_this_repository():
    import sar
    assert Path(sar.__file__).resolve().is_relative_to(REPO)


def test_distribution_is_not_named_sar():
    """`name = "sar"` in pyproject would re-create the shadowing trap."""
    text = (REPO / "pyproject.toml").read_text()
    assert 'name = "sahyog-sar"' in text
    assert '\nname = "sar"' not in text


def test_console_scripts_point_at_real_callables():
    import importlib
    import re
    text = (REPO / "pyproject.toml").read_text()
    block = text.split("[project.scripts]", 1)[1].split("[", 1)[0]
    entries = re.findall(r'^\s*[\w-]+\s*=\s*"([\w\.]+):(\w+)"', block, re.M)
    assert entries, "no console scripts declared"
    for module_name, attr in entries:
        module = importlib.import_module(module_name)
        assert callable(getattr(module, attr)), f"{module_name}:{attr}"


def test_requirements_files_exist_and_are_not_empty():
    for name in ("requirements.txt", "requirements-ai.txt"):
        p = REPO / name
        assert p.is_file() and p.read_text().strip()


def test_doctor_runs_and_reports_json():
    import json
    proc = subprocess.run([sys.executable, "scripts/doctor.py", "--json"],
                          cwd=REPO, capture_output=True, text=True, timeout=180)
    payload = json.loads(proc.stdout)
    checks = {row["check"]: row for row in payload["rows"]}
    assert checks["import sar"]["status"] == "ok"
    assert checks["sar subpackages"]["status"] == "ok", checks["sar subpackages"]
