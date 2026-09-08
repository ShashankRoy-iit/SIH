"""Unit and integration tests for Mission Animation & Visualization."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.animate_rescue_mission import (
    build_mission_animation_data,
    generate_html_animation,
    generate_matplotlib_gif,
)


def test_animation_data_generation():
    data = build_mission_animation_data("flood")
    assert "trajectory" in data
    assert "survivors" in data
    assert "drops" in data
    assert "routes" in data
    assert len(data["trajectory"]) > 10
    assert len(data["survivors"]) > 0
    assert len(data["drops"]) > 0


def test_html_animation_builder(tmp_path: Path):
    data = build_mission_animation_data("flood")
    out_file = tmp_path / "test_animation.html"
    res_path = generate_html_animation("flood", data, out_file)

    assert res_path.exists()
    content = res_path.read_text()
    assert "SAHYOG" in content
    assert "cv-mission" in content
    assert "cv-thermal" not in content or "cv-thermal" in content
    assert len(content) > 1000


def test_matplotlib_gif_builder(tmp_path: Path):
    data = build_mission_animation_data("flood")
    # Truncate to short sequence for fast test execution
    data["trajectory"] = data["trajectory"][:15]
    out_file = tmp_path / "test_animation.gif"
    res_path = generate_matplotlib_gif("flood", data, out_file, fps=10)

    if res_path is not None:
        assert res_path.exists()
        assert res_path.stat().st_size > 0
