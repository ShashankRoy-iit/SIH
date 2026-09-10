#!/usr/bin/env python3
"""Basic repo structure tests."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_base_has_model():
    assert os.path.isfile("base/models/yolov8n_person_thermal.onnx")

def test_plan_has_no_sim_files():
    # Plan should not contain any actual simulation launchers
    for root, dirs, files in os.walk("plan"):
        for f in files:
            assert not ("run_simulation" in f and f.endswith(".py")), f"Simulation launcher in plan: {f}"

def test_emulator_has_gazebo():
    assert os.path.isfile("emulator/gazebo_worlds/rescue_flood.world")

def test_docs_separate():
    assert os.path.isdir("docs")
    assert os.path.isdir("DOCUMENTATION")
