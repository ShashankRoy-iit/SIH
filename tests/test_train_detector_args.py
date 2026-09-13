"""The two training-argument traps that each cost a whole GPU run.

Both were real: a Kaggle run trained SARD (RGB) with ``--channels`` left at its
old default of ``1``, so hue/saturation augmentation was silently off; and
``--p2`` combined with a ``.pt`` did nothing at all, because the old resolution
was a literal ``str.replace("yolo11n.yaml", ...)`` that never matched - so the
documented "fine-tune the P2 model" command trained a stock head from scratch.

Neither failure raises.  The symptom is a bad mAP five GPU-hours later, which
reads as "needs more data".  These tests lock the resolution down.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scripts import train_detector as td

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# augmentation regime
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("data_yaml,expected", [
    # The repo's own dataset table is authoritative: "hit-uav" contains no
    # thermal-looking substring, and it is the primary LWIR set.
    ("datasets/hit-uav/data.yaml", 1),
    ("datasets/airesq/data.yaml", 1),
    ("datasets/sard/data.yaml", 3),
    ("datasets/visdrone/data.yaml", 3),
    ("datasets/heridal/data.yaml", 3),
    # ...then path hints, for datasets the table does not know.
    ("datasets/sim-thermal/data.yaml", 1),
    ("datasets/sim-lwir/data.yaml", 1),
    ("datasets/sim-rgb/data.yaml", 3),
    ("/mnt/data/my-thermal-set/data.yaml", 1),
])
def test_channels_are_inferred_from_the_dataset(data_yaml, expected):
    channels, why = td.infer_channels(data_yaml, None)
    assert channels == expected, why
    assert why                                   # the choice is always explained


def test_explicit_channels_always_win():
    assert td.infer_channels("datasets/sard/data.yaml", 1)[0] == 1
    assert td.infer_channels("datasets/hit-uav/data.yaml", 3)[0] == 3


def test_unrecognised_dataset_defaults_to_rgb_and_says_so():
    """RGB is the safe default; the warning is what stops it being silent."""
    channels, why = td.infer_channels("/tmp/whatever/data.yaml", None)
    assert channels == 3
    assert "UNRECOGNISED" in why and "--channels 1" in why


# --------------------------------------------------------------------------- #
# P2 head + pretrained transfer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model", ["yolo11n.pt", "yolo11n.yaml", "yolo11n-p2.yaml"])
def test_p2_architecture_always_yields_the_p2_yaml(model):
    assert td.p2_architecture(model) == "yolo11n-p2.yaml"


def test_p2_keeps_the_checkpoint_as_the_transfer_source():
    """`--model yolo11n.pt --p2` must build the P2 graph *and* load COCO."""
    resolved = _resolve(["--train", "--data", "datasets/hit-uav/data.yaml",
                         "--p2", "--model", "yolo11n.pt"])
    assert resolved["model"].endswith("configs/models/yolo11n-p2.yaml")
    assert resolved["load"] == "yolo11n.pt"
    assert resolved["channels"] == 1             # hit-uav is LWIR


def test_p2_with_an_explicit_load_does_not_override_it():
    resolved = _resolve(["--train", "--data", "datasets/sim-thermal/data.yaml",
                         "--p2", "--model", "yolo11n.yaml",
                         "--load", "runs/pretrain/weights/best.pt"])
    assert resolved["load"] == "runs/pretrain/weights/best.pt"


def test_default_model_fine_tunes_rather_than_training_from_scratch():
    resolved = _resolve(["--train", "--data", "datasets/sard/data.yaml"])
    assert resolved["model"] == "yolo11n.pt"
    assert resolved["channels"] == 3


def test_p2_for_an_unbundled_scale_fails_loudly():
    """Only the 'n' P2 YAML ships; a silent fall back to stock would be worse."""
    resolved = _resolve(["--train", "--data", "datasets/sard/data.yaml",
                         "--p2", "--model", "yolo11s.pt"], expect_train=False)
    assert "yolo11s-p2.yaml" in resolved["exit"]
    assert "configs/models/" in resolved["exit"]


@pytest.mark.parametrize("degrees,flipud", [(180.0, 0.5), (0.0, 0.0)])
def test_vertical_flip_follows_the_rotation_setting(degrees, flipud):
    """An upside-down person is a real example nadir-aerial and not oblique."""
    resolved = _resolve(["--train", "--data", "datasets/sard/data.yaml",
                         "--degrees", str(degrees)])
    assert resolved["degrees"] == degrees
    assert resolved["flipud"] == flipud


# --------------------------------------------------------------------------- #
# harness: run main() with train() stubbed, so no GPU / ultralytics is needed
# --------------------------------------------------------------------------- #
def _resolve(argv, expect_train=True):
    captured = {}

    def fake_train(args):
        captured.update(
            model=args.model, load=args.load, channels=args.channels,
            degrees=args.degrees,
            flipud=0.5 if args.degrees >= 90 else 0.0,
            patience=args.patience, imgsz=args.imgsz, epochs=args.epochs,
        )
        return 0

    real_train, real_argv = td.train, sys.argv
    td.train = fake_train
    sys.argv = ["train_detector.py"] + argv
    captured["exit"] = ""
    try:
        td.main()
    except SystemExit as exc:
        # argparse errors and our own guards both exit with a message
        captured["exit"] = str(exc.code) if exc.code not in (0, None) else ""
    finally:
        td.train, sys.argv = real_train, real_argv
    if expect_train:
        assert captured.get("model"), f"main() never reached train() for {argv}"
    return captured


def test_bundled_p2_yaml_still_exists():
    assert (REPO / "configs" / "models" / "yolo11n-p2.yaml").is_file()
