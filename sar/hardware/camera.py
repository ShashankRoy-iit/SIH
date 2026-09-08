"""Real cameras: capture, radiometry, and the synchronisation contract.

What makes this non-trivial
---------------------------
The perception stack consumes a *pair* of frames - LWIR and RGB - that it
believes were taken at the same instant from the same pose.  On a real aircraft
they are not.  Two USB cameras free-run at different rates, with different
exposure latencies, and the aircraft moves 8 m/s.  A 120 ms skew at 8 m/s is
one metre of parallax, which is the same order as the geo-tag error budget, and
it silently destroys cross-modal association: the thermal blob and the visible
blob no longer overlap, so the ensemble's agreement bonus never fires and every
detection is treated as single-modality.

So :class:`CameraPair` does three things a naive capture loop does not:

1. timestamps each frame at *capture*, not at delivery;
2. pairs frames only within ``max_skew_s``, and reports the skew it accepted;
3. counts and reports rejected pairs, so a drifting camera shows up in the
   sortie report as a number rather than as mysteriously poor fusion.

Radiometry
----------
A FLIR Lepton 3.5 / Boson in radiometric mode reports 16-bit counts in
centikelvin (Lepton: TLinear, 0.01 K per count).  Converting to degrees Celsius
is a subtraction, not a calibration - but only if the sensor is in TLinear mode
with the correct resolution, which is a runtime configuration the code must
assert rather than assume.  Non-radiometric AGC output is *not* usable for the
physiology model, and the code says so loudly rather than producing plausible
nonsense temperatures.

Ground sample distance
----------------------
``gsd_m`` is computed from AGL and the lens, not guessed.  It drives the
geometry-aware aperture, the extent veto and the coverage grid's per-cell
detection probability, so an incorrect GSD does not merely mislabel a frame -
it corrupts the search map.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

log = logging.getLogger("sar.hardware.camera")


@dataclass
class FrameCapture:
    """One captured frame plus everything the pipeline needs about it.

    Field names match :class:`sar.perception.detector.ImageFrame` so a capture
    can be handed to the detector directly.
    """

    image: np.ndarray
    kind: str                      # 'lwir' | 'rgb'
    t: float                       # monotonic capture timestamp, seconds
    camera: str = "cam"
    gsd_m: float = 0.0
    pose: Any = None
    truth: List[Any] = field(default_factory=list)
    footprint: Tuple[Any, ...] = ()
    radiometric: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)


class CameraSource(Protocol):
    """Anything that produces frames.  Implemented by real and simulated sources."""

    kind: str
    name: str

    def open(self) -> None: ...
    def read(self) -> Optional[FrameCapture]: ...
    def close(self) -> None: ...


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def ground_sample_distance(agl_m: float, hfov_deg: float, width_px: int,
                           tilt_deg: float = 0.0) -> float:
    """Metres per pixel at the centre of the frame.

    ``tilt_deg`` is the camera's angle off nadir; the slant range grows as
    ``1/cos(tilt)`` and so does the GSD.  Ignoring it is why an off-nadir
    detection geo-tags long.
    """
    if agl_m <= 0 or width_px <= 0:
        return 0.0
    slant = agl_m / max(math.cos(math.radians(min(abs(tilt_deg), 80.0))), 0.17)
    swath = 2.0 * slant * math.tan(math.radians(hfov_deg) / 2.0)
    return float(swath / width_px)


# --------------------------------------------------------------------------- #
# Real sources
# --------------------------------------------------------------------------- #
class V4L2CameraSource:
    """RGB capture through OpenCV / V4L2 (or a GStreamer pipeline string).

    On a Qualcomm companion computer the GStreamer path is the one to use -
    ``qtiqmmfsrc`` keeps the frames in the camera ISP's buffers instead of
    copying them through userspace, which is worth several milliseconds and a
    lot of CPU at 1080p.
    """

    kind = "rgb"

    def __init__(self, device: str | int = 0, *, width: int = 1280, height: int = 720,
                 fps: int = 15, hfov_deg: float = 66.0, name: str = "rgb0",
                 gstreamer: Optional[str] = None, fourcc: str = "MJPG") -> None:
        self.device = device
        self.width, self.height, self.fps = width, height, fps
        self.hfov_deg = hfov_deg
        self.name = name
        self.gstreamer = gstreamer
        self.fourcc = fourcc
        self._cap: Any = None
        self.frames = 0
        self.dropped = 0

    def open(self) -> None:
        import cv2  # imported here so the module loads without OpenCV
        if self.gstreamer:
            self._cap = cv2.VideoCapture(self.gstreamer, cv2.CAP_GSTREAMER)
        else:
            self._cap = cv2.VideoCapture(self.device)
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
            # A deep driver queue means the newest frame is minutes-old under
            # load. One buffer: we would rather drop frames than geo-tag stale ones.
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open RGB camera {self.device!r}")
        log.info("RGB camera %s open (%dx%d @ %d fps)", self.name,
                 self.width, self.height, self.fps)

    def read(self) -> Optional[FrameCapture]:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        t = time.monotonic()
        if not ok or frame is None:
            self.dropped += 1
            return None
        self.frames += 1
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return FrameCapture(image=rgb, kind="rgb", t=t, camera=self.name,
                            meta={"hfov_deg": self.hfov_deg})

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class ThermalCameraSource:
    """Radiometric LWIR capture (FLIR Lepton 3.5 / Boson, UVC 16-bit Y16).

    ``counts_to_c`` implements TLinear: ``T[K] = counts * resolution``, with the
    Lepton's default 0.01 K resolution, then Kelvin to Celsius.  If the sensor
    is delivering 8-bit AGC video instead, ``radiometric`` is False and the
    physiology model must not be fed from it - :class:`CameraPair` propagates
    that flag rather than hiding it.
    """

    kind = "lwir"

    def __init__(self, device: str | int = 1, *, width: int = 160, height: int = 120,
                 fps: int = 9, hfov_deg: float = 57.0, name: str = "lwir0",
                 resolution_k: float = 0.01, expect_radiometric: bool = True,
                 gstreamer: Optional[str] = None) -> None:
        self.device = device
        self.width, self.height, self.fps = width, height, fps
        self.hfov_deg = hfov_deg
        self.name = name
        self.resolution_k = resolution_k
        self.expect_radiometric = expect_radiometric
        self.gstreamer = gstreamer
        self._cap: Any = None
        self.radiometric = False
        self.frames = 0
        self.dropped = 0
        self.ffc_count = 0

    def open(self) -> None:
        import cv2
        if self.gstreamer:
            self._cap = cv2.VideoCapture(self.gstreamer, cv2.CAP_GSTREAMER)
        else:
            self._cap = cv2.VideoCapture(self.device)
            # Y16 is what carries radiometry; MJPG or YUYV means AGC 8-bit.
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"Y16 "))
            self._cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open LWIR camera {self.device!r}")
        probe = self.read()
        if probe is None:
            raise RuntimeError("LWIR camera opened but delivered no frame")
        if self.expect_radiometric and not self.radiometric:
            raise RuntimeError(
                "LWIR camera is not delivering 16-bit radiometric frames "
                "(got 8-bit AGC video). Enable TLinear/radiometric mode, or "
                "construct with expect_radiometric=False and accept that the "
                "triage physiology model will be disabled for this sortie.")
        log.info("LWIR camera %s open (%dx%d, radiometric=%s)", self.name,
                 self.width, self.height, self.radiometric)

    def counts_to_c(self, counts: np.ndarray) -> np.ndarray:
        return counts.astype(np.float32) * self.resolution_k - 273.15

    def read(self) -> Optional[FrameCapture]:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        t = time.monotonic()
        if not ok or frame is None:
            self.dropped += 1
            return None
        self.frames += 1
        arr = np.asarray(frame)
        if arr.dtype == np.uint16 or (arr.ndim == 2 and arr.max() > 300):
            self.radiometric = True
            image = self.counts_to_c(arr.reshape(self.height, self.width)
                                     if arr.size == self.height * self.width else arr)
        else:
            self.radiometric = False
            image = arr.astype(np.float32)
            if image.ndim == 3:
                image = image.mean(axis=-1)
        return FrameCapture(image=image, kind="lwir", t=t, camera=self.name,
                            radiometric=self.radiometric,
                            meta={"hfov_deg": self.hfov_deg,
                                  "resolution_k": self.resolution_k})

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class SimulatedCameraSource:
    """Renders from the simulator's world, with the same interface as a real camera.

    This is what makes ``--dry-run`` meaningful: the onboard loop under test is
    byte-identical to the flight one, and only the frame source differs.
    """

    def __init__(self, renderer: Any, state_fn: Any, kind: str = "lwir",
                 name: str = "sim", rate_hz: float = 8.0) -> None:
        self.renderer = renderer
        self.state_fn = state_fn
        self.kind = kind
        self.name = name
        self.period = 1.0 / max(rate_hz, 0.1)
        self._last = 0.0
        self.frames = 0
        self.dropped = 0

    def open(self) -> None:
        self._last = 0.0

    def read(self) -> Optional[FrameCapture]:
        now = time.monotonic()
        if now - self._last < self.period:
            return None
        self._last = now
        st = self.state_fn()
        frame = self.renderer.render(st, t=now)
        self.frames += 1
        return FrameCapture(image=np.asarray(frame.image), kind=self.kind, t=now,
                            camera=self.name, gsd_m=float(getattr(frame, "gsd_m", 0.0)),
                            pose=getattr(frame, "pose", None),
                            truth=list(getattr(frame, "truth", []) or []),
                            radiometric=(self.kind == "lwir"))

    def close(self) -> None:
        return None


# --------------------------------------------------------------------------- #
# Pairing
# --------------------------------------------------------------------------- #
@dataclass
class PairStats:
    pairs: int = 0
    rejected_skew: int = 0
    lwir_only: int = 0
    rgb_only: int = 0
    worst_skew_ms: float = 0.0
    mean_skew_ms: float = 0.0
    _skew_sum: float = 0.0

    def record(self, skew_s: float) -> None:
        self.pairs += 1
        ms = abs(skew_s) * 1000.0
        self._skew_sum += ms
        self.worst_skew_ms = max(self.worst_skew_ms, ms)
        self.mean_skew_ms = self._skew_sum / max(self.pairs, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {"pairs": self.pairs, "rejected_skew": self.rejected_skew,
                "lwir_only": self.lwir_only, "rgb_only": self.rgb_only,
                "mean_skew_ms": round(self.mean_skew_ms, 1),
                "worst_skew_ms": round(self.worst_skew_ms, 1)}


class CameraPair:
    """Two free-running cameras presented as one synchronised source.

    Each source is read in its own thread so a slow USB transaction on one
    cannot stall the other (and so neither can stall the control loop).  The
    latest frame from each is kept; :meth:`latest_pair` returns them only if
    their capture times agree within ``max_skew_s``.
    """

    def __init__(self, lwir: Optional[CameraSource] = None,
                 rgb: Optional[CameraSource] = None, *,
                 max_skew_s: float = 0.08,
                 allow_single_modality: bool = True) -> None:
        self.lwir = lwir
        self.rgb = rgb
        self.max_skew_s = float(max_skew_s)
        self.allow_single_modality = allow_single_modality
        self.stats = PairStats()
        self._latest: Dict[str, FrameCapture] = {}
        self._lock = threading.Lock()
        self._threads: List[threading.Thread] = []
        self._running = False

    # -- lifecycle -------------------------------------------------------- #
    def open(self) -> None:
        for src in (self.lwir, self.rgb):
            if src is not None:
                src.open()
        self._running = True
        for src in (self.lwir, self.rgb):
            if src is None:
                continue
            th = threading.Thread(target=self._pump, args=(src,), daemon=True,
                                  name=f"cam-{src.kind}")
            th.start()
            self._threads.append(th)

    def _pump(self, src: CameraSource) -> None:
        while self._running:
            try:
                frame = src.read()
            except Exception as exc:  # a camera unplugged mid-sortie
                log.warning("%s read failed: %r", src.kind, exc)
                time.sleep(0.2)
                continue
            if frame is None:
                time.sleep(0.002)
                continue
            with self._lock:
                self._latest[frame.kind] = frame

    def close(self) -> None:
        self._running = False
        for th in self._threads:
            th.join(timeout=1.0)
        for src in (self.lwir, self.rgb):
            if src is not None:
                src.close()

    # -- access ----------------------------------------------------------- #
    def latest_pair(self, *, agl_m: float = 0.0, tilt_deg: float = 0.0
                    ) -> List[FrameCapture]:
        """Return the freshest synchronised frames, with GSD filled in."""
        with self._lock:
            lw = self._latest.get("lwir")
            rgb = self._latest.get("rgb")
        out: List[FrameCapture] = []
        if lw is not None and rgb is not None:
            skew = lw.t - rgb.t
            if abs(skew) <= self.max_skew_s:
                self.stats.record(skew)
                out = [lw, rgb]
            else:
                self.stats.rejected_skew += 1
                if self.allow_single_modality:
                    out = [lw]              # thermal is the primary sensor
                    self.stats.lwir_only += 1
        elif lw is not None:
            self.stats.lwir_only += 1
            out = [lw]
        elif rgb is not None:
            self.stats.rgb_only += 1
            out = [rgb]

        for f in out:
            hfov = float(f.meta.get("hfov_deg", 0.0))
            if agl_m > 0 and hfov > 0 and not f.gsd_m:
                width = f.image.shape[1]
                f.gsd_m = ground_sample_distance(agl_m, hfov, width, tilt_deg)
        return out

    def health(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"sync": self.stats.to_dict()}
        for src in (self.lwir, self.rgb):
            if src is None:
                continue
            d[src.kind] = {"name": getattr(src, "name", "?"),
                           "frames": getattr(src, "frames", 0),
                           "dropped": getattr(src, "dropped", 0),
                           "radiometric": getattr(src, "radiometric", None)}
        return d


def build_camera_pair(config: Dict[str, Any]) -> CameraPair:
    """Construct the pair described by ``configs/onboard.yaml``.

    Keeping construction in one factory is what lets the onboard loop be
    identical in simulation and in flight: the loop never names a device.
    """
    lwir_cfg = dict(config.get("lwir") or {})
    rgb_cfg = dict(config.get("rgb") or {})
    lwir = rgb = None
    if lwir_cfg.get("enabled", True):
        lwir = ThermalCameraSource(
            device=lwir_cfg.get("device", 1),
            width=int(lwir_cfg.get("width", 160)),
            height=int(lwir_cfg.get("height", 120)),
            fps=int(lwir_cfg.get("fps", 9)),
            hfov_deg=float(lwir_cfg.get("hfov_deg", 57.0)),
            resolution_k=float(lwir_cfg.get("resolution_k", 0.01)),
            expect_radiometric=bool(lwir_cfg.get("expect_radiometric", True)),
            gstreamer=lwir_cfg.get("gstreamer"))
    if rgb_cfg.get("enabled", True):
        rgb = V4L2CameraSource(
            device=rgb_cfg.get("device", 0),
            width=int(rgb_cfg.get("width", 1280)),
            height=int(rgb_cfg.get("height", 720)),
            fps=int(rgb_cfg.get("fps", 15)),
            hfov_deg=float(rgb_cfg.get("hfov_deg", 66.0)),
            gstreamer=rgb_cfg.get("gstreamer"))
    return CameraPair(lwir=lwir, rgb=rgb,
                      max_skew_s=float(config.get("max_skew_s", 0.08)))


__all__ = ["CameraPair", "CameraSource", "FrameCapture", "PairStats",
           "SimulatedCameraSource", "ThermalCameraSource", "V4L2CameraSource",
           "build_camera_pair", "ground_sample_distance"]
