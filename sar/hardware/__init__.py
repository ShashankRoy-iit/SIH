"""Companion-computer hardware layer: cameras, safety supervision, the onboard loop.

Everything in :mod:`sar.hardware` is code that only matters when the aircraft is
real.  It is kept out of :mod:`sar.sim` and :mod:`sar.mission` so that the
autonomy above it cannot accidentally depend on a device node, and so that the
hardware path can be dry-run on a laptop with ``--dry-run``.

The three pieces:

* :mod:`sar.hardware.camera` - RGB and LWIR capture with an explicit
  synchronisation contract, radiometric conversion and boresight metadata.
* :mod:`sar.hardware.safety` - the supervisor that can end a sortie.  Geofence,
  battery reserve computed against the distance home, link loss, altitude
  limits, and a watchdog on the autonomy loop itself.
* :mod:`sar.hardware.onboard` - the flight-time entry point that wires the
  cameras, the perception pipeline, the MAVLink link, the store-and-forward
  radio and the payload release into one loop.

The rule the whole layer follows: **the same autonomy code runs in simulation
and in flight.**  What changes is which camera source and which MAVLink target
are constructed - one factory call, in one place.
"""

from sar.hardware.camera import (CameraPair, CameraSource, FrameCapture,
                                 SimulatedCameraSource, ThermalCameraSource,
                                 V4L2CameraSource, build_camera_pair)
from sar.hardware.safety import (SafetyDecision, SafetyLimits, SafetySupervisor,
                                 SafetyState)

__all__ = [
    "CameraPair",
    "CameraSource",
    "FrameCapture",
    "SimulatedCameraSource",
    "ThermalCameraSource",
    "V4L2CameraSource",
    "build_camera_pair",
    "SafetyDecision",
    "SafetyLimits",
    "SafetySupervisor",
    "SafetyState",
]
