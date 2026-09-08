"""SAHYOG - a deployable, on-device-AI autonomous search-and-rescue drone stack.

Smart India Hackathon problem statement 26177 (Qualcomm Inc., Robotics & Drones).

The package is organised so that the *same* autonomy code runs against three
transports without modification:

1. the bundled Python simulator (:mod:`sar.sim`) - deterministic, headless,
   used for algorithm development and regression testing;
2. ArduPilot SITL over MAVLink/UDP - the real flight stack, same protocol;
3. the physical vehicle (TBS Lucid H743 wing FC on ArduPilot + ELRS + a
   Qualcomm-class companion computer) over MAVLink telemetry.
"""

__version__ = "0.1.0"
__all__ = ["core", "vehicle", "sim", "perception", "nav", "decision", "comms", "mavlink", "mission", "gcs"]
