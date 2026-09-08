"""MAVLink transport layer.

Bridges the mission and perception code to a flight stack - real ArduPilot SITL,
:class:`~sar.sim.sitl.MiniSITL`, or a physical TBS Lucid H743 - without any of
them knowing which one they are talking to.
"""

from .protocol import (COPTER_MODES, GPS_FIX, MAV_CMD, MAV_FRAME, MAV_MODE,
                       MAV_RESULT, MAV_TYPE, POSITION_MODES, SOURCE_SET,
                       TYPE_MASK, describe_message, mask, result_name)

__all__ = ["protocol", "connection", "COPTER_MODES", "GPS_FIX", "MAV_CMD",
           "MAV_FRAME", "MAV_MODE", "MAV_RESULT", "MAV_TYPE", "POSITION_MODES",
           "SOURCE_SET", "TYPE_MASK", "describe_message", "mask", "result_name"]
