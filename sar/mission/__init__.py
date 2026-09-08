"""Mission execution: the join between flight, perception and reporting.

See :mod:`sar.mission.runner`.  The runner flies a survey by MAVLink command
only, renders both camera modalities at the aircraft's true pose, geo-tags from
its *estimated* pose, and publishes survivors onto the offline-first uplink.
"""

from sar.mission.runner import (MissionReport, MissionRunner, build_pipeline,
                                plant_from_telemetry)

__all__ = ["MissionReport", "MissionRunner", "build_pipeline",
           "plant_from_telemetry"]
