"""Navigation: GPS-denied operation, EKF source management, external nav feeding.

The perception stack answers "who is where". This one answers "where are *we*",
which is the same question under harder constraints: the places a search flies
are the places GNSS works worst.
"""

from .external_nav import (EkfSourceManager, euler_to_quat, quat_rotate, ExternalNavFeeder, ExternalNavSample,
                           NavAssessment, NavQualityMonitor, NavState,
                           SimulatedVioSource, SourceSetPolicy, TelemetryVioSource,
                           VioSource)

__all__ = ["EkfSourceManager", "ExternalNavFeeder", "ExternalNavSample",
           "NavAssessment", "NavQualityMonitor", "NavState", "SimulatedVioSource",
           "SourceSetPolicy", "TelemetryVioSource", "VioSource",
           "euler_to_quat", "quat_rotate"]
