"""Unit and integration tests for Human Identification and Multi-Attribute Classifier."""

import math
import numpy as np
import pytest

from sar.core.geo import GeoPoint
from sar.perception.human_id import (
    DemographicGroup,
    DistressLevel,
    HumanIdentifier,
    HumanPosture,
    HumanProfile,
    RescueEquipmentNeed,
)
from sar.perception.thermal import PriorityTier


class MockObservation:
    def __init__(
        self,
        t: float = 0.0,
        north: float = 50.0,
        east: float = 50.0,
        gsd_m: float = 0.12,
        peak_temp_c: float = 36.5,
        background_temp_c: float = 20.0,
        area_px: float = 40.0,
        modality: str = "fused",
        attributes: dict = None,
    ):
        self.t = t
        self.north = north
        self.east = east
        self.gsd_m = gsd_m
        self.peak_temp_c = peak_temp_c
        self.background_temp_c = background_temp_c
        self.area_px = area_px
        self.modality = modality
        self.confidence = 0.90
        self.attributes = attributes or {"aspect": 1.1, "elongation": 1.2, "rgb_saliency": 50.0}


class MockTrack:
    def __init__(
        self,
        tid: int = 1,
        north: float = 50.0,
        east: float = 50.0,
        confidence: float = 0.88,
        immobility: float = 0.2,
        speed_ms: float = 0.1,
        observations: list = None,
    ):
        self.tid = tid
        self.north = north
        self.east = east
        self.confidence = confidence
        self.immobility = immobility
        self.speed_ms = speed_ms
        self.age_s = 20.0
        self.label = "person"
        self.observations = observations or [MockObservation(0.0), MockObservation(1.0)]

    def geo(self, origin):
        return GeoPoint(origin.lat + 0.0001, origin.lon + 0.0001, origin.alt)


def test_standing_adult_identification():
    identifier = HumanIdentifier()
    tr = MockTrack(tid=1, immobility=0.1, speed_ms=0.2)
    profile = identifier.identify(tr, water_depth_m=0.0, ambient_temp_c=22.0)

    assert profile.is_human
    assert profile.posture == HumanPosture.STANDING
    assert profile.demographic == DemographicGroup.ADULT
    assert profile.group_size == 1
    assert profile.triage_priority in (PriorityTier.DELAYED, PriorityTier.MINOR)
    assert profile.recommended_equipment == RescueEquipmentNeed.FIRST_AID_TRAUMA_KIT


def test_in_water_clinging_and_flotation_need():
    identifier = HumanIdentifier()
    tr = MockTrack(tid=2, immobility=0.8, speed_ms=0.02)
    profile = identifier.identify(tr, water_depth_m=0.85, ambient_temp_c=18.0)

    assert profile.posture == HumanPosture.IN_WATER_CLINGING
    assert profile.recommended_equipment == RescueEquipmentNeed.FLOTATION_BUOY
    assert profile.distress_level == DistressLevel.CRITICAL_ACTIVE_SOS
    assert profile.hypothermia_risk > 0.0


def test_sos_waving_gesture_detection():
    identifier = HumanIdentifier()
    # Create periodic waving oscillations in aspect ratio (1.5 Hz)
    obs_list = []
    for i in range(12):
        t = i * 0.15
        wave_aspect = 1.0 + 0.6 * math.sin(2.0 * math.pi * 1.5 * t)
        obs_list.append(
            MockObservation(
                t=t,
                attributes={"aspect": wave_aspect, "elongation": 1.2, "rgb_saliency": 75.0},
            )
        )
    tr = MockTrack(tid=3, observations=obs_list)
    profile = identifier.identify(tr, water_depth_m=0.0)

    assert profile.sos_waving_detected
    assert profile.posture == HumanPosture.WAVING_SOS
    assert 0.8 <= profile.waving_frequency_hz <= 3.5
    assert profile.clothing_saliency >= 70.0


def test_prone_partial_burial_in_debris():
    identifier = HumanIdentifier()
    tr = MockTrack(
        tid=4,
        immobility=0.95,
        speed_ms=0.0,
        observations=[
            MockObservation(attributes={"aspect": 1.0, "elongation": 1.9}),
            MockObservation(attributes={"aspect": 1.0, "elongation": 2.0}),
        ],
    )
    profile = identifier.identify(
        tr,
        water_depth_m=0.0,
        nearby_hazard_classes=["collapsed_structure"],
    )

    assert profile.posture == HumanPosture.PRONE_PARTIAL_BURIAL
    assert profile.recommended_equipment == RescueEquipmentNeed.LORA_LOCATOR_BEACON
    assert profile.distress_level == DistressLevel.PASSIVE_IMMOBILE


def test_group_clustering_detection():
    identifier = HumanIdentifier()
    # Large area blob (> 1.4 m^2 at 0.12 m/px)
    obs = [MockObservation(area_px=140.0, gsd_m=0.12)]
    tr = MockTrack(tid=5, observations=obs)
    profile = identifier.identify(tr)

    assert profile.demographic == DemographicGroup.GROUP_CLUSTER
    assert profile.group_size >= 2


def test_decoy_rejection():
    identifier = HumanIdentifier()
    # Hot engine decoy (55°C)
    tr_engine = MockTrack(
        tid=6,
        observations=[MockObservation(peak_temp_c=58.0, background_temp_c=20.0)],
    )
    p_engine = identifier.identify(tr_engine)
    assert not p_engine.is_human
    assert any("biology" in note for note in p_engine.decoy_rejection_notes)

    # Fast vehicle decoy (18 m/s)
    tr_car = MockTrack(tid=7, speed_ms=18.0)
    p_car = identifier.identify(tr_car)
    assert not p_car.is_human
    assert any("speed" in note for note in p_car.decoy_rejection_notes)
