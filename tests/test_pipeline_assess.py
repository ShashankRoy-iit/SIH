"""Regression tests for the survivor-assessment stage of the perception pipeline.

``PerceptionPipeline._assess`` is where detection turns into a rescue decision:
it runs the physics-informed human identification (posture, thermal viability,
decoy rejection) and the nearby-hazard context query.  This path shipped a
latent ``NameError`` (``hz`` was never bound) that the existing unit tests did
not reach because they stopped short of assessing a *person* track.  These tests
build a real pipeline, inject a confirmed person track and a registered hazard,
and assert that the assessment returns a physically-informed profile instead of
raising.
"""

import numpy as np

from sar.core.geo import GeoPoint
from sar.perception.geotag import PixelGeoTagger
from sar.perception.hazard import Hazard, HazardMap, HazardSeverity
from sar.perception.human_id import HumanPosture
from sar.perception.pipeline import PerceptionPipeline, PerceptionProduct
from sar.perception.tracker import Track, TrackObservation

ORIGIN = GeoPoint(25.0, 75.0, 0.0)


def _make_track(*, north: float, east: float, elongation: float = 1.2,
                immobility: float = 0.95) -> Track:
    """A confirmed, fused (RGB+LWIR) person track with a radiometric history."""
    obs = [
        TrackObservation(
            t=0.0, north=north, east=east, label="person", score=0.9,
            sigma_n=2.0, sigma_e=2.0, gsd_m=0.12, modality="fused",
            peak_temp_c=36.6, mean_temp_c=34.0, background_temp_c=18.0,
            area_px=42.0, attributes={"elongation": elongation,
                                      "aspect": 1.2, "rgb_saliency": 60.0},
        ),
        TrackObservation(
            t=16.0, north=north + 0.3, east=east - 0.2, label="person",
            score=0.88, sigma_n=2.0, sigma_e=2.0, gsd_m=0.14,
            modality="fused", peak_temp_c=36.2, mean_temp_c=33.8,
            background_temp_c=18.0, area_px=40.0,
            attributes={"elongation": elongation,
                        "aspect": 1.25, "rgb_saliency": 58.0},
        ),
    ]
    return Track(
        tid=1, north=north, east=east, label="person", status="confirmed",
        confidence=0.85, created_t=0.0, updated_t=16.0, propagated_t=16.0,
        n_obs=len(obs), n_modalities={"fused"}, last_gsd_m=0.14,
        observations=obs, votes={"person": 6.0}, person_evidence=0.9,
        immobility=immobility, speed_ms=0.0, n_resolved=2, origin=ORIGIN,
        peak_temp_c=36.6,
    )


def _pipeline(hazard: Hazard) -> PerceptionPipeline:
    hm = HazardMap(ORIGIN, 1000.0, 1000.0)
    hm.hazards.append(hazard)
    hm._stamp(hazard, 0.0)
    tagger = PixelGeoTagger(width=640, height=512, hfov_deg=42.0, origin=ORIGIN)
    return PerceptionPipeline(origin=ORIGIN, tagger=tagger, hazard_map=hm)


def test_assess_with_nearby_collapse_yields_prone_burial_profile():
    """The NameError regression: assessing a person near a collapse must not
    raise, and the hazard context must drive the physics-based posture."""
    hazard = Hazard(
        hid=1, label="collapsed_structure", north=45.0, east=45.0,
        severity=HazardSeverity.DANGEROUS, confidence=0.95, area_m2=120.0,
        extent_m=12.0, first_seen_t=0.0, last_seen_t=0.0, n_obs=1,
        context=["structure collapse"], origin=ORIGIN,
    )
    pipe = _pipeline(hazard)
    track = _make_track(north=50.0, east=50.0, elongation=1.8, immobility=0.95)
    pipe.tracker.person_tracks = lambda: [track]

    product = PerceptionProduct(t=20.0)
    survivors = pipe._assess(20.0, product)

    assert len(survivors) == 1
    s = survivors[0]
    assert s.profile is not None
    assert s.profile.posture == HumanPosture.PRONE_PARTIAL_BURIAL
    # The hazard context reached the assessment verbatim.
    labels = {h.get("label") for h in s.hazards_nearby}
    assert "collapsed_structure" in labels


def test_assess_without_hazards_still_profiles():
    """No hazards on the map is a supported, common case."""
    pipe = _pipeline(Hazard(
        hid=1, label="floodwater", north=400.0, east=400.0,
        severity=HazardSeverity.CAUTION, confidence=0.8, area_m2=500.0,
        extent_m=25.0, first_seen_t=0.0, last_seen_t=0.0, n_obs=1,
        origin=ORIGIN,
    ))
    track = _make_track(north=100.0, east=100.0, elongation=1.1, immobility=0.1)
    pipe.tracker.person_tracks = lambda: [track]

    survivors = pipe._assess(5.0, PerceptionProduct(t=5.0))
    assert len(survivors) == 1
    assert survivors[0].profile is not None
    # Far from the flood, low elongation and mobile -> standing, not in water.
    assert survivors[0].profile.posture == HumanPosture.STANDING


def test_assess_skips_unconfirmed_tracks():
    """Tracks below the alert threshold are not assessed into survivors."""
    pipe = _pipeline(Hazard(
        hid=1, label="debris_field", north=200.0, east=200.0,
        severity=HazardSeverity.CAUTION, confidence=0.7, area_m2=80.0,
        extent_m=10.0, first_seen_t=0.0, last_seen_t=0.0, n_obs=1,
        origin=ORIGIN,
    ))
    track = _make_track(north=50.0, east=50.0)
    track.person_evidence = 0.2  # below alert_threshold=0.45
    pipe.tracker.person_tracks = lambda: [track]

    survivors = pipe._assess(5.0, PerceptionProduct(t=5.0))
    assert survivors == []
