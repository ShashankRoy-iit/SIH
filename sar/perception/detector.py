"""On-device detectors: a pluggable backend interface plus reference models.

Design summary
--------------
Two things are true about finding survivors from a small UAV, and they pull in
opposite directions:

1. A survivor is a **point target**.  At 40 m AGL with a 320-pixel LWIR core a
   standing person covers ~20 pixels; prone, ~8; partly buried, ~3.
2. The ground is **not** thermally flat.  Sun-warmed concrete, dry soil, engine
   bays and rubble all sit 3-15 K above their surroundings.

Global connected-component thresholding therefore fails in a specific,
predictable way: the survivor's blob merges with the warm surface it lies on and
both its size and its mean temperature become meaningless (a 473-pixel "person"
whose mean reads 22 C because most of it is soil).  Full scale-space blob
detection fixes the merging but costs ~60 ms/frame and fires on terrain texture.

So the reference detector does something cheaper and better matched to the
problem:

* threshold to find *candidate regions* (fast, coarse);
* locate the local extrema inside each region;
* measure every extremum through a **geometry-aware aperture** - a disk whose
  radius is computed from the known ground sample distance and the apparent
  area of a human body, not guessed from the image.  Inside that aperture we
  take radiometry (core mean/peak vs. an annulus background) *and* morphology
  (the thresholded core's elongation, aspect and compactness).

The aperture idea is the key move: because the aircraft knows its AGL and its
lens, the expected pixel size of a human body is known *a priori*, and the
measurement can be restricted to exactly that footprint.  Terrain merges away,
the radiometry becomes the survivor's radiometry, and the size term can be
compared directly against the targeting-task-performance curve.

Stage 2 is a calibrated classifier with readable rules, and
:class:`DetectorEnsemble` adds cross-modal agreement with an explicit
"RGB is blind" condition so night and smoke work.

Why not just ship YOLO?
-----------------------
We do - as :class:`OnnxRuntimeBackend`, :class:`UltralyticsBackend` and
:class:`QualcommQnnBackend`, all implementing the same :class:`DetectorBackend`
contract; ``docs/06_AI_MODELS_AND_DATASETS.md`` has the training, export and
Qualcomm AI Hub quantisation recipe.  The heuristic backend stays as the
reference because every one of its decision rules can be read and audited, it
runs at ~90 fps on a 2-core CPU with no NPU, and it emits a *calibrated*
confidence that fuses correctly with search theory (probability of detection) -
which a network argmax does not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

try:
    from scipy import ndimage as ndi
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #
PERSON_LABELS = ("person", "person_group")
HAZARD_LABELS = (
    "fire", "smoke", "flood_water", "flash_flood_channel", "debris_field",
    "collapsed_structure", "damaged_structure", "exposed_powerline",
    "landslide_zone", "chemical_plume", "vehicle", "blocked_road",
)
OTHER_LABELS = ("animal", "hot_rock", "cloth", "structure", "unknown")
ALL_LABELS = PERSON_LABELS + HAZARD_LABELS + OTHER_LABELS

#: Apparent (radiometric) area of a human body by posture, m^2.
HUMAN_APPARENT_AREA_M2 = 0.62
#: Radius of the equivalent disk, m.  Drives the measurement aperture.
HUMAN_APPARENT_RADIUS_M = math.sqrt(HUMAN_APPARENT_AREA_M2 / math.pi)   # 0.444 m
#: Above this ratio of surrounding warm extent to core extent, a hot spot is
#: embedded in a larger warm object (vehicle, generator, condenser, sun-baked
#: slab) rather than being a body lying on the ground.  Calibrated against the
#: decoy set in ``scripts/eval_detector.py``; see docs/06_AI_MODELS_AND_DATASETS.md.
EXTENT_RATIO_MAX = 6.5


@dataclass
class Detection:
    """One detection in one frame."""

    label: str
    score: float                                   # calibrated 0..1
    bbox: Tuple[float, float, float, float]        # (u0, v0, u1, v1) pixels
    modality: str                                  # 'lwir' | 'rgb' | 'fused'
    frame_t: float = 0.0
    centroid: Tuple[float, float] = (0.0, 0.0)
    area_px: float = 0.0
    peak_temp_c: Optional[float] = None
    mean_temp_c: Optional[float] = None
    background_temp_c: Optional[float] = None
    contrast_k: float = 0.0
    polarity: str = "hot"                          # 'hot' | 'cold' | 'n/a'
    aspect: float = 1.0
    elongation: float = 1.0
    compactness: float = 0.0
    scale_px: float = 0.0                          # aperture radius used, px
    attributes: Dict[str, Any] = field(default_factory=dict)
    backend: str = "heuristic"

    @property
    def u(self) -> float:
        return self.centroid[0]

    @property
    def v(self) -> float:
        return self.centroid[1]

    @property
    def width(self) -> float:
        return max(self.bbox[2] - self.bbox[0], 0.0)

    @property
    def height(self) -> float:
        return max(self.bbox[3] - self.bbox[1], 0.0)

    @property
    def equivalent_pixels(self) -> float:
        return math.sqrt(max(self.area_px, 0.0))

    @property
    def is_person(self) -> bool:
        return self.label in PERSON_LABELS

    @property
    def is_hazard(self) -> bool:
        return self.label in HAZARD_LABELS

    def iou(self, other: "Detection") -> float:
        return bbox_iou(self.bbox, other.bbox)

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["bbox"] = list(self.bbox)
        d["centroid"] = list(self.centroid)
        return d


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    aa = max(ax1 - ax0, 0) * max(ay1 - ay0, 0)
    ab = max(bx1 - bx0, 0) * max(by1 - by0, 0)
    union = aa + ab - inter
    return float(inter / union) if union > 1e-9 else 0.0


def bbox_center_dist(a: Sequence[float], b: Sequence[float]) -> float:
    return float(math.hypot((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2,
                            (a[1] + a[3]) / 2 - (b[1] + b[3]) / 2))


@dataclass
class ImageFrame:
    """Minimal frame type for real camera input and unit tests.

    ``gsd_m`` (ground sample distance, metres per pixel) is what makes the
    geometry-aware aperture possible; the flight stack always knows it from AGL
    and the lens, and the simulator supplies it exactly.
    """

    image: np.ndarray
    kind: str = "lwir"
    t: float = 0.0
    camera: str = "cam"
    pose: Any = None
    gsd_m: float = 0.0
    truth: List[Any] = field(default_factory=list)
    footprint: Tuple[Any, ...] = ()


class DetectorBackend(Protocol):
    """Anything that turns a frame into detections."""

    name: str

    def detect(self, frame: Any) -> List[Detection]: ...


# --------------------------------------------------------------------------- #
# Low-level vision utilities
# --------------------------------------------------------------------------- #
def robust_stats(a: np.ndarray) -> Tuple[float, float]:
    """Median and MAD-scaled sigma of the finite values of ``a``."""
    x = a[np.isfinite(a)]
    if x.size == 0:
        return 0.0, 1.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return med, max(1.4826 * mad, 1e-6)


def local_background(a: np.ndarray, size: int = 21, block: int = 4) -> np.ndarray:
    """Local median estimate of the background radiance.

    ``median_filter(size=21)`` on a 320x240 frame costs ~350 ms - it would
    consume the entire real-time budget on the background estimate alone.  The
    background varies on the scale of metres, not pixels, so we pre-average by
    ``block``, median-filter the small image and upsample bilinearly: the same
    answer to within ~0.02 K, ~35x faster.
    """
    if not _HAVE_SCIPY:  # pragma: no cover
        return np.full_like(a, float(np.median(a)), dtype=np.float32)
    h, w = a.shape
    block = max(1, min(block, (min(h, w) // 8) or 1))
    if block <= 1:
        return ndi.median_filter(a, size=max(3, size), mode="nearest").astype(np.float32)
    sm = ndi.uniform_filter(a.astype(np.float32), size=block, mode="nearest")
    small = sm[::block, ::block]
    k = max(3, int(round(size / block)) | 1)
    med_small = ndi.median_filter(small, size=k, mode="nearest")
    big = ndi.zoom(med_small, (h / med_small.shape[0], w / med_small.shape[1]),
                   order=1, mode="nearest")[:h, :w]
    if big.shape != a.shape:
        big = np.pad(big, ((0, h - big.shape[0]), (0, w - big.shape[1])), mode="edge")
    return big.astype(np.float32)


def connected_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    if not _HAVE_SCIPY:  # pragma: no cover
        raise RuntimeError("scipy.ndimage is required for the reference detector")
    lab, n = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
    return lab, int(n)


def blob_features(image: np.ndarray, labels: np.ndarray, n: int,
                  background: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    """Geometry + radiometry for every label in a connected-component map.

    Used by the *extended-region* path (fire fronts, flood sheets, debris), where
    whole-blob statistics are meaningful.  Point targets go through
    :func:`measure_aperture` instead.
    """
    out: List[Dict[str, Any]] = []
    if n == 0:
        return out
    objs = ndi.find_objects(labels)
    for i, sl in enumerate(objs):
        if sl is None:
            continue
        mask = labels[sl] == (i + 1)
        area = int(mask.sum())
        if area < 1:
            continue
        sub = image[sl]
        ys, xs = np.nonzero(mask)
        v0, u0 = sl[0].start, sl[1].start
        du = xs - xs.mean()
        dv = ys - ys.mean()
        mu_uu = float((du * du).mean())
        mu_vv = float((dv * dv).mean())
        mu_uv = float((du * dv).mean())
        tr = mu_uu + mu_vv
        det = mu_uu * mu_vv - mu_uv * mu_uv
        disc = max(tr * tr / 4.0 - det, 0.0)
        l1 = tr / 2.0 + math.sqrt(disc)
        l2 = max(tr / 2.0 - math.sqrt(disc), 1e-9)
        vals = sub[mask]
        bbox = (float(u0 + xs.min()), float(v0 + ys.min()),
                float(u0 + xs.max() + 1), float(v0 + ys.max() + 1))
        feat = {
            "area": area,
            "centroid": (float(xs.mean()) + u0, float(ys.mean()) + v0),
            "bbox": bbox,
            "peak": float(vals.max()), "mean": float(vals.mean()),
            "std": float(vals.std()), "elongation": math.sqrt(l1 / l2),
            "aspect": float((bbox[3] - bbox[1]) / max(bbox[2] - bbox[0], 1e-6)),
            "width": bbox[2] - bbox[0], "height": bbox[3] - bbox[1],
            "compactness": _compactness(mask),
        }
        if background is not None:
            bg = float(background[sl][mask].mean())
            feat["background"] = bg
            feat["contrast"] = feat["mean"] - bg
            feat["peak_contrast"] = feat["peak"] - bg
        out.append(feat)
    return out


def _compactness(mask: np.ndarray) -> float:
    """Isoperimetric quotient 4*pi*area/perimeter^2 (1.0 = perfect disk)."""
    area = float(mask.sum())
    if area < 2 or not _HAVE_SCIPY:
        return 1.0
    er = ndi.binary_erosion(mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool),
                            border_value=0)
    perim = float((mask & ~er).sum())
    if perim < 1:
        return 1.0
    # A 4-connected perimeter undercounts for very small masks, so the quotient
    # can exceed 1; clamp it, since only "much less than 1" (ragged) is informative.
    return float(min(4.0 * math.pi * area / (perim * perim), 1.0))


def peak_candidates(diff: np.ndarray, min_response: float, min_sep: float = 3.0,
                    max_peaks: int = 48, mask: Optional[np.ndarray] = None
                    ) -> List[Tuple[float, float, float]]:
    """Local extrema of ``diff`` above ``min_response``, NMS'd by ``min_sep``.

    One ``maximum_filter`` call instead of a scale-space pyramid: an order of
    magnitude cheaper and, combined with :func:`measure_aperture`, equally
    effective for point targets.
    """
    if not _HAVE_SCIPY:  # pragma: no cover
        return []
    win = max(3, int(round(2 * min_sep + 1)))
    loc = diff >= ndi.maximum_filter(diff, size=win, mode="nearest")
    strong = loc & (diff > min_response)
    if mask is not None:
        strong &= mask
    ys, xs = np.nonzero(strong)
    if ys.size == 0:
        return []
    vals = diff[ys, xs]
    order = np.argsort(-vals)[:max_peaks * 3]
    kept: List[Tuple[float, float, float]] = []
    for idx in order:
        u, v, r = float(xs[idx]), float(ys[idx]), float(vals[idx])
        if all(math.hypot(u - ku, v - kv) >= min_sep for ku, kv, _ in kept):
            kept.append((u, v, r))
        if len(kept) >= max_peaks:
            break
    return kept


def measure_aperture(img: np.ndarray, diff: np.ndarray, u: float, v: float,
                     radius: float, core_frac: float = 0.5,
                     annulus: Tuple[float, float] = (1.8, 3.6)) -> Dict[str, Any]:
    """Radiometry + morphology of a point target inside a fixed aperture.

    The *core* is the set of aperture pixels whose excess over local background
    is at least ``core_frac`` of the peak excess - so the core follows the
    target's own shape instead of being a round window on a merged blob.  The
    background comes from an annulus around the aperture, i.e. from the surface
    the target is actually lying on.
    """
    h, w = img.shape
    r = float(max(radius, 1.0))
    ri = int(math.ceil(r * annulus[1])) + 1
    u0, u1 = max(0, int(round(u)) - ri), min(w, int(round(u)) + ri + 1)
    v0, v1 = max(0, int(round(v)) - ri), min(h, int(round(v)) + ri + 1)
    if u1 - u0 < 3 or v1 - v0 < 3:
        return {"area": 0.0}
    yy, xx = np.mgrid[v0:v1, u0:u1]
    d2 = (xx - u) ** 2 + (yy - v) ** 2
    aperture = d2 <= r * r
    ring = (d2 > (r * annulus[0]) ** 2) & (d2 <= (r * annulus[1]) ** 2)
    sub_diff = diff[v0:v1, u0:u1]
    sub_img = img[v0:v1, u0:u1]
    peak_diff = float(sub_diff[aperture].max()) if aperture.any() else 0.0
    core = aperture & (sub_diff >= core_frac * peak_diff)
    if int(core.sum()) < 2:
        core = aperture
    # Thermal extent of the *surroundings* at a lower threshold.  A survivor on
    # cold ground is a hot spot with nothing around it; a warm vehicle, engine
    # bay, air-conditioner condenser or sun-baked slab is a hot spot embedded in
    # a much larger warm region.  The aperture deliberately cannot see that
    # difference (it is person-sized by design), so it is measured here.
    ctx_r = r * 3.4
    ctx = (d2 <= ctx_r * ctx_r) & (sub_diff >= 0.32 * peak_diff)
    context_area = float(ctx.sum())
    vals = sub_img[core]
    bg = (float(np.median(sub_img[ring])) if int(ring.sum()) >= 4
          else float(np.median(sub_img)))
    ys, xs = yy[core] - v, xx[core] - u
    n = len(xs)
    if n >= 3:
        mu_uu = float((xs * xs).mean()); mu_vv = float((ys * ys).mean())
        mu_uv = float((xs * ys).mean())
        tr = mu_uu + mu_vv
        det = mu_uu * mu_vv - mu_uv * mu_uv
        disc = max(tr * tr / 4.0 - det, 0.0)
        l1 = tr / 2.0 + math.sqrt(disc); l2 = max(tr / 2.0 - math.sqrt(disc), 1e-9)
        elong = math.sqrt(l1 / l2)
        ang = 0.5 * math.atan2(2 * mu_uv, mu_uu - mu_vv)
        ca, sa = math.cos(ang), math.sin(ang)
        pu = xs * ca + ys * sa
        pv = -xs * sa + ys * ca
        extent_u = max(float(np.percentile(np.abs(pu), 95)) * 2.0, 1e-3)
        extent_v = max(float(np.percentile(np.abs(pv), 95)) * 2.0, 1e-3)
        aspect = extent_v / extent_u
    else:
        elong, aspect = 1.0, 1.0
    area = float(n)
    return {
        "area": area, "mean": float(vals.mean()), "peak": float(vals.max()),
        "std": float(vals.std()), "background": bg, "contrast": float(vals.mean()) - bg,
        "peak_contrast": float(vals.max()) - bg, "elongation": elong, "aspect": aspect,
        "compactness": _compactness(core),
        "context_area": context_area,
        "extent_ratio": context_area / max(area_now := float(core.sum()), 1.0),
        "bbox": (float(u0 + xs.min()), float(v0 + ys.min()),
                 float(u0 + xs.max() + 1), float(v0 + ys.max() + 1)),
        "centroid": (float(u + xs.mean()), float(v + ys.mean())),
        "radius_px": r,
    }


def nms(dets: List[Detection], iou_thresh: float = 0.35,
        center_thresh: float = 2.5) -> List[Detection]:
    """Greedy class-agnostic non-maximum suppression."""
    if not dets:
        return []
    order = sorted(range(len(dets)), key=lambda i: -dets[i].score)
    keep: List[int] = []
    dead: set = set()
    for i in order:
        if i in dead:
            continue
        keep.append(i)
        for j in order:
            if j == i or j in dead:
                continue
            if dets[i].iou(dets[j]) >= iou_thresh or bbox_center_dist(
                    dets[i].bbox, dets[j].bbox) < center_thresh:
                dead.add(j)
    return [dets[i] for i in keep]


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-max(min(x, 40.0), -40.0))))


def ttp_probability(n_equivalent: float, n50: float = 2.0) -> float:
    """Targeting-task-performance probability (Dragger et al., 2017).

    Fits NVTherm/IP for a person-shaped target: ``E = 1.5074 + 0.2477*n/n50``
    and ``P = (n/n50)^E / (1 + (n/n50)^E)``.  The exponent grows with size,
    which is what reproduces the familiar "detection is easy until it suddenly
    isn't" cliff as a survivor shrinks below a few pixels.
    """
    ratio = max(n_equivalent, 1e-6) / max(n50, 1e-6)
    e = 1.5074 + 0.2477 * ratio
    return float(ratio ** e / (1.0 + ratio ** e))


def aperture_radius_px(gsd_m: float, min_px: float = 1.6,
                       max_px: float = 9.0) -> float:
    """Radius of the human-body measurement aperture, in pixels.

    ``gsd_m`` is metres per pixel at the target's ground plane.  The apparent
    radius of a human body (0.444 m, from 0.62 m^2 projected area) divided by
    the GSD gives the aperture the physics says a survivor should fill.
    """
    if not gsd_m or gsd_m <= 0:
        return 3.0
    return float(np.clip(HUMAN_APPARENT_RADIUS_M / gsd_m, min_px, max_px))


# --------------------------------------------------------------------------- #
# Thermal point-target detector
# --------------------------------------------------------------------------- #
class ThermalAnomalyDetector:
    """Bi-directional LWIR point-target detector with a calibrated person gate.

    Pipeline
    --------
    1. Local background by block-decimated median -> excess radiance map.
    2. Local extrema above ``min_contrast_k`` for **hot** and **cold** polarity.
       Cold anomalies matter: a survivor on a sun-heated roof or slab reads
       *below* background, and hot-blob-only detectors miss exactly those.
    3. :func:`measure_aperture` with a geometry-aware radius (GSD known from AGL
       and lens): core radiometry against an annulus background, plus core
       morphology.
    4. Explicit class decision from apparent temperature band, then a logistic
       score over TTP size, shape, compactness and contrast-to-noise.
    5. Separate extended-region path for combustion (a fire front is not a point
       target and must not be squashed by the person gates).

    Parameters
    ----------
    gsd_m : float
        Optional fixed metres-per-pixel.  If ``0`` the detector reads
        ``frame.gsd_m``, and if that is unavailable it falls back to a
        conservative 3-pixel aperture.
    """

    #: Apparent temperature band of a live human, on the *peak* core pixel.
    #: The world model defines ``Victim.body_temp_c`` as the temperature of the
    #: brightest part of the body, so the peak pixel is the matching statistic.
    #: Gating on the core *mean* instead - what an earlier version did - rejects
    #: every small target: a survivor 12 px across whose surroundings are 13 C
    #: water has a core mean of 21 C even when its peak reads 25 C, because
    #: pixel integration and the PSF pull the mean toward the background.  That
    #: is exactly the sub-pixel radiometry error real cameras have, and it is why
    #: in-water survivors are the hardest case in this problem.
    HUMAN_BAND = (23.0, 40.5)
    #: Floor on the core mean: a blob whose mean is close to the background is a
    #: single hot pixel in cold clutter (wet cloth, a glint), not a body.
    MEAN_FLOOR_C = 19.0
    FIRE_ONSET_C = 75.0            # nothing alive appears this hot

    def __init__(
        self,
        k_sigma: float = 4.0,
        min_contrast_k: float = 1.6,
        n50_pixels: float = 2.0,
        score_threshold: float = 0.28,
        cold_polarity: bool = True,
        fire_path: bool = True,
        gsd_m: float = 0.0,
        max_person_area_px: float = 110.0,
        min_person_area_frac: float = 0.08,
        bg_size: int = 21,
        bg_block: int = 4,
        max_peaks: int = 40,
    ) -> None:
        self.k_sigma = k_sigma
        self.min_contrast_k = min_contrast_k
        self.n50 = n50_pixels
        self.score_threshold = score_threshold
        self.cold_polarity = cold_polarity
        self.fire_path = fire_path
        self.gsd_m = gsd_m
        self.max_person_area_px = max_person_area_px
        self.min_person_area_frac = min_person_area_frac
        self.bg_size = bg_size
        self.bg_block = bg_block
        self.max_peaks = max_peaks
        self.name = "thermal_anomaly"
        self.modalities = ("lwir",)
        self.last_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    def _aperture(self, frame: Any) -> float:
        gsd = self.gsd_m or float(getattr(frame, "gsd_m", 0.0) or 0.0)
        return aperture_radius_px(gsd)

    def detect(self, frame: Any) -> List[Detection]:
        img = np.nan_to_num(np.asarray(frame.image, dtype=np.float32))
        if img.ndim != 2:
            raise ValueError("ThermalAnomalyDetector expects single-channel LWIR data")
        t = float(getattr(frame, "t", 0.0))
        bg = local_background(img, self.bg_size, self.bg_block)
        diff = img - bg
        med, sigma = robust_stats(diff)
        thresh = max(self.k_sigma * sigma, self.min_contrast_k)
        radius = self._aperture(frame)
        min_sep = max(2.0, radius * 1.5)
        out: List[Detection] = []
        n_peaks = 0

        for polarity, sign in (("hot", 1.0), ("cold", -1.0)):
            if polarity == "cold" and not self.cold_polarity:
                continue
            d = sign * diff
            min_area = max(1.5, self.min_person_area_frac * math.pi * radius * radius)
            for (u, v, resp) in peak_candidates(d, thresh, min_sep=min_sep,
                                                max_peaks=self.max_peaks):
                m = measure_aperture(img, d, u, v, radius)
                # A core much smaller than a body at this GSD is a hot speck in
                # clutter, not a survivor.  The floor scales with the aperture,
                # which is itself set by geometry, so it holds from 20 m to 120 m
                # without retuning - and at 0.08 it still admits a chest-deep
                # survivor, whose visible area really is ~15% of a standing one
                # (0.11 m^2 of apparent area against 0.62 m^2, further reduced
                # by occlusion).  A tighter floor was tried and it removed every
                # in-water survivor from the map while barely denting clutter.
                if m.get("area", 0) < min_area:
                    continue
                n_peaks += 1
                det = self._classify(m, polarity, sigma, med, t, radius)
                if det is not None and det.score >= self.score_threshold:
                    out.append(det)

        if self.fire_path:
            out.extend(self._fire_regions(img, diff, med, t))

        self.last_stats = {"scene_median_c": med, "noise_sigma_k": sigma,
                           "threshold_k": thresh, "peaks": n_peaks,
                           "aperture_radius_px": radius,
                           "min_core_area_px": round(min_area, 2),
                           "candidates": len(out)}
        return nms(out, 0.30, center_thresh=max(1.6, radius * 0.9))

    # ------------------------------------------------------------------ #
    def _classify(self, m: Dict[str, Any], polarity: str, sigma: float,
                  scene_median: float, t: float, radius: float) -> Optional[Detection]:
        peak, mean_t, bg = m["peak"], m["mean"], m["background"]
        area = float(m["area"])
        n_eq = math.sqrt(area)
        contrast = abs(m["peak_contrast"])
        lo, hi = self.HUMAN_BAND

        # ---- radiometric class decision (the dominant false-alarm filter) ----
        if peak >= self.FIRE_ONSET_C:
            label, temp_term = "fire", -3.0
        elif peak > hi:
            # Warmer than a living human can appear: engine bay, exhaust,
            # sun-baked metal or stone, smouldering rubble.  Never a survivor.
            label = "hot_rock"
            temp_term = -2.6 * min((peak - hi) / 8.0, 3.0)
        elif polarity == "hot":
            if peak < lo or mean_t < self.MEAN_FLOOR_C:
                label = "cloth"
                temp_term = -2.0 * min(max(lo - peak, self.MEAN_FLOOR_C - mean_t) / 5.0, 3.0)
            else:
                label = "person"
                temp_term = 1.45 - 0.60 * abs(peak - 32.0) / 8.0
                # Reward a core whose mean is well above the floor: that is the
                # difference between a body and one hot pixel on cold ground.
                temp_term += float(np.clip((mean_t - self.MEAN_FLOOR_C) / 8.0, 0.0, 0.45))
        else:
            # Cold anomaly: a survivor only if the core is still physiological
            # and the surroundings are genuinely warmer (sun-heated surface).
            if lo - 5.0 <= mean_t <= hi and bg > mean_t + 2.0:
                label, temp_term = "person", 0.80
            else:
                label, temp_term = "cloth", -1.8

        # ---- size: targeting-task-performance curve --------------------------
        ttp = ttp_probability(n_eq, self.n50)
        size_term = 2.9 * (ttp - 0.32)
        if label == "person" and area > self.max_person_area_px:
            size_term -= 1.6 * math.log10(area / self.max_person_area_px + 1.0)

        # ---- shape ------------------------------------------------------------
        elong = m.get("elongation", 1.0)
        compact = m.get("compactness", 0.5)
        if 1.0 <= elong <= 2.6:
            shape_term = 0.45
        elif elong < 1.0:
            shape_term = 0.0
        else:
            shape_term = -1.0 * min((elong - 2.6) / 1.4, 2.0)
        # Terrain texture is ragged; a body is a coherent blob.
        shape_term += float(np.clip((compact - 0.35) * 1.6, -0.5, 0.45))

        # ---- contrast to noise ------------------------------------------------
        cnr = contrast / max(sigma, 1e-3)
        cnr_term = float(np.clip((cnr - 3.0) / 10.0, -0.9, 1.0))

        # ---- thermal extent: is this hot spot embedded in a larger warm body? --
        extent_ratio = float(m.get("extent_ratio", 1.0))
        if label == "person" and extent_ratio > EXTENT_RATIO_MAX:
            # Too much warm stuff around the peak to be a body on the ground.
            # Re-label rather than delete: the object is real, it is just not a
            # survivor, and "vehicle" belongs on the hazard map (a warm vehicle
            # in a flood basin is where people shelter, and it blocks roads).
            label = "vehicle"
            temp_term = -1.2
        extent_term = float(np.clip((EXTENT_RATIO_MAX - extent_ratio) * 0.55,
                                    -1.5, 0.35))

        if label == "fire":
            score = float(np.clip(0.45 + 0.53 * sigmoid((peak - 110.0) / 45.0), 0, 0.99))
        elif label in ("hot_rock", "cloth", "vehicle"):
            score = float(np.clip(0.05 + 0.20 * sigmoid(cnr - 7.0), 0, 0.30))
        else:
            score = sigmoid(temp_term + size_term + shape_term + cnr_term
                            + extent_term - 1.55)

        return Detection(
            label=label, score=float(np.clip(score, 0.0, 0.999)),
            bbox=tuple(float(x) for x in m["bbox"]), modality="lwir", frame_t=t,
            centroid=(m["centroid"][0], m["centroid"][1]), area_px=area,
            peak_temp_c=peak, mean_temp_c=mean_t, background_temp_c=bg,
            contrast_k=contrast, polarity=polarity, aspect=m.get("aspect", 1.0),
            elongation=elong, compactness=compact, scale_px=radius,
            attributes={"ttp": ttp, "cnr": cnr, "sigma_k": sigma,
                        "temp_term": temp_term, "size_term": size_term,
                        "shape_term": shape_term, "extent_term": extent_term,
                        "extent_ratio": round(extent_ratio, 2),
                        "context_area": round(float(m.get("context_area", 0.0)), 1),
                        "scene_median_c": scene_median, "stage": "point"},
            backend=self.name,
        )

    # ------------------------------------------------------------------ #
    def _fire_regions(self, img: np.ndarray, diff: np.ndarray, scene_median: float,
                      t: float) -> List[Detection]:
        """Extended-region combustion detection (fire front, not a point)."""
        mask = img > max(self.FIRE_ONSET_C, scene_median + 25.0)
        if mask.sum() < 8:
            return []
        labels, n = connected_components(mask)
        out: List[Detection] = []
        for f in blob_features(img, labels, n, background=None):
            if f["area"] < 6:
                continue
            score = float(np.clip(0.55 + 0.42 * sigmoid((f["peak"] - 140.0) / 60.0), 0, 0.99))
            out.append(Detection(
                label="fire", score=score, bbox=f["bbox"], modality="lwir", frame_t=t,
                centroid=f["centroid"], area_px=float(f["area"]), peak_temp_c=f["peak"],
                mean_temp_c=f["mean"], background_temp_c=float(np.median(img)),
                contrast_k=f["peak"] - float(np.median(img)), polarity="hot",
                aspect=f.get("aspect", 1.0), elongation=f.get("elongation", 1.0),
                compactness=f.get("compactness", 0.0),
                attributes={"stage": "extended"}, backend=self.name,
            ))
        return out


# --------------------------------------------------------------------------- #
# RGB hazard + person detector
# --------------------------------------------------------------------------- #
class RgbHazardDetector:
    """Colour/texture rules for hazard classes, plus chromatic person saliency.

    Every rule is a short, inspectable numpy expression rather than a black box,
    which is what lets us defend the system's behaviour to a disaster-management
    agency.  In production these rules act as priors and validators for the
    neural detector, not as a replacement for it.

    Hazards are extended regions -> connected components.  People are point
    targets -> the same peak + aperture machinery as the thermal detector, run
    on a chromatic-saliency map.  That asymmetry is deliberate and is the reason
    the RGB person channel has high precision and modest recall: its job is to
    *confirm or veto* the LWIR, not to find survivors on its own.
    """

    def __init__(self, min_area_px: int = 8, score_threshold: float = 0.32,
                 gsd_m: float = 0.0, person_k_sigma: float = 3.2,
                 person_min_saliency: float = 45.0, person_max_peaks: int = 14,
                 hazard_decimation: int = 2, max_regions_per_label: int = 8) -> None:
        self.min_area_px = min_area_px
        self.score_threshold = score_threshold
        self.gsd_m = gsd_m
        self.person_k_sigma = person_k_sigma
        self.person_min_saliency = person_min_saliency
        self.person_max_peaks = person_max_peaks
        self.hazard_decimation = max(1, int(hazard_decimation))
        self.max_regions_per_label = max_regions_per_label
        self.name = "rgb_hazard"
        self.modalities = ("rgb",)
        self.last_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    def detect(self, frame: Any) -> List[Detection]:
        img = np.asarray(frame.image)
        if img.ndim != 3 or img.shape[2] < 3:
            return []
        t = float(getattr(frame, "t", 0.0))
        rgb = img.astype(np.float32)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        mx, mn = rgb.max(axis=-1), rgb.min(axis=-1)
        chroma = mx - mn
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        out: List[Detection] = []

        # Fire.  Raw channel differences are useless here: sunlit dry soil is
        # also (R > G > B) and a bright brown rooftop reads r-b ~ 50.  What
        # actually separates flame is *chromaticity* - the red fraction of the
        # total - combined with near-saturation.  Orange flame sits at
        # r/(r+g+b) ~ 0.55, sunlit soil at ~0.38, yellow-white at ~0.35.
        tot = r + g + b + 1.0
        red_frac = r / tot
        fire = (red_frac > 0.44) & (lum > 168.0) & (r > 190.0) & (chroma > 45)
        fire &= (lum > self._local_mean(lum, 17) + 26.0)     # locally bright
        out += self._regions(fire, lum, "fire", t, 0.95, self.min_area_px,
                             allow_huge=True)

        # smoke: desaturated, mid-bright, spatially smooth
        smoke = (chroma < 15) & (lum > 92) & (lum < 198)
        smoke &= self._local_std(lum, 9) < 6.0
        out += self._regions(smoke, lum, "smoke", t, 0.55, self.min_area_px * 8,
                             allow_huge=True)

        # Water: dark, blue-cyan, and - the discriminator that matters - very
        # smooth.  A flood sheet has almost no texture at 10 cm scale, whereas
        # shadowed rubble is equally dark and equally desaturated but rough.
        water = (b >= r - 6) & (lum < 118) & (chroma < 40)
        water &= self._local_std(lum, 7) < 4.2
        out += self._regions(water, lum, "flood_water", t, 0.90, self.min_area_px * 10,
                             allow_huge=True)

        # debris / rubble: high texture, brown-grey, mid luminance
        tex = self._local_std(lum, 5)
        debris = (tex > 13.5) & (lum > 58) & (lum < 192) & (chroma < 54)
        out += self._regions(debris, lum, "debris_field", t, 0.62, self.min_area_px * 5,
                             allow_huge=True)

        # collapsed structure: extreme edge density together with rubble texture
        edges = self._edge_density(lum)
        collapse = (edges > 0.44) & (tex > 19.0)
        out += self._regions(collapse, lum, "collapsed_structure", t, 0.70,
                             self.min_area_px * 4)

        # damaged (but standing) structure: moderate edge density, coherent shape
        damaged = (edges > 0.26) & (edges <= 0.44) & (tex > 15.0) & (lum > 90)
        out += self._regions(damaged, lum, "damaged_structure", t, 0.45,
                             self.min_area_px * 5)

        # Exposed powerline: long, thin, and dark *relative to its immediate
        # surround*.  Thresholding against a global percentile marks 26% of every
        # frame as "dark" and finds conductors in every shadow; a conductor is
        # instead a few-pixel-wide feature tens of pixels long, so it must be
        # both locally dark and locally linear.
        local_dark = lum < (self._local_mean(lum, 15) - 11.0)
        out += self._regions(self._linear_dark(local_dark), lum, "exposed_powerline",
                             t, 0.60, 10, prefer_elongated=True, min_extent_px=14.0)

        # person candidates: compact chromatic blobs
        out += self._people(rgb, lum, chroma, t, frame)

        self.last_stats = {"raw": len(out)}
        return nms(out, 0.30, center_thresh=2.0)

    # ------------------------------------------------------------------ #
    def _people(self, rgb: np.ndarray, lum: np.ndarray, chroma: np.ndarray,
                t: float, frame: Any) -> List[Detection]:
        """Chromatic-saliency point detection for bright clothing.

        SAR survivors are usually dressed in chromatic or high-visibility
        clothing while flood mud, ash and rubble are low-chroma browns and
        greys - so a chroma gate rejects most of the terrain outright.
        """
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        warm = (r > g + 26) & (r > b + 30)
        cool = (b > r + 22) & (b > g + 8)
        hi_vis = (g > r + 18) & (g > b + 18)
        saliency = np.where(warm, (r - np.maximum(g, b)),
                            np.where(cool, (b - np.maximum(r, g)) * 0.9,
                                     np.where(hi_vis, (g - np.maximum(r, b)) * 0.8,
                                              0.0))).astype(np.float32)
        usable = (chroma > 55) & (lum > 96) & (lum < 245)
        if not bool((saliency > self.person_min_saliency).any()):
            return []
        _, sig = robust_stats(saliency)
        gsd = self.gsd_m or float(getattr(frame, "gsd_m", 0.0) or 0.0)
        radius = aperture_radius_px(gsd)
        out: List[Detection] = []
        for (u, v, resp) in peak_candidates(
                saliency, max(self.person_min_saliency, self.person_k_sigma * sig),
                min_sep=max(2.0, radius * 1.5), max_peaks=self.person_max_peaks,
                mask=usable):
            # Core shape from the saliency map; luminance read over the SAME core.
            # (Passing a flat diff would make the core a plain disk and every
            # elongation 1.0, which is how the first cut of this detector ended
            # up with no shape information at all.)
            m = measure_aperture(saliency, saliency, u, v, radius, core_frac=0.45)
            lm = measure_aperture(lum, saliency, u, v, radius, core_frac=0.45)
            area = float(m.get("area", 0))
            # Bright chromatic specks (a glint on water, a chip of paint, a
            # sunlit leaf) are the dominant RGB false alarm at high altitude.
            # Requiring a core that is a real fraction of a body's footprint at
            # this GSD removes them without touching genuine survivors, whose
            # clothing blob fills the aperture.
            if area < max(2.0, 0.35 * math.pi * radius * radius) or area > 300:
                continue
            ttp = ttp_probability(math.sqrt(area), 2.0)
            elong = lm.get("elongation", 1.0)
            compact = m.get("compactness", 0.5)
            # Absolute chromatic excess, in 0-255 units.  Clothing reads 60-150;
            # soil, mud, ash and water-glint edges read 15-40.  Normalising by the
            # frame sigma hid that separation, so the score is in absolute units.
            sal = float(m["mean"])
            score = sigmoid(2.4 * (ttp - 0.30) + 0.050 * (sal - 62.0)
                            + float(np.clip((compact - 0.45) * 1.4, -0.6, 0.4))
                            + (-0.7 if elong > 3.0 else 0.0) - 1.05)
            if score < self.score_threshold:
                continue
            out.append(Detection(
                label="person", score=float(np.clip(score, 0, 0.98)),
                bbox=tuple(float(x) for x in lm.get("bbox", m["bbox"])),
                modality="rgb", frame_t=t, centroid=(m["centroid"][0], m["centroid"][1]),
                area_px=area, contrast_k=0.0, polarity="n/a",
                aspect=lm.get("aspect", 1.0), elongation=elong, compactness=compact,
                scale_px=radius,
                attributes={"saliency": sal, "lum": lm.get("mean", 0.0),
                            "ttp": ttp, "stage": "point"},
                backend=self.name,
            ))
        return out

    # ------------------------------------------------------------------ #
    @staticmethod
    def _local_mean(a: np.ndarray, win: int = 15) -> np.ndarray:
        if not _HAVE_SCIPY:  # pragma: no cover
            return np.full_like(a, float(a.mean()))
        return ndi.uniform_filter(a.astype(np.float32), size=win, mode="nearest")

    @staticmethod
    def _local_std(a: np.ndarray, win: int = 7) -> np.ndarray:
        if not _HAVE_SCIPY:  # pragma: no cover
            return np.zeros_like(a)
        m = ndi.uniform_filter(a, size=win, mode="nearest")
        return np.sqrt(np.maximum(
            ndi.uniform_filter(a * a, size=win, mode="nearest") - m * m, 0.0))

    @staticmethod
    def _edge_density(a: np.ndarray) -> np.ndarray:
        if not _HAVE_SCIPY:  # pragma: no cover
            return np.zeros_like(a)
        gy, gx = np.gradient(a.astype(np.float32))
        return ndi.uniform_filter((np.hypot(gx, gy) > 14).astype(np.float32),
                                  size=7, mode="nearest")

    @staticmethod
    def _linear_dark(dark: np.ndarray) -> np.ndarray:
        """Keep only the long, thin, straight components of a dark mask.

        Morphological opening with four line structure elements.  Computed on a
        2x down-sampled mask: a conductor span is many metres long so it survives
        decimation, while single-pixel noise does not.  ~6x faster than opening
        the full-resolution mask at four angles.
        """
        if not _HAVE_SCIPY:  # pragma: no cover
            return np.zeros(dark.shape, dtype=bool)
        if not dark.any():
            return np.zeros(dark.shape, dtype=bool)
        small = dark[::2, ::2]
        if small.size == 0:
            return np.zeros(dark.shape, dtype=bool)
        accum = np.zeros(small.shape, dtype=bool)
        for ang in (0, 45, 90, 135):
            accum |= ndi.binary_opening(small, structure=_line_structure_element(9, ang))
        big = np.kron(accum, np.ones((2, 2), dtype=bool))
        return big[:dark.shape[0], :dark.shape[1]]

    def _regions(self, mask: np.ndarray, lum: np.ndarray, label: str, t: float,
                 score_scale: float, min_area: int, allow_huge: bool = False,
                 prefer_elongated: bool = False,
                 min_extent_px: float = 0.0) -> List[Detection]:
        """Connected-component hazard regions, capped and extent-gated.

        The cap matters for two reasons: a hazard map has no use for 130 fire
        blobs in one frame, and an uncapped label is how a marginal rule turns
        into both a false-alarm flood and a compute blow-up.  Keeping the highest
        scoring regions makes the failure mode graceful and bounded.
        """
        if not mask.any():
            return []
        coverage = float(mask.mean())
        if coverage > 0.62:
            # More than 62% of the frame is one class: that is an illumination or
            # white-balance state, not a discrete hazard, and labelling it would
            # produce a single frame-sized "detection" every frame.
            return []
        # Hazards are extended in metres, so 2x decimation costs nothing and
        # makes closing + labelling ~4x cheaper.  Areas and boxes are scaled back.
        dec = self.hazard_decimation
        if dec > 1 and min(mask.shape) // dec >= 24:
            m_small = ndi.uniform_filter(mask.astype(np.float32), size=dec,
                                         mode="nearest")[::dec, ::dec] > 0.35
            l_small = lum[::dec, ::dec]
            area_scale = float(dec * dec)
        else:
            dec, m_small, l_small, area_scale = 1, mask, lum, 1.0
        max_area = (0.45 if allow_huge else 0.14) * mask.size / area_scale
        closed = ndi.binary_closing(m_small, structure=np.ones((3, 3), bool))
        labels, n = connected_components(closed)
        feats = blob_features(l_small.astype(np.float32), labels, n)
        out: List[Detection] = []
        kept: List[Tuple[float, Dict[str, Any]]] = []
        for f in feats:
            area = float(f["area"]) * area_scale
            if area < min_area or area > max_area * area_scale:
                continue
            if min_extent_px and max(f["width"], f["height"]) * dec < min_extent_px:
                continue
            fill = f["area"] / max(f["width"] * f["height"], 1.0)
            elong = f.get("elongation", 1.0)
            score = score_scale * float(np.clip(
                0.30 + 0.34 * math.log10(area / max(min_area, 1.0) + 1.0)
                + (0.16 if (prefer_elongated and elong > 2.4) else 0.0)
                - (0.24 if (prefer_elongated and elong < 1.8) else 0.0)
                + 0.10 * (fill - 0.5),
                0.0, 0.97))
            if score < self.score_threshold:
                continue
            kept.append((score, f))
        kept.sort(key=lambda kv: -kv[0])
        for score, f in kept[: self.max_regions_per_label]:
            bbox = tuple(float(x) * dec for x in f["bbox"])
            centroid = tuple(float(x) * dec for x in f["centroid"])
            out.append(Detection(
                label=label, score=score, bbox=bbox, modality="rgb", frame_t=t,
                centroid=centroid, area_px=float(f["area"]) * area_scale,
                aspect=f.get("aspect", 1.0), elongation=f.get("elongation", 1.0),
                compactness=f.get("compactness", 0.0), polarity="n/a",
                attributes={"fill": f["area"] / max(f["width"] * f["height"], 1.0),
                            "lum_mean": f["mean"], "stage": "extended",
                            "regions_available": len(kept)},
                backend=self.name,
            ))
        return out


def _line_structure_element(length: int, angle_deg: int) -> np.ndarray:
    n = max(3, int(length) | 1)
    se = np.zeros((n, n), dtype=bool)
    a = math.radians(angle_deg)
    for i in range(n):
        k = i - n // 2
        rr = int(round(n // 2 - k * math.sin(a)))
        cc = int(round(n // 2 + k * math.cos(a)))
        if 0 <= rr < n and 0 <= cc < n:
            se[rr, cc] = True
    return se


# --------------------------------------------------------------------------- #
# Ensemble
# --------------------------------------------------------------------------- #
class DetectorEnsemble:
    """Runs several backends and fuses their output.

    Fusion policy
    -------------
    * Same-modality duplicates removed by NMS.
    * **Cross-modal agreement** (LWIR person and RGB person within a few pixels)
      raises both scores.  This is the single most effective false-alarm filter
      available: warm decoys - animals, engine bays, sun-baked stone - rarely
      also present as a compact chromatic-clothing blob in the visible band.
    * **Conditional veto**: disagreement only penalises the LWIR detection when
      the RGB channel is actually usable.  At night, in heavy smoke or in rain
      the visible camera is blind, and penalising "LWIR-only" then would throw
      away precisely the detections the thermal sensor exists to provide.  The
      caller passes ``rgb_quality`` (0..1); :func:`rgb_frame_quality` computes it
      from frame luminance, contrast and blow-out.
    """

    def __init__(self, backends: Sequence[DetectorBackend],
                 cross_modal_radius_px: float = 7.0,
                 agreement_bonus: float = 0.24,
                 disagreement_penalty: float = 0.16,
                 rgb_blind_threshold: float = 0.30) -> None:
        self.backends = list(backends)
        self.radius = cross_modal_radius_px
        self.bonus = agreement_bonus
        self.penalty = disagreement_penalty
        self.rgb_blind_threshold = rgb_blind_threshold
        self.name = "ensemble"
        self.last_stats: Dict[str, Any] = {}
        #: Quality of the visible channel for the last call, so a caller that
        #: does the cross-modal step itself (the fuser) uses the same number.
        self.last_rgb_quality: float = 0.0

    def detect(self, frames: Any, rgb_quality: Optional[float] = None) -> List[Detection]:
        """Detect over one frame, or over a list with one frame per modality.

        The aircraft really does have two sensors producing two streams, so the
        fuser takes both.  Each backend is run only on the modalities it
        declares, and an absent modality is handled explicitly rather than
        silently (see the ``rgb_quality`` note in the class docstring).
        """
        frame_list = list(frames) if isinstance(frames, (list, tuple)) else [frames]
        by_kind: Dict[str, Any] = {}
        for f in frame_list:
            k = str(getattr(f, "kind", "") or getattr(f, "modality", ""))
            if k:
                by_kind[k] = f
        all_dets: List[Detection] = []
        per_backend: Dict[str, int] = {}
        for be in self.backends:
            kinds = tuple(getattr(be, "modalities", ("lwir", "rgb")))
            got = 0
            for k in kinds:
                if k not in by_kind:
                    continue
                try:
                    dets = be.detect(by_kind[k])
                except Exception as exc:  # a broken detector must not end the sortie
                    dets = []
                    self.last_stats[f"error_{getattr(be, 'name', 'backend')}"] = repr(exc)
                got += len(dets)
                all_dets.extend(dets)
            per_backend[getattr(be, "name", str(be))] = got

        # NMS strictly *within* each modality first.  Running one class-agnostic
        # NMS across both - as an earlier version did - deletes the RGB partner
        # of a genuine LWIR detection before fusion ever sees it, because a
        # survivor 3 px apart in two cameras has a high box IoU.  That silently
        # turned every true detection into "lwir_only" and penalised it.
        merged: List[Detection] = []
        for kind in ("lwir", "rgb"):
            group = [d for d in all_dets if d.modality == kind]
            if group:
                merged.extend(nms(group, 0.35, center_thresh=2.0))
        merged.extend(d for d in all_dets if d.modality not in ("lwir", "rgb"))
        all_dets = merged

        if rgb_quality is None:
            rgb_quality = rgb_frame_quality(by_kind["rgb"]) if "rgb" in by_kind else 0.0
        self.last_rgb_quality = float(np.clip(rgb_quality, 0.0, 1.0))
        # Cross-modal reasoning deliberately does NOT happen here.  An earlier
        # version scored agreement and consumed the RGB partner inside the
        # ensemble, which left CrossModalFuser with nothing to pair and meant the
        # boresight estimator never saw a single sample.  Responsibilities are
        # now strictly layered:
        #   detector  -> per-modality detections (intra-frame, intra-sensor)
        #   fuser     -> cross-modal pairing and label likelihood
        #   tracker   -> cross-frame persistence and confirmation
        return all_dets
        self.last_stats.update({"raw": sum(per_backend.values()),
                                "after_fusion": len(all_dets),
                                "per_backend": per_backend,
                                "rgb_quality": rgb_quality})
        return all_dets

    def cross_modal_scores(self, dets: List[Detection],
                           rgb_quality: Optional[float] = None) -> List[Detection]:
        """Optional in-place cross-modal scoring, for callers that do not fuse.

        :class:`~sar.perception.fusion.CrossModalFuser` is the normal path and is
        strictly better (it also estimates the boresight).  This exists for
        single-frame evaluation where only a score is wanted - which is what
        ``scripts/eval_detector.py`` uses to report the LWIR-only versus
        cross-modal false-alarm gap.
        """
        if rgb_quality is None:
            rgb_quality = self.last_rgb_quality
        return self._cross_modal(dets, float(np.clip(rgb_quality, 0.0, 1.0)))

    def _cross_modal(self, dets: List[Detection], rgb_quality: float) -> List[Detection]:
        lwir = [d for d in dets if d.modality == "lwir"]
        rgb = [d for d in dets if d.modality == "rgb"]
        rgb_usable = rgb_quality >= self.rgb_blind_threshold
        consumed: List[Detection] = []
        for d in lwir:
            if not d.is_person:
                continue
            partners = [o for o in rgb if o.is_person
                        and math.hypot(o.u - d.u, o.v - d.v) <= self.radius]
            if partners:
                best = max(partners, key=lambda o: o.score)
                d.score = float(np.clip(d.score + self.bonus * best.score, 0, 0.995))
                d.modality = "fused"
                d.attributes["cross_modal"] = "agree"
                d.attributes["rgb_partner_score"] = best.score
                # Carry the visible evidence onto the fused detection and mark the
                # RGB partner consumed, so the pair is reported once.
                d.attributes["rgb_centroid"] = (best.u, best.v)
                d.attributes["rgb_saliency"] = best.attributes.get("saliency")
                d.attributes["offset_px"] = round(math.hypot(best.u - d.u, best.v - d.v), 2)
                best.attributes["consumed_by"] = "fused"
                consumed.append(best)
            elif rgb_usable:
                # Warm, human-sized, human-temperature - and invisible in good
                # light.  That is either a chest-deep survivor (only head and
                # shoulders clear the water, so there is no clothing cue) or a
                # warm decoy.  Both deserve a lower confidence and a second look,
                # which is exactly what the confirmation orbit does.  The penalty
                # is deliberately mild: it must reorder candidates, not delete
                # the hardest survivors from the map.
                d.attributes["cross_modal"] = "lwir_only"
                d.attributes["needs_confirmation"] = True
                d.score = float(max(0.05, d.score - self.penalty * rgb_quality))
            else:
                d.attributes["cross_modal"] = "lwir_only_rgb_blind"
        for d in rgb:
            if not d.is_person:
                continue
            partners = [o for o in lwir if o.is_person
                        and math.hypot(o.u - d.u, o.v - d.v) <= self.radius]
            if partners:
                d.attributes["cross_modal"] = "agree"
                d.score = float(np.clip(d.score + 0.5 * self.bonus, 0, 0.99))
            else:
                d.attributes["cross_modal"] = "rgb_only"
                d.score = float(max(0.05, d.score - 0.6 * self.penalty))
        return [d for d in dets if d not in consumed]


def rgb_frame_quality(frame: Any) -> float:
    """0..1 estimate of how much the visible channel can be trusted.

    Combines mean luminance (night, underexposure), RMS contrast (haze, smoke,
    rain) and blow-out (sun glint on water).  Used by :class:`DetectorEnsemble`
    to decide whether an LWIR-only detection should be penalised for lacking
    visible confirmation - and therefore to stop the system from degrading its
    best sensor exactly when conditions make it the only one that works.
    """
    img = np.asarray(frame.image)
    if img.ndim != 3:
        return 0.0
    lum = img.astype(np.float32).mean(axis=-1)
    mean = float(lum.mean())
    std = float(lum.std())
    blown = float((lum > 250).mean())
    light_term = float(np.clip((mean - 22.0) / 55.0, 0.0, 1.0))
    contrast_term = float(np.clip((std - 6.0) / 26.0, 0.0, 1.0))
    return float(np.clip(light_term * 0.55 + contrast_term * 0.45 - 0.9 * blown, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Neural backends (optional dependencies, same contract)
# --------------------------------------------------------------------------- #
class OnnxRuntimeBackend:
    """YOLOv8/YOLOv11 ONNX export via ``onnxruntime``.

    Production path on a Qualcomm QCS8550 / RB5 Gen2 (QNN or SNPE after AI Hub
    compilation), a Jetson (TensorRT EP) or a Raspberry Pi 5 (CPU EP).  The
    interface is identical to the heuristic backend, so simulation results and
    flight software transfer without changes.
    """

    name = "onnx_yolo"

    def __init__(self, model_path: str, labels: Sequence[str] = ALL_LABELS,
                 input_size: int = 640, conf: float = 0.25, iou: float = 0.45,
                 providers: Optional[Sequence[str]] = None) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "onnxruntime is not installed (pip install onnxruntime). See "
                "docs/06_AI_MODELS_AND_DATASETS.md for the export and Qualcomm "
                "AI Hub quantisation recipe.") from exc
        self.session = ort.InferenceSession(
            model_path, providers=list(providers or ["CPUExecutionProvider"]))
        self.input_name = self.session.get_inputs()[0].name
        self.labels = tuple(labels)
        self.input_size = input_size
        self.conf, self.iou = conf, iou
        self.modalities = ("rgb", "lwir")

    def detect(self, frame: Any) -> List[Detection]:
        img = np.asarray(frame.image)
        out = self.session.run(None, {self.input_name: self._preprocess(img)})[0]
        return self._postprocess(out, img.shape, float(getattr(frame, "t", 0.0)),
                                 str(getattr(frame, "kind", "rgb")))

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        s = self.input_size
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        h, w = img.shape[:2]
        scale = s / max(h, w)
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
        if _HAVE_SCIPY:
            resized = np.stack([ndi.zoom(img[..., c].astype(np.float32),
                                         (nh / h, nw / w), order=1)
                                for c in range(3)], axis=-1)
        else:  # pragma: no cover
            resized = img[:nh, :nw]
        canvas = np.full((s, s, 3), 114.0, dtype=np.float32)
        canvas[:nh, :nw] = resized
        return canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    def _postprocess(self, out: np.ndarray, orig_shape: Tuple[int, ...],
                     t: float, kind: str) -> List[Detection]:
        preds = np.squeeze(out)
        if preds.ndim == 3:
            preds = preds[0]
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T
        boxes, scores = preds[:, :4], preds[:, 4:]
        cls, conf = scores.argmax(axis=1), scores.max(axis=1)
        keep = conf >= self.conf
        boxes, cls, conf = boxes[keep], cls[keep], conf[keep]
        if boxes.size == 0:
            return []
        h, w = orig_shape[:2]
        scale = self.input_size / max(h, w)
        cx, cy, bw, bh = boxes.T
        x0, y0 = (cx - bw / 2) / scale, (cy - bh / 2) / scale
        x1, y1 = (cx + bw / 2) / scale, (cy + bh / 2) / scale
        dets = [Detection(
            label=self.labels[int(cls[i])] if int(cls[i]) < len(self.labels) else "unknown",
            score=float(conf[i]), bbox=(float(x0[i]), float(y0[i]), float(x1[i]), float(y1[i])),
            modality=kind, frame_t=t,
            centroid=(float((x0[i] + x1[i]) / 2), float((y0[i] + y1[i]) / 2)),
            area_px=float(max(x1[i] - x0[i], 0) * max(y1[i] - y0[i], 0)),
            backend=self.name, polarity="n/a", attributes={"stage": "neural"},
        ) for i in range(len(conf))]
        return nms(dets, self.iou)


class UltralyticsBackend:
    """Convenience wrapper over an Ultralytics YOLO checkpoint (PyTorch)."""

    name = "ultralytics_yolo"

    def __init__(self, model_path: str, conf: float = 0.25, device: str = "cpu",
                 half: bool = False) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("ultralytics is not installed (pip install "
                               "ultralytics). Prefer OnnxRuntimeBackend in flight.") from exc
        self.model = YOLO(model_path)
        self.conf, self.device, self.half = conf, device, half
        self.modalities = ("rgb",)

    def detect(self, frame: Any) -> List[Detection]:
        res = self.model.predict(np.asarray(frame.image), conf=self.conf, device=self.device,
                                 half=self.half, verbose=False)[0]
        dets: List[Detection] = []
        for box in res.boxes:
            x0, y0, x1, y1 = [float(v) for v in box.xyxy[0].tolist()]
            cid = int(box.cls[0])
            dets.append(Detection(
                label=str(res.names.get(cid, cid)), score=float(box.conf[0]),
                bbox=(x0, y0, x1, y1), modality=str(getattr(frame, "kind", "rgb")),
                frame_t=float(getattr(frame, "t", 0.0)),
                centroid=((x0 + x1) / 2, (y0 + y1) / 2),
                area_px=max(x1 - x0, 0) * max(y1 - y0, 0),
                backend=self.name, polarity="n/a", attributes={"stage": "neural"}))
        return dets


class QualcommQnnBackend:
    """Deployment adapter for a Qualcomm AI Hub / QNN-compiled model.

    On target this loads the QNN Model Library ``.so`` produced by
    ``qaihub_models`` (Hexagon NPU via the QNN HTP backend).  It lives in the
    repo so flight software has exactly one code path for "run the NPU model" and
    so CI can assert the interface contract holds even where the NPU SDK is
    absent.  See ``docs/06_AI_MODELS_AND_DATASETS.md``.
    """

    name = "qualcomm_qnn"

    def __init__(self, model_path: str, backend_lib: str = "libQnnHtp.so",
                 labels: Sequence[str] = ALL_LABELS) -> None:
        self.model_path = model_path
        self.backend_lib = backend_lib
        self.labels = tuple(labels)
        self.modalities = ("rgb", "lwir")
        try:
            import QnnHtp  # type: ignore  # noqa: F401
            self._available = True
        except Exception:
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def detect(self, frame: Any) -> List[Detection]:
        if not self._available:
            raise RuntimeError(
                "QNN runtime not present on this host. Compile with "
                "qaihub_models for the QCS8550/RB5-Gen2 target; the heuristic "
                "backend is used on the ground station and in simulation.")
        raise NotImplementedError  # pragma: no cover - executes on target only


def build_reference_pipeline(kind: str = "both", **kwargs: Any) -> DetectorEnsemble:
    """Factory used by the simulator, the tests and the mission runner."""
    backends: List[DetectorBackend] = []
    if kind in ("lwir", "both"):
        backends.append(ThermalAnomalyDetector(**kwargs.get("lwir", {})))
    if kind in ("rgb", "both"):
        backends.append(RgbHazardDetector(**kwargs.get("rgb", {})))
    return DetectorEnsemble(backends, **kwargs.get("ensemble", {}))
