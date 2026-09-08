"""Turning network scores into probabilities that can be composed.

The problem
-----------
A detector's confidence is not P(this is a person).  It is a monotone but
arbitrary function of it, and the shape of that function changes with target
size, altitude and modality.  This matters here more than in most vision
systems, because the number does not just get thresholded - it is *composed*:

* the coverage grid accumulates ``1 - Prod(1 - p_i)`` per cell;
* the Bayesian belief map multiplies by ``(1 - p_detect)`` on a negative
  observation;
* the triage clock and the alert tier are driven by the resulting posterior.

Feed an over-confident 0.9 into that machinery and the belief map concludes a
cell has been cleared when it has not, and the aircraft never goes back.  A
false confidence is therefore not a cosmetic error - it removes ground from the
search.

What this module provides
-------------------------
* :func:`fit_platt_scaling` - a two-parameter logistic fit (Platt scaling) on
  held-out detections, optionally with size and altitude as extra features,
  which is enough to correct the dominant over-confidence on small objects
  without needing thousands of samples.
* :func:`expected_calibration_error` and :func:`reliability_table` - how wrong
  the calibration is, in the form that goes into the report.
* :class:`ConfidenceCalibrator` - the fitted object, serialisable to JSON so
  the aircraft loads a file rather than refitting at boot.

Fitting data comes from ``scripts/eval_detector.py``, which already matches
detections against ground truth at several altitudes; the matched/unmatched
flags are exactly the labels this needs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _logit(p: np.ndarray | float, eps: float = 1e-6) -> np.ndarray | float:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def fit_platt_scaling(scores: Sequence[float], labels: Sequence[int], *,
                      features: Optional[np.ndarray] = None,
                      iters: int = 300, lr: float = 0.15,
                      l2: float = 1e-3) -> Dict[str, Any]:
    """Fit ``p = sigmoid(a * logit(s) + b + w . x)`` by gradient descent.

    Plain gradient descent rather than a solver so the only dependency is
    numpy - this has to run on the companion computer, offline, with no scipy
    optimiser and no sklearn.

    ``features`` (n, k) may carry log-area and altitude; they capture the
    size-dependence of over-confidence, which is the part that matters for
    aerial SAR.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if s.size == 0:
        return {"a": 1.0, "b": 0.0, "w": [], "n": 0, "converged": False}
    x0 = _logit(s)
    X = np.column_stack([x0, np.ones_like(x0)])
    if features is not None and len(features):
        F = np.asarray(features, dtype=np.float64).reshape(len(s), -1)
        F = (F - F.mean(axis=0)) / (F.std(axis=0) + 1e-9)
        X = np.column_stack([X, F])
    theta = np.zeros(X.shape[1])
    theta[0] = 1.0
    n = len(y)
    prev = math.inf
    converged = False
    for _ in range(iters):
        p = _sigmoid(X @ theta)
        grad = X.T @ (p - y) / n + l2 * theta
        theta -= lr * grad
        loss = float(-np.mean(y * np.log(np.clip(p, 1e-9, 1)) +
                              (1 - y) * np.log(np.clip(1 - p, 1e-9, 1)))
                     + 0.5 * l2 * float(theta @ theta))
        if abs(prev - loss) < 1e-7:
            converged = True
            break
        prev = loss
    return {"a": float(theta[0]), "b": float(theta[1]),
            "w": [float(v) for v in theta[2:]], "n": int(n),
            "converged": converged, "nll": float(prev)}


def reliability_table(probs: Sequence[float], labels: Sequence[int],
                      bins: int = 10) -> List[Dict[str, float]]:
    """Per-bin mean predicted probability vs observed frequency."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    out: List[Dict[str, float]] = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1.0)
        if not m.any():
            continue
        out.append({"bin_lo": float(edges[i]), "bin_hi": float(edges[i + 1]),
                    "n": int(m.sum()), "mean_pred": float(p[m].mean()),
                    "observed": float(y[m].mean())})
    return out


def expected_calibration_error(probs: Sequence[float], labels: Sequence[int],
                               bins: int = 10) -> float:
    """ECE: the number to quote.  0.0 is perfect; >0.10 means do not compose it."""
    table = reliability_table(probs, labels, bins)
    n_total = sum(row["n"] for row in table) or 1
    return float(sum(row["n"] / n_total * abs(row["mean_pred"] - row["observed"])
                     for row in table))


@dataclass
class ConfidenceCalibrator:
    """A fitted score -> probability map, per label class.

    Serialised to ``models/calibration.json`` and loaded at boot.  Absent file =
    identity mapping with a note in the report; never a silent default.
    """

    params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    use_features: bool = True
    source: str = "identity"
    ece_before: Optional[float] = None
    ece_after: Optional[float] = None

    # -- application ----------------------------------------------------- #
    def apply(self, score: float, *, label: str = "person",
              area_px: float = 0.0, gsd_m: float = 0.0) -> float:
        par = self.params.get(label) or self.params.get("_default")
        if not par:
            return float(score)
        z = par["a"] * float(_logit(score)) + par["b"]
        w = par.get("w") or []
        if w and self.use_features:
            feats = self._features(area_px, gsd_m)
            for wi, fi in zip(w, feats):
                z += wi * fi
        return float(_sigmoid(z))

    @staticmethod
    def _features(area_px: float, gsd_m: float) -> List[float]:
        return [math.log(max(area_px, 1.0)), float(gsd_m)]

    # -- fitting --------------------------------------------------------- #
    @classmethod
    def fit(cls, samples: Iterable[Dict[str, Any]], *,
            use_features: bool = True) -> "ConfidenceCalibrator":
        """``samples``: dicts with ``score``, ``matched`` (0/1), ``label``,
        and optionally ``area_px`` and ``gsd_m``."""
        by_label: Dict[str, List[Dict[str, Any]]] = {}
        rows = list(samples)
        for r in rows:
            by_label.setdefault(str(r.get("label", "person")), []).append(r)
        by_label["_default"] = rows
        params: Dict[str, Dict[str, Any]] = {}
        for label, group in by_label.items():
            if len(group) < 25:          # too few to fit anything trustworthy
                continue
            scores = [float(r["score"]) for r in group]
            labels = [int(bool(r["matched"])) for r in group]
            feats = None
            if use_features:
                feats = np.array([cls._features(float(r.get("area_px", 0.0)),
                                                float(r.get("gsd_m", 0.0)))
                                  for r in group])
            params[label] = fit_platt_scaling(scores, labels, features=feats)
        cal = cls(params=params, use_features=use_features, source="fitted")
        if rows:
            raw = [float(r["score"]) for r in rows]
            y = [int(bool(r["matched"])) for r in rows]
            cal.ece_before = expected_calibration_error(raw, y)
            adj = [cal.apply(float(r["score"]), label=str(r.get("label", "person")),
                             area_px=float(r.get("area_px", 0.0)),
                             gsd_m=float(r.get("gsd_m", 0.0))) for r in rows]
            cal.ece_after = expected_calibration_error(adj, y)
        return cal

    # -- persistence ------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {"params": self.params, "use_features": self.use_features,
                "source": self.source, "ece_before": self.ece_before,
                "ece_after": self.ece_after}

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        return p

    @classmethod
    def load(cls, path: str | Path) -> "ConfidenceCalibrator":
        data = json.loads(Path(path).read_text())
        return cls(params=data.get("params", {}),
                   use_features=bool(data.get("use_features", True)),
                   source=str(data.get("source", "file")),
                   ece_before=data.get("ece_before"),
                   ece_after=data.get("ece_after"))

    @classmethod
    def load_if_present(cls, path: str | Path) -> Optional["ConfidenceCalibrator"]:
        p = Path(path)
        return cls.load(p) if p.is_file() else None


__all__ = ["ConfidenceCalibrator", "expected_calibration_error",
           "fit_platt_scaling", "reliability_table"]
