"""Hurwitz-zeta building blocks for the discrete power law.

These helpers assume validated inputs (``alpha > 1``, ``x_min >= 1`` and
finite integer grids). All user-facing validation lives in
:mod:`pypowercpp._api`.
"""

from __future__ import annotations

import numpy as np
from scipy.special import zeta as _scipy_zeta


def hurwitz_zeta(s, q):
    """Evaluate the Hurwitz zeta function ``zeta(s, q)`` (real branch)."""
    return _scipy_zeta(s, q)


def norm_left(alpha: float, x_min: int) -> float:
    """Return the normalising constant ``zeta(alpha, x_min)`` (Type I law)."""
    return float(_scipy_zeta(alpha, float(x_min)))


def pmf_left(x: np.ndarray, alpha: float, x_min: int) -> np.ndarray:
    """Evaluate ``P(X = x)`` on a finite integer grid (Type I law)."""
    out = np.zeros(x.shape[0], dtype=float)
    ok = x >= x_min
    out[ok] = np.power(x[ok].astype(float), -alpha) / _scipy_zeta(alpha, float(x_min))
    return out


def survival_left(x: np.ndarray, alpha: float, x_min: int) -> np.ndarray:
    """Evaluate ``P(X >= x)`` on a grid with ``x >= x_min`` (Type I law)."""
    denom = _scipy_zeta(alpha, float(x_min))
    return _scipy_zeta(alpha, x.astype(float)) / denom


def cdf_left(x: np.ndarray, alpha: float, x_min: int) -> np.ndarray:
    """Evaluate the standard CDF ``P(X <= x)`` on a finite integer grid.

    Note this is the ordinary CDF, *not* the survival function returned by
    the ``powerlaw_cdf`` wrapper of the R package.
    """
    out = np.zeros(x.shape[0], dtype=float)
    ok = x >= x_min
    denom = _scipy_zeta(alpha, float(x_min))
    out[ok] = 1.0 - _scipy_zeta(alpha, x[ok].astype(float) + 1.0) / denom
    return np.clip(out, 0.0, 1.0)
