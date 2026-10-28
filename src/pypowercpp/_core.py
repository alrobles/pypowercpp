"""Clauset-Shalizi-Newman fitting engine for discrete power laws.

Implements maximum-likelihood estimation of the exponent ``alpha`` (Brent
optimisation of the discrete log-likelihood), lower/upper bound estimation
by Kolmogorov-Smirnov minimisation, exact discrete samplers, and the
semi-parametric bootstrap. Mirrors ``src/DiscreteDistributions.cpp`` of
poweRcpp, with ``scipy.special.zeta`` in place of the bundled
Euler-Maclaurin implementation.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import zeta as _zeta

__all__ = [
    "PowerLawFit",
    "PowerLawGof",
    "prepare_data",
    "invalid_fit",
    "fit_auto",
    "fit_fixed_bound",
    "fit_given_alpha",
    "sample_unbounded",
    "sample_truncated",
]

# Bracket for the Brent MLE search (same bracket as the C++ code).
_BRENT_LO = 1.01
_BRENT_HI = 20.0
# Minimum span of the Type II window (``smallestInterval`` in the C++ code):
# upper-bound candidates start at 1 + _TYPE2_MIN_WIDTH.
_TYPE2_MIN_WIDTH = 20
# Fits with fewer tail observations are reported as invalid (NaN fields).
MIN_TAIL = 10
# Largest support size handled with an explicit CDF table; wider windows use
# a vectorised bisection sampler instead (same law, O(log span) memory).
_TABLE_SPAN_LIMIT = 2_000_000

# User-facing messages, kept equivalent to the R/C++ stop() texts.
NA_MSG_FIT = "NA values found in 'data'. They will be removed before fitting."
NA_MSG_GOF = "NA values found in 'data'. They will be removed."
CAST_MSG_DATA = "Casting non-integer 'data' to integers."
EMPTY_MSG = "'data' must be a non-empty integer vector."
POSITIVE_MSG = "All values in 'data' must be positive integers."
RANGE_MSG = "'data' contains values out of the 64-bit integer range."


def prepare_data(data, *, na_message: str = NA_MSG_FIT) -> np.ndarray:
    """Coerce user input to a validated 1-D int64 array of positive values.

    ``None``/``NaN`` (and infinite) entries are dropped with a warning;
    remaining values are truncated toward zero, warning only when the cast
    actually changes a value.
    """
    try:
        arr = np.asarray(data)
    except Exception:
        raise ValueError(EMPTY_MSG) from None
    if arr.size == 0:
        raise ValueError(EMPTY_MSG)
    flat = np.ravel(arr)
    if flat.dtype.kind == "O":
        vals: list = []
        saw_na = False
        for value in flat.tolist():
            if value is None:
                saw_na = True
            elif isinstance(value, bool):
                vals.append(int(value))
            elif isinstance(value, (int, np.integer)):
                vals.append(int(value))
            elif isinstance(value, (float, np.floating)):
                if math.isnan(value):
                    saw_na = True
                else:
                    vals.append(value)
            else:
                raise ValueError(EMPTY_MSG)
        if saw_na:
            warnings.warn(na_message, UserWarning, stacklevel=3)
        if not vals:
            raise ValueError(EMPTY_MSG)
        flat = np.asarray(vals)
    if flat.dtype.kind == "f":
        finite = np.isfinite(flat)
        if not bool(np.all(finite)):
            warnings.warn(na_message, UserWarning, stacklevel=3)
            flat = flat[finite]
        if flat.size == 0:
            raise ValueError(EMPTY_MSG)
        if bool((np.abs(flat) >= float(2**63)).any()):
            raise ValueError(RANGE_MSG)
        truncated = flat.astype(np.int64)
        if bool((flat != truncated).any()):
            warnings.warn(CAST_MSG_DATA, UserWarning, stacklevel=3)
        ints = truncated
    elif flat.dtype.kind in "iu":
        if flat.dtype.kind == "u" and bool((flat >= np.uint64(2**63)).any()):
            raise ValueError(RANGE_MSG)
        ints = flat.astype(np.int64)
    elif flat.dtype.kind == "b":
        ints = flat.astype(np.int64)
    else:
        raise ValueError(EMPTY_MSG)
    if bool((ints <= 0).any()):
        raise ValueError(POSITIVE_MSG)
    return ints


def _suffix_log_sums(sorted_data: np.ndarray) -> np.ndarray:
    """Return ``suffix[i] = sum(log(sorted_data[i:]))`` (length n + 1)."""
    logs = np.log(sorted_data.astype(float))
    return np.concatenate([np.cumsum(logs[::-1])[::-1], np.zeros(1)])


def _prefix_log_sums(sorted_data: np.ndarray) -> np.ndarray:
    """Return ``prefix[k] = sum(log(sorted_data[:k]))`` (length n + 1)."""
    logs = np.log(sorted_data.astype(float))
    return np.concatenate([np.zeros(1), np.cumsum(logs)])


def _mle_alpha(n: int, sum_log: float, *, kind: str, bound: int, precision: float) -> float:
    """Maximise the discrete log-likelihood over ``alpha`` with Brent's method."""
    if n <= 0:
        return math.nan
    if kind == "left":
        def negloglik(a: float) -> float:
            return n * math.log(float(_zeta(a, float(bound)))) + a * sum_log
    else:
        def negloglik(a: float) -> float:
            den = float(_zeta(a, 1.0) - _zeta(a, float(bound + 1)))
            if not math.isfinite(den) or den <= 0.0:
                return math.inf
            return n * math.log(den) + a * sum_log
    res = minimize_scalar(
        negloglik,
        bounds=(_BRENT_LO, _BRENT_HI),
        method="bounded",
        options={"xatol": precision * 0.1},
    )
    return float(res.x)


def _sorted_unique(values: np.ndarray) -> np.ndarray:
    """Unique values of a sorted array in O(n) (no re-sorting)."""
    if values.size == 0:
        return values
    return values[np.concatenate([[True], np.diff(values) > 0])]


def _ks_left(sorted_data: np.ndarray, x_min: int, alpha: float) -> float:
    """KS distance between the fitted and empirical survival functions.

    Evaluated at the observed tail values only: between consecutive
    observations the empirical survival is constant while the fitted one
    decreases monotonically, so the maximum is always attained at an
    observed point. Identical to the full-grid maximum at O(unique) cost.
    """
    i0 = int(np.searchsorted(sorted_data, x_min, side="left"))
    tail = sorted_data[i0:]
    nt = tail.size
    if nt == 0:
        return math.inf
    xs = _sorted_unique(tail)
    s_emp = (nt - np.searchsorted(tail, xs, side="left")) / nt
    s_theo = _zeta(alpha, xs.astype(float)) / float(_zeta(alpha, float(x_min)))
    return float(np.max(np.abs(s_emp - s_theo)))


def _ks_right(sorted_data: np.ndarray, x_max: int, alpha: float) -> float:
    """KS distance for a right-bounded (Type II) fit over ``[1, x_max]``.

    Evaluated at the observed window values only (same maximum as the
    full grid, by the argument in :func:`_ks_left`).
    """
    i1 = int(np.searchsorted(sorted_data, x_max, side="right"))
    window = sorted_data[:i1]
    n = window.size
    if n == 0:
        return math.inf
    xs = _sorted_unique(window)
    s_emp = (n - np.searchsorted(window, xs, side="left")) / n
    z_cap = float(_zeta(alpha, float(x_max + 1)))
    denom = float(_zeta(alpha, 1.0)) - z_cap
    s_theo = (_zeta(alpha, xs.astype(float)) - z_cap) / denom
    return float(np.max(np.abs(s_emp - s_theo)))


def _fit_left_at(sorted_data, suffix, x_min, precision):
    """Fit a Type I model with fixed ``x_min``; return ``(alpha, ks, n, slog)``."""
    n_total = sorted_data.size
    i0 = int(np.searchsorted(sorted_data, x_min, side="left"))
    nt = n_total - i0
    if nt <= 0:
        return (math.nan, math.inf, 0, 0.0)
    slog = float(suffix[i0])
    alpha = _mle_alpha(nt, slog, kind="left", bound=x_min, precision=precision)
    return (alpha, _ks_left(sorted_data, x_min, alpha), nt, slog)


def _fit_right_at(sorted_data, prefix, x_max, precision):
    """Fit a Type II model with fixed ``x_max``; return ``(alpha, ks, n, slog)``."""
    i1 = int(np.searchsorted(sorted_data, x_max, side="right"))
    n = i1
    if n <= 0:
        return (math.nan, math.inf, 0, 0.0)
    slog = float(prefix[i1])
    alpha = _mle_alpha(n, slog, kind="right", bound=x_max, precision=precision)
    return (alpha, _ks_right(sorted_data, x_max, alpha), n, slog)


def _loglik_left(n: int, slog: float, alpha: float, x_min: int) -> float:
    return -n * math.log(float(_zeta(alpha, float(x_min)))) - alpha * slog


def _loglik_right(n: int, slog: float, alpha: float, x_max: int) -> float:
    den = float(_zeta(alpha, 1.0) - _zeta(alpha, float(x_max + 1)))
    return -n * math.log(den) - alpha * slog


def _show(value) -> str:
    """Format a result field the way the R print methods show NA values."""
    if value is None:
        return "NA"
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.6g}"
    return str(value)


@dataclass
class PowerLawFit:
    """Result of :func:`pypowercpp.fit_powerlaw`.

    Invalid fits (constant data, ``x_min >= max(data)``, tail below
    :data:`MIN_TAIL`) carry ``NaN``/``None`` fields and ``valid == False``
    instead of raising, mirroring the ``NA`` fields of the R package.
    """

    alpha: float
    x_min: int | None
    x_max: int | None
    ks_statistic: float
    standard_error: float
    log_likelihood: float
    n_tail: int | None
    type: str  # "left" (Type I) or "right" (Type II)
    valid: bool

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def to_dict(self) -> dict:
        """Return the fields as a plain ``dict`` of Python scalars."""
        return {
            "alpha": self.alpha,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "ks_statistic": self.ks_statistic,
            "standard_error": self.standard_error,
            "log_likelihood": self.log_likelihood,
            "n_tail": self.n_tail,
            "type": self.type,
            "valid": self.valid,
        }

    def __str__(self) -> str:
        return (
            "Discrete power-law distribution fit\n"
            f"  Type        : {self.type}\n"
            f"  Alpha       : {_show(self.alpha)}\n"
            f"  x_min       : {_show(self.x_min)}\n"
            f"  x_max       : {_show(self.x_max)}\n"
            f"  KS statistic: {_show(self.ks_statistic)}\n"
            f"  Std. error  : {_show(self.standard_error)}\n"
            f"  Log-lik.    : {_show(self.log_likelihood)}\n"
            f"  Tail obs.   : {_show(self.n_tail)}\n"
            f"  Valid       : {self.valid}"
        )

    def summary(self) -> str:
        """Return :meth:`__str__` plus an ``alpha +/- 1 SE`` interval."""
        text = str(self)
        if self.valid:
            lo = self.alpha - self.standard_error
            hi = self.alpha + self.standard_error
            text += f"\n  alpha +/- 1 SE: [{lo:.5g}, {hi:.5g}]"
        return text


@dataclass
class PowerLawGof:
    """Result of :func:`pypowercpp.powerlaw_gof`."""

    gof_pvalue: float
    ks_statistic: float
    n_replicas: int
    alpha: float
    x_min: int | None
    x_max: int | None
    type: str  # "left" (Type I) or "right" (Type II)

    @property
    def p_value(self) -> float:
        """Alias of :attr:`gof_pvalue` (name used by the R package)."""
        return self.gof_pvalue

    @property
    def replicas(self) -> int:
        """Alias of :attr:`n_replicas` (name used by the R package)."""
        return self.n_replicas

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def to_dict(self) -> dict:
        """Return the fields as a plain ``dict`` of Python scalars."""
        return {
            "gof_pvalue": self.gof_pvalue,
            "p_value": self.gof_pvalue,
            "ks_statistic": self.ks_statistic,
            "n_replicas": self.n_replicas,
            "replicas": self.n_replicas,
            "alpha": self.alpha,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "type": self.type,
        }

    def __str__(self) -> str:
        verdict = (
            "No evidence against the power-law hypothesis (p >= 0.10)."
            if self.gof_pvalue >= 0.10
            else "Evidence against the power-law hypothesis (p < 0.10)."
        )
        return (
            "Bootstrap goodness-of-fit for discrete power-law\n"
            f"  Replicas    : {self.n_replicas}\n"
            f"  p-value     : {self.gof_pvalue:.4g}\n"
            f"  KS statistic: {self.ks_statistic:.4g}\n"
            f"  Alpha used  : {self.alpha:.6g}\n"
            f"  x_min used  : {self.x_min}\n"
            f"  -> {verdict}"
        )


def invalid_fit(dtype: str) -> PowerLawFit:
    """Return an invalid fit object (``NaN``/``None`` fields, ``valid=False``)."""
    return PowerLawFit(
        alpha=math.nan,
        x_min=None,
        x_max=None,
        ks_statistic=math.nan,
        standard_error=math.nan,
        log_likelihood=math.nan,
        n_tail=None,
        type=dtype,
        valid=False,
    )


def fit_auto(sorted_data: np.ndarray, precision: float, dtype: str) -> PowerLawFit:
    """Fit a power law estimating the bound by KS minimisation.

    ``sorted_data`` must be sorted ascending. Cost is roughly
    ``O(n_unique * n)`` with vectorised inner steps.
    """
    x_max_data = int(sorted_data[-1])
    if dtype == "left":
        cands = np.unique(sorted_data)
        cands = cands[cands < x_max_data]
        if cands.size == 0:
            return invalid_fit(dtype)
        suffix = _suffix_log_sums(sorted_data)
        best_ks = math.inf
        best = None
        for cand in (int(c) for c in cands):
            alpha, ks, nt, slog = _fit_left_at(sorted_data, suffix, cand, precision)
            if ks < best_ks:
                best_ks = ks
                best = (cand, alpha, nt, slog)
        x_min, alpha, nt, slog = best
        if nt < MIN_TAIL:
            return invalid_fit(dtype)
        ll = _loglik_left(nt, slog, alpha, x_min)
        return PowerLawFit(
            alpha=alpha,
            x_min=x_min,
            x_max=x_max_data,
            ks_statistic=best_ks,
            standard_error=alpha / math.sqrt(nt),
            log_likelihood=ll,
            n_tail=nt,
            type=dtype,
            valid=True,
        )
    lo = 1 + _TYPE2_MIN_WIDTH
    cands = np.unique(sorted_data)
    cands = cands[(cands >= lo) & (cands < x_max_data)]
    if cands.size == 0:
        return invalid_fit(dtype)
    prefix = _prefix_log_sums(sorted_data)
    best_ks = math.inf
    best = None
    for cand in (int(c) for c in cands):
        alpha, ks, n, slog = _fit_right_at(sorted_data, prefix, cand, precision)
        if ks < best_ks:
            best_ks = ks
            best = (cand, alpha, n, slog)
    x_max, alpha, n, slog = best
    if n < MIN_TAIL:
        return invalid_fit(dtype)
    ll = _loglik_right(n, slog, alpha, x_max)
    return PowerLawFit(
        alpha=alpha,
        x_min=1,
        x_max=x_max,
        ks_statistic=best_ks,
        standard_error=alpha / math.sqrt(n),
        log_likelihood=ll,
        n_tail=n,
        type=dtype,
        valid=True,
    )


def fit_fixed_bound(sorted_data: np.ndarray, bound: int, precision: float, dtype: str) -> PowerLawFit:
    """Fit with a fixed bound: ``x_min`` for ``"left"``, ``x_max`` for ``"right"``.

    Note the ``"right"`` convention mirrors the R package, where the ``x_min``
    argument carries the fixed upper bound when ``type == "right"``.
    """
    x_max_data = int(sorted_data[-1])
    x_min_data = int(sorted_data[0])
    if dtype == "left":
        if bound >= x_max_data:
            return invalid_fit(dtype)
        suffix = _suffix_log_sums(sorted_data)
        alpha, ks, nt, slog = _fit_left_at(sorted_data, suffix, bound, precision)
        if nt < MIN_TAIL:
            return invalid_fit(dtype)
        return PowerLawFit(
            alpha=alpha,
            x_min=bound,
            x_max=x_max_data,
            ks_statistic=ks,
            standard_error=alpha / math.sqrt(nt),
            log_likelihood=_loglik_left(nt, slog, alpha, bound),
            n_tail=nt,
            type=dtype,
            valid=True,
        )
    if bound <= x_min_data:
        return invalid_fit(dtype)
    prefix = _prefix_log_sums(sorted_data)
    alpha, ks, n, slog = _fit_right_at(sorted_data, prefix, bound, precision)
    if n < MIN_TAIL:
        return invalid_fit(dtype)
    return PowerLawFit(
        alpha=alpha,
        x_min=1,
        x_max=bound,
        ks_statistic=ks,
        standard_error=alpha / math.sqrt(n),
        log_likelihood=_loglik_right(n, slog, alpha, bound),
        n_tail=n,
        type=dtype,
        valid=True,
    )


def fit_given_alpha(sorted_data: np.ndarray, alpha: float, bound: int, dtype: str) -> PowerLawFit:
    """Build a model from supplied parameters without optimisation (gof path)."""
    x_max_data = int(sorted_data[-1])
    x_min_data = int(sorted_data[0])
    if dtype == "left":
        if bound >= x_max_data:
            return invalid_fit(dtype)
        i0 = int(np.searchsorted(sorted_data, bound, side="left"))
        tail = sorted_data[i0:]
        nt = tail.size
        if nt < MIN_TAIL:
            return invalid_fit(dtype)
        slog = float(np.log(tail.astype(float)).sum())
        return PowerLawFit(
            alpha=alpha,
            x_min=bound,
            x_max=x_max_data,
            ks_statistic=_ks_left(sorted_data, bound, alpha),
            standard_error=alpha / math.sqrt(nt),
            log_likelihood=_loglik_left(nt, slog, alpha, bound),
            n_tail=nt,
            type=dtype,
            valid=True,
        )
    if bound <= x_min_data:
        return invalid_fit(dtype)
    i1 = int(np.searchsorted(sorted_data, bound, side="right"))
    window = sorted_data[:i1]
    n = window.size
    if n < MIN_TAIL:
        return invalid_fit(dtype)
    slog = float(np.log(window.astype(float)).sum())
    return PowerLawFit(
        alpha=alpha,
        x_min=1,
        x_max=bound,
        ks_statistic=_ks_right(sorted_data, bound, alpha),
        standard_error=alpha / math.sqrt(n),
        log_likelihood=_loglik_right(n, slog, alpha, bound),
        n_tail=n,
        type=dtype,
        valid=True,
    )


def sample_unbounded(alpha: float, x_min: int, size: int, rng) -> np.ndarray:
    """Draw exact Type I power-law samples with unbounded support.

    Vectorised inversion of the survival function: each draw is the largest
    ``x`` with ``S(x) > V`` for ``V ~ Uniform(0, 1)``, found by doubling
    followed by bisection. Exact up to floating-point error.
    """
    m = int(size)
    if m <= 0:
        return np.empty(0, dtype=np.int64)
    u = np.asarray(rng.random(m), dtype=float).ravel()
    v = 1.0 - u
    np.minimum(v, 1.0 - 1e-15, out=v)
    z0 = float(_zeta(alpha, float(x_min)))
    lo = np.full(m, x_min - 1, dtype=np.int64)
    hi = np.full(m, x_min, dtype=np.int64)
    cap = np.int64(2**62)
    s_hi = _zeta(alpha, hi.astype(float)) / z0
    active = s_hi > v
    while bool(active.any()):
        grow = active & (hi < cap)
        if not bool(grow.any()):
            break
        hi[grow] *= 2
        s_hi[grow] = _zeta(alpha, hi[grow].astype(float)) / z0
        active[grow] = s_hi[grow] > v[grow]
    gap = hi - lo
    while bool((gap > 1).any()):
        idx = np.nonzero(gap > 1)[0]
        mid = lo[idx] + (hi[idx] - lo[idx]) // 2
        go_left = (_zeta(alpha, mid.astype(float)) / z0) > v[idx]
        left = idx[go_left]
        right = idx[~go_left]
        lo[left] = mid[go_left]
        hi[right] = mid[~go_left]
        gap = hi - lo
    return lo


def _sample_window_bisect(alpha: float, lo: int, hi: int, size: int, rng) -> np.ndarray:
    """Draw exact truncated samples via vectorised bisection (huge windows)."""
    m = int(size)
    if m <= 0:
        return np.empty(0, dtype=np.int64)
    u = np.asarray(rng.random(m), dtype=float).ravel()
    out = np.empty(m, dtype=np.int64)
    at_zero = u <= 0.0
    out[at_zero] = lo
    rem = np.nonzero(~at_zero)[0]
    if rem.size == 0:
        return out
    z_lo = float(_zeta(alpha, float(lo)))
    z_hi1 = float(_zeta(alpha, float(hi + 1)))
    den = z_lo - z_hi1
    t = z_lo - u[rem] * den
    lo_y = np.full(rem.size, lo, dtype=np.int64)
    hi_y = np.full(rem.size, hi + 1, dtype=np.int64)
    gap = hi_y - lo_y
    while bool((gap > 1).any()):
        idx = np.nonzero(gap > 1)[0]
        mid = lo_y[idx] + (hi_y[idx] - lo_y[idx]) // 2
        go_left = _zeta(alpha, mid.astype(float)) > t[idx]
        left = idx[go_left]
        right = idx[~go_left]
        lo_y[left] = mid[go_left]
        hi_y[right] = mid[~go_left]
        gap = hi_y - lo_y
    out[rem] = hi_y - 1
    return out


def sample_truncated(alpha: float, lo: int, hi: int, size: int, rng) -> np.ndarray:
    """Draw exact power-law samples truncated to ``[lo, hi]``.

    Narrow windows use CDF-table inversion; windows wider than
    ``_TABLE_SPAN_LIMIT`` use vectorised bisection (same law).
    """
    span = hi - lo + 1
    if span > _TABLE_SPAN_LIMIT:
        return _sample_window_bisect(alpha, lo, hi, int(size), rng)
    xs = np.arange(lo, hi + 1)
    weights = np.power(xs.astype(float), -alpha)
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Power-law weights underflowed; the window is numerically degenerate.")
    cdf = np.cumsum(weights / total)
    cdf[-1] = 1.0
    u = rng.random(int(size))
    return xs[np.searchsorted(cdf, u, side="left")].astype(np.int64)
