"""Public API of pypowercpp (ports of the R wrappers in ``R/poweRcpp.R``)."""

from __future__ import annotations

import math
import warnings

import numpy as np

from ._core import (
    NA_MSG_GOF,
    PowerLawFit,
    PowerLawGof,
    fit_auto,
    fit_fixed_bound,
    fit_given_alpha,
    invalid_fit,
    prepare_data,
    sample_truncated,
    sample_unbounded,
)
from ._zeta import cdf_left, pmf_left

__all__ = [
    "PowerLawFit",
    "PowerLawGof",
    "fit_powerlaw",
    "powerlaw_pdf",
    "powerlaw_cdf",
    "powerlaw_generate",
    "powerlaw_gof",
]

# Precision used by the internal fits of the gof bootstrap (the C++ code
# hardcodes 0.01 there as well).
_GOF_PRECISION = 0.01
_X_CAST_MSG = "Casting non-integer 'x' to integers."
_BOUND_CAP = 2**62


def _check_type(value) -> str:
    if isinstance(value, str) and value in ("left", "right"):
        return value
    raise ValueError("'type' must be \"left\" or \"right\".")


def _check_precision(value) -> float:
    try:
        precision = float(value)
    except Exception:
        raise ValueError("'alpha_precision' must be in (0, 1].") from None
    if math.isnan(precision) or not (0.0 < precision <= 1.0):
        raise ValueError("'alpha_precision' must be in (0, 1].")
    return precision


def _coerce_alpha(value) -> float:
    try:
        alpha = float(value)
    except Exception:
        raise ValueError("'alpha' must be > 1.") from None
    if not (alpha > 1.0):
        raise ValueError("'alpha' must be > 1.")
    return alpha


def _coerce_bound(value, message: str, *, name: str, check_positive: bool) -> int:
    """Coerce an ``x_min``/``x_max`` scalar (R ``as.integer`` truncation)."""
    if isinstance(value, bool):
        bound = int(value)
    elif isinstance(value, (int, np.integer)):
        bound = int(value)
    elif isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            raise ValueError(message)
        bound = int(value)
    else:
        raise ValueError(message)
    if abs(bound) >= _BOUND_CAP:
        raise ValueError(f"'{name}' is out of the supported range.")
    if check_positive and bound < 1:
        raise ValueError(message)
    return bound


def _coerce_count(value, message: str) -> int:
    """Coerce an ``n``/``replicas`` scalar (R ``as.integer`` truncation)."""
    if isinstance(value, bool):
        raise ValueError(message)
    if isinstance(value, (int, np.integer)):
        count = int(value)
    elif isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            raise ValueError(message)
        count = int(value)
    else:
        raise ValueError(message)
    if count <= 0:
        raise ValueError(message)
    return count


def _coerce_rng(rng, *, name: str):
    """Accept ``None`` (fresh entropy), an integer seed, or a numpy RNG."""
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, (int, np.integer)) and not isinstance(rng, bool):
        return np.random.default_rng(int(rng))
    if isinstance(rng, (np.random.Generator, np.random.RandomState)):
        return rng
    raise ValueError(f"'{name}' must be None, an integer seed, or a numpy random Generator.")


def _eval_grid(primitive, x, alpha: float, x_min: int, *, pos_inf: float, neg_inf: float):
    """Evaluate a pmf/cdf primitive on scalar or array input.

    Returns a ``float`` for scalar input and an ``ndarray`` otherwise.
    ``NaN`` entries propagate as ``NaN`` (as in the R wrappers) and
    non-integral values are truncated with a warning.
    """
    scalar = np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0)
    try:
        arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("'x' must be numeric.") from None
    flat = arr.ravel()
    out = np.empty(flat.shape[0], dtype=float)
    is_nan = np.isnan(flat)
    out[is_nan] = math.nan
    huge_pos = np.isposinf(flat) | (flat >= float(2**63))
    huge_neg = np.isneginf(flat) | (flat <= -float(2**63))
    out[huge_pos] = pos_inf
    out[huge_neg] = neg_inf
    finite = ~(is_nan | huge_pos | huge_neg)
    if finite.any():
        vals = flat[finite]
        ints = vals.astype(np.int64)
        if bool((vals != ints).any()):
            warnings.warn(_X_CAST_MSG, UserWarning, stacklevel=3)
        out[finite] = primitive(ints, alpha, x_min)
    if scalar:
        return float(out[0])
    return out.reshape(arr.shape)


def fit_powerlaw(data, x_min=None, alpha_precision=0.01, type="left") -> PowerLawFit:
    """Fit a discrete power-law distribution (Clauset-Shalizi-Newman).

    Parameters
    ----------
    data:
        Positive integer observations (array-like). ``None``/``NaN`` entries
        are dropped with a warning; non-integer values are truncated toward
        zero.
    x_min:
        Lower cut-off for ``type="left"``. When ``None`` (default) it is
        estimated by minimising the KS statistic over the unique candidate
        values. For ``type="right"`` this argument carries the fixed *upper*
        bound instead (same convention as the R package).
    alpha_precision:
        Tolerance of the Brent MLE search for ``alpha`` (in ``(0, 1]``).
    type:
        ``"left"`` fits a left-bounded (Type I) law, ``"right"`` a
        right-bounded (Type II) law.

    Returns
    -------
    PowerLawFit
        Object with ``alpha``, ``x_min``, ``x_max``, ``ks_statistic``,
        ``standard_error`` (``alpha / sqrt(n_tail)``), ``log_likelihood``,
        ``n_tail``, ``type`` and ``valid``. Unfittable inputs (constant
        data, ``x_min >= max(data)``, tail below 10 observations) yield
        ``NaN``/``None`` fields with ``valid == False`` instead of raising.

    Notes
    -----
    Bound estimation costs roughly ``O(n_unique * n)`` with vectorised
    inner steps (``O(n^2)`` in the worst case of all-distinct data).
    """
    dtype = _check_type(type)
    values = prepare_data(data)
    precision = _check_precision(alpha_precision)
    bound = None
    if x_min is not None:
        bound = _coerce_bound(x_min, "'x_min' must be >= 1.", name="x_min", check_positive=True)
    ordered = np.sort(values)
    if ordered[0] == ordered[-1]:
        return invalid_fit(dtype)
    if bound is None:
        return fit_auto(ordered, precision, dtype)
    return fit_fixed_bound(ordered, bound, precision, dtype)


def powerlaw_pdf(x, alpha, x_min=1):
    """Evaluate ``P(X = x)`` for a Type I discrete power law.

    Returns a ``float`` for scalar ``x`` and a numpy array otherwise.
    Values below ``x_min`` map to ``0.0``; ``NaN`` entries propagate.
    """
    alpha = _coerce_alpha(alpha)
    x_min = _coerce_bound(x_min, "'x_min' must be >= 1.", name="x_min", check_positive=True)
    return _eval_grid(pmf_left, x, alpha, x_min, pos_inf=0.0, neg_inf=0.0)


def powerlaw_cdf(x, alpha, x_min=1):
    """Evaluate the standard CDF ``P(X <= x)`` for a Type I discrete power law.

    Returns a ``float`` for scalar ``x`` and a numpy array otherwise.
    Values below ``x_min`` map to ``0.0``; ``NaN`` entries propagate.

    Note this is the ordinary CDF. The R ``powerlaw_cdf`` wrapper returns
    the *survival* function ``P(X >= x)`` instead; use ``1 - powerlaw_cdf(x - 1)``
    for ``x > x_min`` to recover it.
    """
    alpha = _coerce_alpha(alpha)
    x_min = _coerce_bound(x_min, "'x_min' must be >= 1.", name="x_min", check_positive=True)
    return _eval_grid(cdf_left, x, alpha, x_min, pos_inf=1.0, neg_inf=0.0)


def powerlaw_generate(n, alpha, x_min=1, x_max=None, rng=None) -> np.ndarray:
    """Draw samples from a discrete power law by CDF inversion.

    Parameters
    ----------
    n:
        Number of samples (positive integer).
    alpha:
        Power-law exponent (``> 1``).
    x_min:
        Lower cut-off (positive integer).
    x_max:
        Upper cut-off. Defaults to ``x_min * 1000`` (same default as the R
        package) and must be strictly greater than ``x_min``.
    rng:
        ``None`` (fresh entropy), an integer seed, or a numpy
        ``Generator``/``RandomState``.

    Returns
    -------
    numpy.ndarray
        Integer array of length ``n`` with values in ``[x_min, x_max]``.
    """
    count = _coerce_count(n, "'n' must be a positive integer.")
    alpha = _coerce_alpha(alpha)
    x_min = _coerce_bound(x_min, "'x_min' must be >= 1.", name="x_min", check_positive=True)
    if x_max is None:
        x_max = x_min * 1000
    else:
        x_max = _coerce_bound(
            x_max, "'x_max' must be greater than 'x_min'.", name="x_max", check_positive=False
        )
    if x_max <= x_min:
        raise ValueError("'x_max' must be greater than 'x_min'.")
    gen = _coerce_rng(rng, name="rng")
    return sample_truncated(alpha, x_min, x_max, count, gen)


def powerlaw_gof(data, alpha=None, x_min=None, replicas=1000, type="left", seed=None) -> PowerLawGof:
    """Bootstrap goodness-of-fit test for a discrete power law.

    Uses the semi-parametric bootstrap of Clauset-Shalizi-Newman (2009):
    each replica combines a parametric power-law tail with non-parametric
    resampling of the body, is refitted, and the fraction of replicas with
    ``KS >= KS_observed`` is returned as the p-value. The loop is serial;
    start with a few hundred replicas for exploratory runs.

    Parameters
    ----------
    data:
        Positive integer observations (array-like).
    alpha:
        Exponent to test. Used directly only together with ``x_min``;
        otherwise it is ignored (with a warning) and estimated from ``data``.
    x_min:
        Bound to test (lower cut-off for ``type="left"``, upper cut-off for
        ``type="right"``). When ``None`` (default) it is estimated.
    replicas:
        Number of bootstrap replicas (positive integer).
    type:
        ``"left"`` (Type I) or ``"right"`` (Type II).
    seed:
        ``None`` (fresh entropy), an integer seed, or a numpy RNG.

    Returns
    -------
    PowerLawGof
        Object with ``gof_pvalue`` (alias ``p_value``), ``ks_statistic``,
        ``n_replicas`` (alias ``replicas``), ``alpha``, ``x_min``,
        ``x_max`` and ``type``.
    """
    dtype = _check_type(type)
    values = prepare_data(data, na_message=NA_MSG_GOF)
    n_reps = _coerce_count(replicas, "'replicas' must be a positive integer.")
    use_alpha = None if alpha is None else _coerce_alpha(alpha)
    bound = None
    if x_min is not None:
        bound = _coerce_bound(x_min, "'x_min' must be >= 1.", name="x_min", check_positive=True)
    gen = _coerce_rng(seed, name="seed")
    ordered = np.sort(values)
    if use_alpha is not None and bound is not None:
        fit = fit_given_alpha(ordered, use_alpha, bound, dtype)
    elif bound is not None:
        fit = fit_fixed_bound(ordered, bound, _GOF_PRECISION, dtype)
    else:
        if use_alpha is not None:
            warnings.warn(
                "'alpha' is ignored when 'x_min' is not supplied; "
                "alpha will be estimated from data.",
                UserWarning,
                stacklevel=2,
            )
        fit = fit_auto(ordered, _GOF_PRECISION, dtype)
    if not fit.valid:
        raise ValueError("Power-law model fitting failed. Check that data contains positive integers.")
    n = values.size
    alpha_hat = fit.alpha
    if dtype == "left":
        body = values[values < fit.x_min]
    else:
        body = values[values > fit.x_max]
    model_n = min(max(int((1.0 - body.size / n) * n), 0), n)
    if model_n > 0:
        if dtype == "left":
            parametric = sample_unbounded(alpha_hat, fit.x_min, model_n * n_reps, gen)
        else:
            parametric = sample_truncated(alpha_hat, 1, fit.x_max, model_n * n_reps, gen)
        parametric = parametric.reshape(n_reps, model_n)
    else:
        parametric = np.empty((n_reps, 0), dtype=np.int64)
    ks_syn = np.empty(n_reps, dtype=float)
    for i in range(n_reps):
        parts = []
        if model_n:
            parts.append(parametric[i])
        n_body = n - model_n
        if n_body:
            parts.append(body[gen.integers(0, body.size, n_body)])
        replica = np.sort(np.concatenate(parts))
        refit = fit_auto(replica, _GOF_PRECISION, dtype)
        ks_syn[i] = refit.ks_statistic if refit.valid else math.inf
    p_value = float(np.mean(ks_syn >= fit.ks_statistic))
    return PowerLawGof(
        gof_pvalue=p_value,
        ks_statistic=fit.ks_statistic,
        n_replicas=n_reps,
        alpha=alpha_hat,
        x_min=fit.x_min,
        x_max=fit.x_max,
        type=dtype,
    )
