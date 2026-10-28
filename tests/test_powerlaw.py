"""Tests for pypowercpp (no external data; fixed seeds throughout)."""

import math
import warnings

import numpy as np
import pytest

from pypowercpp import (
    PowerLawFit,
    PowerLawGof,
    fit_powerlaw,
    powerlaw_cdf,
    powerlaw_generate,
    powerlaw_gof,
    powerlaw_pdf,
)


def _rpowerlaw(n, alpha, x_min=1, seed=0):
    """Independent continuous-approximation sampler (test helper only)."""
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    x = np.floor(x_min * (1.0 - u) ** (-1.0 / (alpha - 1.0)))
    return np.maximum(x, x_min).astype(np.int64)


# --- alpha recovery and x_min estimation ------------------------------------


def test_alpha_recovery():
    data = powerlaw_generate(5000, alpha=2.5, x_min=1, rng=42)
    fit = fit_powerlaw(data)
    assert isinstance(fit, PowerLawFit)
    assert fit.valid
    assert fit.alpha == pytest.approx(2.5, abs=0.15)
    assert fit.type == "left"


def test_alpha_recovery_independent_sampler():
    data = _rpowerlaw(3000, alpha=2.5, x_min=1, seed=3)
    fit = fit_powerlaw(data, x_min=1)
    assert fit.valid
    assert 2.0 < fit.alpha < 3.0


def test_known_xmin():
    data = powerlaw_generate(5000, alpha=2.5, x_min=1, rng=7)
    fit = fit_powerlaw(data[data >= 3])
    assert fit.valid
    assert fit.x_min in (2, 3, 4)


def test_fixed_xmin_tail_count():
    data = _rpowerlaw(300, alpha=2.5, x_min=1, seed=5)
    fit = fit_powerlaw(data, x_min=3)
    assert fit.valid
    assert fit.n_tail == int(np.sum(data >= 3))
    assert fit["n_tail"] == fit.n_tail


def test_fit_result_helpers():
    data = powerlaw_generate(300, alpha=2.5, x_min=1, rng=9)
    fit = fit_powerlaw(data, x_min=1)
    text = str(fit)
    assert "Alpha" in text and "KS statistic" in text
    assert "alpha +/- 1 SE" in fit.summary()
    assert fit.to_dict()["alpha"] == fit.alpha
    with pytest.raises(KeyError):
        fit["no_such_field"]


# --- pdf / cdf ---------------------------------------------------------------


def test_pdf_normalises():
    total = float(np.sum(powerlaw_pdf(np.arange(1, 20001), alpha=3.0, x_min=1)))
    assert total == pytest.approx(1.0, abs=1e-6)


def test_pdf_monotone_and_bounded():
    vals = powerlaw_pdf(np.arange(1, 21), alpha=2.5, x_min=1)
    assert bool(np.all(vals >= 0.0)) and bool(np.all(vals <= 1.0))
    assert bool(np.all(np.diff(vals) < 0.0))


def test_pdf_below_xmin_is_zero():
    assert list(powerlaw_pdf([1, 2], alpha=2.5, x_min=3)) == [0.0, 0.0]


def test_cdf_tends_to_one():
    assert powerlaw_cdf(100000, alpha=2.5, x_min=1) > 0.99999
    assert powerlaw_cdf(10**12, alpha=2.5, x_min=1) == pytest.approx(1.0)


def test_cdf_is_standard_cdf():
    # P(X <= 1) == P(X == 1) for x_min == 1; increasing; 0 below x_min.
    assert powerlaw_cdf(1, alpha=2.5, x_min=1) == pytest.approx(
        powerlaw_pdf(1, alpha=2.5, x_min=1)
    )
    vals = powerlaw_cdf(np.arange(1, 51), alpha=2.5, x_min=1)
    assert bool(np.all(np.diff(vals) > 0.0))
    assert list(powerlaw_cdf([0, 1, 2], alpha=2.5, x_min=3)) == [0.0, 0.0, 0.0]


def test_pdf_cdf_scalar_and_nan():
    assert isinstance(powerlaw_pdf(2, alpha=2.5, x_min=1), float)
    assert isinstance(powerlaw_cdf(2, alpha=2.5, x_min=1), float)
    assert math.isnan(powerlaw_pdf(float("nan"), alpha=2.5, x_min=1))
    out = powerlaw_cdf([1, float("nan")], alpha=2.5, x_min=1)
    assert math.isnan(out[1]) and out[0] > 0.0


# --- generation --------------------------------------------------------------


def test_generate_shape_bounds_dtype():
    smp = powerlaw_generate(500, alpha=2.5, x_min=3, rng=11)
    assert isinstance(smp, np.ndarray) and smp.shape == (500,)
    assert smp.dtype.kind == "i"
    assert int(smp.min()) >= 3 and int(smp.max()) <= 3000


def test_generate_seed_reproducible():
    a = powerlaw_generate(200, alpha=2.5, x_min=1, rng=123)
    b = powerlaw_generate(200, alpha=2.5, x_min=1, rng=123)
    assert list(a) == list(b)


def test_generate_custom_xmax():
    smp = powerlaw_generate(500, alpha=2.5, x_min=1, x_max=20, rng=50)
    assert int(smp.min()) >= 1 and int(smp.max()) <= 20


# --- right-bounded fits ------------------------------------------------------


def test_right_bounded_fit():
    data = powerlaw_generate(2000, alpha=2.5, x_min=1, x_max=60, rng=11)
    fit = fit_powerlaw(data, type="right")
    assert fit.valid
    assert fit.type == "right"
    assert fit.x_min == 1
    assert 1.5 < fit.alpha < 4.0
    assert 21 <= fit.x_max <= int(data.max())
    assert math.isfinite(fit.log_likelihood)


def test_right_bounded_no_candidates_is_invalid():
    fit = fit_powerlaw(np.arange(1, 21), type="right")
    assert not fit.valid and math.isnan(fit.alpha)


# --- goodness of fit ---------------------------------------------------------


def test_gof_pure_powerlaw():
    data = powerlaw_generate(500, alpha=2.5, x_min=1, rng=21)
    gof = powerlaw_gof(data, replicas=100, seed=22)
    assert isinstance(gof, PowerLawGof)
    assert 0.0 <= gof.gof_pvalue <= 1.0
    assert gof.gof_pvalue > 0.01
    assert gof.gof_pvalue == gof.p_value
    assert gof.n_replicas == 100 == gof.replicas
    assert gof.ks_statistic >= 0.0
    assert "p-value" in str(gof)


def test_gof_supplied_params():
    data = powerlaw_generate(300, alpha=2.5, x_min=1, rng=70)
    fit = fit_powerlaw(data, x_min=1)
    gof = powerlaw_gof(data, alpha=fit.alpha, x_min=fit.x_min, replicas=30, seed=71)
    assert 0.0 <= gof.gof_pvalue <= 1.0
    assert gof.alpha == pytest.approx(fit.alpha)


def test_gof_alpha_only_warns():
    data = powerlaw_generate(200, alpha=2.5, x_min=1, rng=72)
    with pytest.warns(UserWarning, match="'alpha' is ignored"):
        gof = powerlaw_gof(data, alpha=2.5, replicas=20, seed=73)
    assert 0.0 <= gof.gof_pvalue <= 1.0


def test_gof_right():
    data = powerlaw_generate(400, alpha=2.5, x_min=1, x_max=60, rng=80)
    gof = powerlaw_gof(data, replicas=30, type="right", seed=81)
    assert gof.type == "right"
    assert 0.0 <= gof.gof_pvalue <= 1.0


# --- edge cases --------------------------------------------------------------


def test_empty_input_errors():
    with pytest.raises(ValueError, match="non-empty"):
        fit_powerlaw([])
    with pytest.raises(ValueError, match="non-empty"):
        powerlaw_gof([], replicas=10)


def test_na_removed_with_warning():
    data = np.arange(1, 101, dtype=float)
    data = np.append(data, [np.nan, np.nan])
    with pytest.warns(UserWarning, match="NA values"):
        fit = fit_powerlaw(data, x_min=1)
    assert fit.valid
    with pytest.warns(UserWarning, match="NA values"):
        gof = powerlaw_gof(np.append(data, np.nan), x_min=1, replicas=10, seed=1)
    assert 0.0 <= gof.gof_pvalue <= 1.0
    with pytest.warns(UserWarning, match="NA values"), pytest.raises(
        ValueError, match="non-empty"
    ):
        fit_powerlaw([float("nan"), float("nan")])


def test_constant_data_nan_fields():
    fit = fit_powerlaw([7] * 50)
    assert not fit.valid
    assert math.isnan(fit.alpha)
    assert math.isnan(fit.ks_statistic)
    assert fit.x_min is None and fit.x_max is None and fit.n_tail is None
    single = fit_powerlaw([5])
    assert not single.valid and math.isnan(single.alpha)


def test_xmin_above_max_nan_fields():
    data = powerlaw_generate(200, alpha=2.5, x_min=1, rng=5)
    fit = fit_powerlaw(data, x_min=int(data.max()))
    assert not fit.valid and math.isnan(fit.alpha) and fit.x_min is None


def test_small_tail_is_invalid():
    fit = fit_powerlaw([1, 2, 3, 4, 5])
    assert not fit.valid and math.isnan(fit.alpha)


def test_noninteger_cast_warns():
    data = np.array([1.5, 2.5] * 50)
    with pytest.warns(UserWarning, match="Casting non-integer"):
        fit = fit_powerlaw(data, x_min=1)
    assert fit.valid
    assert fit.n_tail == 100


def test_integral_floats_cast_silently():
    data = np.array([1.0, 2.0, 3.0] * 40)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit = fit_powerlaw(data, x_min=1)
    assert fit.valid


def test_validation_errors():
    data = _rpowerlaw(100, alpha=2.5, seed=60)
    with pytest.raises(ValueError, match="positive integers"):
        fit_powerlaw([-1, 2, 3])
    with pytest.raises(ValueError, match='"left" or "right"'):
        fit_powerlaw(data, type="foo")
    with pytest.raises(ValueError, match="alpha_precision"):
        fit_powerlaw(data, alpha_precision=0.0)
    with pytest.raises(ValueError, match="'alpha' must be > 1"):
        powerlaw_pdf([1, 2], alpha=1.0)
    with pytest.raises(ValueError, match="'alpha' must be > 1"):
        powerlaw_cdf([1, 2], alpha=0.5)
    with pytest.raises(ValueError, match="'n' must be a positive integer"):
        powerlaw_generate(0, alpha=2.5)
    with pytest.raises(ValueError, match="'x_max' must be greater than 'x_min'"):
        powerlaw_generate(10, alpha=2.5, x_min=5, x_max=5)
    with pytest.raises(ValueError, match="'x_min' must be >= 1"):
        powerlaw_generate(10, alpha=2.5, x_min=0)
    with pytest.raises(ValueError, match="'replicas' must be"):
        powerlaw_gof(data, replicas=0)
    with pytest.raises(ValueError, match="Power-law model fitting failed"):
        powerlaw_gof([4] * 60, replicas=10)
