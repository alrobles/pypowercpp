# pypowercpp

Pure-Python port of [poweRcpp](https://github.com/alrobles/poweRcpp): fitting of
discrete power-law distributions following Clauset, Shalizi & Newman (2009),
*Power-law distributions in empirical data* (SIAM Review 51(4), 661–703).

This package is the backend of the power-law validation step of
**interseek** / **interseekr**.

## Install

```bash
pip install git+https://github.com/alrobles/pypowercpp.git
```

From a checkout (runtime deps are numpy + scipy only):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"   # dev extras: pytest, build
```

## Minimal example

```python
from pypowercpp import fit_powerlaw, powerlaw_generate, powerlaw_gof

data = powerlaw_generate(5000, alpha=2.5, x_min=1, rng=42)
fit = fit_powerlaw(data)
print(fit)
# Discrete power-law distribution fit
#   Type        : left
#   Alpha       : 2.49901
#   x_min       : 1
#   ...

gof = powerlaw_gof(data, replicas=200, seed=1)
print("p-value:", gof.gof_pvalue)
```

## R → Python API equivalence

| R (poweRcpp) | Python (pypowercpp) | Notes |
|---|---|---|
| `fit_powerlaw(data, x_min, alpha_precision, type)` | `fit_powerlaw(data, x_min=None, alpha_precision=0.01, type="left")` | Returns `PowerLawFit` with `.alpha .x_min .x_max .ks_statistic .standard_error .log_likelihood .n_tail .type` (+`.valid`); `print`/`summary` → `str()`/`.summary()` |
| `powerlaw_pdf(x, alpha, x_min)` | `powerlaw_pdf(x, alpha, x_min=1)` | Same PMF; scalar → `float`, vector → `ndarray` |
| `powerlaw_cdf(x, alpha, x_min)` | `powerlaw_cdf(x, alpha, x_min=1)` | **Different**: Python returns the standard CDF `P(X ≤ x)`; R returns the survival function `P(X ≥ x)` |
| `powerlaw_generate(n, alpha, x_min, x_max)` | `powerlaw_generate(n, alpha, x_min=1, x_max=None, rng=None)` | Exact discrete CDF inversion; default `x_max = x_min * 1000` as in R; `rng` accepts a seed or numpy RNG |
| `powerlaw_gof(data, alpha, x_min, replicas, type)` | `powerlaw_gof(data, alpha=None, x_min=None, replicas=1000, type="left", seed=None)` | Returns `PowerLawGof` with `.gof_pvalue` (alias `.p_value`), `.ks_statistic`, `.n_replicas` (alias `.replicas`); serial bootstrap (R uses RcppParallel) |

Semantic notes:

- Result `.type` echoes `"left"`/`"right"` (R returns `"Left bounded (Type I)"`
  style labels). As in R, the `x_min` argument carries the fixed *upper* bound
  when `type="right"`.
- `standard_error = alpha / sqrt(n_tail)`; the KS statistic, MLE (Brent) and
  bound search follow the C++ implementation, with `scipy.special.zeta` as the
  Hurwitz-zeta backend.
- Unfittable inputs (constant data, `x_min >= max(data)`, tail under 10
  observations) yield `NaN`/`None` fields with `.valid == False` instead of
  raising, mirroring R's `NA` fields. Empty input and non-positive data raise.
- Bound estimation is roughly `O(n_unique * n)` with vectorised inner steps
  (`O(n²)` worst case for all-distinct data).

## Tests and build

```bash
python -m pytest -q
python -m build
```

## License

MIT © Angel Luis Robles Fernandez. See [LICENSE](LICENSE).
poweRcpp itself builds on the DiscretePowerlawFitter work of Carlos Manuel
Rodriguez.
