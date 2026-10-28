"""pypowercpp: discrete power-law fitting (Clauset-Shalizi-Newman, 2009).

Pure-Python port of the R package poweRcpp: maximum-likelihood estimation
of the exponent ``alpha`` with Brent optimisation, bound estimation by KS
minimisation, and semi-parametric bootstrap goodness-of-fit.
"""

from __future__ import annotations

from ._api import (
    PowerLawFit,
    PowerLawGof,
    fit_powerlaw,
    powerlaw_cdf,
    powerlaw_generate,
    powerlaw_gof,
    powerlaw_pdf,
)

__version__ = "0.1.0"

__all__ = [
    "PowerLawFit",
    "PowerLawGof",
    "__version__",
    "fit_powerlaw",
    "powerlaw_cdf",
    "powerlaw_generate",
    "powerlaw_gof",
    "powerlaw_pdf",
]
