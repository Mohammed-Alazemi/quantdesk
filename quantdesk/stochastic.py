"""Terminal-price simulators. GBM (constant vol) and Heston (stochastic vol, full-truncation Euler).

Heston matters for options because vol moves WITH price (rho<0) and vol-of-vol fattens tails;
a flat-vol GBM understates both. We also return an implied-vol proxy at the exit date so the
re-pricing step sees a path-dependent IV rather than a constant."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Heston:
    v0: float      # current variance (annualised)
    kappa: float   # mean-reversion speed
    theta: float   # long-run variance
    xi: float      # vol of vol
    rho: float     # spot/vol correlation

    @staticmethod
    def from_vols(spot_vol, long_run_vol, kappa=3.0, xi=0.6, rho=-0.7):
        return Heston(spot_vol**2, kappa, long_run_vol**2, xi, rho)


def gbm_terminal(S, mu, vol, horizon, n=50_000, seed=0):
    """Real-world (drift mu) lognormal terminal prices with antithetic variates. horizon in years."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n // 2)
    z = np.concatenate([z, -z])
    return S * np.exp((mu - 0.5 * vol**2) * horizon + vol * np.sqrt(horizon) * z)


def heston_terminal(S, mu, horizon, p: Heston, n=50_000, seed=0, steps_per_year=252):
    """Returns (S_T, v_T). Antithetic, full-truncation Euler."""
    rng = np.random.default_rng(seed)
    steps = max(int(round(horizon * steps_per_year)), 5)
    dt = horizon / steps
    half = n // 2
    lnS = np.full(2 * half, np.log(S))
    v = np.full(2 * half, p.v0)
    sq = np.sqrt(1 - p.rho**2)
    for _ in range(steps):
        z = rng.standard_normal((2, half))
        z1 = np.concatenate([z[0], -z[0]])
        z2 = np.concatenate([z[1], -z[1]])
        w2 = p.rho * z1 + sq * z2
        vp = np.maximum(v, 0.0)
        lnS += (mu - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1
        v += p.kappa * (p.theta - vp) * dt + p.xi * np.sqrt(vp * dt) * w2
    return np.exp(lnS), np.maximum(v, 0.0)


def expected_avg_iv(v, tau, p: Heston):
    """ATM-IV proxy: sqrt of expected average variance over the remaining life tau (years)
    given instantaneous variance v. Exact for the Heston variance process."""
    tau = max(tau, 1e-6)
    avg = p.theta + (np.asarray(v) - p.theta) * (1 - np.exp(-p.kappa * tau)) / (p.kappa * tau)
    return np.sqrt(np.maximum(avg, 1e-8))
