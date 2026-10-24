"""Volatility toolkit: realised vol, EWMA, GARCH(1,1) with variance targeting,
IV rank, variance risk premium, and an SVI smile fit with a butterfly-arbitrage check."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.signal import lfilter


def log_returns(prices):
    return np.diff(np.log(np.asarray(prices, float)))


def hist_vol(prices, window=30):
    r = log_returns(prices)[-window:]
    return float(np.std(r, ddof=1) * np.sqrt(252))


def ewma_vol(prices, lam=0.94):
    r = log_returns(prices)
    var = np.var(r[:30]) if len(r) > 30 else np.var(r)
    for x in r:
        var = lam * var + (1 - lam) * x * x
    return float(np.sqrt(var * 252))


def iv_rank(iv_history, iv_now):
    """Percentile (0-100) of today's IV within its own history."""
    h = np.asarray(iv_history, float)
    return float(100.0 * np.mean(h <= iv_now))


def variance_risk_premium(iv, forecast_vol):
    """IV^2 - forecast RV^2 (annualised variance points). Positive = options rich."""
    return float(iv**2 - forecast_vol**2)


# ------------------------------------------------------------------ GARCH(1,1)
@dataclass
class GarchFit:
    omega: float
    alpha: float
    beta: float
    last_var: float
    last_r2: float

    @property
    def persistence(self):
        return self.alpha + self.beta

    @property
    def long_run_vol(self):
        return float(np.sqrt(self.omega / (1 - self.persistence) * 252))


def _garch_filter(r2, omega, alpha, beta, h0):
    x = np.empty_like(r2)
    x[0] = h0
    x[1:] = omega + alpha * r2[:-1]
    return lfilter([1.0], [1.0, -beta], x)


def garch_fit(returns) -> GarchFit:
    """MLE with variance targeting (omega = V(1-a-b)); fits (alpha, beta) only, which is far more
    stable than a free 3-parameter fit on a few hundred observations."""
    r = np.asarray(returns, float)
    r = r - r.mean()
    r2 = r * r
    V = r2.mean()

    def nll(p):
        a, b = p
        if a < 1e-6 or b < 0 or a + b >= 0.9995:
            return 1e12
        h = _garch_filter(r2, V * (1 - a - b), a, b, V)
        if np.any(h <= 0):
            return 1e12
        return 0.5 * np.sum(np.log(h) + r2 / h)

    best = min((minimize(nll, x0, method="Nelder-Mead",
                         options=dict(xatol=1e-7, fatol=1e-9, maxiter=2000))
                for x0 in ([0.08, 0.90], [0.05, 0.93], [0.15, 0.75])), key=lambda s: s.fun)
    a, b = best.x
    omega = V * (1 - a - b)
    h = _garch_filter(r2, omega, a, b, V)
    return GarchFit(omega, a, b, float(h[-1]), float(r2[-1]))


def garch_forecast_vol(returns, horizon_days):
    """Annualised average vol forecast over the next `horizon_days`."""
    g = garch_fit(returns)
    V = g.omega / (1 - g.persistence)
    h_next = g.omega + g.alpha * g.last_r2 + g.beta * g.last_var
    k = np.arange(horizon_days)
    path = V + g.persistence**k * (h_next - V)
    return float(np.sqrt(path.mean() * 252))


# ------------------------------------------------------------------ SVI smile
def svi_w(k, a, b, rho, m, s):
    """Raw SVI total variance w(k) = a + b(rho(k-m) + sqrt((k-m)^2 + s^2)), k = ln(K/F)."""
    k = np.asarray(k, float)
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s**2))


def svi_fit(k, w):
    """Least-squares SVI fit to market total variance w = IV^2 * T. Returns (params, rmse)."""
    k, w = np.asarray(k, float), np.asarray(w, float)
    lo = [-1.0, 1e-6, -0.999, k.min() - 1.0, 1e-4]
    hi = [max(w.max(), 1e-3) * 2, 10.0, 0.999, k.max() + 1.0, 5.0]
    best = None
    for rho0 in (-0.6, 0.0, 0.6):
        for s0 in (0.05, 0.2):
            x0 = [w.min() * 0.5, 0.1, rho0, k[np.argmin(w)], s0]
            x0 = np.clip(x0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
            res = least_squares(lambda p: svi_w(k, *p) - w, x0, bounds=(lo, hi))
            if best is None or res.cost < best.cost:
                best = res
    rmse = float(np.sqrt(2 * best.cost / len(k)))
    return tuple(best.x), rmse


def svi_butterfly_ok(params, k_grid=None):
    """Gatheral's g(k) >= 0 condition (no butterfly arbitrage). Returns (ok, min_g)."""
    a, b, rho, m, s = params
    k = np.linspace(-1.5, 1.5, 601) if k_grid is None else np.asarray(k_grid, float)
    root = np.sqrt((k - m) ** 2 + s**2)
    w = svi_w(k, *params)
    w1 = b * (rho + (k - m) / root)
    w2 = b * s**2 / root**3
    g = (1 - k * w1 / (2 * w)) ** 2 - w1**2 / 4 * (1 / w + 0.25) + w2 / 2
    return bool(np.all(g >= -1e-9)), float(g.min())


def svi_iv(k, T, params):
    return np.sqrt(np.maximum(svi_w(k, *params), 0) / T)
