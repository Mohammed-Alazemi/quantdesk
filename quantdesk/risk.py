"""Risk and sizing: Kelly (point estimate, bootstrap, drift-sensitivity), VaR/CVaR, bootstrap CIs,
stress grids and portfolio Greeks."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar
from .strategies import value, net_greeks


def kelly_fraction(pnl, risk_capital, n_grid=300, max_n=6000, seed=0):
    """argmax_f E[log(1 + f R)], R = pnl / risk_capital. Returns (f*, growth rate at f*).
    f is the fraction of bankroll placed at risk (premium). 0 if no positive edge."""
    R = np.asarray(pnl, float) / risk_capital
    if len(R) > max_n:
        R = np.random.default_rng(seed).choice(R, max_n, replace=False)
    rmin = R.min()
    f_hi = 0.999 if rmin >= -1 else 0.999 / (-rmin)
    fs = np.linspace(0.0, f_hi, n_grid)
    g = np.array([np.log1p(f * R).mean() for f in fs])
    i = int(np.argmax(g))
    if i == 0:                                   # no positive edge: E[R] <= 0 means f* = 0
        return 0.0, 0.0
    # The growth curve is concave in f, so refine the grid optimum with a bounded 1-D search
    # (the grid alone is only accurate to f_hi / n_grid).
    lo, hi = fs[i - 1], fs[min(i + 1, n_grid - 1)]
    res = minimize_scalar(lambda f: -np.log1p(f * R).mean(), bounds=(lo, hi), method="bounded",
                          options=dict(xatol=1e-7))
    return float(res.x), float(-res.fun)


def kelly_bootstrap(pnl, risk_capital, n_boot=100, sub=4000, seed=1):
    """Sampling distribution of f*. A median of 8% with a 25th percentile of 0% means
    the edge is not statistically solid."""
    rng = np.random.default_rng(seed)
    pnl = np.asarray(pnl, float)
    fs = np.array([kelly_fraction(rng.choice(pnl, min(sub, len(pnl)), replace=True),
                                  risk_capital, n_grid=150, max_n=sub)[0]
                   for _ in range(n_boot)])
    return dict(median=float(np.median(fs)), p25=float(np.percentile(fs, 25)),
                p75=float(np.percentile(fs, 75)), mean=float(fs.mean()))


def kelly_vs_drift(sim_fn, drifts):
    """sim_fn(mu) -> (pnl, risk_capital). Shows how fragile the 'edge' is to the drift you assume."""
    rows = []
    for mu in drifts:
        pnl, cap = sim_fn(mu)
        f, _ = kelly_fraction(pnl, cap)
        rows.append(dict(drift=mu, pop=float((pnl > 0).mean()), mean_pnl=float(pnl.mean()), kelly=f))
    return rows


def var_cvar(pnl, alpha=0.05):
    pnl = np.asarray(pnl, float)
    var = float(np.percentile(pnl, 100 * alpha))
    return var, float(pnl[pnl <= var].mean())


def bootstrap_ci(x, stat=np.mean, n=2000, alpha=0.05, seed=2):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    s = np.array([stat(rng.choice(x, len(x), replace=True)) for _ in range(n)])
    return float(np.percentile(s, 100 * alpha / 2)), float(np.percentile(s, 100 * (1 - alpha / 2)))


def stress_grid(legs, S, T, r, iv, dS=(-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15),
                dIV=(-0.10, -0.05, 0, 0.05, 0.10), dt_days=0, q=0.0):
    """P&L ($, mid prices) for spot shocks (rows, %) x absolute IV shocks (cols, vol points)."""
    base = value(legs, S, T, r, iv, q)
    T2 = max(T - dt_days / 365, 1e-6)
    grid = np.array([[value(legs, S * (1 + ds), T2, r, max(iv + dv, 0.01), q) - base
                      for dv in dIV] for ds in dS])
    return np.array(dS), np.array(dIV), grid


def portfolio_greeks(positions, q=0.0):
    """positions: list of dicts(legs, S, T, r, iv, contracts). Greeks in $ terms, summed."""
    tot = {}
    for p in positions:
        g = net_greeks(p["legs"], p["S"], p["T"], p["r"], p["iv"], q)
        for k, v in g.items():
            tot[k] = tot.get(k, 0.0) + v * p.get("contracts", 1)
    return tot
