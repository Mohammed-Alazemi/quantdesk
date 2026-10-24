"""Semi-analytic Heston (1993) pricing and calibration to an implied-volatility surface.

Model (risk-neutral):
    dS = (r - q) S dt + sqrt(v) S dW1
    dv = kappa (theta - v) dt + xi sqrt(v) dW2,     dW1 dW2 = rho dt

Prices come from the characteristic function of X = ln(S_T / F) through Lewis's (2001)
single-integral formula

    C = S e^{-qT} - sqrt(F K) e^{-rT} / pi * Int_0^inf Re[ e^{i u x} phi(u - i/2) ] / (u^2 + 1/4) du,

with x = ln(F / K). The characteristic function uses the "little trap" form of Albrecher et al.
(2007), which avoids the branch-cut discontinuity of Heston's original formula at long maturities.

The integral is computed with fixed Gauss-Legendre nodes on [0, 1) after the map
u = c t / (1 - t). The scale c is set from the model's total variance, so the nodes sit where
the integrand lives for both short and long maturities. Fixed nodes make the price a smooth
function of the parameters, which keeps the least-squares calibration well behaved.
"""
from __future__ import annotations
from dataclasses import dataclass, astuple
import numpy as np
from scipy.optimize import least_squares
from .pricing import bs_price, greeks, implied_vol
from .stochastic import Heston

_N_NODES = 256
_t, _w = np.polynomial.legendre.leggauss(_N_NODES)
_t, _w = 0.5 * (_t + 1.0), 0.5 * _w          # nodes and weights on [0, 1]


def char_func(u, T, p: Heston):
    """E[exp(i u X)], X = ln(S_T / F), for complex u (vectorised)."""
    u = np.asarray(u, dtype=complex)
    a = p.kappa - p.rho * p.xi * 1j * u
    d = np.sqrt(a * a + p.xi**2 * (1j * u + u * u))
    g = (a - d) / (a + d)
    e = np.exp(-d * T)
    C = p.kappa * p.theta / p.xi**2 * ((a - d) * T - 2.0 * np.log((1 - g * e) / (1 - g)))
    D = (a - d) / p.xi**2 * (1 - e) / (1 - g * e)
    return np.exp(C + D * p.v0)


def heston_price(S, K, T, r, p: Heston, q=0.0, kind="call"):
    """European option price under Heston. Vectorised over K (scalar T)."""
    scalar = np.ndim(K) == 0
    K = np.atleast_1d(np.asarray(K, float))
    F = S * np.exp((r - q) * T)
    x = np.log(F / K)[:, None]
    var_scale = max(min(p.v0, p.theta) * T, 1e-4)
    c = 2.0 / np.sqrt(var_scale)                    # integrand decays on a scale ~ 1/sqrt(total var)
    u = c * _t / (1 - _t)
    jac = c / (1 - _t) ** 2
    phi = char_func(u - 0.5j, T, p)
    integrand = np.real(np.exp(1j * u * x) * phi) / (u * u + 0.25) * jac
    integral = integrand @ _w
    call = S * np.exp(-q * T) - np.sqrt(F * K) * np.exp(-r * T) / np.pi * integral
    if kind == "call":
        out = call
    elif kind == "put":
        out = call - S * np.exp(-q * T) + K * np.exp(-r * T)   # put-call parity
    else:
        raise ValueError("kind must be 'call' or 'put'")
    return float(out[0]) if scalar else out


def heston_smile(S, K, T, r, p: Heston, q=0.0):
    """Black-Scholes implied vols of Heston prices (OTM side used for accuracy)."""
    K = np.atleast_1d(np.asarray(K, float))
    F = S * np.exp((r - q) * T)
    calls = heston_price(S, K, T, r, p, q, "call")
    puts = calls - S * np.exp(-q * T) + K * np.exp(-r * T)
    otm_call = K >= F
    iv = np.where(otm_call,
                  implied_vol(calls, S, K, T, r, q, "call"),
                  implied_vol(puts, S, K, T, r, q, "put"))
    return iv


# ------------------------------------------------------------------ calibration
@dataclass
class HestonFit:
    params: Heston
    rmse_iv: float            # root-mean-square implied-vol error (decimal vol)
    feller: float             # 2 kappa theta - xi^2; >= 0 means v never reaches 0
    n_quotes: int
    success: bool


def calibrate_heston(S, r, quotes, q=0.0, x0: Heston | None = None) -> HestonFit:
    """Fit (v0, kappa, theta, xi, rho) to a surface of implied vols.

    quotes: iterable of (T, K, iv) with T in years and iv in decimal. Residuals are price errors
    divided by Black-Scholes vega, which is a first-order implied-vol error but avoids inverting
    every model price inside the optimiser. Several starting points guard against local minima.
    """
    Q = np.array(list(quotes), float)
    Ts, Ks, ivs = Q[:, 0], Q[:, 1], Q[:, 2]
    mkt = np.array([bs_price(S, k, t, r, v, q, "call") for t, k, v in Q])
    vega = np.array([greeks(S, k, t, r, v, q, "call")["vega"] * 100 for t, k, v in Q])
    vega = np.maximum(vega, 1e-3 * S)
    groups = [(t, np.where(Ts == t)[0]) for t in np.unique(Ts)]

    def resid(x):
        p = Heston(*x)
        model = np.empty_like(mkt)
        for t, idx in groups:
            model[idx] = heston_price(S, Ks[idx], t, r, p, q, "call")
        return (model - mkt) / vega

    lo = [1e-4, 0.05, 1e-4, 0.01, -0.99]
    hi = [2.0, 20.0, 2.0, 3.0, 0.99]
    atm = float(np.median(ivs)) ** 2
    starts = [x0] if x0 is not None else []
    starts += [Heston(atm, 2.0, atm, 0.5, -0.6), Heston(atm, 5.0, atm, 1.0, -0.3),
               Heston(atm, 1.0, atm, 0.3, -0.8)]
    best = None
    for s in starts:
        res = least_squares(resid, np.clip(astuple(s), lo, hi), bounds=(lo, hi),
                            x_scale="jac", max_nfev=300)
        if best is None or res.cost < best.cost:
            best = res
    p = Heston(*(float(v) for v in best.x))
    model_iv = np.concatenate([heston_smile(S, Ks[idx], t, r, p, q) for t, idx in groups])
    order = np.concatenate([idx for _, idx in groups])
    rmse = float(np.sqrt(np.nanmean((model_iv - ivs[order]) ** 2)))
    return HestonFit(p, rmse, float(2 * p.kappa * p.theta - p.xi**2), len(Q), bool(best.success))
