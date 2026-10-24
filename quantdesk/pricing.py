"""Option pricing: vectorised Black-Scholes-Merton, full Greeks (incl. vanna/volga),
robust implied vol, and a CRR binomial tree for American exercise."""
from __future__ import annotations
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def _arr(*xs):
    return [np.asarray(x, dtype=float) for x in xs]


def _unwrap(x):
    x = np.asarray(x)
    return float(x) if x.ndim == 0 else x


def _check_kind(kind):
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    return kind == "call"


def d1_d2(S, K, T, r, sigma, q=0.0):
    S, K, T, r, sigma, q = _arr(S, K, T, r, sigma, q)
    with np.errstate(all="ignore"):
        sqT = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqT)
        d2 = d1 - sigma * sqT
    return d1, d2


def bs_price(S, K, T, r, sigma, q=0.0, kind="call"):
    """European price with continuous dividend yield q. Vectorised over every argument."""
    S, K, T, r, sigma, q = _arr(S, K, T, r, sigma, q)
    call = _check_kind(kind)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    with np.errstate(all="ignore"):
        if call:
            p = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            p = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    intrinsic = np.maximum(S - K, 0.0) if call else np.maximum(K - S, 0.0)
    return _unwrap(np.where((T > 0) & (sigma > 0), p, intrinsic))


def greeks(S, K, T, r, sigma, q=0.0, kind="call"):
    """Units: delta per $1, gamma per $1^2, vega/vanna per 1 vol point (0.01),
    volga per (1 vol point)^2, theta per calendar day, rho per 1% rate move.
    Requires T > 0 and sigma > 0 (Greeks are undefined at expiry)."""
    _check_kind(kind)
    S, K, T, r, sigma, q = _arr(S, K, T, r, sigma, q)
    if np.any(T <= 0) or np.any(sigma <= 0):
        raise ValueError("greeks need T > 0 and sigma > 0")
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    pdf = norm.pdf(d1)
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    sqT = np.sqrt(T)
    vega_raw = S * dq * pdf * sqT
    gamma = dq * pdf / (S * sigma * sqT)
    vanna = -dq * pdf * d2 / sigma / 100.0
    volga = vega_raw * d1 * d2 / sigma / 1e4
    base_theta = -S * dq * pdf * sigma / (2 * sqT)
    if kind == "call":
        delta = dq * norm.cdf(d1)
        theta = (base_theta - r * K * dr * norm.cdf(d2) + q * S * dq * norm.cdf(d1)) / 365
        rho = K * T * dr * norm.cdf(d2) / 100
    else:
        delta = -dq * norm.cdf(-d1)
        theta = (base_theta + r * K * dr * norm.cdf(-d2) - q * S * dq * norm.cdf(-d1)) / 365
        rho = -K * T * dr * norm.cdf(-d2) / 100
    out = dict(delta=delta, gamma=gamma, vega=vega_raw / 100, theta=theta, rho=rho,
               vanna=vanna, volga=volga)
    return {k: _unwrap(v) for k, v in out.items()}


def implied_vol(price, S, K, T, r, q=0.0, kind="call"):
    """Brent root-find on [1e-6, 10]. Returns NaN when price violates no-arbitrage bounds.
    Accepts arrays (e.g. a whole option chain) and broadcasts them element by element."""
    _check_kind(kind)
    if any(np.ndim(x) for x in (price, S, K, T, r, q)):
        f = np.vectorize(lambda *a: _implied_vol_scalar(*a, kind=kind), otypes=[float])
        return f(price, S, K, T, r, q)
    return _implied_vol_scalar(price, S, K, T, r, q, kind=kind)


def _implied_vol_scalar(price, S, K, T, r, q=0.0, kind="call"):
    if T <= 0:
        return float("nan")
    dr, dq = np.exp(-r * T), np.exp(-q * T)
    if kind == "call":
        lo, hi = max(S * dq - K * dr, 0.0), S * dq
    else:
        lo, hi = max(K * dr - S * dq, 0.0), K * dr
    if not (lo < price < hi):
        return float("nan")
    f = lambda s: bs_price(S, K, T, r, s, q, kind) - price
    try:
        return float(brentq(f, 1e-6, 10.0, xtol=1e-12, maxiter=300))
    except ValueError:
        return float("nan")


def american_crr(S, K, T, r, sigma, q=0.0, kind="call", steps=500):
    """Cox-Ross-Rubinstein binomial tree with early exercise (vectorised backward pass)."""
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    j = np.arange(steps + 1)
    ST = S * u ** (2 * j - steps)
    V = np.maximum(ST - K, 0.0) if kind == "call" else np.maximum(K - ST, 0.0)
    for i in range(steps - 1, -1, -1):
        V = disc * (p * V[1:] + (1 - p) * V[:-1])
        Si = S * u ** (2 * np.arange(i + 1) - i)
        ex = np.maximum(Si - K, 0.0) if kind == "call" else np.maximum(K - Si, 0.0)
        V = np.maximum(V, ex)
    return float(V[0])
