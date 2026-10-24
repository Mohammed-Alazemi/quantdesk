"""What the option market is pricing in: risk-neutral densities, variance swaps and event vol.

* Breeden-Litzenberger (1978): the risk-neutral density of S_T is  e^{rT} d^2C/dK^2.
* Carr-Madan replication: a variance swap's fair strike is a 1/K^2-weighted strip of OTM options,
      K_var = (2 e^{rT} / T) [ Int_0^F P(K)/K^2 dK + Int_F^inf C(K)/K^2 dK ].
* Event vol: variance is additive in time, so an earnings day's vol can be backed out of the
  term structure, and the IV crush after the print can be predicted before it happens.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import trapezoid
from .pricing import bs_price
from .vol import svi_iv, svi_w


# ------------------------------------------------------------------ densities
def bl_density(K, calls, r, T):
    """Risk-neutral density on the strike grid K from call prices (second finite difference).
    K must be evenly spaced. Returns (K_inner, density)."""
    K, C = np.asarray(K, float), np.asarray(calls, float)
    h = K[1] - K[0]
    if not np.allclose(np.diff(K), h):
        raise ValueError("bl_density needs an evenly spaced strike grid")
    dens = np.exp(r * T) * (C[2:] - 2 * C[1:-1] + C[:-2]) / h**2
    return K[1:-1], dens


def _wing(params, sign, width):
    """Smallest |k| that is `width` local standard deviations from the money on one side."""
    ks = np.linspace(0.01, 8.0, 800)
    w = np.maximum(svi_w(sign * ks, *params), 1e-8)
    ok = ks > width * np.sqrt(w)
    return float(ks[np.argmax(ok)]) if ok.any() else 8.0


def svi_density(S, T, r, params, q=0.0, n=4001, width=7.0):
    """Density implied by an SVI smile (params in total-variance form, k = ln(K/F)).
    The strike grid reaches `width` local standard deviations into each wing, so a heavy
    left tail is not cut off. Negative values flag butterfly arbitrage in the fitted smile."""
    F = S * np.exp((r - q) * T)
    K = np.linspace(F * np.exp(-_wing(params, -1, width)), F * np.exp(_wing(params, +1, width)), n)
    iv = svi_iv(np.log(K / F), T, params)
    return bl_density(K, bs_price(S, K, T, r, iv, q, "call"), r, T)


def density_moments(K, dens):
    """Mass, mean, standard deviation, skewness and excess kurtosis of a density on a grid."""
    m0 = trapezoid(dens, K)
    p = dens / m0
    mean = trapezoid(K * p, K)
    var = trapezoid((K - mean) ** 2 * p, K)
    sd = np.sqrt(var)
    skew = trapezoid(((K - mean) / sd) ** 3 * p, K)
    kurt = trapezoid(((K - mean) / sd) ** 4 * p, K) - 3
    return dict(mass=float(m0), mean=float(mean), sd=float(sd), skew=float(skew),
                excess_kurt=float(kurt))


# ------------------------------------------------------------------ variance swap
def variance_swap_strike(S, T, r, strikes, ivs, q=0.0):
    """Fair variance (annualised, decimal^2) from a strip of implied vols by Carr-Madan.
    Uses OTM puts below the forward and OTM calls above it. The strip should extend far enough
    into both wings that the tails contribute negligibly."""
    K = np.asarray(strikes, float)
    iv = np.asarray(ivs, float)
    order = np.argsort(K)
    K, iv = K[order], iv[order]
    F = S * np.exp((r - q) * T)
    otm = np.where(K < F, bs_price(S, K, T, r, iv, q, "put"), bs_price(S, K, T, r, iv, q, "call"))
    return float(2 * np.exp(r * T) / T * trapezoid(otm / K**2, K))


# ------------------------------------------------------------------ event volatility
def event_vol(term_iv, n_days, regular_iv):
    """Annualised vol of the event day, backed out of an expiry that contains the event.

    term_iv: implied vol of the expiry that spans the event (annualised)
    n_days: trading days to that expiry, event included
    regular_iv: vol of an ordinary day (an expiry before the event, or the back month)
    Variance is additive: term_iv^2 n = event^2 + regular^2 (n - 1).
    """
    ev2 = term_iv**2 * n_days - regular_iv**2 * (n_days - 1)
    if ev2 <= 0:
        raise ValueError("term vol does not exceed regular vol: no event premium priced in")
    return float(np.sqrt(ev2))


def implied_event_move(event_vol_annual):
    """Expected absolute one-day move priced in for the event: sigma_day * sqrt(2/pi)."""
    return float(event_vol_annual / np.sqrt(252) * np.sqrt(2 / np.pi))


def iv_after_event(term_iv, n_days, event_vol_annual):
    """Predicted implied vol of the same expiry the day after the print (the IV crush):
    the event's variance leaves the option all at once."""
    rest = (term_iv**2 * n_days - event_vol_annual**2) / (n_days - 1)
    return float(np.sqrt(max(rest, 0.0)))
