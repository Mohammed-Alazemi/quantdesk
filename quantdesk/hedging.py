"""Discrete delta hedging: where an option trader's P&L actually comes from.

Buy an option at implied vol sigma_i, then hedge it by shorting delta shares, rebalanced n times
before expiry, while the stock really moves with volatility sigma_r. Two classic results
(Wilmott ch. 10; Carr 2005; Derman & Kamal) are reproduced and tested here:

1. Hedge with the delta computed at the REALISED vol, and the P&L converges to the deterministic
   amount  [V(sigma_r) - V(sigma_i)] e^{rT}  as n -> infinity. The vol edge is locked in.
2. Hedge with the delta at the IMPLIED vol, and the expected P&L has the same sign but is path
   dependent:  P&L = Int e^{r(T-t)} 1/2 Gamma_i S^2 (sigma_r^2 - sigma_i^2) dt.

With sigma_r = sigma_i the P&L is pure hedging error. Its standard deviation falls like
1/sqrt(n): for an at-the-money option, std ~= sqrt(pi/4) * vega * sigma / sqrt(n).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .pricing import bs_price, greeks, d1_d2
from scipy.stats import norm


@dataclass
class HedgeResult:
    pnl: np.ndarray             # final P&L per option, valued at expiry ($ per share of underlying)
    premium: float              # option price paid at t = 0
    traded_shares: np.ndarray   # total |shares| traded per path (turnover)

    @property
    def mean(self):
        return float(self.pnl.mean())

    @property
    def std(self):
        return float(self.pnl.std(ddof=1))


def _delta(S, K, tau, r, sigma, q, kind):
    d1, _ = d1_d2(S, K, tau, r, sigma, q)
    dq = np.exp(-q * tau)
    return dq * norm.cdf(d1) if kind == "call" else -dq * norm.cdf(-d1)


def delta_hedge(S0, K, T, r, sigma_implied, sigma_real, n_steps=252, n_paths=20_000,
                sigma_hedge=None, mu=None, q=0.0, kind="call", cost_per_share_pct=0.0,
                seed=0) -> HedgeResult:
    """Long one option bought at sigma_implied, delta-hedged n_steps times until expiry.

    sigma_hedge: vol used to compute the hedge ratio (default: sigma_implied).
    mu: real-world drift of the stock (default r). In the continuous limit the hedged P&L does
        not depend on mu, which is the whole point of hedging.
    cost_per_share_pct: proportional transaction cost on every share traded (e.g. 0.0005 = 5 bp).
    """
    sigma_hedge = sigma_implied if sigma_hedge is None else sigma_hedge
    mu = r if mu is None else mu
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    half = n_paths // 2
    premium = float(bs_price(S0, K, T, r, sigma_implied, q, kind))

    S = np.full(2 * half, float(S0))
    shares = -_delta(S, K, T, r, sigma_hedge, q, kind)          # short delta
    cash = -premium - shares * S                                 # buy option, sell shares
    cash -= cost_per_share_pct * np.abs(shares) * S
    turnover = np.abs(shares)
    growth = np.exp(r * dt)
    for i in range(1, n_steps + 1):
        z = rng.standard_normal(half)
        z = np.concatenate([z, -z])                              # antithetic
        S = S * np.exp((mu - q - 0.5 * sigma_real**2) * dt + sigma_real * np.sqrt(dt) * z)
        cash = cash * growth + shares * S * (np.exp(q * dt) - 1)   # interest + dividends on shares
        tau = T - i * dt
        if i < n_steps:
            new = -_delta(S, K, tau, r, sigma_hedge, q, kind)
            trade = new - shares
            cash -= trade * S + cost_per_share_pct * np.abs(trade) * S
            turnover += np.abs(trade)
            shares = new
    payoff = np.maximum(S - K, 0.0) if kind == "call" else np.maximum(K - S, 0.0)
    cash += shares * S - cost_per_share_pct * np.abs(shares) * S     # unwind the hedge
    turnover += np.abs(shares)
    return HedgeResult(cash + payoff, premium, turnover)


def locked_in_pnl(S0, K, T, r, sigma_implied, sigma_real, q=0.0, kind="call"):
    """Continuous-limit P&L when hedging at the realised vol: [V(sigma_r) - V(sigma_i)] e^{rT}."""
    return float((bs_price(S0, K, T, r, sigma_real, q, kind)
                  - bs_price(S0, K, T, r, sigma_implied, q, kind)) * np.exp(r * T))


def hedging_error_std(S0, K, T, r, sigma, n_steps, q=0.0):
    """Approximate std of the hedging error with sigma_r = sigma_i (Derman-Kamal / Kamal & Derman
    1999): sqrt(pi/4) * vega * sigma / sqrt(n). Most accurate near the money."""
    vega = greeks(S0, K, T, r, sigma, q)["vega"] * 100          # $ per unit of vol
    return float(np.sqrt(np.pi / 4) * vega * sigma / np.sqrt(n_steps))


def gamma_theta_breakeven_move(S, K, T, r, sigma, q=0.0, kind="call"):
    """Daily stock move at which gamma P&L exactly pays the day's theta: sqrt(2|theta|/gamma).
    It equals about one daily implied standard deviation, S * sigma / sqrt(365)."""
    g = greeks(S, K, T, r, sigma, q, kind)
    return float(np.sqrt(2 * abs(g["theta"]) / g["gamma"]))
