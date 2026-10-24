"""Multi-leg positions, realistic transaction costs, expiry payoffs and scenario P&L."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .pricing import bs_price, greeks

MULT = 100


@dataclass(frozen=True)
class Leg:
    kind: str            # "call" | "put"
    K: float
    qty: int             # +long / -short (contracts)
    iv_offset: float = 0.0   # skew term: this leg's IV = base IV + offset


@dataclass
class CostModel:
    """Half the bid-ask spread is paid on every fill (entry and exit)."""
    half_spread_pct: float = 0.03   # of option mid
    min_half_spread: float = 0.01   # $ per share floor (penny-wide quotes are rare)
    commission: float = 0.65        # $ per contract per side

    def half_spread(self, px):
        return np.maximum(np.asarray(px) * self.half_spread_pct, self.min_half_spread)


def bull_call_spread(K_long, K_short, skew=0.0, n=1):
    """Long K_long call, short K_short call (K_short > K_long). `skew` = IV(short) - IV(long);
    typically negative for equities (higher strikes trade at lower IV)."""
    return [Leg("call", K_long, +n), Leg("call", K_short, -n, iv_offset=skew)]


def value(legs, S, T, r, iv, q=0.0):
    """Mid value of the position in $ (x100 multiplier), vectorised over S and/or iv."""
    return sum(l.qty * bs_price(S, l.K, T, r, np.asarray(iv) + l.iv_offset, q, l.kind)
               for l in legs) * MULT


def net_greeks(legs, S, T, r, iv, q=0.0):
    tot = None
    for l in legs:
        g = greeks(S, l.K, T, r, iv + l.iv_offset, q, l.kind)
        g = {k: v * l.qty * MULT for k, v in g.items()}
        tot = g if tot is None else {k: tot[k] + g[k] for k in tot}
    return tot


def payoff_at_expiry(legs, ST):
    ST = np.asarray(ST, float)
    out = 0.0
    for l in legs:
        intr = np.maximum(ST - l.K, 0) if l.kind == "call" else np.maximum(l.K - ST, 0)
        out = out + l.qty * intr
    return out * MULT


def breakevens(legs, S, T, r, iv, cm: CostModel | None = None, q=0.0):
    """Expiry breakevens on a fine grid (debit positions)."""
    cm = cm or CostModel(0.0, 0.0, 0.0)
    entry, _ = _entry(legs, S, T, r, iv, q, cm)
    grid = np.linspace(S * 0.3, S * 2.0, 20001)
    pnl = payoff_at_expiry(legs, grid) - entry
    sign = np.sign(pnl)
    idx = np.where(np.diff(sign) != 0)[0]
    return [float(grid[i]) for i in idx]


def _entry(legs, S, T, r, iv, q, cm):
    cash = 0.0
    for l in legs:
        p = float(bs_price(S, l.K, T, r, iv + l.iv_offset, q, l.kind))
        hs = float(cm.half_spread(p))
        cash += l.qty * (p + np.sign(l.qty) * hs) * MULT
    comm = cm.commission * sum(abs(l.qty) for l in legs)
    return cash, comm


def position_pnl(legs, S0, T_entry, T_exit, r, ST, iv_entry, iv_T, q=0.0,
                 cm: CostModel | None = None):
    """Net-of-cost P&L array at the exit date, one value per simulated price.

    Returns (pnl, entry_cash, risk_capital). risk_capital = debit + all commissions, so every
    simulated return pnl/risk_capital is >= -1 (defined-risk debit positions only)."""
    cm = cm or CostModel()
    ST = np.asarray(ST, float)
    iv_T = np.broadcast_to(np.asarray(iv_T, float), ST.shape)
    entry_cash, comm_side = _entry(legs, S0, T_entry, r, iv_entry, q, cm)
    if entry_cash <= 0:
        raise ValueError("position_pnl supports debit positions only (net premium paid > 0).")
    exit_cash = 0.0
    for l in legs:
        v = bs_price(ST, l.K, T_exit, r, iv_T + l.iv_offset, q, l.kind)
        exit_cash = exit_cash + l.qty * (v - np.sign(l.qty) * cm.half_spread(v)) * MULT
    pnl = exit_cash - entry_cash - 2 * comm_side
    return pnl, float(entry_cash), float(entry_cash + 2 * comm_side)
