"""Inventory-aware market making (Avellaneda & Stoikov, 2008).

A market maker quotes a bid and an ask around the mid-price s and earns the spread, but every
fill changes inventory q, and inventory is exposed to price moves. With exponential utility
(risk aversion gamma), mid-price volatility sigma and fill intensity lambda(delta) = A e^{-k delta}
for a quote delta away from the mid, the optimal quotes are:

    reservation price   r = s - q gamma sigma^2 (T - t)
    total spread        delta_a + delta_b = gamma sigma^2 (T - t) + (2 / gamma) ln(1 + gamma / k)

Long inventory pushes both quotes down (sell more readily, buy less), which pulls inventory back
towards zero. The simulator below reproduces the paper's comparison: the inventory strategy gives
up a little mean profit for a much tighter P&L distribution than quoting symmetrically around
the mid.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class ASParams:
    s0: float = 100.0
    T: float = 1.0
    sigma: float = 2.0      # mid-price vol, $ per sqrt(unit time) (arithmetic Brownian motion)
    dt: float = 0.005
    gamma: float = 0.1      # risk aversion
    k: float = 1.5          # fill-intensity decay
    A: float = 140.0        # fill intensity at zero distance


def optimal_quotes(s, q, t, p: ASParams):
    """Returns (bid, ask, reservation_price) for mid s, inventory q, time t."""
    tau = p.T - t
    r = s - q * p.gamma * p.sigma**2 * tau
    spread = p.gamma * p.sigma**2 * tau + (2 / p.gamma) * np.log(1 + p.gamma / p.k)
    return r - spread / 2, r + spread / 2, r


@dataclass
class MMResult:
    pnl: np.ndarray            # terminal wealth: cash + inventory marked at the mid
    inventory: np.ndarray      # terminal inventory
    spread_mean: float         # average quoted spread

    def summary(self):
        return dict(mean_pnl=float(self.pnl.mean()), std_pnl=float(self.pnl.std(ddof=1)),
                    mean_q=float(self.inventory.mean()), std_q=float(self.inventory.std(ddof=1)),
                    spread=self.spread_mean)


def simulate(p: ASParams = None, strategy="inventory", n_paths=1000, seed=0) -> MMResult:
    """strategy: 'inventory' (Avellaneda-Stoikov quotes) or 'symmetric' (same average spread,
    centred on the mid, ignoring inventory)."""
    p = p or ASParams()
    rng = np.random.default_rng(seed)
    n_steps = int(round(p.T / p.dt))
    s = np.full(n_paths, p.s0)
    q = np.zeros(n_paths)
    cash = np.zeros(n_paths)
    # The symmetric benchmark uses the time-averaged optimal spread, as in the original paper.
    avg_spread = np.mean([p.gamma * p.sigma**2 * (p.T - i * p.dt)
                          + (2 / p.gamma) * np.log(1 + p.gamma / p.k) for i in range(n_steps)])
    spreads = []
    for i in range(n_steps):
        t = i * p.dt
        if strategy == "inventory":
            bid, ask, _ = optimal_quotes(s, q, t, p)
        elif strategy == "symmetric":
            bid, ask = s - avg_spread / 2, s + avg_spread / 2
        else:
            raise ValueError("strategy must be 'inventory' or 'symmetric'")
        spreads.append(float(np.mean(ask - bid)))
        db, da = s - bid, ask - s
        hit_bid = rng.random(n_paths) < p.A * np.exp(-p.k * db) * p.dt
        hit_ask = rng.random(n_paths) < p.A * np.exp(-p.k * da) * p.dt
        q += hit_bid.astype(float) - hit_ask.astype(float)
        cash += np.where(hit_ask, ask, 0.0) - np.where(hit_bid, bid, 0.0)
        s = s + p.sigma * np.sqrt(p.dt) * np.where(rng.random(n_paths) < 0.5, 1.0, -1.0)
    return MMResult(cash + q * s, q, float(np.mean(spreads)))
