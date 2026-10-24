"""Seven-gate pre-trade checklist (+ position-count gate) driven by an explicit, editable Rules object.
Hard gates veto the trade. Soft gates warn. Sizing is only produced when every hard gate passes."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .risk import kelly_fraction, bootstrap_ci


@dataclass
class Rules:
    max_open_positions: int = 2
    min_dte: int = 45
    max_size_frac: float = 0.10
    min_pop: float = 0.45
    min_sharpe: float = 0.10
    max_iv_hv: float = 1.15
    kelly_haircut: float = 0.5            # trade half-Kelly of the *median bootstrap* Kelly
    entry_window_days: tuple = (28, 56)   # pre-earnings strategy: enter 4-8 weeks before the print


@dataclass
class Gate:
    name: str
    passed: bool
    value: str
    hard: bool = True


@dataclass
class Verdict:
    gates: list = field(default_factory=list)
    allowed: bool = False
    kelly_median: float = 0.0
    recommended_fraction: float = 0.0
    recommended_premium: float = 0.0


def run_gates(pnl, risk_capital, premium_total, portfolio, iv, hv30, dte, hold_days,
              days_to_earnings=None, open_positions=0, kelly_boot=None,
              pre_earnings_strategy=False, rules: Rules | None = None) -> Verdict:
    rules = rules or Rules()
    pnl = np.asarray(pnl, float)
    g = []
    pop = float((pnl > 0).mean())
    g.append(Gate("1 Probability of profit", pop >= rules.min_pop, f"{pop:.1%} (min {rules.min_pop:.0%})"))
    sh = float(pnl.mean() / pnl.std()) if pnl.std() > 0 else 0.0
    g.append(Gate("2 Sharpe per trade", sh >= rules.min_sharpe, f"{sh:.2f} (min {rules.min_sharpe})"))
    f_star, _ = kelly_fraction(pnl, risk_capital)
    g.append(Gate("3 Kelly > 0", f_star > 0, f"{f_star:.1%}"))
    ratio = iv / hv30
    g.append(Gate("4 IV / HV30 edge", ratio <= rules.max_iv_hv, f"{ratio:.2f} (max {rules.max_iv_hv})"))
    g.append(Gate("5 DTE minimum", dte >= rules.min_dte, f"{dte}d (min {rules.min_dte})"))
    if days_to_earnings is None:
        g.append(Gate("6 Earnings timing", True, "no print supplied: VERIFY MANUALLY", hard=False))
    else:
        ok = hold_days <= days_to_earnings - 1
        g.append(Gate("6 Earnings: exit day before", ok,
                      f"print in {days_to_earnings}d, hold {hold_days}d"))
        if pre_earnings_strategy:
            lo, hi = rules.entry_window_days
            g.append(Gate("6b Entry window 4-8 wks pre-print", lo <= days_to_earnings <= hi,
                          f"{days_to_earnings}d (window {lo}-{hi})", hard=False))
    frac = premium_total / portfolio
    g.append(Gate("7 Size cap", frac <= rules.max_size_frac, f"{frac:.1%} (max {rules.max_size_frac:.0%})"))
    g.append(Gate("8 Open positions", open_positions < rules.max_open_positions,
                  f"{open_positions} open (max {rules.max_open_positions})"))
    lo_ci, hi_ci = bootstrap_ci(pnl, n=500)
    g.append(Gate("S1 EV 95% CI excludes 0", lo_ci > 0, f"[{lo_ci:.0f}, {hi_ci:.0f}] $", hard=False))
    if kelly_boot:
        g.append(Gate("S2 Kelly robust (p25 > 0)", kelly_boot["p25"] > 0,
                      f"p25 {kelly_boot['p25']:.1%}, median {kelly_boot['median']:.1%}", hard=False))

    allowed = all(x.passed for x in g if x.hard)
    km = kelly_boot["median"] if kelly_boot else f_star
    rec_f = min(rules.max_size_frac, rules.kelly_haircut * km) if allowed else 0.0
    return Verdict(g, allowed, km, rec_f, rec_f * portfolio)
