"""One call that runs the whole desk on a trade idea and formats a verdict-first report."""
from __future__ import annotations
import numpy as np
from .pricing import bs_price, greeks, implied_vol
from .vol import log_returns, hist_vol, garch_fit, garch_forecast_vol
from .stochastic import Heston, gbm_terminal, heston_terminal, expected_avg_iv
from .strategies import Leg, CostModel, position_pnl, net_greeks, value
from .risk import (kelly_fraction, kelly_bootstrap, kelly_vs_drift, var_cvar, stress_grid)
from .checklist import Rules, run_gates


def full_analysis(prices, K, dte, hold_days, iv, S=None, legs=None, r=0.045, q=0.0,
                  drift=0.08, model="heston", iv_exit_shift=0.0, days_to_earnings=None,
                  portfolio=5000.0, open_positions=0, pre_earnings_strategy=False,
                  cm: CostModel | None = None, rules: Rules | None = None, n_paths=40_000, seed=0):
    """prices: daily closes (>= ~250 for a sensible GARCH). iv: market ATM IV of the option (decimal).
    iv_exit_shift: expected change in IV by exit in vol points (e.g. +0.03 pre-earnings ramp,
    -0.10 if you would be holding through a crush). drift: ANNUAL real-world drift assumption."""
    cm = cm or CostModel()
    prices = np.asarray(prices, float)
    S = float(prices[-1]) if S is None else float(S)
    legs = legs or [Leg("call", K, 1)]
    rets = log_returns(prices)
    hv30 = hist_vol(prices, 30)
    g = garch_fit(rets)
    gvol = garch_forecast_vol(rets, hold_days)
    T0, T1, h = dte / 365, (dte - hold_days) / 365, hold_days / 365

    hp = Heston.from_vols(gvol, g.long_run_vol)

    def simulate(mu, n=n_paths):
        if model == "heston":
            ST, vT = heston_terminal(S, mu, h, hp, n=n, seed=seed)
            lvl = iv - float(expected_avg_iv(hp.v0, T0, hp))      # anchor model IV to the market
            ivT = np.maximum(expected_avg_iv(vT, T1, hp) + lvl + iv_exit_shift, 0.03)
        else:
            ST = gbm_terminal(S, mu, gvol, h, n=n, seed=seed)
            ivT = np.full_like(ST, max(iv + iv_exit_shift, 0.03))
        return position_pnl(legs, S, T0, T1, r, ST, iv, ivT, q, cm)

    pnl, entry_cash, risk_cap = simulate(drift)
    f_star, growth = kelly_fraction(pnl, risk_cap)
    boot = kelly_bootstrap(pnl, risk_cap)
    sens = kelly_vs_drift(lambda mu: simulate(mu, 15_000)[::2], [-0.20, 0.0, 0.08, 0.20, 0.40])
    var5, cvar5 = var_cvar(pnl)
    verdict = run_gates(pnl, risk_cap, entry_cash, portfolio, iv, hv30, dte, hold_days,
                        days_to_earnings, open_positions, boot, pre_earnings_strategy, rules)
    dS, dIV, grid = stress_grid(legs, S, T0, r, iv, q=q)
    return dict(S=S, hv30=hv30, garch=g, garch_vol=gvol, heston=hp, pnl=pnl, entry_cash=entry_cash,
                risk_capital=risk_cap, kelly=f_star, kelly_boot=boot, drift_sens=sens,
                var5=var5, cvar5=cvar5, verdict=verdict, greeks=net_greeks(legs, S, T0, r, iv, q),
                stress=(dS, dIV, grid), mean_pnl=float(pnl.mean()), pop=float((pnl > 0).mean()),
                legs=legs, dte=dte, hold_days=hold_days, iv=iv, drift=drift, model=model)


def format_report(a) -> str:
    v = a["verdict"]
    L = []
    L.append("=" * 66)
    L.append(f"  VERDICT: {'TRADE ALLOWED' if v.allowed else 'NO TRADE'}"
             + (f"   size <= {v.recommended_fraction:.1%} of portfolio (${v.recommended_premium:,.0f} premium)"
                if v.allowed else ""))
    L.append("=" * 66)
    for gt in v.gates:
        tag = "PASS" if gt.passed else ("FAIL" if gt.hard else "WARN")
        L.append(f"  [{tag}] {gt.name:<34} {gt.value}")
    L.append("")
    L.append(f"  Spot {a['S']:.2f} | IV {a['iv']:.1%} | HV30 {a['hv30']:.1%} | GARCH fwd vol {a['garch_vol']:.1%}"
             f" (persistence {a['garch'].persistence:.3f})")
    L.append(f"  Model: {a['model']} | drift assumed {a['drift']:.0%} | DTE {a['dte']} | hold {a['hold_days']}d")
    L.append(f"  Cost of position ${a['entry_cash']:.0f} | mean P&L ${a['mean_pnl']:.0f} (net of costs)"
             f" | PoP {a['pop']:.1%}")
    L.append(f"  VaR5 ${a['var5']:.0f} | CVaR5 ${a['cvar5']:.0f}")
    gk = a["greeks"]
    L.append(f"  Greeks ($): delta {gk['delta']:.1f} gamma {gk['gamma']:.2f} vega/pt {gk['vega']:.2f}"
             f" theta/day {gk['theta']:.2f} vanna {gk['vanna']:.2f}")
    L.append("")
    L.append("  Kelly sensitivity to the drift you assume (this is the honesty check):")
    for s in a["drift_sens"]:
        L.append(f"    drift {s['drift']:+.0%}: PoP {s['pop']:.1%}  mean ${s['mean_pnl']:+.0f}  Kelly {s['kelly']:.1%}")
    kb = a["kelly_boot"]
    L.append(f"  Kelly bootstrap: p25 {kb['p25']:.1%} / median {kb['median']:.1%} / p75 {kb['p75']:.1%}")
    L.append("")
    dS, dIV, grid = a["stress"]
    L.append("  Stress P&L ($): rows = spot move, cols = IV change (vol pts)")
    L.append("          " + "".join(f"{x*100:>+8.0f}" for x in dIV))
    for ds, row in zip(dS, grid):
        L.append(f"  {ds:>+6.0%}  " + "".join(f"{x:>8.0f}" for x in row))
    return "\n".join(L)
