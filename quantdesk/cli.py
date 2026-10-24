"""Command line:
    python -m quantdesk demo
    python -m quantdesk research          # Heston calibration, hedging, densities, overfitting, MM
    python -m quantdesk backtest [--pump 0.25] [--tickers 20]
    python -m quantdesk analyze NVDA --dte 60 --hold 30 --portfolio 5000      (needs network + yfinance)
    python -m quantdesk analyze --csv closes.csv --strike 100 --iv 0.32 --dte 60 --hold 30   (offline)
"""
from __future__ import annotations
import argparse
import numpy as np
from .analysis import full_analysis, format_report
from .backtest import (run_universe, summarize, calibration_table, synthetic_market, null_test,
                       run_backtest)


def _backtest(pump, tickers):
    print(f"\nUniverse backtest: {tickers} synthetic tickers, pump={pump:.0%}, costs ON")
    tr = run_universe(n_tickers=tickers, pump=pump)
    s = summarize(tr)
    print(f"  trades {s['n']} | win rate {s['win_rate']:.1%} | mean return on premium {s['mean_R']:+.1%} "
          f"(95% CI {s['mean_R_ci95'][0]:+.1%}..{s['mean_R_ci95'][1]:+.1%}) | t={s['t_stat']:.2f}")
    print(f"  per-trade Sharpe {s['sharpe_trade']:.2f} | 10%-sizing equity x{s['final_equity']:.2f} "
          f"| max drawdown {s['max_drawdown']:.0%} | Brier {s['brier']:.3f}")
    print(f"  probabilistic Sharpe P(true Sharpe > 0) = {s['psr']:.1%}")
    print("\n  Calibration of the model's own PoP forecast:")
    print(calibration_table(tr["pop_pred"], tr["win"]).round(3).to_string(index=False))
    m = synthetic_market(seed=0, pump=pump)
    ev = run_backtest(m, n_calib=0)
    R, means = null_test(m, len(ev))
    p = float((means >= ev["R"].mean()).mean())
    print(f"\n  Null test (ticker 0): event-timed mean {ev['R'].mean():+.1%} vs random-window mean "
          f"{means.mean():+.1%}; p-value {p:.3f}")


def _garch_prices(n=600, seed=5, omega=4e-6, alpha=0.08, beta=0.90):
    """Synthetic closes with realistic volatility clustering (GARCH(1,1) returns)."""
    rng = np.random.default_rng(seed)
    h, r = omega / (1 - alpha - beta), np.empty(n)
    for t in range(n):
        r[t] = 0.0004 + np.sqrt(h) * rng.standard_normal()
        h = omega + alpha * (r[t] - 0.0004) ** 2 + beta * h
    return 100 * np.exp(np.cumsum(r))


def _research():
    from .stochastic import Heston
    from .heston import heston_smile, calibrate_heston
    from .hedging import delta_hedge, locked_in_pnl, hedging_error_std
    from .density import (svi_density, density_moments, variance_swap_strike, event_vol,
                          implied_event_move, iv_after_event)
    from .vol import svi_w
    from .stats import probabilistic_sharpe, deflated_sharpe
    from .marketmaking import simulate
    S, r = 100.0, 0.04
    line = "-" * 66

    print(line + "\n  1. Heston calibration to a noisy implied-vol surface (45 quotes, 0.2 vol-pt noise)")
    true = Heston(0.045, 2.5, 0.06, 0.7, -0.7)
    rng = np.random.default_rng(0)
    quotes = []
    for T in (0.1, 0.25, 0.5, 1.0, 2.0):
        K = S * np.exp(np.linspace(-0.4, 0.3, 9) * np.sqrt(T))
        quotes += [(T, k, v + rng.normal(0, 0.002)) for k, v in zip(K, heston_smile(S, K, T, r, true))]
    fit = calibrate_heston(S, r, quotes)
    for name in ("v0", "kappa", "theta", "xi", "rho"):
        print(f"     {name:<6} true {getattr(true, name):>7.3f}   fitted {getattr(fit.params, name):>7.3f}")
    print(f"     IV RMSE {fit.rmse_iv * 100:.2f} vol pts | Feller 2*kappa*theta - xi^2 = {fit.feller:+.3f}")

    print(line + "\n  2. Delta hedging: buy at 20% IV, stock realises 30%, ATM 6-month call")
    print(f"     locked-in edge if hedged at realised vol: ${locked_in_pnl(S, 100, 0.5, r, 0.2, 0.3):.3f}")
    print("     rebalances   hedge@realised mean/std    hedge@implied mean/std")
    for n in (12, 50, 200):
        a = delta_hedge(S, 100, 0.5, r, 0.2, 0.3, n_steps=n, sigma_hedge=0.3, n_paths=20_000)
        b = delta_hedge(S, 100, 0.5, r, 0.2, 0.3, n_steps=n, n_paths=20_000)
        print(f"     {n:>10}   {a.mean:>8.3f} / {a.std:<8.3f}       {b.mean:>8.3f} / {b.std:.3f}")
    h = delta_hedge(S, 100, 0.5, 0.0, 0.2, 0.2, n_steps=100, n_paths=20_000)
    print(f"     no edge (20% vs 20%), 100 rebalances: hedging-error std {h.std:.3f} "
          f"vs Derman-Kamal {hedging_error_std(S, 100, 0.5, 0.0, 0.2, 100):.3f}")

    print(line + "\n  3. What the smile prices in (SVI smile, 6 months)")
    params, T = (0.01, 0.1, -0.6, 0.0, 0.15), 0.5
    K, d = svi_density(S, T, r, params)
    m = density_moments(K, d)
    print(f"     risk-neutral density: mass {m['mass']:.4f}, mean {m['mean']:.2f} "
          f"(forward {S * np.exp(r * T):.2f}), skew {m['skew']:+.2f}, excess kurtosis {m['excess_kurt']:.2f}")
    Ks = np.linspace(5, 600, 4000)
    iv = np.sqrt(svi_w(np.log(Ks / (S * np.exp(r * T))), *params) / T)
    kv = variance_swap_strike(S, T, r, Ks, iv)
    print(f"     variance-swap strike {np.sqrt(kv):.1%} vs ATM IV {np.sqrt(svi_w(0, *params) / T):.1%}"
          " (the skew premium)")
    e = event_vol(0.36, 40, 0.24)
    print(f"     earnings: 40-day IV 36%, normal-day vol 24% -> event vol {e:.0%}, implied move "
          f"{implied_event_move(e):.1%}, IV the day after {iv_after_event(0.36, 40, e):.0%}")

    print(line + "\n  4. Backtest overfitting: try 100 random strategies, keep the best")
    R = rng.normal(0, 0.01, (100, 250))
    srs = R.mean(1) / R.std(1, ddof=1)
    best = R[np.argmax(srs)]
    print(f"     best Sharpe {srs.max() * np.sqrt(252):.2f} annualised | naive P(skill) "
          f"{probabilistic_sharpe(best):.1%} | deflated P(skill) {deflated_sharpe(best, 100, srs):.1%}")

    print(line + "\n  5. Market making (Avellaneda-Stoikov, gamma = 0.1, 1000 days)")
    for st in ("inventory", "symmetric"):
        x = simulate(strategy=st, n_paths=1000).summary()
        print(f"     {st:<10} spread {x['spread']:.2f} | P&L {x['mean_pnl']:.1f} +/- {x['std_pnl']:.1f}"
              f" | final inventory sd {x['std_q']:.1f}")


def _demo():
    prices = _garch_prices()
    a = full_analysis(prices, K=round(prices[-1] * 1.0), dte=60, hold_days=30, iv=0.33,
                      days_to_earnings=70, portfolio=5000)
    print(format_report(a))
    _backtest(0.25, 8)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="quantdesk")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo")
    sub.add_parser("research")
    b = sub.add_parser("backtest")
    b.add_argument("--pump", type=float, default=0.25)
    b.add_argument("--tickers", type=int, default=20)
    a = sub.add_parser("analyze")
    a.add_argument("ticker", nargs="?")
    a.add_argument("--csv")
    a.add_argument("--strike", type=float)
    a.add_argument("--iv", type=float)
    a.add_argument("--dte", type=int, default=60)
    a.add_argument("--hold", type=int, default=30)
    a.add_argument("--portfolio", type=float, default=5000)
    a.add_argument("--open", type=int, default=0)
    a.add_argument("--drift", type=float, default=0.08)
    a.add_argument("--earnings", type=int, default=None, help="days to next earnings")
    a.add_argument("--model", choices=["heston", "gbm"], default="heston")
    args = ap.parse_args(argv)

    if args.cmd == "demo":
        _demo()
    elif args.cmd == "research":
        _research()
    elif args.cmd == "backtest":
        _backtest(args.pump, args.tickers)
    else:
        if args.csv:
            from .data import load_closes
            prices = load_closes(args.csv)
            K, iv, earn = args.strike, args.iv, args.earnings
            if K is None or iv is None:
                raise SystemExit("--csv mode needs --strike and --iv")
        else:
            from . import data
            prices = data.get_prices(args.ticker)
            opt = data.get_atm_call(args.ticker, args.dte)
            K, iv, args.dte = opt["K"], opt["iv"], opt["dte"]
            earn = args.earnings if args.earnings is not None else data.days_to_next_earnings(args.ticker)
        res = full_analysis(prices, K=K, dte=args.dte, hold_days=args.hold, iv=iv, drift=args.drift,
                            days_to_earnings=earn, portfolio=args.portfolio,
                            open_positions=args.open, model=args.model)
        print(format_report(res))


if __name__ == "__main__":
    main()
