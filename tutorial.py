# %% [markdown]
# quantdesk tutorial: run this file one cell at a time.
# In Spyder, click inside a cell (the blocks that start with "# %%") and press Shift+Enter.
# In VS Code, click "Run Cell" above each block. Or just run the whole file.
#
# The examples use the CRDO Jan $200 call from my desk notes: stock $190, strike $200,
# 87 days to expiry, implied vol 68%, risk-free rate 4%, portfolio $25,000.

# %% 0. Setup ---------------------------------------------------------------------------
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
sys.path.insert(0, HERE)
import quantdesk as qd

S, K, DTE, IV, R = 190.0, 200.0, 87, 0.68, 0.04
T = DTE / 365
print("quantdesk", qd.__version__, "loaded")

# %% 1. Price an option and read its Greeks --------------------------------------------
call = qd.bs_price(S, K, T, R, IV, kind="call")
put = qd.bs_price(S, K, T, R, IV, kind="put")
g = qd.greeks(S, K, T, R, IV, kind="call")
print(f"Call ${call:.2f} (= ${call * 100:,.0f} per contract)   Put ${put:.2f}")
print(f"Delta {g['delta']:.3f} (~{g['delta'] * 100:.0f} shares)   Gamma {g['gamma']:.4f}")
print(f"Theta ${g['theta'] * 100:.2f}/day per contract   Vega ${g['vega'] * 100:.2f} per vol point")
print(f"Leverage: the call moves {g['delta'] * S / call:.1f}x as much as the stock in % terms")

# %% 2. Implied vol from a market price ------------------------------------------------
market_mid = 21.73            # type the option's mid price from your broker here
iv = qd.implied_vol(market_mid, S, K, T, R, kind="call")
print(f"A mid of ${market_mid} means the market is pricing {iv:.1%} volatility")
# A price outside the no-arbitrage bounds gives nan instead of a fake number:
print("Impossible price ->", qd.implied_vol(0.01, S, 100, T, R))

# %% 3. Load real price history -----------------------------------------------------------
# Get a CSV: finance.yahoo.com -> search the ticker -> Historical Data -> Time period 2-3 years
# -> Download. Save it next to this file and put its name below (e.g. "CRDO.csv").
CSV = None   # e.g. "CRDO.csv"

if CSV:
    from quantdesk.data import load_closes
    prices = load_closes(os.path.join(HERE, CSV))
    print(f"Loaded {len(prices)} closes, last {prices[-1]:.2f}")
else:
    # No file yet: use synthetic prices with realistic volatility clustering, ending at $190.
    from quantdesk.cli import _garch_prices
    prices = _garch_prices(n=600, omega=4e-5, alpha=0.08, beta=0.90)
    prices = prices / prices[-1] * S
    print("Using synthetic prices (set CSV above to use real data)")

print(f"HV30 {qd.hist_vol(prices, 30):.1%}   GARCH forecast over 41 days "
      f"{qd.garch_forecast_vol(np.diff(np.log(prices)), 41):.1%}   vs IV {IV:.0%}")

# %% 4. The full desk: one call, verdict first ---------------------------------------------
a = qd.full_analysis(
    prices, K=K, S=S, dte=DTE, hold_days=41, iv=IV,
    days_to_earnings=42,       # days to the next print (the exit must come before it)
    portfolio=25_000, open_positions=0,
    drift=0.15,                # YOUR assumed annual drift: be honest, the report tests it
    iv_exit_shift=0.05,        # expected IV change by exit (+ for a pre-earnings ramp)
    pre_earnings_strategy=True,
)
print(qd.format_report(a))

# %% 5. Change the rulebook ------------------------------------------------------------------
rules = qd.Rules(max_size_frac=0.08, min_pop=0.40, max_open_positions=3)
a2 = qd.full_analysis(prices, K=K, S=S, dte=DTE, hold_days=41, iv=IV, days_to_earnings=42,
                      portfolio=25_000, drift=0.15, rules=rules, n_paths=20_000)
print("Allowed with the edited rules?", a2["verdict"].allowed)

# %% 6. Spread versus naked call ---------------------------------------------------------------
naked = [qd.Leg("call", 200, 1)]
spread = qd.bull_call_spread(200, 230, skew=-0.02)     # sell the 230 call at 2 vol pts lower IV
for name, legs in (("naked 200C", naked), ("200/230 spread", spread)):
    from quantdesk.strategies import value
    gk = qd.net_greeks(legs, S, T, R, IV)
    print(f"{name:<15} cost ${value(legs, S, T, R, IV):>6,.0f}  delta {gk['delta']:>5.1f}  "
          f"vega/pt ${gk['vega']:>5.1f}  theta/day ${gk['theta']:>6.1f}")

dS, dIV, grid = qd.stress_grid(spread, S, T, R, IV, dt_days=20)
print("\nSpread P&L in 20 days: rows = stock move, cols = IV change (vol pts)")
print("        " + "".join(f"{x * 100:>+7.0f}" for x in dIV))
for ds, row in zip(dS, grid):
    print(f"{ds:>+6.0%}  " + "".join(f"{x:>7.0f}" for x in row))

# %% 7. Earnings: what move is priced in, and how big the IV crush will be ---------------------
term_iv, days, normal_vol = 0.36, 40, 0.24   # IV of the expiry holding the print, its trading days,
                                             # and the IV of an expiry BEFORE the print
ev = qd.event_vol(term_iv, days, normal_vol)
print(f"Event-day vol {ev:.0%} -> market expects a {qd.implied_event_move(ev):.1%} move on the print")
print(f"IV the day after the print: {qd.iv_after_event(term_iv, days, ev):.0%} (from {term_iv:.0%})")

# %% 8. Where your P&L comes from: delta-hedging experiment ---------------------------------------
locked = qd.locked_in_pnl(S, K, T, R, IV, 0.80)
h = qd.delta_hedge(S, K, T, R, sigma_implied=IV, sigma_real=0.80, n_steps=60, n_paths=10_000)
print(f"Buy at 68% IV, stock realises 80%, hedge 60 times:")
print(f"  mean hedged P&L ${h.mean * 100:,.0f}/contract (theory ${locked * 100:,.0f}), "
      f"std ${h.std * 100:,.0f}, P(loss) {np.mean(h.pnl < 0):.0%}")

# %% 9. Heston: fit the smile from your broker's option chain ------------------------------------
# Each quote is (years to expiry, strike, implied vol). Copy them from your broker's chain.
quotes = [(0.24, k, v) for k, v in [(150, 0.70), (170, 0.672), (190, 0.649), (210, 0.629), (230, 0.614)]]
quotes += [(0.49, k, v) for k, v in [(140, 0.685), (170, 0.65), (190, 0.63), (220, 0.607), (250, 0.589)]]
fit = qd.calibrate_heston(S, R, quotes)
p = fit.params
print(f"v0 {p.v0:.3f} (spot vol {np.sqrt(p.v0):.0%})  long-run vol {np.sqrt(p.theta):.0%}  "
      f"kappa {p.kappa:.2f}  vol-of-vol {p.xi:.2f}  rho {p.rho:+.2f}")
print(f"Fit error {fit.rmse_iv * 100:.2f} vol pts. rho < 0 means the market fears the downside.")

# %% 10. Is your own track record skill or luck? -----------------------------------------------
# Put your closed trades' returns on premium here (0.25 = +25%, -1.0 = lost it all).
my_trades = [0.42, -0.35, 0.18, -1.0, 0.65, 0.10, -0.22, 0.31, -0.48, 0.90, 0.05, -0.15]
r_ = np.array(my_trades)
print(f"{len(r_)} trades, mean {r_.mean():+.1%}, win rate {np.mean(r_ > 0):.0%}")
print(f"P(true Sharpe > 0) = {qd.probabilistic_sharpe(r_):.0%}")
print(f"Trades needed for 95% confidence at this Sharpe: {qd.min_track_record(r_):.0f}")
print(f"If you picked this strategy out of 5 you tried: {qd.deflated_sharpe(r_, 5):.0%}")

# %% 11. Optional: live data (needs internet and `pip install yfinance`) -------------------------
LIVE = False
if LIVE:
    from quantdesk import data
    px = data.get_prices("NVDA")
    opt = data.get_atm_call("NVDA", target_dte=60)
    print(opt["S"], opt["K"], opt["dte"], opt["iv"])
    print(qd.format_report(qd.full_analysis(px, K=opt["K"], dte=opt["dte"], hold_days=30,
                                            iv=opt["iv"], portfolio=25_000)))
