import base64, hashlib, os, sys, tempfile
import numpy as np

TICKER                      = "CRDO"
STOCK_PRICE                 = 190.00
STRIKE                      = 200.00
DAYS_TO_EXPIRY              = 87
IMPLIED_VOL                 = 0.68
OPTION_MID                  = 21.73
RATE                        = 0.04

HOLD_DAYS                   = 41
DAYS_TO_EARNINGS            = 42
PORTFOLIO                   = 25_000
OPEN_POSITIONS              = 0
MY_DRIFT                    = 0.15
IV_CHANGE_BY_EXIT           = 0.05

PRICE_HISTORY_CSV           = None

SPREAD_SHORT_STRIKE         = 230.00

IV_EXPIRY_AFTER_EARNINGS    = 0.36
TRADING_DAYS_TO_THAT_EXPIRY = 40
IV_EXPIRY_BEFORE_EARNINGS   = 0.24

MY_TRADES = [0.42, -0.35, 0.18, -1.0, 0.65, 0.10, -0.22, 0.31, -0.48, 0.90, 0.05, -0.15]

_BUNDLE = "__BUNDLE__"

def _load_quantdesk():
    raw = base64.b64decode(_BUNDLE)
    path = os.path.join(tempfile.gettempdir(),
                        f"quantdesk_{hashlib.sha1(raw).hexdigest()[:10]}.zip")
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(raw)
    for name in [m for m in sys.modules if m == "quantdesk" or m.startswith("quantdesk.")]:
        del sys.modules[name]
    sys.path.insert(0, path)
    import quantdesk
    return quantdesk

qd = _load_quantdesk()
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
T = DAYS_TO_EXPIRY / 365
W = 70

def section(title):
    print("\n" + "=" * W + f"\n  {title}\n" + "=" * W)

def row(label, value):
    print(f"  {label:<34}{value}")

print("=" * W)
print(f"  QUANTDESK {qd.__version__}   |   {TICKER} ${STRIKE:g} call, "
      f"{DAYS_TO_EXPIRY} days, IV {IMPLIED_VOL:.0%}")
print("=" * W)

section("1  PRICE AND GREEKS   what the option is worth and how it moves")
call = qd.bs_price(STOCK_PRICE, STRIKE, T, RATE, IMPLIED_VOL, kind="call")
put = qd.bs_price(STOCK_PRICE, STRIKE, T, RATE, IMPLIED_VOL, kind="put")
g = qd.greeks(STOCK_PRICE, STRIKE, T, RATE, IMPLIED_VOL, kind="call")
row("Call fair value", f"${call:.2f}   (${call * 100:,.0f} per contract)")
row("Put fair value", f"${put:.2f}")
row("Delta", f"{g['delta']:.3f}   (acts like ~{g['delta'] * 100:.0f} shares)")
row("Gamma", f"{g['gamma']:.4f}")
row("Theta (time decay)", f"-${abs(g['theta']) * 100:.2f} per day per contract")
row("Vega", f"${g['vega'] * 100:.2f} per 1 vol point")
row("Leverage", f"{g['delta'] * STOCK_PRICE / call:.1f}x the stock's % move")

section("2  IMPLIED VOL   what volatility the market price implies")
iv_mkt = qd.implied_vol(OPTION_MID, STOCK_PRICE, STRIKE, T, RATE, kind="call")
if np.isnan(iv_mkt):
    row("Implied vol", "n/a - price is outside no-arbitrage bounds, check the inputs")
else:
    row(f"Mid ${OPTION_MID:.2f} implies", f"{iv_mkt:.1%} volatility")
    row("Versus the IV you typed", f"{IMPLIED_VOL:.1%}")

section("3  PRICE HISTORY   realised vol and GARCH forecast")
if PRICE_HISTORY_CSV:
    from quantdesk.data import load_closes
    prices = load_closes(os.path.join(HERE, PRICE_HISTORY_CSV))
    row("Loaded", f"{len(prices)} daily closes from {PRICE_HISTORY_CSV}, last ${prices[-1]:.2f}")
else:
    from quantdesk.cli import _garch_prices
    prices = _garch_prices(n=600, omega=4e-5, alpha=0.08, beta=0.90)
    prices = prices / prices[-1] * STOCK_PRICE
    row("Using", "demo prices (set PRICE_HISTORY_CSV to use real data)")

hv30 = qd.hist_vol(prices, 30)
garch = qd.garch_forecast_vol(np.diff(np.log(prices)), HOLD_DAYS)
row("Realised vol, last 30 days", f"{hv30:.1%}")
row(f"GARCH forecast, next {HOLD_DAYS} days", f"{garch:.1%}")
row("Implied vol", f"{IMPLIED_VOL:.1%}   (IV / HV30 = {IMPLIED_VOL / hv30:.2f})")

section("4  THE VERDICT   full trade check, answer first")
report = qd.full_analysis(
    prices, K=STRIKE, S=STOCK_PRICE, dte=DAYS_TO_EXPIRY, hold_days=HOLD_DAYS, iv=IMPLIED_VOL,
    r=RATE, days_to_earnings=DAYS_TO_EARNINGS, portfolio=PORTFOLIO,
    open_positions=OPEN_POSITIONS, drift=MY_DRIFT, iv_exit_shift=IV_CHANGE_BY_EXIT,
    pre_earnings_strategy=DAYS_TO_EARNINGS is not None,
)
print(qd.format_report(report))

section(f"5  SPREAD   ${STRIKE:g}/${SPREAD_SHORT_STRIKE:g} call spread vs the naked call")
from quantdesk.strategies import value

naked = [qd.Leg("call", STRIKE, 1)]
spread = qd.bull_call_spread(STRIKE, SPREAD_SHORT_STRIKE, skew=-0.02)
print(f"  {'':<18}{'cost':>9}{'delta':>9}{'vega/pt':>10}{'theta/day':>11}")
for name, legs in ((f"naked {STRIKE:g}C", naked),
                   (f"{STRIKE:g}/{SPREAD_SHORT_STRIKE:g} spread", spread)):
    gk = qd.net_greeks(legs, STOCK_PRICE, T, RATE, IMPLIED_VOL)
    cost = value(legs, STOCK_PRICE, T, RATE, IMPLIED_VOL)
    print(f"  {name:<18}{cost:>9,.0f}{gk['delta']:>9.1f}{gk['vega']:>10.1f}{gk['theta']:>11.1f}")

dS, dIV, grid = qd.stress_grid(spread, STOCK_PRICE, T, RATE, IMPLIED_VOL, dt_days=20)
print("\n  Spread P&L ($) in 20 days   rows: stock move   columns: IV change (vol pts)")
print("  " + " " * 7 + "".join(f"{x * 100:>+8.0f}" for x in dIV))
for ds, line in zip(dS, grid):
    print(f"  {ds:>+6.0%} " + "".join(f"{x:>8.0f}" for x in line))

section("6  EARNINGS   the move the market expects and the IV crush after")
ev = qd.event_vol(IV_EXPIRY_AFTER_EARNINGS, TRADING_DAYS_TO_THAT_EXPIRY, IV_EXPIRY_BEFORE_EARNINGS)
row("Earnings-day volatility", f"{ev:.0%}")
row("Expected move on the report", f"+/- {qd.implied_event_move(ev):.1%}")
row("IV the day after the report",
    f"{qd.iv_after_event(IV_EXPIRY_AFTER_EARNINGS, TRADING_DAYS_TO_THAT_EXPIRY, ev):.0%}"
    f"   (from {IV_EXPIRY_AFTER_EARNINGS:.0%})")

section("7  HEDGING   buy at implied vol, stock realises a different vol")
REALISED = IMPLIED_VOL + 0.12
locked = qd.locked_in_pnl(STOCK_PRICE, STRIKE, T, RATE, IMPLIED_VOL, REALISED)
h = qd.delta_hedge(STOCK_PRICE, STRIKE, T, RATE, sigma_implied=IMPLIED_VOL,
                   sigma_real=REALISED, n_steps=60, n_paths=10_000)
row("Bought at / stock realises", f"{IMPLIED_VOL:.0%} / {REALISED:.0%}, hedged 60 times")
row("Average hedged P&L", f"${h.mean * 100:,.0f} per contract   (theory ${locked * 100:,.0f})")
row("Spread of outcomes", f"${h.std * 100:,.0f}   P(loss) {np.mean(h.pnl < 0):.0%}")

section("8  HESTON   fit a stochastic-vol model to the smile")

QUOTES = [(0.24, k, v) for k, v in [(150, .700), (170, .672), (190, .649), (210, .629), (230, .614)]]
QUOTES += [(0.49, k, v) for k, v in [(140, .685), (170, .650), (190, .630), (220, .607), (250, .589)]]
fit = qd.calibrate_heston(STOCK_PRICE, RATE, QUOTES)
p = fit.params
row("Spot vol / long-run vol", f"{np.sqrt(p.v0):.0%} / {np.sqrt(p.theta):.0%}")
row("Mean reversion (kappa)", f"{p.kappa:.2f}")
row("Vol of vol (xi)", f"{p.xi:.2f}")
row("Spot-vol correlation (rho)", f"{p.rho:+.2f}" + ("   market fears the downside" if p.rho < 0 else ""))
row("Fit error", f"{fit.rmse_iv * 100:.2f} vol points")

section("9  TRACK RECORD   is it skill or luck?")
r_ = np.array(MY_TRADES, float)
row("Trades / mean return / win rate", f"{len(r_)} / {r_.mean():+.1%} / {np.mean(r_ > 0):.0%}")
row("P(true Sharpe > 0)", f"{qd.probabilistic_sharpe(r_):.0%}")
row("Trades needed for 95% confidence", f"{qd.min_track_record(r_):.0f}")
row("If best of 5 strategies tried", f"{qd.deflated_sharpe(r_, 5):.0%}   (deflated Sharpe)")
print("\n" + "=" * W + "\n  Done. Change the inputs at the top and press F5 again.\n" + "=" * W)
