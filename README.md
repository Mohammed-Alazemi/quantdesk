# quantdesk

[![tests](https://github.com/Mohammed-Alazemi/quantdesk/actions/workflows/tests.yml/badge.svg)](https://github.com/Mohammed-Alazemi/quantdesk/actions/workflows/tests.yml)

An options research desk written from first principles in NumPy, SciPy and pandas. It covers:

- pricing and Greeks;
- implied-volatility surfaces and Heston calibration;
- delta-hedging P&L;
- risk-neutral densities, variance swaps and earnings event vol;
- position sizing and backtesting with honest statistics;
- inventory-aware market making.

**Every module is tested against an exact or published result: 43 tests, about 11 seconds.**

```bash
pip install -r requirements.txt
python -m pytest -q                 # 43 tests
python -m quantdesk research        # the experiments below, in about 2 seconds
python -m quantdesk demo            # verdict-first trade report + universe backtest
python -m quantdesk analyze --csv closes.csv --strike 100 --iv 0.32 --dte 60 --hold 30 --earnings 75
```

**Using an editor such as Spyder or VS Code instead of a terminal?**

1. Unzip or clone the project first. Python cannot import the package from inside a `.zip` file that your editor opened.
2. Install the requirements once: `pip install -r requirements.txt`. In Spyder's console, use `%pip install pytest` (numpy, scipy and pandas are usually already installed).
3. Open and run `quickstart.py` to see every experiment, or `run_tests.py` to run all 43 tests. Each test file also runs on its own.

## What it shows

**1. Heston calibration.** The model is calibrated to a noisy implied-volatility surface: 45 quotes across 5 expiries, with 0.2 vol points of noise.

| | v0 | κ | θ | ξ | ρ |
|---|---|---|---|---|---|
| true | 0.045 | 2.50 | 0.060 | 0.70 | −0.70 |
| fitted | 0.045 | 2.55 | 0.060 | 0.69 | −0.71 |

The fit error is 0.16 vol points, which is below the noise level, as it should be. With noise-free quotes, all five parameters are recovered to 10⁻⁴. The pricer is checked three ways: against adaptive quadrature (to 10⁻⁸), against Monte Carlo (within the standard error), and against Black–Scholes in the ξ → 0 limit.

**2. Where an option trader's P&L comes from.** The experiment buys an ATM call at 20% implied volatility while the stock realises 30%.

| rebalances | hedged at realised vol: mean / std | hedged at implied vol: mean / std |
|---|---|---|
| 12 | 2.83 / 2.08 | 2.82 / 2.45 |
| 50 | 2.81 / 1.05 | 2.81 / 1.65 |
| 200 | 2.82 / 0.52 | 2.82 / 1.38 |

- **Hedging at the realised vol locks in exactly V(σ_r) − V(σ_i) = $2.82.** The noise falls like 1/√n.
- **Hedging at the implied vol earns the same on average, but stays path-dependent** however often you rebalance.
- **With no vol edge, the P&L is pure hedging error.** Its spread matches the Derman–Kamal formula √(π/4)·vega·σ/√n to within a few percent.
- **The hedged P&L does not depend on the stock's drift,** which is the point of hedging.

**3. What the smile prices in.**
- **The risk-neutral density** (Breeden–Litzenberger) integrates to 1 and has the forward as its mean. A skewed SVI smile turns into a left-skewed, fat-tailed density.
- **The variance-swap strike** (Carr–Madan strip) is 28.3%, against 22.4% ATM IV. The difference is the price of the skew.
- **Earnings event vol** is backed out of the term structure. With a 40-day IV of 36% and 24% on a normal day, the earnings day carries 171% vol, which implies an expected move of 8.6%. The IV crush to 24% is predictable before the print.

**4. Is the backtest lying?**
- **Picking the best of 100 random strategies** gives an annualised Sharpe of 2.6 and a naive P(skill) of 99.5%. The deflated Sharpe ratio (Bailey & López de Prado) puts it at 47%. In the test suite, the naive test is fooled in more than 90 of 100 trials; the deflated test is fooled in at most 5.
- **Every backtest summary now reports the probabilistic Sharpe ratio.** Walk-forward splits with an embargo keep test windows strictly out of sample.

**5. Market making (Avellaneda–Stoikov 2008).** This reproduces the paper's comparison:

| strategy | spread | P&L | final inventory sd |
|---|---|---|---|
| inventory-aware | 1.49 | 64.9 ± 6.6 | 3.0 |
| symmetric | 1.49 | 68.5 ± 13.2 | 8.4 |

Skewing quotes against inventory gives up about 5% of mean profit and halves the risk.

## Modules

| Module | Contents |
|---|---|
| `pricing.py` | Vectorised Black–Scholes–Merton with dividend yield. Delta, gamma, vega, theta, rho, vanna and volga, all checked against finite differences. Brent IV solver with no-arbitrage bounds, vectorised over a chain. American CRR tree. |
| `heston.py` | Heston characteristic function in the "little trap" form, priced by Lewis's single integral on fixed Gauss–Legendre nodes (0.3 ms for 30 strikes). Implied smile. Vega-weighted least-squares calibration from several starting points. |
| `hedging.py` | Discrete delta-hedging simulator with transaction costs, the locked-in P&L, the Derman–Kamal hedging-error formula, and the gamma–theta breakeven move. |
| `density.py` | Breeden–Litzenberger density from prices or from an SVI smile, density moments, variance-swap strike, earnings event vol, implied move, and the predicted IV crush. |
| `vol.py` | Historical vol, EWMA, GARCH(1,1) by maximum likelihood with variance targeting, IV rank, variance risk premium, SVI fit, and Gatheral's butterfly no-arbitrage check. |
| `stochastic.py` | GBM and Heston Monte Carlo (full-truncation Euler, antithetic variates), with a path-dependent exit IV. |
| `strategies.py` | Multi-leg positions, a bid–ask and commission model, expiry payoffs, breakevens, net Greeks, and net-of-cost scenario P&L. |
| `risk.py` | Kelly sizing (grid search refined by a bounded 1-D search), bootstrap Kelly, Kelly versus assumed drift, VaR/CVaR, bootstrap confidence intervals, spot × IV stress grid. |
| `stats.py` | Probabilistic and deflated Sharpe ratios, minimum track record length, walk-forward splits with an embargo. |
| `marketmaking.py` | Avellaneda–Stoikov optimal quotes and a fill simulator. |
| `checklist.py` | A rulebook written as an editable `Rules` object. Hard gates veto a trade; soft gates warn. A size is suggested only when every hard gate passes. |
| `backtest.py` | Pre-earnings IV-pump backtest with costs, bootstrap confidence interval, t-statistic, probabilistic Sharpe, calibration table with Brier score, and a random-window null test. |
| `analysis.py`, `cli.py` | One call runs everything and prints a verdict-first report. |

## Findings, stated honestly

- **The machinery works; the edge is not proven.** The synthetic market builds the pre-earnings IV pump in. With a 35% pump and realistic costs, the strategy breaks even: −0.2% mean return on premium, 95% CI −5.8% to +5.2%, 780 trades, P(Sharpe > 0) = 47%. With no pump it loses 9.2% per trade. Costs and theta are the default outcome.
- **Expiry choice matters more than entry timing.** On synthetic data the Sharpe rises steadily from −0.09 (expiry 14 days after the print) to +0.11 (120 days after). But that comes from searching over six expiries, which is exactly what `deflated_sharpe` is for, and the pump was put there by construction.
- **Kelly is fragile.** In the demo, the suggested size is 0% at every assumed drift up to 20% and jumps to 33% at 40% drift. An edge that exists only under an optimistic drift is not an edge.
- **The real test needs real data.** The next step is historical option IV (ORATS, CBOE DataShop, or saved daily chains) run through the same `run_backtest`, with walk-forward splits.

## Limits

- The Monte Carlo exit IV is an ATM proxy (expected average variance plus a level anchor), not a full surface re-price.
- `position_pnl` handles debit positions only.
- The backtest uses one constant-maturity IV series.
- `data.py` (yfinance) needs network access and is not covered by the tests.

**This is research code for learning, not investment advice.**

## References

- Black & Scholes (1973); Merton (1973)
- Heston (1993); Albrecher, Mayer, Schoutens & Tistaert (2007), "The little Heston trap"; Lewis (2001)
- Gatheral, *The Volatility Surface* (2006)
- Breeden & Litzenberger (1978); Carr & Madan (1998)
- Kamal & Derman (1999), "When you cannot hedge continuously"; Wilmott, *Paul Wilmott Introduces Quantitative Finance*, ch. 10
- Bailey & López de Prado (2012, 2014), on the probabilistic and deflated Sharpe ratios
- Avellaneda & Stoikov (2008), "High-frequency trading in a limit order book"

## Author

Mohammed Alazemi, BSc Physics, The University of Manchester. Released under the MIT licence.
