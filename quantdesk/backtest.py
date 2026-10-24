"""Backtest machinery for the pre-earnings IV-pump strategy (buy ATM call ~6 weeks before the print,
sell the day before), with realistic costs, bootstrap CIs, a random-window null test and a
calibration check of the model's own probability-of-profit forecasts.

IMPORTANT: `synthetic_market` BUILDS the pump in (you choose its size), so on synthetic data a
positive result only proves the machinery works. The real test is running `run_backtest` on
historical IV data (ORATS / CBOE DataShop / your own saved chains) - the code is identical."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .strategies import Leg, CostModel, position_pnl
from .stochastic import gbm_terminal
from .vol import hist_vol
from .risk import bootstrap_ci
from .stats import probabilistic_sharpe


@dataclass
class Market:
    df: pd.DataFrame          # columns: close, iv   (iv = constant-maturity ATM IV, decimal)
    earnings_idx: np.ndarray  # integer positions of earnings days


def synthetic_market(n_days=2520, seed=0, mu=0.08, vol=0.30, iv_prem=1.05, earnings_every=63,
                     earn_sd=0.06, pump=0.25, ramp_days=40, start="2016-01-04") -> Market:
    """GBM prices with earnings gaps; IV = noisy base level x (1 + pump ramp into each print),
    dropping back to base the day of the print (IV crush). pump=0 means NO exploitable edge."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    dt = 1 / 252
    rets = (mu - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * rng.standard_normal(n_days)
    earn = np.arange(60, n_days - 1, earnings_every)
    rets[earn] += rng.normal(0, earn_sd, len(earn))
    close = 100 * np.exp(np.cumsum(rets))
    ou = np.zeros(n_days)
    for t in range(1, n_days):
        ou[t] = 0.97 * ou[t - 1] + 0.012 * rng.standard_normal()
    base = vol * iv_prem * np.exp(ou)
    bump = np.zeros(n_days)
    for e in earn:
        t = np.arange(max(e - ramp_days, 0), e)
        bump[t] = np.maximum(bump[t], pump * (1 - (e - t) / ramp_days))
    return Market(pd.DataFrame({"close": close, "iv": base * (1 + bump)}, index=idx), earn)


def _trade(close, iv, i0, i1, dte_cal, r, cm, strike_pct, seed, n_calib):
    S0 = close[i0]
    K = round(S0 * strike_pct, 2)
    elapsed = (i1 - i0) * 365 / 252
    T0, T1 = dte_cal / 365, (dte_cal - elapsed) / 365
    legs = [Leg("call", K, 1)]
    pnl, entry, risk = position_pnl(legs, S0, T0, T1, r, np.array([close[i1]]), iv[i0],
                                    np.array([iv[i1]]), 0.0, cm)
    out = dict(S0=S0, K=K, iv0=iv[i0], iv1=iv[i1], entry=entry, pnl=float(pnl[0]),
               R=float(pnl[0] / risk))
    if n_calib:
        hv = hist_vol(close[max(i0 - 30, 0):i0 + 1])
        ST = gbm_terminal(S0, 0.0, hv, (i1 - i0) / 252, n=n_calib, seed=seed)
        sim, _, _ = position_pnl(legs, S0, T0, T1, r, ST, iv[i0], np.full_like(ST, iv[i0]), 0.0, cm)
        out["pop_pred"] = float((sim > 0).mean())
    return out


def run_backtest(mkt: Market, entry_days_before=30, dte_buffer_cal=45, r=0.045, strike_pct=1.0,
                 cm: CostModel | None = None, n_calib=2000, seed=0) -> pd.DataFrame:
    cm = cm or CostModel()
    close, iv = mkt.df["close"].to_numpy(), mkt.df["iv"].to_numpy()
    dte = entry_days_before * 365 / 252 + dte_buffer_cal
    rows = []
    for k, e in enumerate(mkt.earnings_idx):
        i0, i1 = e - entry_days_before, e - 1
        if i0 < 31:
            continue
        t = _trade(close, iv, i0, i1, dte, r, cm, strike_pct, seed + k, n_calib)
        t["date"] = mkt.df.index[i0]
        t["win"] = t["pnl"] > 0
        rows.append(t)
    return pd.DataFrame(rows)


def run_universe(n_tickers=20, seed=0, **kw) -> pd.DataFrame:
    """Same strategy across many independent synthetic tickers (bigger sample, honest stats)."""
    mk_kw = {k: kw.pop(k) for k in list(kw) if k in
             ("n_days", "mu", "vol", "iv_prem", "earnings_every", "earn_sd", "pump", "ramp_days")}
    frames = []
    for i in range(n_tickers):
        m = synthetic_market(seed=seed + i, **mk_kw)
        t = run_backtest(m, seed=seed + 1000 * i, **kw)
        t["ticker"] = i
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def _equity_stats(R, frac):
    eq = np.cumprod(1 + frac * R)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    return float(eq[-1]), float(((eq - peak) / peak).min())


def summarize(trades: pd.DataFrame, frac=0.10):
    """Trade statistics. Equity/drawdown are computed PER TICKER (one account trading one name)
    and averaged, because chaining different tickers into one sequence would overstate trade count."""
    R = np.clip(trades["R"].to_numpy(), -1.0, None)
    groups = ([np.clip(g["R"].to_numpy(), -1.0, None) for _, g in trades.groupby("ticker")]
              if "ticker" in trades else [R])
    eqs = [_equity_stats(g, frac) for g in groups]
    lo, hi = bootstrap_ci(R)
    out = dict(n=len(R), win_rate=float((R > 0).mean()), mean_R=float(R.mean()),
               median_R=float(np.median(R)), std_R=float(R.std(ddof=1)),
               sharpe_trade=float(R.mean() / R.std(ddof=1)),
               t_stat=float(R.mean() / (R.std(ddof=1) / np.sqrt(len(R)))),
               mean_R_ci95=(lo, hi), final_equity=float(np.mean([e[0] for e in eqs])),
               max_drawdown=float(np.mean([e[1] for e in eqs])),
               psr=probabilistic_sharpe(R))
    if "pop_pred" in trades:
        out["brier"] = float(np.mean((trades["pop_pred"] - trades["win"].astype(float)) ** 2))
    return out


def calibration_table(pred, outcome, bins=5):
    """Does 'a 40% chance of profit' actually win ~40% of the time?"""
    df = pd.DataFrame(dict(p=np.asarray(pred, float), y=np.asarray(outcome, float)))
    df["bin"] = pd.qcut(df["p"], bins, duplicates="drop")
    t = df.groupby("bin", observed=True).agg(mean_pred=("p", "mean"), observed=("y", "mean"),
                                             n=("y", "size")).reset_index(drop=True)
    return t


def null_test(mkt: Market, n_events: int, entry_days_before=30, dte_buffer_cal=45, r=0.045,
              cm: CostModel | None = None, n_windows=400, n_resamples=2000, seed=0):
    """Distribution of the mean return of the same trade placed on RANDOM windows (any date).
    Compare with the earnings-timed mean to see whether timing adds anything."""
    cm = cm or CostModel()
    close, iv = mkt.df["close"].to_numpy(), mkt.df["iv"].to_numpy()
    rng = np.random.default_rng(seed)
    dte = entry_days_before * 365 / 252 + dte_buffer_cal
    starts = rng.integers(31, len(close) - entry_days_before - 2, n_windows)
    R = np.array([_trade(close, iv, i0, i0 + entry_days_before - 1, dte, r, cm, 1.0, 0, 0)["R"]
                  for i0 in starts])
    means = np.array([rng.choice(R, n_events, replace=True).mean() for _ in range(n_resamples)])
    return R, means
