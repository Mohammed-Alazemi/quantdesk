"""Optional live-data helpers (yfinance). Not exercised by the tests: they need network access.
Everything else in the package takes plain numpy arrays so it also works fully offline."""
from __future__ import annotations
import numpy as np


def _yf():
    try:
        import yfinance as yf
        return yf
    except ImportError as e:
        raise RuntimeError("pip install yfinance to use live data") from e


def get_prices(ticker, period="3y"):
    h = _yf().Ticker(ticker).history(period=period, auto_adjust=True)
    if h.empty:
        raise RuntimeError(f"No price history returned for {ticker} (network blocked or bad ticker).")
    return h["Close"].to_numpy(float)


def get_atm_call(ticker, target_dte=60):
    """Nearest-expiry ATM call to target_dte: returns dict(S, K, dte, iv, bid, ask, strikes, ivs)."""
    import pandas as pd
    tk = _yf().Ticker(ticker)
    S = float(tk.history(period="5d")["Close"].iloc[-1])
    today = pd.Timestamp.today().normalize()
    exps = tk.options
    if not exps:
        raise RuntimeError("No option expiries returned.")
    dtes = [(pd.Timestamp(e) - today).days for e in exps]
    i = int(np.argmin([abs(d - target_dte) for d in dtes]))
    calls = tk.option_chain(exps[i]).calls
    row = calls.iloc[(calls["strike"] - S).abs().argsort().iloc[0]]
    return dict(S=S, K=float(row["strike"]), dte=dtes[i], iv=float(row["impliedVolatility"]),
                bid=float(row["bid"]), ask=float(row["ask"]),
                strikes=calls["strike"].to_numpy(float), ivs=calls["impliedVolatility"].to_numpy(float))


def days_to_next_earnings(ticker):
    """Best effort; yfinance's calendar format changes. Returns None if it cannot be determined."""
    import pandas as pd
    try:
        cal = _yf().Ticker(ticker).calendar
        d = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if d:
            d = d[0] if isinstance(d, (list, tuple)) else d
            return (pd.Timestamp(d).normalize() - pd.Timestamp.today().normalize()).days
    except Exception:
        pass
    return None
