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


def load_closes(path):
    """Daily closes from a CSV file, oldest first.

    Works with a Yahoo Finance "Historical data" download (Date, Open, High, Low, Close,
    Adj Close, Volume) and with simple two-column files (Date, Close). Uses "Adj Close" when
    present (splits and dividends adjusted), otherwise "Close", otherwise the only numeric column.
    """
    import pandas as pd
    df = pd.read_csv(path)
    cols = {c.strip().lower(): c for c in df.columns}
    for name in ("adj close", "adj_close", "adjclose", "close", "price"):
        if name in cols:
            col = cols[name]
            break
    else:
        numeric = df.select_dtypes("number").columns
        if len(numeric) != 1:
            raise ValueError(f"Could not tell which column is the price in {path}: {list(df.columns)}")
        col = numeric[0]
    if "date" in cols:
        df = df.sort_values(cols["date"], key=lambda d: pd.to_datetime(d, errors="coerce"))
    prices = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(float)
    if len(prices) < 60:
        raise ValueError(f"Only {len(prices)} prices in {path}; download at least a year of daily data.")
    return prices


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
