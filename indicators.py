import pandas as pd
import numpy as np


# =========================================================
# RSI
# =========================================================
def compute_rsi(df: pd.DataFrame, period: int = 14):
    """
    Compute RSI using ONLY the close series.
    Returns the latest RSI value as float, or None if unavailable.

    This avoids errors caused by including datetime columns in diff().
    """
    if df is None or df.empty:
        return None

    if "close" not in df.columns:
        return None

    close = pd.to_numeric(df["close"], errors="coerce").dropna()

    if len(close) < period + 2:
        return None

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder-style smoothing (EMA with alpha=1/period)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    # Avoid divide-by-zero
    avg_loss = avg_loss.replace(0, np.nan)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    latest = rsi.iloc[-1]
    return float(latest) if pd.notna(latest) else None


# =========================================================
# ATR (used by Supertrend)
# =========================================================
def _compute_atr(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """
    Compute ATR from OHLC.
    """
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr


# =========================================================
# Supertrend (string output)
# =========================================================
def compute_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    """
    Compute Supertrend direction and return ONLY:
    'Up' or 'Down'

    This function is intentionally designed to return a clean string
    so scanner logic never receives a DataFrame/Series unexpectedly.
    """
    st_df = compute_supertrend_df(df, period=period, multiplier=multiplier)

    if st_df is None or st_df.empty or "in_uptrend" not in st_df.columns:
        return "Unknown"

    last_val = st_df["in_uptrend"].dropna()
    if last_val.empty:
        return "Unknown"

    return "Up" if bool(last_val.iloc[-1]) else "Down"


def compute_supertrend_df(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Full Supertrend calculation returning a DataFrame.

    Columns:
    - upperband
    - lowerband
    - in_uptrend

    You can use this for debugging if you ever want.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    required = {"high", "low", "close"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    data = df.copy()

    # Ensure numeric
    data["high"] = pd.to_numeric(data["high"], errors="coerce")
    data["low"] = pd.to_numeric(data["low"], errors="coerce")
    data["close"] = pd.to_numeric(data["close"], errors="coerce")

    data = data.dropna(subset=["high", "low", "close"])
    if len(data) < period + 2:
        return pd.DataFrame()

    hl2 = (data["high"] + data["low"]) / 2.0
    atr = _compute_atr(data, period=period)

    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)

    in_uptrend = [True] * len(data)

    # Convert to numpy-friendly indexing
    close = data["close"].values
    ub = upperband.values
    lb = lowerband.values

    for i in range(1, len(data)):
        # Trend switch rules
        if close[i] > ub[i - 1]:
            in_uptrend[i] = True
        elif close[i] < lb[i - 1]:
            in_uptrend[i] = False
        else:
            in_uptrend[i] = in_uptrend[i - 1]

            # Band adjustment rules
            if in_uptrend[i] and lb[i] < lb[i - 1]:
                lb[i] = lb[i - 1]

            if not in_uptrend[i] and ub[i] > ub[i - 1]:
                ub[i] = ub[i - 1]

    out = pd.DataFrame(index=data.index)
    out["upperband"] = ub
    out["lowerband"] = lb
    out["in_uptrend"] = in_uptrend

    return out.reset_index(drop=True)
