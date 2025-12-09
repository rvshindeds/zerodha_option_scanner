import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI using EMA smoothing.
    Returns a Series aligned with close index.
    """
    delta = close.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ATR using EMA smoothing.
    df must contain: high, low, close
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return atr


def compute_supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0
) -> pd.DataFrame:
    """
    Supertrend calculation.
    Adds:
      - supertrend
      - st_direction (1 = Up, -1 = Down)

    df must contain: open, high, low, close
    """
    df = df.copy()

    atr = compute_atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2.0

    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)

    final_upper = upperband.copy()
    final_lower = lowerband.copy()

    # Build final bands
    for i in range(1, len(df)):
        if (
            upperband.iloc[i] < final_upper.iloc[i - 1]
            or df["close"].iloc[i - 1] > final_upper.iloc[i - 1]
        ):
            final_upper.iloc[i] = upperband.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        if (
            lowerband.iloc[i] > final_lower.iloc[i - 1]
            or df["close"].iloc[i - 1] < final_lower.iloc[i - 1]
        ):
            final_lower.iloc[i] = lowerband.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

    supertrend = pd.Series(index=df.index, dtype="float64")
    direction = pd.Series(index=df.index, dtype="int64")

    supertrend.iloc[0] = np.nan
    direction.iloc[0] = 1

    # Assign trend
    for i in range(1, len(df)):
        close = df["close"].iloc[i]
        prev_dir = direction.iloc[i - 1]

        if prev_dir == 1:
            if close <= final_upper.iloc[i]:
                direction.iloc[i] = -1
                supertrend.iloc[i] = final_upper.iloc[i]
            else:
                direction.iloc[i] = 1
                supertrend.iloc[i] = final_lower.iloc[i]
        else:
            if close >= final_lower.iloc[i]:
                direction.iloc[i] = 1
                supertrend.iloc[i] = final_lower.iloc[i]
            else:
                direction.iloc[i] = -1
                supertrend.iloc[i] = final_upper.iloc[i]

        # Safety fallback
        if pd.isna(supertrend.iloc[i]):
            supertrend.iloc[i] = (
                final_lower.iloc[i] if direction.iloc[i] == 1 else final_upper.iloc[i]
            )

    df["supertrend"] = supertrend
    df["st_direction"] = direction

    return df
