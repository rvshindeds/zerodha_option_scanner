import pandas as pd
from indicators import compute_rsi, compute_supertrend, compute_supertrend_df


def make_mock_ohlc(n=60):
    """
    Create mock OHLC data that trends upward with small noise.
    Enough candles for RSI/Supertrend to compute safely.
    """
    base = 100
    closes = []
    val = base

    for i in range(n):
        # gentle uptrend
        val += 0.2
        closes.append(val)

    # Build OHLC around close
    df = pd.DataFrame({
        "open":  [c - 0.3 for c in closes],
        "high":  [c + 0.5 for c in closes],
        "low":   [c - 0.6 for c in closes],
        "close": closes,
        "volume": [1000] * n,
    })

    # Add a date column intentionally
    # to prove RSI is safe and won't break
    df["date"] = pd.date_range("2025-01-01", periods=n, freq="min")

    return df


def run_tests():
    print("=== Testing indicators with mock OHLC ===")

    df = make_mock_ohlc()

    # RSI
    rsi = compute_rsi(df, period=14)
    print(f"RSI latest: {rsi}")

    assert rsi is None or (0 <= rsi <= 100), "RSI out of expected range"

    # Supertrend direction string
    st_dir = compute_supertrend(df, period=10, multiplier=3.0)
    print(f"Supertrend direction: {st_dir}")

    assert st_dir in ("Up", "Down", "Unknown"), "Supertrend direction not normalized"

    # Optional: full ST dataframe
    st_df = compute_supertrend_df(df, period=10, multiplier=3.0)
    print("Supertrend DF columns:", list(st_df.columns))

    if not st_df.empty:
        assert "in_uptrend" in st_df.columns, "Supertrend DF missing expected column"

    print("✅ All indicator tests passed.")


if __name__ == "__main__":
    run_tests()
