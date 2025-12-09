import pandas as pd
import numpy as np
from indicators import compute_rsi, compute_supertrend

# Create dummy OHLC data
np.random.seed(42)

n = 100
price = np.cumsum(np.random.randn(n)) + 100

df = pd.DataFrame({
    "open": price + np.random.randn(n) * 0.2,
    "high": price + np.random.rand(n) * 0.5,
    "low":  price - np.random.rand(n) * 0.5,
    "close": price,
    "volume": np.random.randint(100, 1000, n),
})

df["rsi"] = compute_rsi(df["close"], 14)
df = compute_supertrend(df, 10, 3)

print(df.tail(5)[["close", "rsi", "supertrend", "st_direction"]])
print("\nLatest RSI:", df["rsi"].iloc[-1])
print("Latest ST direction:", "Down" if df["st_direction"].iloc[-1] == -1 else "Up")
