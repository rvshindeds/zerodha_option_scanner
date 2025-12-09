import os
from dotenv import load_dotenv
from kiteconnect import KiteConnect

from scanner import scan_options

load_dotenv()

API_KEY = os.getenv("KITE_API_KEY")
API_SECRET = os.getenv("KITE_API_SECRET")

# ---- IMPORTANT ----
# For this test, we temporarily paste access token manually.
# We'll remove this in Step 4 when we build secure Streamlit login flow.

ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN", "")

if not ACCESS_TOKEN:
    raise RuntimeError(
        "Set KITE_ACCESS_TOKEN in your environment for this test only.\n"
        "Example (PowerShell):\n"
        "$env:KITE_ACCESS_TOKEN='your_daily_access_token'"
    )

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# Update values:
UNDERLYING = "NIFTY"
EXPIRY = "2025-04-24"  # change to your actual nearest expiry

result = scan_options(
    kite=kite,
    underlying=UNDERLYING,
    expiry_date=EXPIRY,
    interval="5minute",
    lookback_days=5,
    rsi_period=14,
    rsi_threshold=60,
    st_period=10,
    st_mult=3
)

print(result.to_string(index=False) if not result.empty else "No matches.")
