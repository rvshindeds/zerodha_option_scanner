import os
import inspect
from datetime import datetime
from typing import Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

try:
    from kiteconnect import KiteConnect
except Exception:
    KiteConnect = None

import scanner  # your existing scanner.py

# -----------------------------
# App config
# -----------------------------
st.set_page_config(
    page_title="Zerodha Options Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

APP_VERSION = "v0.1.0"

API_KEY = os.getenv("KITE_API_KEY", "")
API_SECRET = os.getenv("KITE_API_SECRET", "")

# -----------------------------
# Safety checks
# -----------------------------
def require_kite():
    if KiteConnect is None:
        st.error("kiteconnect is not installed in this environment.")
        st.stop()
    if not API_KEY or not API_SECRET:
        st.error("Missing KITE_API_KEY / KITE_API_SECRET in .env.")
        st.stop()

def make_kite(access_token: Optional[str] = None):
    k = KiteConnect(api_key=API_KEY)
    if access_token:
        k.set_access_token(access_token)
    return k

# -----------------------------
# Cached data helpers
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_instruments_cached():
    require_kite()
    k = make_kite()
    try:
        return k.instruments()
    except Exception:
        try:
            return k.instruments("NFO")
        except Exception:
            return []

def extract_expiries(instruments, underlying: str):
    expiries = set()
    u = underlying.upper()

    for ins in instruments:
        try:
            if ins.get("instrument_type") not in ("CE", "PE"):
                continue
            name = (ins.get("name") or "").upper()
            ts = (ins.get("tradingsymbol") or "").upper()
            if u in name or ts.startswith(u):
                exp = ins.get("expiry")
                if exp:
                    expiries.add(exp)
        except Exception:
            continue

    return sorted(list(expiries))

def get_underlying_symbol_for_ltp(underlying: str):
    u = underlying.upper()
    if u == "NIFTY":
        return "NSE:NIFTY 50"
    if u == "BANKNIFTY":
        return "NSE:NIFTY BANK"
    if u == "FINNIFTY":
        return "NSE:NIFTY FIN SERVICE"
    return f"NSE:{underlying}"

def safe_ltp(kite, underlying: str):
    try:
        sym = get_underlying_symbol_for_ltp(underlying)
        data = kite.ltp(sym)
        return data.get(sym, {}).get("last_price")
    except Exception:
        return None

def compute_strike_step_from_instruments(instruments, underlying: str, expiry):
    u = underlying.upper()
    strikes = []

    for ins in instruments:
        try:
            if ins.get("instrument_type") not in ("CE", "PE"):
                continue
            name = (ins.get("name") or "").upper()
            ts = (ins.get("tradingsymbol") or "").upper()
            if u not in name and not ts.startswith(u):
                continue
            if expiry and ins.get("expiry") != expiry:
                continue
            strike = ins.get("strike")
            if strike:
                strikes.append(float(strike))
        except Exception:
            continue

    strikes = sorted(set(strikes))
    if len(strikes) < 3:
        return None

    diffs = []
    for i in range(1, len(strikes)):
        d = strikes[i] - strikes[i - 1]
        if d > 0:
            diffs.append(d)

    if not diffs:
        return None

    diffs = sorted(diffs)
    return diffs[0]

def compute_atm(spot: Optional[float], step: Optional[float]):
    if spot is None or step is None or step == 0:
        return None
    return round(spot / step) * step

def build_action_summary(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    if "OI Read" not in df.columns or "Action" not in df.columns:
        return None

    cols = ["Type", "OI Read", "Action"]
    try:
        summary = (
            df.groupby(cols)
              .size()
              .reset_index(name="Count")
              .sort_values(["Type", "Count"], ascending=[True, False])
        )
        return summary
    except Exception:
        return None

# -----------------------------
# Header
# -----------------------------
st.title("Zerodha Options Scanner")
st.caption(
    f"Version: {APP_VERSION} • Two-layer labels (Momentum + Trade) with confidence • "
    "Access token stored only in memory."
)

# -----------------------------
# Core logic explainer panel
# -----------------------------
with st.expander("How to read these signals (CE/PE core logic)", expanded=False):
    st.markdown("""
### The 2-layer model

Each strike is explained in **two layers**:

1) **Momentum label**  
   What **Price Δ + OI Δ** suggests about option premium momentum.

2) **Trade label**  
   The suggested bias using your intended style:  
   **primarily mean-reversion / premium-selling**  
   with **BUY** signals reserved for rare "panic momentum" cases.

---

## Core idea for **PE**

### Put Long buildup (price ↑ + OI ↑)
**Trade label:** ✅ **SELL PUT (Mean reversion candidate)** — Higher confidence  
**Momentum label:** ✅ **PUT MOMENTUM UP (Long buildup)**

### Put Short covering (price ↑ + OI ↓)
**Trade label:** ✅ **SELL PUT (Mean reversion candidate)** — Medium/Low confidence  
or ✅ **WATCH / WAIT**  
**Momentum label:** ✅ **PUT MOMENTUM UP (Short covering)**

### Put Long unwinding (price ↓ + OI ↓)
**Trade label:** ✅ **Avoid sell (premium shrinking)**  
**Momentum label:** ✅ **PUT MOMENTUM DOWN (Long unwinding)**

### Put Short buildup (price ↓ + OI ↑)
**Trade label:** ✅ **BUY PUT (Panic momentum)** — Low frequency  
**Momentum label:** ✅ **PUT MOMENTUM DOWN (Short buildup)**

---

## Core idea for **CE (mirror model)**

### Call Long buildup (price ↑ + OI ↑)
**Trade label:** ✅ **SELL CALL (Mean reversion candidate)** — Higher confidence  
**Momentum label:** ✅ **CALL MOMENTUM UP (Long buildup)**

### Call Short covering (price ↑ + OI ↓)
**Trade label:** ✅ **SELL CALL (Mean reversion candidate)** — Medium/Low confidence  
or ✅ **WATCH / WAIT**  
**Momentum label:** ✅ **CALL MOMENTUM UP (Short covering)**

### Call Long unwinding (price ↓ + OI ↓)
**Trade label:** ✅ **Avoid sell (premium shrinking)**  
**Momentum label:** ✅ **CALL MOMENTUM DOWN (Long unwinding)**

### Call Short buildup (price ↓ + OI ↑)
**Trade label:** ✅ **BUY CALL (Panic momentum)** — Low frequency  
**Momentum label:** ✅ **CALL MOMENTUM DOWN (Short buildup)**

---

### Note
This scanner prioritizes **clarity + premium-selling logic**.  
**BUY CALL / BUY PUT** should appear rarely by design.
""")

st.divider()

# -----------------------------
# Login block
# -----------------------------
require_kite()

kite_public = make_kite()
login_url = kite_public.login_url()

st.subheader("Login (required daily)")

l1, l2 = st.columns([1.2, 2.0])

with l1:
    st.markdown(f"🔐 **Step 1:** [Login to Zerodha]({login_url})")
    st.caption("After login, Zerodha redirects to your redirect URL with a `request_token`.")

with l2:
    request_token = st.text_input(
        "Paste request_token",
        type="password",
        key="request_token_input"
    )

    gen_clicked = st.button("Generate session", use_container_width=True, key="gen_session_btn")

    st.caption("Security: Access token is stored only in memory for this session. Nothing is written to disk.")

    if gen_clicked:
        if not request_token:
            st.warning("Please paste the request_token first.")
        else:
            try:
                session_data = kite_public.generate_session(request_token, api_secret=API_SECRET)
                access_token = session_data.get("access_token")

                if not access_token:
                    st.error("Access token not returned. Please re-login.")
                else:
                    st.session_state["access_token"] = access_token
                    st.success("Session generated. You can scan now.")
            except Exception as e:
                st.error(f"Session generation failed: {e}")

st.divider()

# -----------------------------
# Require login for scanning UI
# -----------------------------
access_token = st.session_state.get("access_token")
if not access_token:
    st.info("Login first to enable scanning.")
    st.stop()

kite = make_kite(access_token=access_token)

# -----------------------------
# Layout columns
# -----------------------------
left, right = st.columns([0.9, 2.1])

# -----------------------------
# Controls (LEFT)
# -----------------------------
with left:
    st.subheader("Quick Controls")

    underlying = st.selectbox(
        "Underlying",
        options=["NIFTY", "BANKNIFTY", "FINNIFTY"],
        index=0,
        key="underlying_select"
    )

    instruments = fetch_instruments_cached()
    expiries = extract_expiries(instruments, underlying)

    if not expiries:
        st.warning("Could not detect expiries automatically.")
        expiry = None
    else:
        expiry = st.selectbox(
            "Expiry",
            options=expiries,
            format_func=lambda d: d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d),
            index=0,
            key="expiry_select"
        )

    timeframe = st.selectbox(
        "Timeframe",
        options=["5minute", "15minute", "30minute", "60minute"],
        index=1,
        key="timeframe_select"
    )

    rsi_period = st.number_input(
        "RSI period",
        min_value=5, max_value=50, value=14, step=1,
        key="rsi_period"
    )

    rsi_threshold = st.slider(
        "RSI threshold (idea filter)",
        min_value=40, max_value=80, value=60,
        key="rsi_threshold"
    )

    st_period = st.number_input(
        "Supertrend period",
        min_value=5, max_value=30, value=10, step=1,
        key="st_period"
    )

    st_mult = st.number_input(
        "Supertrend multiplier",
        min_value=1.0, max_value=5.0, value=3.0, step=0.5,
        key="st_mult"
    )

    st.markdown("### Strike Range")
    use_atm_filter = st.checkbox("ATM ± X filter", value=True, key="use_atm_filter")
    atm_steps = st.number_input(
        "X strike steps",
        min_value=1, max_value=50, value=10, step=1,
        key="atm_steps"
    )

    st.markdown("### Confidence")
    min_conf = st.slider(
        "Minimum Action Confidence",
        min_value=0, max_value=100, value=50,
        key="min_conf"
    )

    st.markdown("### Debug")
    debug_show_strikes = st.checkbox(
        "Debug: show included strikes",
        value=False,
        key="debug_show_strikes"
    )
    debug_no_match_details = st.checkbox(
        "Debug: show no-match diagnostics",
        value=False,
        key="debug_no_match_details"
    )

# -----------------------------
# Right side header
# -----------------------------
with right:
    st.subheader("Trade Ideas")
    st.caption("Set controls on the left and click **Run Scan**.")
    run = st.button("Run Scan", type="primary", use_container_width=True, key="run_scan_btn")

    # Everything related to results should render inside this container
    results_area = st.container()


# -----------------------------
# Transparency row (spot/step/ATM)
# -----------------------------
spot = safe_ltp(kite, underlying)
step = compute_strike_step_from_instruments(instruments, underlying, expiry)
atm = compute_atm(spot, step)

t1, t2, t3 = st.columns(3)
with t1:
    st.metric("Detected spot", f"{spot:.2f}" if spot else "N/A")
with t2:
    st.metric("Strike step", f"{step:.0f}" if step else "N/A")
with t3:
    st.metric("Computed ATM", f"{atm:.0f}" if atm else "N/A")

# -----------------------------
# Run scan button
# -----------------------------
#run = st.button("Run Scan", type="primary", use_container_width=True, key="run_scan_btn")

results_df = None

# -----------------------------
# Dynamic scan call (prevents keyword errors)
# -----------------------------
if run:
    with st.spinner("Scanning option strikes..."):
        try:
            fn = scanner.scan_options_trade_ideas
            params = inspect.signature(fn).parameters

            kwargs = {}

            # Core
            if "kite" in params:
                kwargs["kite"] = kite
            if "underlying" in params:
                kwargs["underlying"] = underlying

            # Normalize expiry to string for scanner compatibility
            expiry_str = None
            if expiry is not None:
                try:
                    expiry_str = expiry.strftime("%Y-%m-%d")  # if it's a date/datetime
                except Exception:
                    expiry_str = str(expiry)  # fallback

            # Expiry mapping (supports multiple scanner versions)
            if "expiry_date" in params:
                kwargs["expiry_date"] = expiry_str
            elif "expiry" in params:
                kwargs["expiry"] = expiry_str

            # Timeframe mapping
            if "timeframe" in params:
                kwargs["timeframe"] = timeframe
            elif "interval" in params:
                kwargs["interval"] = timeframe
            elif "timeframe_str" in params:
                kwargs["timeframe_str"] = timeframe
            elif "candle_interval" in params:
                kwargs["candle_interval"] = timeframe

            # RSI
            if "rsi_period" in params:
                kwargs["rsi_period"] = int(rsi_period)
            if "rsi_threshold" in params:
                kwargs["rsi_threshold"] = int(rsi_threshold)

            # Supertrend
            if "st_period" in params:
                kwargs["st_period"] = int(st_period)
            if "st_multiplier" in params:
                kwargs["st_multiplier"] = float(st_mult)
            elif "st_mult" in params:
                kwargs["st_mult"] = float(st_mult)

            # ATM filter
            if "use_atm_filter" in params:
                kwargs["use_atm_filter"] = bool(use_atm_filter)
            if "atm_steps" in params:
                kwargs["atm_steps"] = int(atm_steps)

            # Confidence
            if "min_confidence" in params:
                kwargs["min_confidence"] = int(min_conf)
            elif "min_conf" in params:
                kwargs["min_conf"] = int(min_conf)

            # Debug
            if "debug" in params:
                kwargs["debug"] = bool(debug_no_match_details)

            results_df = fn(**kwargs)

        except Exception as e:
            st.error(f"Scan failed: {e}")
            results_df = None

# -----------------------------
# Render results (RIGHT column)
# -----------------------------
# -----------------------------
# Render results (RIGHT column)
# -----------------------------
with results_area:
    if isinstance(results_df, pd.DataFrame):

        # ✅ HARD enforce Minimum Confidence in UI (guaranteed)
        if not results_df.empty and "Action Confidence" in results_df.columns:
            results_df = results_df[
                results_df["Action Confidence"].fillna(0) >= int(min_conf)
            ].copy()

        if results_df.empty:
            st.warning("No matching strikes found for the selected settings.")
            if debug_no_match_details:
                st.info(
                    "Debug tips: Increase ATM ± X range, lower RSI threshold, "
                    "lower minimum confidence, or try a different timeframe."
                )
        else:
            # Top mini counters
            action_counts = {}
            if "Action" in results_df.columns:
                action_counts = results_df["Action"].value_counts().to_dict()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("BUY CALL", action_counts.get("BUY CALL", 0))
            c2.metric("SELL CALL", action_counts.get("SELL CALL", 0))
            c3.metric("BUY PUT", action_counts.get("BUY PUT", 0))
            c4.metric("SELL PUT", action_counts.get("SELL PUT", 0))

            # Action Summary by OI Read
            summary = build_action_summary(results_df)
            with st.expander("Action Summary by OI Read", expanded=False):
                if summary is None or summary.empty:
                    st.caption("Summary appears when both 'OI Read' and 'Action' are available.")
                else:
                    st.dataframe(summary, use_container_width=True, hide_index=True)

            # Debug: show strikes included
            if debug_show_strikes and "Strike" in results_df.columns:
                strikes = sorted(results_df["Strike"].dropna().unique().tolist())
                st.caption(f"Filtered scan strikes ({len(strikes)}):")
                st.write(strikes)

            # Preferred display order
            preferred = [
                "Action", "Action Confidence", "Trade Label", "Momentum Label",
                "Strike", "Type", "RSI", "ST Trend",
                "Price Δ", "OI Δ", "OI Read", "Symbol"
            ]
            cols = [c for c in preferred if c in results_df.columns]
            remaining = [c for c in results_df.columns if c not in cols]
            final_cols = cols + remaining

            st.dataframe(results_df[final_cols], use_container_width=True, hide_index=True)

            # CSV export
            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results (CSV)",
                data=csv,
                file_name=f"{underlying}_options_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_csv_btn"
            )

    else:
        st.caption("Ready. Click **Run Scan**.")
