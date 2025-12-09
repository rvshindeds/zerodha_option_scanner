import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from kiteconnect.connect import KiteConnect

from scanner import (
    scan_options_trade_ideas,
    scan_options_with_indicators,
    fetch_instruments_df,
    get_expiries_for_underlying,
    get_atm_context,
)

# =========================================================
# Page
# =========================================================
st.set_page_config(page_title="Options Signal Scanner", layout="wide")

# =========================================================
# Env
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("KITE_API_KEY", "")
API_SECRET = os.getenv("KITE_API_SECRET", "")

if not API_KEY or not API_SECRET:
    raise RuntimeError("Missing KITE_API_KEY or KITE_API_SECRET in .env")


# =========================================================
# Kite
# =========================================================
kite = KiteConnect(api_key=API_KEY)
login_url = kite.login_url()

# =========================================================
# Defaults (production)
# =========================================================
DEFAULTS = {
    "interval": "15minute",
    "lookback_days": 7,
    "rsi_period": 14,
    "rsi_threshold": 55.0,
    "st_period": 10,
    "st_mult": 3.0,
    "use_atm_filter": True,
    "atm_steps": 10,
    "min_conf": 40,
    "show_all_rows": False,
    "diagnostic_fallback": True,
    "show_atm_details": False,
}

# =========================================================
# Instruments cache
# =========================================================
def get_instruments_cached(force_refresh: bool = False) -> pd.DataFrame:
    """
    Cache NFO instruments in session_state for speed/stability.
    Refresh every 60 minutes unless forced.
    """
    now = time.time()
    cache = st.session_state.get("instruments_cache")

    if (not force_refresh) and cache and (now - cache["ts"] < 3600):
        return cache["df"]

    df = fetch_instruments_df(kite)
    st.session_state["instruments_cache"] = {"df": df, "ts": now}
    return df


# =========================================================
# Header
# =========================================================
h1, h2 = st.columns([3, 1])

with h1:
    st.title("Options Signal Scanner")
    st.caption("Trade Ideas powered by RSI + Supertrend + option OI/Price behavior.")

with h2:
    session_active = bool(st.session_state.get("access_token"))
    st.markdown("✅ **Session Active**" if session_active else "⚠️ **Not Logged In**")

    refresh_clicked = st.button("Refresh instruments", use_container_width=True)

st.divider()


# =========================================================
# Login (compact + safe)
# =========================================================
with st.expander("Login (required daily)", expanded=not session_active):
    st.markdown(f"[Login to Zerodha]({login_url})")

    # ✅ Safe clear mechanism (pre-widget)
    if st.session_state.get("clear_request_token"):
        st.session_state["request_token_input"] = ""
        st.session_state["clear_request_token"] = False

    st.text_input(
        "Paste request_token",
        type="password",
        key="request_token_input",
        placeholder="request_token from Zerodha redirect URL",
    )

    c1, c2 = st.columns([1, 2])

    with c1:
        if st.button("Generate session", use_container_width=True):
            request_token = (st.session_state.get("request_token_input") or "").strip()

            if not request_token:
                st.warning("Please paste the request_token.")
            else:
                try:
                    data = kite.generate_session(request_token, api_secret=API_SECRET)
                    access_token = data.get("access_token")

                    if not access_token:
                        st.error("No access_token received.")
                    else:
                        st.session_state["access_token"] = access_token

                        # ✅ Set flag instead of mutating widget state directly
                        st.session_state["clear_request_token"] = True

                        st.success("Session generated for this app session.")
                        st.rerun()

                except Exception as e:
                    st.error(f"Session generation failed: {e}")

    with c2:
        st.caption(
            "Security: Access token is stored only in memory for this session. "
            "Nothing is written to disk."
        )


# Must have access token to scan
if not st.session_state.get("access_token"):
    st.warning("Login first to enable scanning.")
    st.stop()

kite.set_access_token(st.session_state["access_token"])


# Refresh instruments if user clicked button
if refresh_clicked:
    try:
        _ = get_instruments_cached(force_refresh=True)
        st.success("Instruments refreshed.")
    except Exception as e:
        st.error(f"Instrument refresh failed: {e}")


# =========================================================
# Main Layout
# =========================================================
left, right = st.columns([1, 2.2], gap="large")

# =========================
# LEFT: Quick Controls
# =========================
with left:
    st.subheader("Quick Controls")

    underlying = st.selectbox("Underlying", ["NIFTY", "BANKNIFTY", "FINNIFTY", "SENSEX"])

    instruments_df = get_instruments_cached()
    expiries = get_expiries_for_underlying(instruments_df, underlying)

    if expiries:
        expiry_date = st.selectbox("Expiry", expiries, index=0)
    else:
        expiry_date = st.text_input("Expiry (YYYY-MM-DD)", placeholder="No expiries found")

    interval = st.selectbox(
        "Timeframe",
        ["5minute", "15minute"],
        index=1 if DEFAULTS["interval"] == "15minute" else 0,
    )

    st.markdown("### Strike Range")
    use_atm_filter = st.checkbox("ATM ± X filter", value=DEFAULTS["use_atm_filter"])
    atm_steps = st.number_input("X strike steps", min_value=1, max_value=50, value=DEFAULTS["atm_steps"])

    st.markdown("### Signal Filter")
    min_conf = st.slider("Min confidence", 0, 100, DEFAULTS["min_conf"], 5)

    show_all_rows = st.checkbox("Show all scanned rows", value=DEFAULTS["show_all_rows"])

    run_scan = st.button("Run Scan", use_container_width=True)

    with st.expander("Advanced", expanded=False):
        lookback_days = st.number_input(
            "Lookback days", min_value=2, max_value=15, value=DEFAULTS["lookback_days"]
        )
        rsi_period = st.number_input(
            "RSI period", min_value=5, max_value=50, value=DEFAULTS["rsi_period"]
        )
        rsi_threshold = st.number_input(
            "RSI threshold", min_value=50.0, max_value=90.0, value=DEFAULTS["rsi_threshold"]
        )
        st_period = st.number_input(
            "Supertrend period", min_value=5, max_value=50, value=DEFAULTS["st_period"]
        )
        st_mult = st.number_input(
            "Supertrend multiplier", min_value=1.0, max_value=6.0,
            value=DEFAULTS["st_mult"], step=0.5
        )

        diagnostic_fallback = st.checkbox(
            "If no trade ideas, show diagnostic table",
            value=DEFAULTS["diagnostic_fallback"]
        )

        show_atm_details = st.checkbox(
            "Show ATM detection details",
            value=DEFAULTS["show_atm_details"]
        )

# If Advanced was never opened, ensure variables exist with defaults
if "lookback_days" not in locals():
    lookback_days = DEFAULTS["lookback_days"]
if "rsi_period" not in locals():
    rsi_period = DEFAULTS["rsi_period"]
if "rsi_threshold" not in locals():
    rsi_threshold = DEFAULTS["rsi_threshold"]
if "st_period" not in locals():
    st_period = DEFAULTS["st_period"]
if "st_mult" not in locals():
    st_mult = DEFAULTS["st_mult"]
if "diagnostic_fallback" not in locals():
    diagnostic_fallback = DEFAULTS["diagnostic_fallback"]
if "show_atm_details" not in locals():
    show_atm_details = DEFAULTS["show_atm_details"]


# =========================
# RIGHT: Results
# =========================
with right:
    st.subheader("Trade Ideas")

    # Optional ATM detail banner (only when enabled)
    if show_atm_details and expiry_date:
        try:
            atm_ctx = get_atm_context(
                kite, instruments_df, underlying, expiry_date, int(atm_steps)
            )
            b1, b2, b3, b4 = st.columns(4)

            spot_val = atm_ctx.get("spot")
            step_val = atm_ctx.get("step")
            atm_val = atm_ctx.get("atm")

            with b1:
                st.metric("Spot", f"{spot_val:.2f}" if spot_val else "N/A")
            with b2:
                st.metric("Strike Step", f"{step_val:.0f}" if step_val else "N/A")
            with b3:
                st.metric("Computed ATM", f"{atm_val:.0f}" if atm_val else "N/A")
            with b4:
                st.metric("Contracts in Expiry", str(atm_ctx.get("count", 0)))

            low = atm_ctx.get("low")
            high = atm_ctx.get("high")
            if use_atm_filter and low is not None and high is not None:
                st.caption(f"ATM band active: **{low:.0f} → {high:.0f}**")
        except Exception:
            st.caption("ATM details unavailable for this selection.")

    if run_scan:
        if not expiry_date:
            st.warning("Please select a valid expiry.")
            st.stop()

        with st.spinner("Scanning strikes..."):
            result = scan_options_trade_ideas(
                kite=kite,
                underlying=underlying,
                expiry_date=expiry_date,
                interval=interval,
                lookback_days=int(lookback_days),
                rsi_period=int(rsi_period),
                rsi_threshold=float(rsi_threshold),
                st_period=int(st_period),
                st_mult=float(st_mult),
                use_atm_filter=bool(use_atm_filter),
                atm_steps=int(atm_steps),
            )

        if result.empty:
            st.info("No data returned for these settings.")
            st.caption("Try 15minute timeframe and lookback 7. Ensure market is open.")
            st.stop()

        # ---------------------------
        # Ideas filter + confidence
        # ---------------------------
        ideas = result.copy()
        if "Action" in ideas.columns:
            ideas = ideas[ideas["Action"] != "IGNORE"]
        if "Action Confidence" in ideas.columns:
            ideas = ideas[ideas["Action Confidence"] >= int(min_conf)]

        # ---------------------------
        # Action breakdown bar
        # ---------------------------
        counts = ideas["Action"].value_counts().to_dict() if (not ideas.empty and "Action" in ideas.columns) else {}

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("BUY CALL", counts.get("BUY CALL", 0))
        c2.metric("SELL CALL", counts.get("SELL CALL", 0))
        c3.metric("BUY PUT", counts.get("BUY PUT", 0))
        c4.metric("SELL PUT", counts.get("SELL PUT", 0))

        # ---------------------------
        # Sort & column order (clean)
        # ---------------------------
        if "Action Confidence" in ideas.columns:
            ideas = ideas.sort_values("Action Confidence", ascending=False)

        preferred_order = [
            "Action",
            "Action Confidence",
            "Strike",
            "Type",
            "RSI",
            "ST Trend",
            "Price Δ",
            "OI Δ",
            "OI Read",
            "Symbol",
        ]

        def reorder(df: pd.DataFrame) -> pd.DataFrame:
            cols = list(df.columns)
            ordered = [c for c in preferred_order if c in cols]
            remaining = [c for c in cols if c not in ordered]
            return df[ordered + remaining]

        ideas = reorder(ideas)
        result = reorder(result)

        # ---------------------------
        # Display
        # ---------------------------
        if ideas.empty:
            st.info("No trade ideas met the confidence filter right now.")
            if diagnostic_fallback:
                with st.spinner("Loading diagnostic view..."):
                    diag = scan_options_with_indicators(
                        kite=kite,
                        underlying=underlying,
                        expiry_date=expiry_date,
                        interval=interval,
                        lookback_days=int(lookback_days),
                        rsi_period=int(rsi_period),
                        rsi_threshold=float(rsi_threshold),
                        st_period=int(st_period),
                        st_mult=float(st_mult),
                        use_atm_filter=bool(use_atm_filter),
                        atm_steps=int(atm_steps),
                    )
                if not diag.empty:
                    st.markdown("#### Diagnostic (why no ideas?)")
                    st.dataframe(diag, use_container_width=True, hide_index=True)
        else:
            st.dataframe(ideas, use_container_width=True, hide_index=True)

        if show_all_rows:
            with st.expander("All scanned rows", expanded=False):
                st.dataframe(result, use_container_width=True, hide_index=True)

        # ---------------------------
        # CSV download
        # ---------------------------
        csv_bytes = result.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name=f"{underlying}_{expiry_date}_trade_ideas.csv",
            mime="text/csv",
        )

    else:
        st.caption("Set your controls on the left and click **Run Scan**.")


# =========================================================
# Footer
# =========================================================
st.caption(
    "Production defaults: 15m timeframe, RSI(14) threshold 55, Supertrend(10,3), "
    "ATM filter ON, min confidence 40."
)
