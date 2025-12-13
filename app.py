import os
import datetime as dt
import inspect
from datetime import datetime
from typing import Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import trade_manager

try:
    from kiteconnect import KiteConnect
except Exception:
    KiteConnect = None

import scanner  # scanner.py


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
# Full-chain OI loader
# -----------------------------
def load_full_chain_oi(kite, underlying: str, expiry_str: str):
    return scanner.fetch_full_chain_oi(kite, underlying, expiry_str)


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


def render_rejected_summary(raw_df: pd.DataFrame, after_conf_df: pd.DataFrame, after_rsi_df: pd.DataFrame, strict_rsi: bool):
    total = len(raw_df)
    after_conf = len(after_conf_df)
    after_rsi = len(after_rsi_df)

    rejected_conf = total - after_conf
    rejected_rsi = after_conf - after_rsi if strict_rsi else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows from scanner", total)
    c2.metric("Rejected by confidence", max(0, rejected_conf))
    c3.metric("Rejected by Strict RSI", max(0, rejected_rsi) if strict_rsi else 0)

    st.caption(
        "These counts reflect only **UI filters**. "
        "Scanner-side rejections (ATM range / candles / quotes / liquidity) are not counted here."
    )


def _toast(msg: str):
    if hasattr(st, "toast"):
        st.toast(msg)
    else:
        st.success(msg)


def apply_preset(preset: str):
    presets = {
        "discovery": {
            "timeframe_select": "15minute",
            "rsi_period": 14,
            "rsi_threshold": 55,
            "min_conf": 30,
            "use_atm_filter": True,
            "atm_steps": 20,
            "st_period": 10,
            "st_mult": 3.0,
            "strict_rsi": False,
            "enable_liquidity": False,
            "min_oi": 0,
            "min_volume": 0,
            "min_ltp": 0.0,
            "max_spread_pct": 999.0,
        },
        "execution": {
            "timeframe_select": "15minute",
            "rsi_period": 14,
            "rsi_threshold": 60,
            "min_conf": 60,
            "use_atm_filter": True,
            "atm_steps": 15,
            "st_period": 10,
            "st_mult": 3.0,
            "strict_rsi": True,
            "enable_liquidity": True,
            "min_oi": 50000,
            "min_volume": 1000,
            "min_ltp": 20.0,
            "max_spread_pct": 4.0,
        },
        "reset": {
            "timeframe_select": "15minute",
            "rsi_period": 14,
            "rsi_threshold": 55,
            "min_conf": 35,
            "use_atm_filter": True,
            "atm_steps": 15,
            "st_period": 10,
            "st_mult": 3.0,
            "strict_rsi": False,
            "enable_liquidity": True,
            "min_oi": 10000,
            "min_volume": 100,
            "min_ltp": 5.0,
            "max_spread_pct": 8.0,
        },
    }

    cfg = presets.get(preset)
    if not cfg:
        return

    for k, v in cfg.items():
        st.session_state[k] = v

    _toast(f"Preset applied: {preset.title()}")

    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.warning("Preset applied. Please refresh once.")


# =============================
# STEP 5 — Scoring / Ranking
# =============================
def compute_idea_score(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()
    base = out["Action Confidence"].fillna(0) if "Action Confidence" in out.columns else 0
    out["Score"] = base.astype(float)

    if "Action" in out.columns:
        out.loc[out["Action"].isin(["SELL PUT", "SELL CALL"]), "Score"] += 5
        out.loc[out["Action"].isin(["BUY PUT", "BUY CALL"]), "Score"] -= 5

    if "OI Read" in out.columns:
        out.loc[out["OI Read"].isin(["Long buildup", "Short covering"]), "Score"] += 3
        out.loc[out["OI Read"].isin(["Long unwinding", "Short buildup"]), "Score"] -= 2

    if "RSI" in out.columns:
        r = out["RSI"].fillna(0)
        out.loc[(r >= 55) & (r <= 68), "Score"] += 2
        out.loc[(r >= 70) | (r <= 35), "Score"] -= 2

    if "Trade Label" in out.columns:
        tl = out["Trade Label"].fillna("").astype(str)
        out.loc[tl.str.contains("ST conflict", case=False, na=False), "Score"] -= 8
        out.loc[tl.str.contains("Against SPOT trend", case=False, na=False), "Score"] -= 8

    if "SPOT Trend" in out.columns and "Action" in out.columns:
        spot = out["SPOT Trend"].fillna("Unknown").astype(str).str.title()
        act = out["Action"].fillna("").astype(str)
        out.loc[(act == "SELL PUT") & (spot == "Up"), "Score"] += 3
        out.loc[(act == "SELL CALL") & (spot == "Down"), "Score"] += 3

    if "OI Δ" in out.columns:
        oid = out["OI Δ"].fillna(0).abs()
        out.loc[oid >= 100000, "Score"] += 1

    out["Score"] = out["Score"].round(1)
    return out


# -----------------------------
# Header
# -----------------------------
st.title("Zerodha Options Scanner")
st.caption(
    f"Version: {APP_VERSION} • Two-layer labels (Momentum + Trade) with confidence • "
    "Access token stored only in memory."
)

with st.expander("How to read these signals (CE/PE core logic)", expanded=False):
    st.markdown("""
### The 2-layer model
Each strike is explained in **two layers**:
1) **Momentum label**: Price Δ + OI Δ  
2) **Trade label**: your premium-selling bias (BUY is rare)
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
    request_token = st.text_input("Paste request_token", type="password", key="request_token_input")
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

access_token = st.session_state.get("access_token")
if not access_token:
    st.info("Login first to enable scanning.")
    st.stop()

kite = make_kite(access_token=access_token)
trade_manager.init_trade_state(st)

tab_scan, tab_summary, tab_trades = st.tabs(["Scanner", "Market / OI Summary", "Open Trades / Square-off"])

# show last action toast
msg = st.session_state.pop("last_action_msg", None)
if msg:
    st.success(msg)


# =============================
# TAB 1: SCANNER
# =============================
with tab_scan:
    left, right = st.columns([0.9, 2.1])

    with left:
        st.subheader("Quick Controls")

        p1, p2, p3 = st.columns(3)
        with p1:
            if st.button("Discovery", use_container_width=True):
                apply_preset("discovery")
        with p2:
            if st.button("Execution", use_container_width=True):
                apply_preset("execution")
        with p3:
            if st.button("Reset", use_container_width=True):
                apply_preset("reset")

        st.caption("Use presets to switch between exploring ideas vs trading-only shortlists.")

        underlying = st.selectbox(
            "Underlying",
            options=["NIFTY", "BANKNIFTY", "FINNIFTY"],
            index=0,
            key="underlying_select",
        )
        st.caption("Ideal: pick the instrument you actually trade that day (avoid switching too often).")

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
                key="expiry_select",
            )
            st.caption("Ideal: nearest weekly expiry for intraday; avoid far expiries unless you intend to trade them.")

        timeframe = st.selectbox(
            "Timeframe",
            options=["5minute", "15minute", "30minute", "60minute"],
            index=["5minute", "15minute", "30minute", "60minute"].index(st.session_state.get("timeframe_select", "15minute"))
            if st.session_state.get("timeframe_select") in ["5minute", "15minute", "30minute", "60minute"] else 1,
            key="timeframe_select",
        )
        st.caption("Ideal: 15m for stability. 5m for faster signals but more noise.")

        rsi_period = st.number_input("RSI period", min_value=5, max_value=50, value=int(st.session_state.get("rsi_period", 14)), step=1, key="rsi_period")
        st.caption("Ideal: 14 (standard).")

        rsi_threshold = st.slider(
            "RSI threshold (idea filter)",
            min_value=40, max_value=80,
            value=int(st.session_state.get("rsi_threshold", 55)),
            key="rsi_threshold",
        )
        st.caption("Ideal: 55 for discovery; 60–65 for execution-only filtering.")

        strict_rsi = st.checkbox(
            "Strict RSI filter (hide RSI below threshold)",
            value=bool(st.session_state.get("strict_rsi", False)),
            key="strict_rsi",
            help="ON = hard filter; OFF = soft idea filter.",
        )
        st.caption("Ideal: OFF (discovery). Turn ON only when you want a tight shortlist.")

        st_period = st.number_input("Supertrend period", min_value=5, max_value=30, value=int(st.session_state.get("st_period", 10)), step=1, key="st_period")
        st.caption("Ideal: 10 for intraday trend gating.")

        st_mult = st.number_input("Supertrend multiplier", min_value=1.0, max_value=5.0, value=float(st.session_state.get("st_mult", 3.0)), step=0.5, key="st_mult")
        st.caption("Ideal: 3.0 (common). Lower = more signals; higher = fewer, cleaner signals.")

        st.markdown("### Strike Range")
        use_atm_filter = st.checkbox("ATM ± X filter", value=bool(st.session_state.get("use_atm_filter", True)), key="use_atm_filter")
        st.caption("Ideal: ON (keeps scan relevant; avoids deep OTM junk).")

        atm_steps = st.number_input("X strike steps", min_value=1, max_value=50, value=int(st.session_state.get("atm_steps", 15)), step=1, key="atm_steps")
        st.caption("Ideal: NIFTY 12–20, BANKNIFTY 15–25 depending on volatility.")

        st.markdown("### Confidence")
        min_conf = st.slider("Minimum Action Confidence", min_value=0, max_value=100, value=int(st.session_state.get("min_conf", 35)), key="min_conf")
        st.caption("Ideal: 30–40 discovery; 55–65 execution shortlist.")

        st.markdown("### Execution / Liquidity (Step 4)")
        enable_liquidity = st.checkbox(
            "Enable liquidity filters",
            value=bool(st.session_state.get("enable_liquidity", True)),
            key="enable_liquidity",
            help="Filters illiquid strikes (low OI/volume, low LTP, wide spreads).",
        )
        st.caption("Ideal: ON for real trading; OFF only for debugging/learning.")

        min_oi = st.number_input("Min OI", min_value=0, value=int(st.session_state.get("min_oi", 10000)), step=1000, key="min_oi")
        st.caption("Ideal: 10k–50k. Increase on expiry day; reduce for far expiries.")

        min_volume = st.number_input("Min Volume", min_value=0, value=int(st.session_state.get("min_volume", 100)), step=50, key="min_volume")
        st.caption("Ideal: 100–1000. Increase after 10:00 AM when volume stabilizes.")

        min_ltp = st.number_input("Min LTP", min_value=0.0, value=float(st.session_state.get("min_ltp", 5.0)), step=1.0, key="min_ltp")
        st.caption("Ideal: 5–20. Increase if you want only tradable premiums with better fills.")

        max_spread_pct = st.number_input("Max spread %", min_value=0.0, value=float(st.session_state.get("max_spread_pct", 8.0)), step=0.5, key="max_spread_pct")
        st.caption("Ideal: 2–5% for execution quality. Use 6–10% if you’re only scouting.")

        st.markdown("### Debug")
        debug_show_strikes = st.checkbox("Debug: show included strikes", value=False, key="debug_show_strikes")
        debug_no_match_details = st.checkbox("Debug: show no-match diagnostics", value=False, key="debug_no_match_details")

    with right:
        st.subheader("Trade Ideas")
        st.caption("Set controls on the left and click **Run Scan**.")
        run = st.button("Run Scan", type="primary", use_container_width=True, key="run_scan_btn")
        scan_placeholder = st.empty()
        results_area = st.container()

    # Transparency row
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

    results_df = None

    if run:
        try:
            with scan_placeholder.container():
                with st.spinner("Scanning option strikes..."):
                    fn = scanner.scan_options_trade_ideas
                    params = inspect.signature(fn).parameters
                    kwargs = {}

                    kwargs["kite"] = kite
                    kwargs["underlying"] = underlying

                    expiry_str = None
                    if expiry is not None:
                        try:
                            expiry_str = expiry.strftime("%Y-%m-%d")
                        except Exception:
                            expiry_str = str(expiry)

                    if "expiry_date" in params:
                        kwargs["expiry_date"] = expiry_str
                    elif "expiry" in params:
                        kwargs["expiry"] = expiry_str

                    if "interval" in params:
                        kwargs["interval"] = timeframe
                    elif "timeframe" in params:
                        kwargs["timeframe"] = timeframe

                    if "rsi_period" in params:
                        kwargs["rsi_period"] = int(rsi_period)
                    if "rsi_threshold" in params:
                        kwargs["rsi_threshold"] = float(rsi_threshold)

                    if "st_period" in params:
                        kwargs["st_period"] = int(st_period)
                    if "st_mult" in params:
                        kwargs["st_mult"] = float(st_mult)

                    if "use_atm_filter" in params:
                        kwargs["use_atm_filter"] = bool(use_atm_filter)
                    if "atm_steps" in params:
                        kwargs["atm_steps"] = int(atm_steps)

                    if enable_liquidity:
                        if "min_oi" in params:
                            kwargs["min_oi"] = int(min_oi)
                        if "min_volume" in params:
                            kwargs["min_volume"] = int(min_volume)
                        if "min_ltp" in params:
                            kwargs["min_ltp"] = float(min_ltp)
                        if "max_spread_pct" in params:
                            kwargs["max_spread_pct"] = float(max_spread_pct)
                    else:
                        if "min_oi" in params:
                            kwargs["min_oi"] = 0
                        if "min_volume" in params:
                            kwargs["min_volume"] = 0
                        if "min_ltp" in params:
                            kwargs["min_ltp"] = 0.0
                        if "max_spread_pct" in params:
                            kwargs["max_spread_pct"] = 999.0

                    results_df = fn(**kwargs)

            scan_placeholder.empty()
        except Exception as e:
            scan_placeholder.empty()
            st.error(f"Scan failed: {e}")
            results_df = None

    with results_area:
        # If not running now, reuse last scan
        if (not run) and ("last_scan_df" in st.session_state):
            results_df = st.session_state["last_scan_df"].copy()

        if isinstance(results_df, pd.DataFrame):
            raw_df = results_df.copy()

            # confidence filter
            if not results_df.empty and "Action Confidence" in results_df.columns:
                after_conf_df = results_df[results_df["Action Confidence"].fillna(0) >= int(min_conf)].copy()
            else:
                after_conf_df = results_df.copy()

            # strict RSI filter
            if not after_conf_df.empty and strict_rsi and "RSI" in after_conf_df.columns:
                after_rsi_df = after_conf_df[after_conf_df["RSI"].fillna(0) >= float(rsi_threshold)].copy()
            else:
                after_rsi_df = after_conf_df.copy()

            results_df = after_rsi_df

            if results_df.empty:
                st.warning("No matching strikes found for the selected settings.")
                if len(raw_df) > 0:
                    with st.expander("Rejected count summary", expanded=True):
                        render_rejected_summary(raw_df, after_conf_df, after_rsi_df, strict_rsi)
                else:
                    st.caption(
                        "Scanner returned 0 rows. Likely causes: ATM range too narrow, "
                        "liquidity filters too strict, or candle/quote availability issues."
                    )
                if debug_no_match_details:
                    st.info("Try: raise ATM steps, lower min_conf, lower RSI threshold, or disable liquidity filters.")
            else:
                results_scored = compute_idea_score(results_df)
                results_scored = results_scored.sort_values(["Score", "Action Confidence"], ascending=[False, False])

                # persist scored scan for reruns + other tabs
                st.session_state["last_scan_df"] = results_scored.copy()

                with st.expander("Top Ranked Ideas (Step 5)", expanded=True):
                    st.caption(
                        "Ranking uses a simple score: Confidence + selling preference + OI-read quality "
                        "+ small bonuses/penalties from labels. Use this as a shortlist, not as auto-trade."
                    )

                    top_n = st.number_input("Show Top N", min_value=5, max_value=50, value=10, step=1, key="top_n")
                    prefer_sells = st.checkbox("Show only SELL actions", value=True, key="prefer_sells")

                    view = results_scored.copy()
                    if prefer_sells and "Action" in view.columns:
                        view = view[view["Action"].isin(["SELL PUT", "SELL CALL"])].copy()

                    top = view.head(int(top_n)).copy()

                    preferred_cols = [
                        "Score", "Action", "Action Confidence",
                        "Trade Label", "Momentum Label",
                        "Strike", "Type", "RSI", "ST Trend", "SPOT Trend",
                        "Price Δ", "OI Δ", "OI Read", "Symbol",
                        "instrument_token", "LTP"
                    ]
                    cols = [c for c in preferred_cols if c in top.columns]
                    st.dataframe(top[cols], use_container_width=True, hide_index=True)

                    # persist top list too (optional)
                    st.session_state["last_top_df"] = top.copy()

                    # -----------------------------
                    # STEP 7: One-click Add Trade (FORM)
                    # -----------------------------
                    st.markdown("### ➕ Add selected strike to Open Trades (Step 7)")
                    st.caption("Recommend-only: this does NOT place orders. It just adds to the Open Trades tab.")

                    required_cols = {"Symbol", "instrument_token"}
                    if top.empty:
                        st.info("Top list is empty after filters. Change Top N / SELL-only filter.")
                    elif not required_cols.issubset(set(top.columns)):
                        st.error("Top table missing required columns. Ensure scanner returns Symbol + instrument_token.")
                    else:
                        def _label(r: dict) -> str:
                            return f"{r.get('Symbol')} | {r.get('Action','')} | Strike {r.get('Strike','')} | Score {r.get('Score','')}"

                        options = top.to_dict("records")

                        with st.form("add_trade_form", clear_on_submit=False):
                            pick = st.selectbox(
                                "Pick strike to add",
                                options=options,
                                format_func=_label,
                                key="add_trade_pick_form",
                            )

                            c1, c2, c3 = st.columns(3)
                            with c1:
                                entry_price = st.number_input(
                                    "Entry premium (your fill)",
                                    min_value=0.0,
                                    step=0.5,
                                    value=float(pick.get("LTP", 0) or 0),
                                    key="add_trade_entry_form",
                                )
                            with c2:
                                qty = st.number_input(
                                    "Quantity",
                                    min_value=1,
                                    value=15,
                                    step=1,
                                    key="add_trade_qty_form",
                                )
                            with c3:
                                submitted = st.form_submit_button("Add Trade")

                            if submitted:
                                action = str(pick.get("Action", "")).upper()
                                side = "SELL" if action.startswith("SELL") else "BUY"

                                try:
                                    trade_manager.add_trade(
                                        st,
                                        {
                                            "symbol": str(pick["Symbol"]),
                                            "instrument_token": int(pick["instrument_token"]),
                                            "side": side,
                                            "qty": int(qty),
                                            "entry_price": float(entry_price),
                                            "sl_pct": 30.0,
                                            "tp_pct": 40.0,
                                            "max_hold_minutes": 120,
                                            "square_off_time": dt.time(15, 15),
                                            "note": f"From scanner | {pick.get('Trade Label','')}",
                                        }
                                    )
                                    st.session_state["last_action_msg"] = f"Trade added: {pick['Symbol']} ({side})"
                                    st.success(f"Trade added: {pick['Symbol']} ({side})")
                                except Exception as e:
                                    st.error(f"Failed to add trade: {e}")

                # counters
                action_counts = results_scored["Action"].value_counts().to_dict() if "Action" in results_scored.columns else {}
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("BUY CALL", action_counts.get("BUY CALL", 0))
                c2.metric("SELL CALL", action_counts.get("SELL CALL", 0))
                c3.metric("BUY PUT", action_counts.get("BUY PUT", 0))
                c4.metric("SELL PUT", action_counts.get("SELL PUT", 0))

                summary = build_action_summary(results_scored)
                with st.expander("Action Summary by OI Read", expanded=False):
                    if summary is None or summary.empty:
                        st.caption("Summary appears when both 'OI Read' and 'Action' are available.")
                    else:
                        st.dataframe(summary, use_container_width=True, hide_index=True)

                if debug_show_strikes and "Strike" in results_scored.columns:
                    strikes = sorted(results_scored["Strike"].dropna().unique().tolist())
                    st.caption(f"Filtered scan strikes ({len(strikes)}):")
                    st.write(strikes)

                # full table
                preferred = [
                    "Score", "Action", "Action Confidence", "Trade Label", "Momentum Label",
                    "Strike", "Type", "RSI", "ST Trend", "SPOT Trend",
                    "Price Δ", "OI Δ", "OI Read", "Symbol", "instrument_token", "LTP"
                ]
                cols = [c for c in preferred if c in results_scored.columns]
                remaining = [c for c in results_scored.columns if c not in cols]
                final_cols = cols + remaining
                st.dataframe(results_scored[final_cols], use_container_width=True, hide_index=True)

                csv = results_scored.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download results (CSV)",
                    data=csv,
                    file_name=f"{underlying}_options_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_csv_btn",
                )
        else:
            st.caption("Ready. Click **Run Scan**.")


# =============================
# TAB 2: MARKET / OI SUMMARY
# =============================
with tab_summary:
    st.markdown("### 1️⃣ Market / Directional View (from OI)")

    expiry_str = None
    try:
        expiry_str = expiry.strftime("%Y-%m-%d") if "expiry" in locals() and expiry is not None else None
    except Exception:
        expiry_str = str(expiry) if "expiry" in locals() and expiry is not None else None

    detected_spot = spot if "spot" in locals() else None

    if kite is None or expiry_str is None:
        st.info("Login and select an expiry to view OI summary.")
    else:
        full_chain_df = load_full_chain_oi(kite, underlying, expiry_str)
        oi_summary = scanner.summarize_oi_chain(full_chain_df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Underlying", underlying)
        c2.metric("Expiry", expiry_str)
        c3.metric("Spot (approx)", f"{detected_spot:,.2f}" if detected_spot else "N/A")
        pcr_display = f"{oi_summary['pcr_oi']:.2f}" if oi_summary["pcr_oi"] is not None else "N/A"
        c4.metric("PCR (OI-based)", pcr_display)

        if full_chain_df is None or full_chain_df.empty:
            st.warning("Not enough OI data to form a directional view.")
        else:
            st.write("Directional OI view: full-chain data loaded (you can add richer logic here).")

        st.markdown("### 2️⃣ Key OI Levels")

        if full_chain_df is None or full_chain_df.empty:
            st.info("No OI data available for this expiry.")
        else:
            ce_strike = oi_summary["highest_ce_strike"]
            pe_strike = oi_summary["highest_pe_strike"]

            ce_text = f"**Highest CE OI** at strike **{ce_strike:,.0f}**" if ce_strike is not None else "No CE OI data."
            pe_text = f"**Highest PE OI** at strike **{pe_strike:,.0f}**" if pe_strike is not None else "No PE OI data."

            st.markdown(f"- {ce_text}")
            st.markdown(f"- {pe_text}")
            st.markdown(
                f"- Total CE OI: **{oi_summary['total_ce_oi']:,.0f}**, "
                f"Total PE OI: **{oi_summary['total_pe_oi']:,.0f}**"
            )

            with st.expander("Show raw option-chain snapshot (full chain for this expiry)"):
                st.dataframe(full_chain_df.sort_values(["type", "strike"]), use_container_width=True, height=400)

    st.markdown("### 3️⃣ Intraday Option-Selling Structures")
    st.markdown("""
- **Setup A — Short Call at Resistance**
- **Setup B — Short Strangle / Iron Condor (Range-bound)**
- **Setup C — Short Put near Support**
""")


# =============================
# TAB 3: OPEN TRADES / SQUARE-OFF
# =============================
with tab_trades:
    st.subheader("Open Trades / Square-off Recommendations")
    st.caption("Recommend-only. No orders are placed.")

    auto_refresh = st.checkbox("Auto-refresh (every 30s)", value=False)
    if auto_refresh:
        st.info("Auto-refresh is ON. The app will refresh every ~30 seconds while this tab is open.")

    with st.expander("➕ Add a trade (manual)", expanded=True):
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            t_symbol = st.text_input("Option Symbol (tradingsymbol)", value="")
        with c2:
            t_token = st.number_input("Instrument Token", min_value=0, value=0, step=1)
        with c3:
            t_side = st.selectbox("Side", options=["SELL", "BUY"], index=0)
        with c4:
            t_qty = st.number_input("Quantity", min_value=1, value=15, step=1)

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            t_entry = st.number_input("Entry Premium", min_value=0.0, value=100.0, step=0.5)
        with c6:
            t_sl = st.number_input("SL %", min_value=1.0, value=30.0, step=1.0)
        with c7:
            t_tp = st.number_input("Target %", min_value=1.0, value=40.0, step=1.0)
        with c8:
            t_max_hold = st.number_input("Max hold (mins)", min_value=5, value=120, step=5)

        c9, c10 = st.columns(2)
        with c9:
            sq_time_str = st.text_input("Square-off time (HH:MM)", value="15:15")
        with c10:
            t_note = st.text_input("Notes (optional)", value="")

        add_clicked = st.button("Add Trade", type="primary", use_container_width=True)

        if add_clicked:
            try:
                if not t_symbol.strip():
                    st.error("Please enter Option Symbol.")
                elif int(t_token) <= 0:
                    st.error("Please enter a valid instrument token.")
                else:
                    hh, mm = [int(x) for x in sq_time_str.strip().split(":")]
                    sq_time = dt.time(hour=hh, minute=mm)

                    trade_manager.add_trade(
                        st,
                        {
                            "symbol": t_symbol.strip(),
                            "instrument_token": int(t_token),
                            "side": t_side,
                            "qty": int(t_qty),
                            "entry_price": float(t_entry),
                            "sl_pct": float(t_sl),
                            "tp_pct": float(t_tp),
                            "max_hold_minutes": float(t_max_hold),
                            "square_off_time": sq_time,
                            "note": t_note.strip(),
                        }
                    )
                    st.success("Trade added.")
            except Exception as e:
                st.error(f"Failed to add trade: {e}")

    st.divider()

    open_trades = [t for t in st.session_state.get("open_trades", []) if t.get("status") == "OPEN"]
    if not open_trades:
        st.info("No open trades yet. Add one above or from Scanner (Step 7).")
    else:
        rows = []
        now = dt.datetime.now()

        for t in open_trades:
            token = int(t["instrument_token"])
            ltp = trade_manager.safe_ltp_by_token(kite, token) or 0.0

            spot_tr = "Unknown"
            opt_tr = "Unknown"
            oi_read = "Neutral"

            try:
                spot_tr = scanner.get_spot_trend(kite, underlying, interval="15minute", lookback_days=5, st_period=10, st_mult=3.0)
            except Exception:
                spot_tr = "Unknown"

            try:
                candles = kite.historical_data(token, now - dt.timedelta(days=5), now, "15minute", oi=False)
                if candles:
                    dfc = pd.DataFrame(candles)
                    from indicators import compute_supertrend
                    opt_tr = compute_supertrend(dfc, period=10, multiplier=3.0)
            except Exception:
                opt_tr = "Unknown"

            action, confidence, reasons = trade_manager.recommend_square_off(
                trade=t,
                ltp=float(ltp),
                spot_trend=spot_tr,
                option_trend=opt_tr,
                oi_read=oi_read,
                now=now,
            )

            entry = float(t["entry_price"])
            side = str(t["side"]).upper()
            qty = int(t["qty"])

            pnl_points = (entry - ltp) if side == "SELL" else (ltp - entry)
            pnl = pnl_points * qty

            hold_mins = (now - t["entry_time"]).total_seconds() / 60.0 if isinstance(t.get("entry_time"), dt.datetime) else None

            rows.append({
                "Trade ID": t["trade_id"],
                "Symbol": t["symbol"],
                "Side": side,
                "Qty": qty,
                "Entry": round(entry, 2),
                "LTP": round(float(ltp), 2),
                "PnL (approx)": round(float(pnl), 2),
                "Hold (mins)": round(hold_mins, 1) if hold_mins is not None else None,
                "Recommendation": action,
                "Confidence": int(confidence),
                "Reasons": "; ".join(reasons) if reasons else "",
                "Partial Done": bool(t.get("partial_done", False)),
                "Note": t.get("note", ""),
            })

        df = pd.DataFrame(rows).sort_values(["Confidence"], ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("### Actions (recommend-only)")
        c1, c2, c3 = st.columns(3)

        with c1:
            tid_partial = st.text_input("Trade ID for Partial Done", value="", key="tid_partial")
            if st.button("Mark Partial Done (50%)", use_container_width=True):
                ok = trade_manager.mark_partial_done(st, tid_partial.strip())
                st.success("Marked partial done.") if ok else st.error("Trade ID not found.")

        with c2:
            tid_exit = st.text_input("Trade ID to mark exited", value="", key="tid_exit")
            if st.button("Mark Exited", use_container_width=True):
                ok = trade_manager.mark_exited(st, tid_exit.strip())
                st.success("Marked exited.") if ok else st.error("Trade ID not found.")

        with c3:
            if st.button("Remove all EXITED trades", use_container_width=True):
                trade_manager.purge_exited(st)
                st.success("Purged exited trades.")

    if auto_refresh:
        if hasattr(st, "rerun"):
            st.rerun()
        elif hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
