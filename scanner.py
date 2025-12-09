import datetime as dt
from typing import Dict, List, Tuple

import pandas as pd

from indicators import compute_rsi, compute_supertrend


# =========================================================
# Instruments
# =========================================================
def fetch_instruments_df(kite) -> pd.DataFrame:
    """
    Fetch NFO instruments and return as DataFrame with normalized fields.
    """
    instruments = kite.instruments("NFO")
    df = pd.DataFrame(instruments)

    if df.empty:
        return df

    # Normalize expiry to date
    if "expiry" in df.columns:
        df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce").dt.date

    # Normalize strike numeric
    if "strike" in df.columns:
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")

    return df


def get_expiries_for_underlying(instruments_df: pd.DataFrame, underlying: str) -> List[str]:
    """
    Return sorted unique expiry ISO strings for the given underlying options.
    """
    if instruments_df is None or instruments_df.empty:
        return []

    underlying = underlying.upper().strip()

    df = instruments_df.copy()
    df = df[df.get("segment", "").astype(str).str.contains("OPT", na=False)]

    if "name" in df.columns:
        df = df[df["name"].astype(str).str.upper() == underlying]
    else:
        # fallback
        df = df[df["tradingsymbol"].astype(str).str.upper().str.startswith(underlying)]

    expiries = sorted({e for e in df.get("expiry", []) if pd.notna(e)})
    return [e.isoformat() for e in expiries]


# =========================================================
# Spot / ATM context
# =========================================================
def _get_spot_symbol(underlying: str) -> str:
    u = underlying.upper().strip()
    if u == "NIFTY":
        return "NSE:NIFTY 50"
    if u == "BANKNIFTY":
        return "NSE:NIFTY BANK"
    if u == "FINNIFTY":
        return "NSE:NIFTY FIN SERVICE"
    if u == "SENSEX":
        return "BSE:SENSEX"
    return f"NSE:{u}"


def get_atm_context(
    kite,
    instruments_df: pd.DataFrame,
    underlying: str,
    expiry_date: str,
    atm_steps: int = 10,
) -> Dict:
    """
    Detect spot, strike step, ATM and the ATM band [low, high].
    """
    underlying = underlying.upper().strip()
    expiry = dt.date.fromisoformat(expiry_date)

    # Spot
    spot_symbol = _get_spot_symbol(underlying)
    spot_quote = kite.quote([spot_symbol])
    spot = float(spot_quote[spot_symbol]["last_price"])

    # Filter options for underlying + expiry
    df = instruments_df.copy()
    df = df[df.get("segment", "").astype(str).str.contains("OPT", na=False)]

    if "name" in df.columns:
        df = df[df["name"].astype(str).str.upper() == underlying]

    df = df[df["expiry"] == expiry]
    df = df[pd.notna(df["strike"])]

    strikes = sorted(df["strike"].unique().tolist())

    # Strike step
    if len(strikes) < 2:
        step = 50 if underlying in ("NIFTY", "FINNIFTY") else 100
    else:
        diffs = [round(strikes[i + 1] - strikes[i], 2) for i in range(len(strikes) - 1)]
        step = int(pd.Series(diffs).mode().iloc[0]) if diffs else 50

    # ATM
    atm = min(strikes, key=lambda s: abs(s - spot)) if strikes else round(spot / step) * step

    low = atm - (atm_steps * step)
    high = atm + (atm_steps * step)

    return {
        "spot": spot,
        "step": float(step),
        "atm": float(atm),
        "low": float(low),
        "high": float(high),
        "count": int(len(df)),
        "spot_symbol": spot_symbol,
    }


# =========================================================
# Option chain filtering
# =========================================================
def _get_option_chain_for_expiry(
    instruments_df: pd.DataFrame,
    underlying: str,
    expiry_date: str,
) -> pd.DataFrame:
    """
    Return a minimal chain table for CE/PE of given underlying + expiry.
    """
    underlying = underlying.upper().strip()
    expiry = dt.date.fromisoformat(expiry_date)

    df = instruments_df.copy()
    df = df[df.get("segment", "").astype(str).str.contains("OPT", na=False)]

    if "name" in df.columns:
        df = df[df["name"].astype(str).str.upper() == underlying]

    df = df[df["expiry"] == expiry]
    df = df[pd.notna(df["strike"])]

    # Standardize type
    if "instrument_type" in df.columns:
        df["Type"] = df["instrument_type"].astype(str).str.upper()
    else:
        df["Type"] = df["tradingsymbol"].astype(str).str[-2:].str.upper()

    df = df[df["Type"].isin(["CE", "PE"])]

    df = df.rename(
        columns={
            "tradingsymbol": "Symbol",
            "instrument_token": "instrument_token",
            "strike": "Strike",
        }
    )

    return df[["Symbol", "instrument_token", "Strike", "Type"]].dropna()


# =========================================================
# Candle fetch + deltas
# =========================================================
def _fetch_ohlc(kite, instrument_token: int, interval: str, lookback_days: int) -> pd.DataFrame:
    """
    Fetch historical candles with OI.
    """
    to_dt = dt.datetime.now()
    from_dt = to_dt - dt.timedelta(days=lookback_days)

    candles = kite.historical_data(
        instrument_token=instrument_token,
        from_date=from_dt,
        to_date=to_dt,
        interval=interval,
        continuous=False,
        oi=True,
    )

    if not candles:
        return pd.DataFrame()

    df = pd.DataFrame(candles)
    df.rename(columns={"oi": "OI"}, inplace=True)
    return df


def _compute_price_oi_delta(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Price Δ and OI Δ from last two candles.
    """
    if df is None or df.empty or len(df) < 2:
        return 0.0, 0.0

    df = df.sort_values("date")
    last = df.iloc[-1]
    prev = df.iloc[-2]

    price_delta = float(last["close"]) - float(prev["close"])
    oi_delta = float(last.get("OI", 0) or 0) - float(prev.get("OI", 0) or 0)

    return price_delta, oi_delta


# =========================================================
# Two-layer action model (PE + CE)
# =========================================================
def classify_action_two_layer(
    opt_type: str,
    st_trend: str,
    rsi: float,
    price_delta: float,
    oi_delta: float,
    rsi_threshold: float = 55.0,
) -> Dict:
    """
    Two-layer interpretation model.
    Returns:
      Action, Action Confidence, OI Read, Momentum Label, Trade Label

    PE philosophy:
      - Long buildup (price↑ + OI↑) -> SELL PUT (higher confidence)
      - Short covering (price↑ + OI↓) -> SELL PUT (medium/low)
      - Long unwinding (price↓ + OI↓) -> WATCH / avoid sell
      - BUY PUT only for rare panic momentum

    CE mirror:
      - Long buildup -> SELL CALL (higher)
      - Short covering -> SELL CALL (medium/low)
      - Long unwinding -> WATCH / avoid sell
      - BUY CALL only for rare panic momentum
    """

    opt_type = (opt_type or "").upper().strip()
    st_trend = (st_trend or "Unknown").title().strip()

    # ---------------------------
    # OI Read
    # ---------------------------
    if price_delta > 0 and oi_delta > 0:
        oi_read = "Long buildup"
    elif price_delta > 0 and oi_delta < 0:
        oi_read = "Short covering"
    elif price_delta < 0 and oi_delta > 0:
        oi_read = "Short buildup"
    elif price_delta < 0 and oi_delta < 0:
        oi_read = "Long unwinding"
    else:
        oi_read = "Neutral"

    # RSI helpers
    rsi_ok = (rsi is not None) and (rsi >= rsi_threshold)
    rsi_high = (rsi is not None) and (rsi >= max(60, rsi_threshold + 5))
    rsi_very_high = (rsi is not None) and (rsi >= 70)

    # Side word for labels
    side_word = "PUT" if opt_type == "PE" else "CALL"

    # Momentum label
    if oi_read in ("Long buildup", "Short covering"):
        momentum_label = f"{side_word} MOMENTUM UP ({oi_read})"
    elif oi_read in ("Short buildup", "Long unwinding"):
        momentum_label = f"{side_word} MOMENTUM DOWN ({oi_read})"
    else:
        momentum_label = f"{side_word} MOMENTUM MIXED ({oi_read})"

    # Defaults
    action = "WATCH"
    confidence = 25
    trade_label = "WATCH / WAIT (No clean edge)"

    # =========================================================
    # PE rules
    # =========================================================
    if opt_type == "PE":

        if oi_read == "Long buildup":
            action = "SELL PUT"
            confidence = 75
            trade_label = "SELL PUT (Bullish premium + OI expansion) — Higher confidence"

            # Conservative ST/RSI boosts
            if st_trend == "Up":
                confidence += 8
            if rsi_high:
                confidence += 8
            confidence = min(confidence, 92)

            # Rare panic override -> BUY PUT
            # Only if strong demand surge is suspected
            if st_trend == "Up" and rsi_very_high and abs(oi_delta) > 500000:
                action = "BUY PUT"
                confidence = 68
                momentum_label = "PUT PANIC MOMENTUM (Demand surge)"
                trade_label = "BUY PUT (Rare panic momentum) — Use strict risk control"

        elif oi_read == "Short covering":
            action = "SELL PUT"
            confidence = 50
            trade_label = "SELL PUT (Mean reversion candidate) — Medium confidence"

            if st_trend == "Up":
                confidence += 5
            if rsi_ok:
                confidence += 5
            confidence = min(confidence, 65)

        elif oi_read == "Long unwinding":
            action = "WATCH"
            confidence = 35
            trade_label = "AVOID SELL PUT (Premium shrinking) — Wait for stabilization"

        elif oi_read == "Short buildup":
            # Put premium falling with OI rising:
            # Can be bearish pressure; DON'T encourage SELL PUT here.
            action = "WATCH"
            confidence = 45 if st_trend == "Down" else 35
            trade_label = "WATCH (Rising put activity / potential downside pressure)"

        else:
            action = "WATCH"
            confidence = 25
            trade_label = "WATCH / WAIT"

    # =========================================================
    # CE rules (mirrored)
    # =========================================================
    elif opt_type == "CE":

        if oi_read == "Long buildup":
            action = "SELL CALL"
            confidence = 75
            trade_label = "SELL CALL (Mean reversion candidate) — Higher confidence"

            if st_trend == "Up":
                confidence += 8
            if rsi_high:
                confidence += 8
            confidence = min(confidence, 92)

            # Rare panic override -> BUY CALL
            if st_trend == "Up" and rsi_very_high and abs(oi_delta) > 500000:
                action = "BUY CALL"
                confidence = 68
                momentum_label = "CALL PANIC MOMENTUM (Demand surge)"
                trade_label = "BUY CALL (Rare panic momentum) — Use strict risk control"

        elif oi_read == "Short covering":
            action = "SELL CALL"
            confidence = 50
            trade_label = "SELL CALL (Mean reversion candidate) — Medium confidence"

            if st_trend == "Up":
                confidence += 5
            if rsi_ok:
                confidence += 5
            confidence = min(confidence, 65)

        elif oi_read == "Long unwinding":
            action = "WATCH"
            confidence = 35
            trade_label = "AVOID SELL CALL (Premium shrinking) — Wait for stabilization"

        elif oi_read == "Short buildup":
            action = "WATCH"
            confidence = 45 if st_trend == "Down" else 35
            trade_label = "WATCH (Rising call activity / potential upside squeeze risk)"

        else:
            action = "WATCH"
            confidence = 25
            trade_label = "WATCH / WAIT"

    else:
        action = "IGNORE"
        confidence = 0
        trade_label = "IGNORE"
        momentum_label = "UNKNOWN"

    # Cap confidence bounds
    confidence = max(0, min(100, int(confidence)))

    return {
        "Action": action,
        "Action Confidence": confidence,
        "OI Read": oi_read,
        "Momentum Label": momentum_label,
        "Trade Label": trade_label,
    }


# =========================================================
# Diagnostic scan (full rows)
# =========================================================
def scan_options_with_indicators(
    kite,
    underlying: str,
    expiry_date: str,
    interval: str = "15minute",
    lookback_days: int = 7,
    rsi_period: int = 14,
    rsi_threshold: float = 55.0,
    st_period: int = 10,
    st_mult: float = 3.0,
    use_atm_filter: bool = True,
    atm_steps: int = 10,
) -> pd.DataFrame:
    """
    Returns raw indicator rows for all strikes in the filtered scan range.
    Used for diagnostics when no trade ideas appear.
    """

    instruments_df = fetch_instruments_df(kite)
    chain = _get_option_chain_for_expiry(instruments_df, underlying, expiry_date)

    if chain.empty:
        return pd.DataFrame()

    # ATM range filter
    if use_atm_filter:
        atm_ctx = get_atm_context(kite, instruments_df, underlying, expiry_date, atm_steps=int(atm_steps))
        low, high = atm_ctx["low"], atm_ctx["high"]
        chain = chain[(chain["Strike"] >= low) & (chain["Strike"] <= high)]

    rows = []

    for _, c in chain.iterrows():
        token = int(c["instrument_token"])
        sym = c["Symbol"]
        strike = float(c["Strike"])
        opt_type = c["Type"]

        ohlc = _fetch_ohlc(kite, token, interval, lookback_days)
        if ohlc.empty or len(ohlc) < 3:
            continue

        price_delta, oi_delta = _compute_price_oi_delta(ohlc)

        rsi_val = compute_rsi(ohlc, period=int(rsi_period))
        st_dir = compute_supertrend(ohlc, period=int(st_period), multiplier=float(st_mult))

        cls = classify_action_two_layer(
            opt_type=opt_type,
            st_trend=st_dir,
            rsi=rsi_val,
            price_delta=price_delta,
            oi_delta=oi_delta,
            rsi_threshold=float(rsi_threshold),
        )

        rows.append(
            {
                "Symbol": sym,
                "Strike": int(strike) if strike.is_integer() else strike,
                "Type": opt_type,
                "ST Trend": st_dir,
                "RSI": round(float(rsi_val), 2) if rsi_val is not None else None,
                "Price Δ": round(float(price_delta), 2),
                "OI Δ": int(oi_delta),
                "OI Read": cls["OI Read"],
                "Momentum Label": cls["Momentum Label"],
                "Trade Label": cls["Trade Label"],
                "Action": cls["Action"],
                "Action Confidence": cls["Action Confidence"],
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if "Action Confidence" in df.columns:
        df = df.sort_values(["Action Confidence", "Strike"], ascending=[False, True])

    return df


# =========================================================
# Production scan (ideas)
# =========================================================
def scan_options_trade_ideas(
    kite,
    underlying: str,
    expiry_date: str,
    interval: str = "15minute",
    lookback_days: int = 7,
    rsi_period: int = 14,
    rsi_threshold: float = 55.0,
    st_period: int = 10,
    st_mult: float = 3.0,
    use_atm_filter: bool = True,
    atm_steps: int = 10,
) -> pd.DataFrame:
    """
    Production scan wrapper used by app.py.
    Returns the full scan DataFrame, letting the UI filter by confidence.
    """
    df = scan_options_with_indicators(
        kite=kite,
        underlying=underlying,
        expiry_date=expiry_date,
        interval=interval,
        lookback_days=lookback_days,
        rsi_period=rsi_period,
        rsi_threshold=rsi_threshold,
        st_period=st_period,
        st_mult=st_mult,
        use_atm_filter=use_atm_filter,
        atm_steps=atm_steps,
    )
    return df
