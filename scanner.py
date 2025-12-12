import datetime as dt
from typing import Dict, List, Tuple, Optional

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
# SPOT trend (Step 2)
# =========================================================
_SPOT_TOKEN_CACHE: Dict[str, int] = {}


def _resolve_spot_token(kite, underlying: str) -> Optional[int]:
    """
    Resolve instrument_token for the underlying's spot index via NSE/BSE instruments.
    Caches result per underlying to avoid repeated downloads.
    """
    u = underlying.upper().strip()
    if u in _SPOT_TOKEN_CACHE:
        return _SPOT_TOKEN_CACHE[u]

    spot_symbol = _get_spot_symbol(u)  # e.g. "NSE:NIFTY 50"
    exch, ts = spot_symbol.split(":", 1)

    try:
        inst = kite.instruments(exch)  # NSE / BSE
    except Exception:
        return None

    df = pd.DataFrame(inst)
    if df.empty:
        return None

    m = df[df.get("tradingsymbol", "").astype(str) == ts]
    if m.empty:
        m = df[df.get("name", "").astype(str) == ts]

    if m.empty:
        return None

    token = int(m.iloc[0]["instrument_token"])
    _SPOT_TOKEN_CACHE[u] = token
    return token


def get_spot_trend(
    kite,
    underlying: str,
    interval: str = "15minute",
    lookback_days: int = 7,
    st_period: int = 10,
    st_mult: float = 3.0,
) -> str:
    """
    Compute SPOT Supertrend direction ('Up'/'Down') for the underlying index.
    Returns 'Unknown' if token/candles not available.
    """
    token = _resolve_spot_token(kite, underlying)
    if token is None:
        return "Unknown"

    to_dt = dt.datetime.now()
    from_dt = to_dt - dt.timedelta(days=lookback_days)

    try:
        candles = kite.historical_data(
            instrument_token=token,
            from_date=from_dt,
            to_date=to_dt,
            interval=interval,
            continuous=False,
            oi=False,
        )
    except Exception:
        return "Unknown"

    if not candles:
        return "Unknown"

    df = pd.DataFrame(candles)
    try:
        return compute_supertrend(df, period=int(st_period), multiplier=float(st_mult))
    except Exception:
        return "Unknown"


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
# Step 3 — Liquidity / execution filters
# =========================================================
def _passes_liquidity_filters(
    quote: dict,
    min_oi: int,
    min_volume: int,
    max_spread_pct: float,
    min_ltp: float,
) -> bool:
    """
    Hard execution-quality filters.
    Tolerant: spread check applies only if depth data is available.
    """
    try:
        ltp = float(quote.get("last_price", 0) or 0)
        oi = float(quote.get("oi", 0) or 0)
        vol = float(quote.get("volume", 0) or 0)

        if ltp < min_ltp:
            return False
        if oi < min_oi:
            return False
        if vol < min_volume:
            return False

        depth = quote.get("depth") or {}
        buy = depth.get("buy") or []
        sell = depth.get("sell") or []

        if buy and sell:
            bid = float((buy[0] or {}).get("price", 0) or 0)
            ask = float((sell[0] or {}).get("price", 0) or 0)
            if bid > 0 and ask > 0 and ltp > 0:
                spread_pct = ((ask - bid) / ltp) * 100.0
                if spread_pct > float(max_spread_pct):
                    return False

        return True
    except Exception:
        return False


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
# Two-layer action model (Step 1 + Step 2)
# =========================================================
def classify_action_two_layer(
    opt_type: str,
    st_trend: str,
    rsi: float,
    price_delta: float,
    oi_delta: float,
    rsi_threshold: float = 55.0,
    spot_trend: str = "Unknown",
) -> Dict:
    """
    Two-layer interpretation model.
    Returns:
      Action, Action Confidence, OI Read, Momentum Label, Trade Label
    """
    opt_type = (opt_type or "").upper().strip()
    st_trend = (st_trend or "Unknown").title().strip()
    spot_trend = (spot_trend or "Unknown").title().strip()

    # OI Read
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

    # PE rules
    if opt_type == "PE":
        if oi_read == "Long buildup":
            action = "SELL PUT"
            confidence = 75
            trade_label = "SELL PUT (Bullish premium + OI expansion) — Higher confidence"
            if st_trend == "Up":
                confidence += 8
            if rsi_high:
                confidence += 8
            confidence = min(confidence, 92)

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
            action = "WATCH"
            confidence = 45 if st_trend == "Down" else 35
            trade_label = "WATCH (Rising put activity / potential downside pressure)"
        else:
            action = "WATCH"
            confidence = 25
            trade_label = "WATCH / WAIT"

    # CE rules
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

    # Step 1: Momentum vs ST conflict
    momentum_dir = None
    if "MOMENTUM UP" in momentum_label:
        momentum_dir = "Up"
    elif "MOMENTUM DOWN" in momentum_label:
        momentum_dir = "Down"

    if momentum_dir in ("Up", "Down") and st_trend in ("Up", "Down") and momentum_dir != st_trend:
        confidence -= 12
        if action != "WATCH":
            trade_label = f"{trade_label} (ST conflict)"

    # Step 2: SPOT trend gate
    if action == "SELL CALL" and spot_trend == "Up":
        confidence -= 20
        trade_label = f"{trade_label} (Against SPOT trend)"
    if action == "SELL PUT" and spot_trend == "Down":
        confidence -= 20
        trade_label = f"{trade_label} (Against SPOT trend)"

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
    # STEP 3 — Liquidity filters (tune later from app.py)
    min_oi: int = 10_000,
    min_volume: int = 100,
    max_spread_pct: float = 8.0,
    min_ltp: float = 5.0,
) -> pd.DataFrame:
    """
    Returns raw indicator rows for all strikes in the filtered scan range.
    """

    instruments_df = fetch_instruments_df(kite)
    chain = _get_option_chain_for_expiry(instruments_df, underlying, expiry_date)
    if chain.empty:
        return pd.DataFrame()

    # Step 2: SPOT trend once per scan
    spot_tr = get_spot_trend(
        kite=kite,
        underlying=underlying,
        interval=interval,
        lookback_days=int(lookback_days),
        st_period=int(st_period),
        st_mult=float(st_mult),
    )

    # ATM filter
    if use_atm_filter:
        atm_ctx = get_atm_context(kite, instruments_df, underlying, expiry_date, atm_steps=int(atm_steps))
        low, high = atm_ctx["low"], atm_ctx["high"]
        chain = chain[(chain["Strike"] >= low) & (chain["Strike"] <= high)]

    if chain.empty:
        return pd.DataFrame()

    # Step 3: batch quote once (VERY IMPORTANT)
    tokens = chain["instrument_token"].astype(int).tolist()
    try:
        quotes_map = kite.quote(tokens)  # one call
    except Exception:
        quotes_map = {}

    rows = []

    for _, c in chain.iterrows():
        token = int(c["instrument_token"])
        sym = c["Symbol"]
        strike = float(c["Strike"])
        opt_type = c["Type"]

        quote = quotes_map.get(str(token)) or quotes_map.get(token)
        if not quote:
            continue

        # Step 3: liquidity gate
        if not _passes_liquidity_filters(
            quote=quote,
            min_oi=int(min_oi),
            min_volume=int(min_volume),
            max_spread_pct=float(max_spread_pct),
            min_ltp=float(min_ltp),
        ):
            continue

        ohlc = _fetch_ohlc(kite, token, interval, lookback_days)
        if ohlc.empty or len(ohlc) < 3:
            continue

        price_delta, oi_delta = _compute_price_oi_delta(ohlc)

        rsi_val = compute_rsi(ohlc, period=int(rsi_period))
        st_dir = compute_supertrend(ohlc, period=int(st_period), multiplier=float(st_mult))

        cls = classify_action_two_layer(
            opt_type=opt_type,
            st_trend=st_dir,
            spot_trend=spot_tr,
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
                "SPOT Trend": spot_tr,
                "RSI": round(float(rsi_val), 2) if rsi_val is not None else None,
                "Price Δ": round(float(price_delta), 2),
                "OI Δ": int(oi_delta),
                "OI Read": cls["OI Read"],
                "Momentum Label": cls["Momentum Label"],
                "Trade Label": cls["Trade Label"],
                "Action": cls["Action"],
                "Action Confidence": cls["Action Confidence"],
                # Optional: expose execution metrics for debugging
                "LTP": float(quote.get("last_price", 0) or 0),
                "VOL": float(quote.get("volume", 0) or 0),
                "OI": float(quote.get("oi", 0) or 0),
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
    # Step 3 pass-through
    min_oi: int = 10_000,
    min_volume: int = 100,
    max_spread_pct: float = 8.0,
    min_ltp: float = 5.0,
) -> pd.DataFrame:
    """
    Production scan wrapper used by app.py.
    Returns the full scan DataFrame, letting the UI filter by confidence.
    """
    return scan_options_with_indicators(
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
        min_oi=min_oi,
        min_volume=min_volume,
        max_spread_pct=max_spread_pct,
        min_ltp=min_ltp,
    )


# =============================================================================
# Full option-chain OI utilities (for Market / OI Summary tab)
# =============================================================================
def fetch_full_chain_oi(kite, underlying: str, expiry_date: str) -> pd.DataFrame:
    """
    Fetch a *full* option-chain snapshot (CE + PE) for a given underlying + expiry.
    """
    try:
        expiry = dt.datetime.fromisoformat(expiry_date).date()
    except Exception:
        return pd.DataFrame(columns=["strike", "type", "oi", "ltp"])

    try:
        instruments = kite.instruments("NFO")
    except Exception:
        return pd.DataFrame(columns=["strike", "type", "oi", "ltp"])

    chain = [
        inst for inst in instruments
        if inst.get("segment") == "NFO-OPT"
        and inst.get("name") == underlying
        and inst.get("expiry") == expiry
    ]

    if not chain:
        return pd.DataFrame(columns=["strike", "type", "oi", "ltp"])

    tokens = [inst["instrument_token"] for inst in chain]

    try:
        quotes = kite.quote(tokens)
    except Exception:
        return pd.DataFrame(columns=["strike", "type", "oi", "ltp"])

    rows = []
    for inst in chain:
        token = inst["instrument_token"]
        q = quotes.get(str(token)) or quotes.get(token)
        if not q:
            continue

        oi = q.get("oi", 0) or 0
        ltp = q.get("last_price", 0) or 0.0

        rows.append(
            {
                "strike": float(inst.get("strike", 0)),
                "type": (inst.get("instrument_type", "") or "").upper(),
                "oi": float(oi),
                "ltp": float(ltp),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["strike", "type", "oi", "ltp"])

    df = pd.DataFrame(rows)
    df["type"] = df["type"].astype(str).str.upper()
    return df


def summarize_oi_chain(df: pd.DataFrame) -> dict:
    """
    Build a simple OI summary from a full-chain DataFrame.
    """
    summary = {
        "total_ce_oi": 0.0,
        "total_pe_oi": 0.0,
        "pcr_oi": None,
        "highest_ce_strike": None,
        "highest_ce_oi": 0.0,
        "highest_pe_strike": None,
        "highest_pe_oi": 0.0,
    }

    if df is None or df.empty:
        return summary

    ce = df[df["type"] == "CE"]
    pe = df[df["type"] == "PE"]

    summary["total_ce_oi"] = float(ce["oi"].sum()) if not ce.empty else 0.0
    summary["total_pe_oi"] = float(pe["oi"].sum()) if not pe.empty else 0.0

    if summary["total_ce_oi"] > 0 and not ce.empty:
        idx = ce["oi"].idxmax()
        summary["highest_ce_strike"] = float(ce.loc[idx, "strike"])
        summary["highest_ce_oi"] = float(ce.loc[idx, "oi"])

    if summary["total_pe_oi"] > 0 and not pe.empty:
        idx = pe["oi"].idxmax()
        summary["highest_pe_strike"] = float(pe.loc[idx, "strike"])
        summary["highest_pe_oi"] = float(pe.loc[idx, "oi"])

    if summary["total_ce_oi"] > 0:
        summary["pcr_oi"] = summary["total_pe_oi"] / summary["total_ce_oi"]

    return summary
