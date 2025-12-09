# scanner.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from collections import Counter

import pandas as pd

from indicators import compute_rsi, compute_supertrend


# -----------------------
# Data Model
# -----------------------
@dataclass
class OptionMeta:
    tradingsymbol: str
    strike: float
    option_type: str  # CE / PE
    expiry: str
    instrument_token: int


# -----------------------
# Instruments + Expiries
# -----------------------
def fetch_instruments_df(kite) -> pd.DataFrame:
    """Fetch NFO instruments from Zerodha."""
    inst = kite.instruments("NFO")
    return pd.DataFrame(inst)


def get_expiries_for_underlying(instruments_df: pd.DataFrame, underlying: str) -> List[str]:
    """Returns sorted unique expiry dates (as YYYY-MM-DD strings) for CE/PE of the underlying."""
    df = instruments_df.copy()
    df = df[
        (df["name"].str.upper() == underlying.upper())
        & (df["instrument_type"].isin(["CE", "PE"]))
    ]
    return sorted(df["expiry"].astype(str).unique().tolist())


def filter_options_for_expiry(
    instruments_df: pd.DataFrame,
    underlying: str,
    expiry_date: str
) -> List[OptionMeta]:
    """Filter all CE/PE of a given underlying + exact expiry date."""
    df = instruments_df.copy()
    df = df[
        (df["name"].str.upper() == underlying.upper())
        & (df["instrument_type"].isin(["CE", "PE"]))
        & (df["expiry"].astype(str) == str(expiry_date))
    ]

    metas: List[OptionMeta] = []
    for _, row in df.iterrows():
        metas.append(
            OptionMeta(
                tradingsymbol=row["tradingsymbol"],
                strike=float(row["strike"]),
                option_type=row["instrument_type"],
                expiry=str(row["expiry"]),
                instrument_token=int(row["instrument_token"]),
            )
        )
    return metas


# -----------------------
# ATM Helpers
# -----------------------
def infer_strike_step(strikes: List[float]) -> float:
    """Infer the common strike interval from strikes."""
    strikes = sorted(set([float(s) for s in strikes if s is not None]))
    if len(strikes) < 3:
        return 50.0

    diffs = [round(strikes[i] - strikes[i - 1], 2) for i in range(1, len(strikes))]
    diffs = [d for d in diffs if d > 0]
    if not diffs:
        return 50.0

    return float(Counter(diffs).most_common(1)[0][0])


def get_underlying_ltp(kite, underlying: str) -> Optional[float]:
    """Fetch underlying spot LTP using likely symbol mappings."""
    mapping = {
        "NIFTY": "NSE:NIFTY 50",
        "BANKNIFTY": "NSE:NIFTY BANK",
        "FINNIFTY": "NSE:NIFTY FIN SERVICE",
        "SENSEX": "BSE:SENSEX",
    }
    sym = mapping.get(underlying.upper())
    if not sym:
        return None

    try:
        ltp_data = kite.ltp([sym])
        return float(ltp_data[sym]["last_price"])
    except Exception:
        return None


def compute_atm_strike(spot: float, step: float) -> Optional[float]:
    """Approx ATM by rounding spot to nearest strike step."""
    if not spot or not step:
        return None
    return round(spot / step) * step


def _apply_atm_filter(
    kite,
    metas: List[OptionMeta],
    underlying: str,
    atm_steps: int
) -> Tuple[List[OptionMeta], dict]:
    """Apply ATM ± X filter to metas and return (filtered_metas, context)."""
    ctx = {
        "spot": None,
        "step": None,
        "atm": None,
        "low": None,
        "high": None,
        "count_all": len(metas),
        "count_filtered": len(metas),
    }

    if not metas:
        return metas, ctx

    strikes = [m.strike for m in metas]
    step = infer_strike_step(strikes)
    spot = get_underlying_ltp(kite, underlying)
    atm = compute_atm_strike(spot, step) if spot else None

    ctx.update({"spot": spot, "step": step, "atm": atm})

    if atm is not None and step is not None:
        low = atm - (atm_steps * step)
        high = atm + (atm_steps * step)
        ctx.update({"low": low, "high": high})
        metas = [m for m in metas if low <= m.strike <= high]

    ctx["count_filtered"] = len(metas)
    return metas, ctx


def get_atm_context(
    kite,
    instruments_df: pd.DataFrame,
    underlying: str,
    expiry_date: str,
    atm_steps: int = 10
) -> dict:
    """Compute spot, inferred strike step, computed ATM and range for UI."""
    metas = filter_options_for_expiry(instruments_df, underlying, expiry_date)

    if not metas:
        return {
            "spot": None, "step": None, "atm": None,
            "low": None, "high": None, "count": 0
        }

    strikes = [m.strike for m in metas]
    step = infer_strike_step(strikes)
    spot = get_underlying_ltp(kite, underlying)
    atm = compute_atm_strike(spot, step) if spot else None

    low = high = None
    if atm is not None and step is not None:
        low = atm - (atm_steps * step)
        high = atm + (atm_steps * step)

    return {
        "spot": spot,
        "step": step,
        "atm": atm,
        "low": low,
        "high": high,
        "count": len(metas),
    }


# -----------------------
# Candles
# -----------------------
def fetch_candles(
    kite,
    instrument_token: int,
    interval: str,
    lookback_days: int
) -> pd.DataFrame:
    """Fetch historical OHLC candles for an option instrument token (without OI)."""
    to_dt = datetime.now()
    from_dt = to_dt - timedelta(days=lookback_days)

    data = kite.historical_data(
        instrument_token=instrument_token,
        from_date=from_dt,
        to_date=to_dt,
        interval=interval,
        continuous=False,
        oi=False,
    )

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df.rename(columns={"date": "datetime"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)

    cols = ["open", "high", "low", "close", "volume"]
    available = [c for c in cols if c in df.columns]
    return df[available].copy()


def fetch_candles_with_oi(
    kite,
    instrument_token: int,
    interval: str,
    lookback_days: int
) -> pd.DataFrame:
    """Fetch historical OHLC candles including OI."""
    to_dt = datetime.now()
    from_dt = to_dt - timedelta(days=lookback_days)

    data = kite.historical_data(
        instrument_token=instrument_token,
        from_date=from_dt,
        to_date=to_dt,
        interval=interval,
        continuous=False,
        oi=True,
    )

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df.rename(columns={"date": "datetime", "oi": "open_interest"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)

    cols = ["open", "high", "low", "close", "volume", "open_interest"]
    available = [c for c in cols if c in df.columns]
    return df[available].copy()


# -----------------------
# Indicator helpers
# -----------------------
def analyze_option(
    df: pd.DataFrame,
    rsi_period: int,
    st_period: int,
    st_mult: float
):
    """Returns latest (rsi_value, st_direction). st_direction: 1 = Up, -1 = Down"""
    if df.empty:
        return None, None

    min_len = max(rsi_period, st_period) + 5
    if len(df) < min_len:
        return None, None

    df = df.copy()
    df["rsi"] = compute_rsi(df["close"], rsi_period)
    df = compute_supertrend(df, st_period, st_mult)

    latest = df.iloc[-1]
    rsi_val = latest.get("rsi", None)
    st_dir = latest.get("st_direction", None)

    if rsi_val is None or pd.isna(rsi_val) or st_dir is None:
        return None, None

    return float(rsi_val), int(st_dir)


# -----------------------
# OI + Price interpretation
# -----------------------
def oi_price_read(price_change: float, oi_change: float) -> str:
    """Classic OI + price interpretation (on option itself)."""
    if price_change > 0 and oi_change > 0:
        return "Long buildup"
    if price_change > 0 and oi_change < 0:
        return "Short covering"
    if price_change < 0 and oi_change > 0:
        return "Short buildup"
    if price_change < 0 and oi_change < 0:
        return "Long unwinding"
    return "Neutral"


def _confidence_score(
    rsi_val: float,
    rsi_threshold: float,
    st_dir: int,
    price_change: float,
    oi_change: Optional[float],
    favorable_reads: tuple[str, ...],
    require_st: Optional[int] = None,
) -> int:
    """
    Lightweight confidence scoring (0-100).
    Components:
    - RSI strength vs threshold (0-40)
    - OI Read alignment (0-30)
    - Price direction (0-15)
    - Supertrend alignment (0-15)
    """
    score = 0

    # RSI contribution
    if rsi_val is not None:
        rsi_gap = max(0.0, rsi_val - rsi_threshold)
        score += int(min(40, rsi_gap * 4))  # 10 RSI points above threshold => +40

    # OI contribution
    if oi_change is not None:
        read = oi_price_read(price_change, oi_change)
        if read in favorable_reads:
            score += 30
        elif read == "Neutral":
            score += 10

    # Price contribution
    if price_change > 0:
        score += 15

    # ST contribution
    if require_st is None:
        score += 8 if st_dir in (1, -1) else 0
    else:
        score += 15 if st_dir == require_st else 0

    return max(0, min(100, score))


def classify_action(
    option_type: str,
    st_dir: int,
    rsi_val: float,
    price_change: float,
    oi_change: Optional[float],
    rsi_threshold: float
) -> tuple[str, int, str]:
    """
    Returns (Action, Confidence, OI_Read)

    Relaxed SELL logic:
    - SELL signals allow:
        Short buildup OR Long unwinding
      instead of only Short buildup.
    """
    if oi_change is None:
        return "IGNORE", 0, "N/A"

    read = oi_price_read(price_change, oi_change)

    # CALL side
    if option_type == "CE":
        # BUY CALL
        if price_change > 0 and read in ("Long buildup", "Short covering") and rsi_val >= rsi_threshold:
            conf = _confidence_score(
                rsi_val, rsi_threshold, st_dir, price_change, oi_change,
                favorable_reads=("Long buildup", "Short covering"),
                require_st=1
            )
            return "BUY CALL", conf, read

        # RELAXED SELL CALL
        if st_dir == -1 and price_change < 0 and read in ("Short buildup", "Long unwinding"):
            conf = _confidence_score(
                rsi_val, rsi_threshold, st_dir, price_change, oi_change,
                favorable_reads=("Short buildup", "Long unwinding"),
                require_st=-1
            )
            conf = max(conf, 40)
            return "SELL CALL", conf, read

    # PUT side
    if option_type == "PE":
        # BUY PUT
        if price_change > 0 and read in ("Long buildup", "Short covering") and rsi_val >= rsi_threshold:
            conf = _confidence_score(
                rsi_val, rsi_threshold, st_dir, price_change, oi_change,
                favorable_reads=("Long buildup", "Short covering"),
                require_st=1
            )
            return "BUY PUT", conf, read

        # RELAXED SELL PUT
        if st_dir == -1 and price_change < 0 and read in ("Short buildup", "Long unwinding"):
            conf = _confidence_score(
                rsi_val, rsi_threshold, st_dir, price_change, oi_change,
                favorable_reads=("Short buildup", "Long unwinding"),
                require_st=-1
            )
            conf = max(conf, 40)
            return "SELL PUT", conf, read

    return "IGNORE", 0, read


# -----------------------
# Scanner: Diagnostic (RSI+ST for ALL scanned)
# -----------------------
def scan_options_with_indicators(
    kite,
    underlying: str,
    expiry_date: str,
    interval: str = "5minute",
    lookback_days: int = 5,
    rsi_period: int = 14,
    rsi_threshold: float = 60.0,
    st_period: int = 10,
    st_mult: float = 3.0,
    use_atm_filter: bool = True,
    atm_steps: int = 10,
) -> pd.DataFrame:
    """
    Diagnostic table with indicators for each scanned option.
    Returns:
      Symbol | Strike | Type | ST Trend | RSI | Match
    """
    instruments_df = fetch_instruments_df(kite)
    metas = filter_options_for_expiry(instruments_df, underlying, expiry_date)

    if use_atm_filter and metas:
        metas, _ = _apply_atm_filter(kite, metas, underlying, atm_steps)

    rows = []
    for meta in sorted(metas, key=lambda x: (x.strike, x.option_type)):
        try:
            df = fetch_candles(
                kite=kite,
                instrument_token=meta.instrument_token,
                interval=interval,
                lookback_days=lookback_days
            )

            rsi_val, st_dir = analyze_option(df, rsi_period, st_period, st_mult)
            if rsi_val is None or st_dir is None:
                continue

            st_trend = "Down" if st_dir == -1 else "Up"
            match = (st_dir == -1) and (rsi_val > rsi_threshold)

            rows.append({
                "Symbol": meta.tradingsymbol,
                "Strike": meta.strike,
                "Type": meta.option_type,
                "ST Trend": st_trend,
                "RSI": round(rsi_val, 2),
                "Match": match,
            })
        except Exception:
            continue

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(by=["Strike", "Type"]).reset_index(drop=True)
    return result


# -----------------------
# Scanner: Trade Ideas (OI + Price Δ + Action + Confidence)
# -----------------------
def scan_options_trade_ideas(
    kite,
    underlying: str,
    expiry_date: str,
    interval: str = "5minute",
    lookback_days: int = 5,
    rsi_period: int = 14,
    rsi_threshold: float = 60.0,
    st_period: int = 10,
    st_mult: float = 3.0,
    use_atm_filter: bool = True,
    atm_steps: int = 10,
) -> pd.DataFrame:
    """
    Trade-ideas scanner based on option-level:
      - RSI
      - Supertrend direction
      - Price Δ (latest close - previous close)
      - OI Δ (latest OI - previous OI)
      - OI Read (classic grid)
      - Action tag
      - Action Confidence

    Returns:
      Symbol | Strike | Type | ST Trend | RSI | Price Δ | OI Δ | OI Read | Action | Action Confidence
    """
    instruments_df = fetch_instruments_df(kite)
    metas = filter_options_for_expiry(instruments_df, underlying, expiry_date)

    if use_atm_filter and metas:
        metas, _ = _apply_atm_filter(kite, metas, underlying, atm_steps)

    rows = []

    for meta in sorted(metas, key=lambda x: (x.strike, x.option_type)):
        try:
            df = fetch_candles_with_oi(
                kite=kite,
                instrument_token=meta.instrument_token,
                interval=interval,
                lookback_days=lookback_days
            )
            if df.empty:
                continue

            min_len = max(rsi_period, st_period) + 5
            if len(df) < min_len:
                continue

            df = df.copy()
            df["rsi"] = compute_rsi(df["close"], rsi_period)
            df = compute_supertrend(df, st_period, st_mult)

            latest = df.iloc[-1]
            prev = df.iloc[-2]

            rsi_val = latest.get("rsi", None)
            st_dir = latest.get("st_direction", None)

            if rsi_val is None or pd.isna(rsi_val) or st_dir is None:
                continue

            price_change = float(latest["close"]) - float(prev["close"])

            oi_change = None
            if "open_interest" in df.columns:
                oi_latest = latest.get("open_interest")
                oi_prev = prev.get("open_interest")
                if pd.notna(oi_latest) and pd.notna(oi_prev):
                    oi_change = float(oi_latest) - float(oi_prev)

            st_trend = "Down" if int(st_dir) == -1 else "Up"

            action, confidence, read = classify_action(
                option_type=meta.option_type,
                st_dir=int(st_dir),
                rsi_val=float(rsi_val),
                price_change=price_change,
                oi_change=oi_change,
                rsi_threshold=rsi_threshold
            )

            rows.append({
                "Symbol": meta.tradingsymbol,
                "Strike": meta.strike,
                "Type": meta.option_type,
                "ST Trend": st_trend,
                "RSI": round(float(rsi_val), 2),
                "Price Δ": round(price_change, 2),
                "OI Δ": round(oi_change, 0) if oi_change is not None else None,
                "OI Read": read,
                "Action": action,
                "Action Confidence": confidence,
            })

        except Exception:
            continue

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(by=["Strike", "Type"]).reset_index(drop=True)
    return result
