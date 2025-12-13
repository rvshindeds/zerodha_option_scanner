# trade_manager.py
import uuid
import datetime as dt
from typing import Dict, Tuple, List, Optional

# -----------------------------
# State helpers
# -----------------------------
def init_trade_state(st):
    if "open_trades" not in st.session_state:
        st.session_state["open_trades"] = []

def _new_trade_id() -> str:
    return str(uuid.uuid4())[:8]

def add_trade(st, trade: Dict):
    """
    trade must include:
      symbol, instrument_token, side (BUY/SELL), qty, entry_price
    optional:
      sl_pct, tp_pct, max_hold_minutes, square_off_time, note
    """
    trade = dict(trade)
    trade["trade_id"] = _new_trade_id()
    trade["entry_time"] = trade.get("entry_time") or dt.datetime.now()
    trade["status"] = "OPEN"
    trade["partial_done"] = bool(trade.get("partial_done", False))
    trade["note"] = trade.get("note", "")
    st.session_state["open_trades"].append(trade)

def mark_partial_done(st, trade_id: str):
    for t in st.session_state.get("open_trades", []):
        if t.get("trade_id") == trade_id:
            t["partial_done"] = True
            return True
    return False

def mark_exited(st, trade_id: str):
    for t in st.session_state.get("open_trades", []):
        if t.get("trade_id") == trade_id:
            t["status"] = "EXITED"
            t["exit_mark_time"] = dt.datetime.now()
            return True
    return False

def purge_exited(st):
    st.session_state["open_trades"] = [
        t for t in st.session_state.get("open_trades", [])
        if t.get("status") != "EXITED"
    ]


# -----------------------------
# Market data helpers
# -----------------------------
def safe_quote_by_token(kite, token: int) -> Dict:
    try:
        q = kite.quote([token])
        return q.get(str(token)) or q.get(token) or {}
    except Exception:
        return {}

def safe_ltp_by_token(kite, token: int) -> Optional[float]:
    q = safe_quote_by_token(kite, token)
    try:
        return float(q.get("last_price", 0) or 0)
    except Exception:
        return None

def safe_historical(kite, token: int, interval: str, lookback_days: int, oi: bool = False) -> List[Dict]:
    to_dt = dt.datetime.now()
    from_dt = to_dt - dt.timedelta(days=lookback_days)
    try:
        return kite.historical_data(
            instrument_token=token,
            from_date=from_dt,
            to_date=to_dt,
            interval=interval,
            continuous=False,
            oi=oi,
        ) or []
    except Exception:
        return []


# -----------------------------
# Exit logic
# -----------------------------
def compute_hard_exit(trade: Dict, ltp: float, now: dt.datetime) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Hard exits override everything.
    """
    entry = float(trade["entry_price"])
    side = str(trade["side"]).upper()
    sl_pct = float(trade.get("sl_pct", 30.0)) / 100.0
    tp_pct = float(trade.get("tp_pct", 40.0)) / 100.0

    # Time square-off
    sq_time = trade.get("square_off_time")  # datetime.time
    if sq_time is not None and now.time() >= sq_time:
        return "EXIT NOW", 95, "Time square-off reached"

    if side == "SELL":
        if ltp >= entry * (1 + sl_pct):
            return "EXIT NOW", 100, "Premium SL hit"
        if ltp <= entry * (1 - tp_pct):
            return "EXIT NOW", 90, "Target achieved"
    else:  # BUY
        if ltp <= entry * (1 - sl_pct):
            return "EXIT NOW", 100, "Premium SL hit"
        if ltp >= entry * (1 + tp_pct):
            return "EXIT NOW", 90, "Target achieved"

    return None, None, None


def recommend_square_off(
    trade: Dict,
    ltp: float,
    spot_trend: str = "Unknown",
    option_trend: str = "Unknown",
    oi_read: str = "Neutral",
    now: Optional[dt.datetime] = None,
) -> Tuple[str, int, List[str]]:
    """
    Returns: (action, confidence_score, reasons[])
    """
    now = now or dt.datetime.now()

    # Hard exits first
    action, conf, reason = compute_hard_exit(trade, ltp, now)
    if action:
        return action, conf or 90, [reason] if reason else []

    score = 0
    reasons = []

    side = str(trade.get("side", "")).upper()

    def is_against(tr: str) -> bool:
        tr = (tr or "Unknown").title()
        if tr not in ("Up", "Down"):
            return False
        # For SELL, we don't want underlying/option to trend UP strongly (premium expansion risk)
        if side == "SELL":
            return tr == "Up"
        # For BUY, we don't want trend DOWN against our long premium
        return tr == "Down"

    if is_against(spot_trend):
        score += 30
        reasons.append("SPOT trend flipped against position")

    if is_against(option_trend):
        score += 25
        reasons.append("Option trend flipped against position")

    if oi_read in ("Short buildup", "Long unwinding"):
        score += 20
        reasons.append(f"OI adverse: {oi_read}")

    entry_time = trade.get("entry_time")
    max_hold = float(trade.get("max_hold_minutes", 120))
    if isinstance(entry_time, dt.datetime):
        hold_mins = (now - entry_time).total_seconds() / 60.0
        if max_hold > 0 and hold_mins >= max_hold * 0.75:
            score += 10
            reasons.append("Late in holding window")

    # Decision thresholds
    partial_done = bool(trade.get("partial_done", False))

    if score >= 70:
        return "EXIT NOW", score, reasons
    if score >= 50 and not partial_done:
        return "PARTIAL EXIT (50%)", score, reasons
    if score >= 35:
        return "TIGHTEN SL", score, reasons

    return "HOLD", max(score, 10), reasons
