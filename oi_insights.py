# oi_insights.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd


@dataclass
class OIStats:
    underlying: str
    expiry: str
    spot: float

    highest_ce_strike: Optional[float]
    highest_ce_oi: Optional[int]

    highest_pe_strike: Optional[float]
    highest_pe_oi: Optional[int]

    total_ce_oi: int
    total_pe_oi: int
    pcr_oi: Optional[float]

    iv_hint: Optional[str] = None  # we’ll just store a qualitative note if we want


def _build_option_chain_from_kite(
    kite,
    underlying: str,
    expiry: str,
) -> Tuple[pd.DataFrame, float]:
    """
    Build a basic option-chain DataFrame for a given index + expiry using Kite.

    Columns returned: ['tradingsymbol', 'strike', 'instrument_type', 'oi', 'iv', 'last_price'].

    NOTE:
    - This uses `kite.instruments("NFO")` + `kite.quote`.
    - It is intentionally conservative: we pull only options for the given index + expiry.
    """

    # 1) Get all NFO instruments and filter for our underlying + expiry
    instruments = kite.instruments("NFO")
    rows = []
    for inst in instruments:
        if inst.get("segment") != "NFO-OPT":
            continue

        # Underlying name (e.g. NIFTY, BANKNIFTY, FINNIFTY etc.)
        if inst.get("name") != underlying:
            continue

        # Expiry filter
        if str(inst.get("expiry"))[:10] != expiry:
            continue

        rows.append(inst)

    if not rows:
        return pd.DataFrame(), 0.0

    inst_df = pd.DataFrame(rows)
    symbols = inst_df["tradingsymbol"].tolist()

    # 2) Pull live quotes (OI, IV etc.)
    quotes = kite.quote(symbols)

    data = []
    for sym in symbols:
        q = quotes.get(sym, {})
        oi = q.get("oi")
        last_price = q.get("last_price")
        iv = None
        try:
            # Some brokers embed IV in "greeks" or "depth" – if not, we keep None.
            iv = q.get("greeks", {}).get("iv")
        except Exception:
            iv = None

        row = inst_df.loc[inst_df["tradingsymbol"] == sym].iloc[0]
        data.append(
            {
                "tradingsymbol": sym,
                "strike": float(row["strike"]),
                "instrument_type": row["instrument_type"],  # CE / PE
                "oi": int(oi) if oi is not None else 0,
                "iv": iv,
                "last_price": float(last_price) if last_price is not None else 0.0,
            }
        )

    chain_df = pd.DataFrame(data)

    # 3) Spot from underlying index future/spot symbol
    # You already compute/know spot in your main app, but we still return a fallback here.
    # If you have spot handy, you can override it.
    spot = 0.0
    try:
        if underlying.upper() == "NIFTY":
            spot_sym = "NSE:NIFTY 50"
        elif underlying.upper() == "BANKNIFTY":
            spot_sym = "NSE:NIFTY BANK"
        else:
            spot_sym = f"NSE:{underlying}"

        spot_q = kite.quote([spot_sym])
        spot = float(list(spot_q.values())[0]["last_price"])
    except Exception:
        spot = 0.0

    return chain_df, spot


def compute_oi_stats(
    kite,
    underlying: str,
    expiry: str,
    spot_override: Optional[float] = None,
) -> Tuple[Optional[OIStats], Optional[pd.DataFrame]]:
    """
    High-level helper:
    - Fetch option chain
    - Compute key OI stats
    - Return OIStats dataclass + the underlying chain DataFrame
    """

    chain_df, spot_from_chain = _build_option_chain_from_kite(kite, underlying, expiry)
    if chain_df.empty:
        return None, chain_df

    spot = float(spot_override) if spot_override else float(spot_from_chain)

    ce = chain_df[chain_df["instrument_type"] == "CE"].copy()
    pe = chain_df[chain_df["instrument_type"] == "PE"].copy()

    highest_ce_strike = None
    highest_ce_oi = None
    if not ce.empty:
        idx = ce["oi"].idxmax()
        highest_ce_strike = float(ce.loc[idx, "strike"])
        highest_ce_oi = int(ce.loc[idx, "oi"])

    highest_pe_strike = None
    highest_pe_oi = None
    if not pe.empty:
        idx = pe["oi"].idxmax()
        highest_pe_strike = float(pe.loc[idx, "strike"])
        highest_pe_oi = int(pe.loc[idx, "oi"])

    total_ce_oi = int(ce["oi"].sum()) if not ce.empty else 0
    total_pe_oi = int(pe["oi"].sum()) if not pe.empty else 0

    pcr_oi = None
    if total_ce_oi > 0:
        pcr_oi = total_pe_oi / total_ce_oi

    stats = OIStats(
        underlying=underlying,
        expiry=expiry,
        spot=spot,
        highest_ce_strike=highest_ce_strike,
        highest_ce_oi=highest_ce_oi,
        highest_pe_strike=highest_pe_strike,
        highest_pe_oi=highest_pe_oi,
        total_ce_oi=total_ce_oi,
        total_pe_oi=total_pe_oi,
        pcr_oi=pcr_oi,
        iv_hint=None,
    )

    return stats, chain_df


def build_directional_view(stats: OIStats) -> str:
    """
    Turn OIStats into a short directional narrative.
    This is a deterministic mapping of your framework.
    """

    if not stats or stats.pcr_oi is None:
        return "Not enough OI data to form a directional view."

    pcr = stats.pcr_oi
    lines = []

    # PCR based view
    if pcr > 1.2:
        lines.append(
            f"PCR (OI) is {pcr:.2f} (>1.2): put OI dominates, "
            f"suggesting supportive flows from put writers — slightly bullish-to-neutral bias."
        )
    elif pcr < 0.8:
        lines.append(
            f"PCR (OI) is {pcr:.2f} (<0.8): call OI dominates, "
            f"indicating overhead supply from call writers — slightly bearish bias."
        )
    else:
        lines.append(
            f"PCR (OI) is {pcr:.2f} (~1): put and call OI are relatively balanced — neutral / range-bound bias."
        )

    if stats.highest_ce_strike:
        lines.append(
            f"Highest CE OI at {stats.highest_ce_strike:.0f} → likely resistance zone from call writers."
        )
    if stats.highest_pe_strike:
        lines.append(
            f"Highest PE OI at {stats.highest_pe_strike:.0f} → likely support zone from put writers."
        )

    return " ".join(lines)


def build_trade_structures_text(stats: Optional[OIStats]) -> Dict[str, str]:
    """
    Returns text templates for:
    - Short Call at Resistance
    - Short Strangle / Iron Condor
    - Short Put near Support
    with the actual strikes plugged in (where possible).
    """

    if not stats:
        return {}

    ce_strike = stats.highest_ce_strike
    pe_strike = stats.highest_pe_strike
    mid_zone = None
    if ce_strike and pe_strike:
        mid_zone = (ce_strike + pe_strike) / 2.0

    setup_a = (
        f"**Setup A — Short Call at Resistance**  \n"
        f"- Candidate CE strike: **{ce_strike:.0f}** (highest CE OI, potential resistance).  \n"
        f"- Entry: Consider SELLING this OTM call if spot trades below this zone with weak momentum.  \n"
        f"- Invalidation: Spot closes convincingly above this strike (or next strike) with volume + rising IV.  \n"
        f"- Profit-taking: Target 40–60% premium decay or a drift back toward the middle of the OI cluster."
        if ce_strike
        else "Setup A — Short Call at Resistance: CE OI data not available."
    )

    setup_b = (
        f"**Setup B — Short Strangle / Iron Condor (Range-Bound)**  \n"
        f"- Upper call leg: around **{ce_strike:.0f} CE**.  \n"
        f"- Lower put leg: around **{pe_strike:.0f} PE**.  \n"
        f"- Spot is approx **{stats.spot:.0f}**, mid-zone near **{mid_zone:.0f}**.  \n"
        f"- Entry: When price oscillates near the middle of this band and PCR is around 1.  \n"
        f"- Invalidation: Strong breakout beyond either wing with volume or IV spike.  \n"
        f"- Profit-taking: Close once 30–50% of total credit is captured."
        if ce_strike and pe_strike
        else "Setup B — Short Strangle / Iron Condor: need both CE and PE OI clusters."
    )

    setup_c = (
        f"**Setup C — Short Put near Support**  \n"
        f"- Candidate PE strike: **{pe_strike:.0f}** (highest PE OI, potential support).  \n"
        f"- Entry: Consider SELLING this OTM put if intraday price bounces from lower levels and PCR>1.  \n"
        f"- Invalidation: Spot breaks below this strike with rising volume and IV.  \n"
        f"- Profit-taking: Exit on sizeable premium decay or drift back toward mid-zone."
        if pe_strike
        else "Setup C — Short Put near Support: PE OI data not available."
    )

    return {
        "setup_a": setup_a,
        "setup_b": setup_b,
        "setup_c": setup_c,
    }
