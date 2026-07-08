"""
Rule-based match explanation — reads model JSON, never invents data.

Optional LLM layer can use SYSTEM_PROMPT + to_llm_payload() output.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

from intelligence.match_analyzer import MatchIntelligence
from intelligence.report import to_llm_payload


def explain_from_payload(payload: Dict) -> str:
    """Generate human-readable analysis from LLM-ready JSON."""
    match = payload.get("match", "Unknown match")
    surface = payload.get("surface", "hard")
    probs = payload.get("model_probability", {})
    fair = payload.get("fair_odds", {})
    market = payload.get("market_odds", {})
    edges = payload.get("edge", {})
    fatigue = payload.get("fatigue", {})
    ue = payload.get("ue_context", {})
    args = payload.get("key_arguments", [])
    risks = payload.get("risks", [])
    rec = payload.get("recommendation", "NO BET")
    conf = payload.get("confidence", "low")

    players = list(probs.keys())
    p1 = players[0] if players else "Player A"
    p2 = players[1] if len(players) > 1 else "Player B"

    value_side = _value_side(edges, p1, p2)
    min_odds = _minimum_odds(fair, value_side, edges)

    lines = [
        f"🎾 Match: {match}",
        f"Surface: {surface}",
        "",
        "MODEL PRICE",
        f"{p1}: {probs.get(p1, 0):.0%}",
        f"Fair odds: {fair.get(p1, 0):.2f}",
        "",
        f"{p2}: {probs.get(p2, 0):.0%}",
        f"Fair odds: {fair.get(p2, 0):.2f}",
        "",
        "MARKET",
        f"{p1}: {market.get(p1, 'n/a')}",
        f"{p2}: {market.get(p2, 'n/a')}",
        "",
        "VALUE",
    ]

    if value_side and edges.get(value_side, 0) >= 0.03:
        lines.append(
            f"{value_side} heeft mogelijk value boven {min_odds:.2f} "
            f"(edge {edges.get(value_side, 0):+.1%})."
        )
    else:
        lines.append("Geen duidelijke value — NO BET aanbevolen.")

    lines.extend(["", "BELANGRIJKSTE REDENEN"])
    for arg in args[:6] or ["Onvoldoende contextuele data"]:
        lines.append(f"- {arg}")

    lines.extend(["", "RISICO"])
    for risk in risks or ["Standaard matchvariance"]:
        lines.append(f"- {risk}")

    _append_context_lines(lines, fatigue, ue, p1, p2)

    lines.extend([
        "",
        "ACTIE",
        _action_text(rec, conf, value_side, edges),
        f"Confidence: {conf}",
    ])
    if min_odds:
        lines.append(f"Minimum odds ({value_side or 'n/a'}): {min_odds:.2f}")

    return "\n".join(lines)


def explain_match(report: Union[MatchIntelligence, Dict]) -> str:
    if isinstance(report, MatchIntelligence):
        return explain_from_payload(to_llm_payload(report))
    return explain_from_payload(report)


def _value_side(edges: Dict, p1: str, p2: str) -> Optional[str]:
    e1 = edges.get(p1, 0)
    e2 = edges.get(p2, 0)
    if e1 <= 0 and e2 <= 0:
        return None
    return p1 if e1 >= e2 else p2


def _minimum_odds(fair: Dict, side: Optional[str], edges: Dict) -> float:
    if not side or side not in fair:
        return 0.0
    return fair[side] * 1.08


def _action_text(rec: str, conf: str, side: Optional[str], edges: Dict) -> str:
    edge = edges.get(side, 0) if side else 0
    if "NO BET" in rec.upper() or edge < 0.03:
        return "NO BET — geen edge of onvoldoende data."
    stake = "0.25%" if conf == "low" else "0.50%" if conf == "medium" else "0.75%"
    rec_up = rec.upper()
    if "HEDGE" in rec_up:
        return f"SMALL VALUE + live hedge mogelijk. Stake: {stake}."
    if "LIVE WAIT" in rec_up or rec_up == "LIVE WAIT":
        return f"LIVE WAIT — wacht op betere prijs. Max stake {stake}."
    if "LIVE" in rec_up:
        return f"LIVE WAIT — kleine pre-match value mogelijk, stake {stake} max."
    return f"{rec}. Stake: {stake}."


def _append_context_lines(
    lines: List[str],
    fatigue: Dict,
    ue: Dict,
    p1: str,
    p2: str,
) -> None:
    if fatigue:
        lines.extend(["", "FATIGUE"])
        for player, data in fatigue.items():
            risk = data.get("collapse_risk", "unknown")
            lines.append(f"- {player}: collapse risk {risk}")

    if ue:
        lines.extend(["", "ERROR PROFILE"])
        for player, data in ue.items():
            risk = data.get("ue_risk", "unknown")
            expected = data.get("expected_ue_pp")
            actual = data.get("actual_ue_pp")
            detail = f"UE risk {risk}"
            if expected and actual:
                detail += f" (expected {expected:.2f}, actual {actual:.2f})"
            lines.append(f"- {player}: {detail}")