"""WimbledonAce match intelligence reports — rule-based, LLM-ready."""

from __future__ import annotations

import json
from typing import Dict

from intelligence.match_analyzer import MatchIntelligence


def format_report(report: MatchIntelligence) -> str:
    lines = [
        f"🎾 {report.match}",
        f"Tournament: {report.tournament or 'N/A'} | Surface: {report.surface}",
        "",
        "1. Model price",
        f"   {report.player1}: {report.model_prob_p1:.0%} → fair odds {report.fair_odds_p1:.2f}",
        f"   {report.player2}: {report.model_prob_p2:.0%} → fair odds {report.fair_odds_p2:.2f}",
        "",
        "2. Market",
        f"   {report.player1}: {report.market_odds_p1 or 'n/a'}",
        f"   {report.player2}: {report.market_odds_p2 or 'n/a'}",
        "",
        "3. Value",
        f"   {report.player1} edge: {report.edge_p1:+.1%} (min odds {report.minimum_odds_p1:.2f})",
        f"   {report.player2} edge: {report.edge_p2:+.1%} (min odds {report.minimum_odds_p2:.2f})",
        "",
        "4. Factor table",
    ]
    for f in report.factor_table:
        lines.append(f"   {f.name:16s} {f.edge:4s}  {f.detail}")

    lines.extend([
        "",
        "5. UE / Fatigue context",
        f"   {report.player1}: UE risk {report.ue_analysis_p1.get('ue_risk')} | "
        f"fatigue {report.fatigue_p1.get('collapse_risk')}",
        f"   {report.player2}: UE risk {report.ue_analysis_p2.get('ue_risk')} | "
        f"fatigue {report.fatigue_p2.get('collapse_risk')}",
    ])

    if report.weather_notes:
        lines.extend(["", "6. Weather / court conditions"])
        for n in report.weather_notes:
            lines.append(f"   - {n}")

    if report.quote_notes:
        lines.extend(["", "7. Press / player quotes"])
        for n in report.quote_notes:
            lines.append(f"   - {n}")

    lines.extend([
        "",
        "8. Data confidence",
        f"   {json.dumps(report.data_confidence)}",
        "",
        "9. Key arguments",
    ])
    for arg in report.key_arguments:
        lines.append(f"   - {arg}")

    lines.extend(["", "10. Risks"])
    for r in report.risks or ["Standard match variance"]:
        lines.append(f"   - {r}")

    if report.live_notes:
        lines.extend(["", "11. Live notes"])
        for n in report.live_notes:
            lines.append(f"   - {n}")

    lines.extend([
        "",
        f"ADVICE: {report.recommended_action}",
        f"Stake: {report.stake_pct_range} | Confidence: {report.confidence}",
    ])
    return "\n".join(lines)


def to_llm_payload(report: MatchIntelligence) -> Dict:
    """JSON payload for optional LLM explanation layer — model does NOT guess."""
    return {
        "match": report.match,
        "surface": report.surface,
        "model_probability": {
            report.player1: report.model_prob_p1,
            report.player2: report.model_prob_p2,
        },
        "fair_odds": {
            report.player1: report.fair_odds_p1,
            report.player2: report.fair_odds_p2,
        },
        "market_odds": {
            report.player1: report.market_odds_p1,
            report.player2: report.market_odds_p2,
        },
        "edge": {
            report.player1: report.edge_p1,
            report.player2: report.edge_p2,
        },
        "ue_context": {
            report.player1: report.ue_analysis_p1,
            report.player2: report.ue_analysis_p2,
        },
        "fatigue": {
            report.player1: report.fatigue_p1,
            report.player2: report.fatigue_p2,
        },
        "data_confidence": report.data_confidence,
        "recommendation": report.recommended_action,
        "confidence": report.confidence,
        "key_arguments": report.key_arguments,
        "risks": report.risks,
        "weather_notes": report.weather_notes,
        "quote_notes": report.quote_notes,
    }