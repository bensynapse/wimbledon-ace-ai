"""Post-match player quotes — psychology signals for intelligence layer."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

QUOTES_PATH = Path(__file__).resolve().parent.parent / "data" / "intelligence" / "player_quotes.json"

SIGNAL_LABELS = {
    "confidence_high": "high confidence in press",
    "composed": "composed under pressure",
    "pressure_coping": "handles nerves well",
    "momentum": "emotional momentum / crowd energy",
    "home_crowd_boost": "home crowd advantage cited",
    "serving_well": "serve clicking",
    "experience": "big-match experience",
    "crowd_neutral": "unfazed by hostile crowd",
    "fatigue_high": "fatigue mentioned post-match",
    "injury_minor": "minor physical issue flagged",
    "injury_context": "opponent played hurt (context)",
    "mental_toughness": "mental resilience highlighted",
    "recovery_confident": "confident about recovery window",
    "low_ue_mindset": "disciplined — avoids cheap points",
    "fresh": "physically fresh path",
    "emotional_high": "elevated emotional state",
    "opponent_form": "opponent context supports form read",
}


class NewsQuotesSource:
    def __init__(self, path: Optional[Path] = None):
        self.path = path or QUOTES_PATH
        self._quotes: List[Dict] = []
        self._loaded = False

    def warm(self) -> None:
        if self._loaded:
            return
        if not self.path.exists():
            logger.warning("Player quotes file missing: %s", self.path)
            self._loaded = True
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                self._quotes = json.load(f)
        except Exception as exc:
            logger.warning("Failed to load player quotes: %s", exc)
            self._quotes = []
        self._loaded = True

    def for_player(
        self,
        player: str,
        tournament: str = "",
        max_age_days: int = 14,
    ) -> List[Dict]:
        self.warm()
        key = player.lower().strip()
        t_key = tournament.lower().strip() if tournament else ""
        out = []
        for q in self._quotes:
            if q.get("player", "").lower().strip() != key:
                continue
            if t_key and t_key not in q.get("tournament", "").lower():
                continue
            out.append(q)
        return out

    def apply_to_context(
        self,
        ctx: Dict,
        player1: str,
        player2: str,
        tournament: str = "",
    ) -> None:
        for prefix, player in (("player1", player1), ("player2", player2)):
            quotes = self.for_player(player, tournament=tournament)
            if not quotes:
                continue
            ctx[f"{prefix}_quotes"] = quotes
            signals = _aggregate_signals(quotes)
            ctx[f"{prefix}_quote_signals"] = signals
            ctx[f"{prefix}_quote_summary"] = _build_summary(player, quotes, signals)
            if _has_signal(signals, "injury_minor", "fatigue_high"):
                flags = list(ctx.get(f"{prefix}_context_flags", []))
                if _has_signal(signals, "injury_minor") and "injury_mentioned" not in flags:
                    flags.append("injury_mentioned")
                if _has_signal(signals, "fatigue_high") and "fatigue_mentioned" not in flags:
                    flags.append("fatigue_mentioned")
                ctx[f"{prefix}_context_flags"] = flags

        matchup_notes = _matchup_quote_notes(player1, player2, ctx)
        if matchup_notes:
            ctx["quote_matchup_notes"] = matchup_notes


def _aggregate_signals(quotes: List[Dict]) -> List[str]:
    seen = set()
    out = []
    for q in quotes:
        for sig in q.get("signals", []):
            if sig not in seen:
                seen.add(sig)
                out.append(sig)
    return out


def _has_signal(signals: List[str], *names: str) -> bool:
    return any(s in signals for s in names)


def _build_summary(player: str, quotes: List[Dict], signals: List[str]) -> str:
    if not quotes:
        return ""
    latest = quotes[0]
    labels = [SIGNAL_LABELS.get(s, s) for s in signals[:4]]
    q_short = latest.get("quote", "")[:120]
    if len(latest.get("quote", "")) > 120:
        q_short += "…"
    parts = [f'{player} ({latest.get("source", "press")}): "{q_short}"']
    if labels:
        parts.append("Signals: " + ", ".join(labels))
    if latest.get("stats_note"):
        parts.append(latest["stats_note"])
    return " | ".join(parts)


def _matchup_quote_notes(p1: str, p2: str, ctx: Dict) -> List[str]:
    notes = []
    p1_sigs = set(ctx.get("player1_quote_signals", []))
    p2_sigs = set(ctx.get("player2_quote_signals", []))

    if "home_crowd_boost" in p1_sigs or "momentum" in p1_sigs:
        if "crowd_neutral" in p2_sigs:
            notes.append(f"{p2} says crowd noise won't affect him — test vs {p1} home support")

    if "fatigue_high" in p1_sigs and "fresh" in p2_sigs:
        notes.append(f"{p1} flagged fatigue; {p2} reports fresh — rest edge to {p2}")
    elif "fatigue_high" in p2_sigs and "fresh" in p1_sigs:
        notes.append(f"{p2} flagged fatigue; {p1} reports fresh — rest edge to {p1}")

    if "injury_minor" in p1_sigs:
        notes.append(f"{p1} mentioned minor injury — monitor movement in warm-up")
    if "injury_minor" in p2_sigs:
        notes.append(f"{p2} mentioned minor injury — monitor movement in warm-up")

    if "low_ue_mindset" in p1_sigs and "serving_well" in p2_sigs:
        notes.append("Press contrast: disciplined ball-striker vs confident server")
    elif "low_ue_mindset" in p2_sigs and "serving_well" in p1_sigs:
        notes.append("Press contrast: disciplined ball-striker vs confident server")

    return notes