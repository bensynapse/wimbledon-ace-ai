"""
Tennis Market Intelligence — full match factor analysis.

Pipeline: Data → model probability → fair odds → value → contextual flags → report
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from intelligence.confidence import ConfidenceLevel, DataConfidence
from intelligence.expected_ue import ExpectedUEModel, PlayerShotProfile
from intelligence.fatigue import FatigueInput, FatigueModel


@dataclass
class FactorScore:
    """Single factor assessment for one player."""
    name: str
    player1: str
    player2: str
    edge: str  # ++ | + | = | - | --
    detail: str


@dataclass
class MatchIntelligence:
    """Complete intelligence output for one match."""
    match: str
    tournament: str
    surface: str
    player1: str
    player2: str
    model_prob_p1: float
    model_prob_p2: float
    fair_odds_p1: float
    fair_odds_p2: float
    market_odds_p1: float
    market_odds_p2: float
    edge_p1: float
    edge_p2: float
    minimum_odds_p1: float
    minimum_odds_p2: float
    confidence: str
    recommended_action: str
    stake_pct_range: str
    factor_table: List[FactorScore]
    ue_analysis_p1: Dict
    ue_analysis_p2: Dict
    fatigue_p1: Dict
    fatigue_p2: Dict
    data_confidence: Dict
    key_arguments: List[str]
    risks: List[str]
    live_notes: List[str] = field(default_factory=list)
    weather_notes: List[str] = field(default_factory=list)
    quote_notes: List[str] = field(default_factory=list)


class MatchIntelligenceAnalyzer:
    """
    Combines Elo/model output with contextual UE, fatigue and market data.
    AI layer should only explain this output — never invent probabilities.
    """

    def __init__(self, min_edge: float = 0.05, min_odds_buffer: float = 0.08):
        self.min_edge = min_edge
        self.min_odds_buffer = min_odds_buffer
        self.ue_model = ExpectedUEModel()
        self.fatigue_model = FatigueModel()

    def analyze(
        self,
        player1: str,
        player2: str,
        surface: str,
        model_prob_p1: float,
        market_odds_p1: float = 0.0,
        market_odds_p2: float = 0.0,
        tournament: str = "",
        context: Optional[Dict] = None,
    ) -> MatchIntelligence:
        ctx = context or {}
        prob_p2 = 1.0 - model_prob_p1
        fair_p1 = 1.0 / max(model_prob_p1, 0.01)
        fair_p2 = 1.0 / max(prob_p2, 0.01)

        implied_p1 = 1.0 / market_odds_p1 if market_odds_p1 > 1 else 0.0
        implied_p2 = 1.0 / market_odds_p2 if market_odds_p2 > 1 else 0.0
        edge_p1 = model_prob_p1 - implied_p1 if implied_p1 else 0.0
        edge_p2 = prob_p2 - implied_p2 if implied_p2 else 0.0

        min_odds_p1 = fair_p1 * (1 + self.min_odds_buffer)
        min_odds_p2 = fair_p2 * (1 + self.min_odds_buffer)

        ue_p1 = self._ue_for_player(player1, surface, ctx, "player1")
        ue_p2 = self._ue_for_player(player2, surface, ctx, "player2")
        fat_p1 = self._fatigue_for_player(player1, ctx, "player1")
        fat_p2 = self._fatigue_for_player(player2, ctx, "player2")

        confidence = self._build_confidence(ctx)
        factors = self._build_factor_table(
            player1, player2, surface, ctx, ue_p1, ue_p2, fat_p1, fat_p2, edge_p1, edge_p2
        )
        weather_notes, quote_notes = self._extract_context_notes(ctx)
        action, stake, args, risks, live = self._decide(
            player1, player2, edge_p1, edge_p2, min_odds_p1, min_odds_p2,
            market_odds_p1, market_odds_p2, ue_p1, ue_p2, fat_p1, fat_p2, confidence, ctx,
            weather_notes, quote_notes,
        )

        return MatchIntelligence(
            match=f"{player1} vs {player2}",
            tournament=tournament or ctx.get("tournament", ""),
            surface=surface,
            player1=player1,
            player2=player2,
            model_prob_p1=round(model_prob_p1, 4),
            model_prob_p2=round(prob_p2, 4),
            fair_odds_p1=round(fair_p1, 2),
            fair_odds_p2=round(fair_p2, 2),
            market_odds_p1=market_odds_p1,
            market_odds_p2=market_odds_p2,
            edge_p1=round(edge_p1, 4),
            edge_p2=round(edge_p2, 4),
            minimum_odds_p1=round(min_odds_p1, 2),
            minimum_odds_p2=round(min_odds_p2, 2),
            confidence=confidence,
            recommended_action=action,
            stake_pct_range=stake,
            factor_table=factors,
            ue_analysis_p1=self._ue_to_dict(ue_p1),
            ue_analysis_p2=self._ue_to_dict(ue_p2),
            fatigue_p1=self._fatigue_to_dict(fat_p1),
            fatigue_p2=self._fatigue_to_dict(fat_p2),
            data_confidence=ctx.get("data_confidence", DataConfidence().to_dict()),
            key_arguments=args,
            risks=risks,
            live_notes=live,
            weather_notes=weather_notes,
            quote_notes=quote_notes,
        )

    def _ue_for_player(self, player: str, surface: str, ctx: Dict, key: str):
        prefix = f"{key}_"
        profile = None
        if ctx.get(f"{prefix}ue_pp") or ctx.get(f"{prefix}w_pp"):
            profile = PlayerShotProfile(
                ue_per_point=ctx.get(f"{prefix}ue_pp", 0.19),
                winners_per_point=ctx.get(f"{prefix}w_pp", 0.17),
                source=ctx.get(f"{prefix}ue_source", "estimated"),
            )
        return self.ue_model.analyze(
            player=player,
            surface=surface,
            profile=profile,
            actual_ue_pp=ctx.get(f"{prefix}actual_ue_pp"),
            actual_w_pp=ctx.get(f"{prefix}actual_w_pp"),
            opponent_tags=ctx.get(f"{prefix}opponent_tags", ctx.get("opponent_tags_p2" if key == "player1" else "opponent_tags_p1", [])),
            context_flags=ctx.get(f"{prefix}context_flags", []),
        )

    def _fatigue_for_player(self, player: str, ctx: Dict, key: str):
        prefix = f"{key}_"
        return self.fatigue_model.analyze(
            player,
            FatigueInput(
                minutes_last_7_days=ctx.get(f"{prefix}minutes_7d", 0),
                sets_last_7_days=ctx.get(f"{prefix}sets_7d", 0),
                matches_last_7_days=ctx.get(f"{prefix}matches_7d", 0),
                rest_days=ctx.get(f"{prefix}rest_days", 2),
                five_set_recent=ctx.get(f"{prefix}five_set_recent", False),
                medical_timeout_recent=ctx.get(f"{prefix}medical_timeout", False),
                retirement_recent=ctx.get(f"{prefix}retirement_recent", False),
                age=ctx.get(f"{prefix}age", 27),
                temperature_c=ctx.get("temperature_c"),
                humidity_pct=ctx.get("humidity_pct"),
                wind_kmh=ctx.get("wind_kmh"),
            ),
        )

    def _build_confidence(self, ctx: Dict) -> str:
        dc = DataConfidence()
        if ctx.get("odds_player1") or ctx.get("market_odds_p1"):
            dc.odds = ConfidenceLevel.HIGH
        if ctx.get("player1_actual_ue_pp"):
            dc.ue_stats = ConfidenceLevel.MEDIUM
        if ctx.get("player1_minutes_7d"):
            dc.fatigue = ConfidenceLevel.HIGH
        if ctx.get("temperature_c"):
            dc.weather = ConfidenceLevel.MEDIUM
        if ctx.get("injury_flag"):
            dc.injury = ConfidenceLevel.LOW
        score = dc.overall_score()
        ctx["data_confidence"] = dc.to_dict()
        if score >= 0.7:
            return "high"
        if score >= 0.45:
            return "medium"
        return "low"

    def _build_factor_table(self, p1, p2, surface, ctx, ue1, ue2, f1, f2, e1, e2):
        def edge_label(diff: float) -> str:
            if diff > 0.08:
                return "++"
            if diff > 0.03:
                return "+"
            if diff < -0.08:
                return "--"
            if diff < -0.03:
                return "-"
            return "="

        rows = [
            FactorScore("Surface Elo", p1, p2, edge_label(ctx.get("surface_elo_diff", 0) / 200), f"diff {ctx.get('surface_elo_diff', 0):+.0f}"),
            FactorScore("UE risk", p1, p2, self._ue_edge(ue1.ue_risk, ue2.ue_risk), f"{p1} {ue1.ue_risk} / {p2} {ue2.ue_risk}"),
            FactorScore("Fatigue", p1, p2, self._fatigue_edge(f1.collapse_risk, f2.collapse_risk), f"{p1} {f1.collapse_risk} / {p2} {f2.collapse_risk}"),
            FactorScore("Market value", p1, p2, edge_label(e1 - e2), f"edge {e1:+.1%} / {e2:+.1%}"),
        ]
        if ctx.get("weather_summary"):
            w_edge = self._weather_edge(ctx, p1, p2)
            rows.append(FactorScore(
                "Weather",
                p1, p2, w_edge,
                ctx.get("weather_summary", ""),
            ))
        return rows

    def _weather_edge(self, ctx: Dict, p1: str, p2: str) -> str:
        """Rough edge from wind/heat — favors flatter hitters & fresh legs in heat."""
        wind = ctx.get("weather_wind_level", "low")
        heat = ctx.get("weather_heat_level", "low")
        p1_fatigue = ctx.get("player1_fatigue_score", 0)
        p2_fatigue = ctx.get("player2_fatigue_score", 0)
        if heat in ("high", "extreme") and abs(p1_fatigue - p2_fatigue) > 0.15:
            return "+" if p1_fatigue < p2_fatigue else "-"
        if wind == "high":
            return "="
        return "="

    def _extract_context_notes(self, ctx: Dict) -> tuple:
        weather = []
        if ctx.get("weather_summary"):
            weather.append(ctx["weather_summary"])
        weather.extend(ctx.get("weather_impact_notes", []))

        quotes = []
        for key in ("player1_quote_summary", "player2_quote_summary"):
            if ctx.get(key):
                quotes.append(ctx[key])
        quotes.extend(ctx.get("quote_matchup_notes", []))
        return weather, quotes

    def _ue_edge(self, r1: str, r2: str) -> str:
        order = {"low": 0, "medium": 1, "high": 2}
        if order[r1] < order[r2]:
            return "+"
        if order[r1] > order[r2]:
            return "-"
        return "="

    def _fatigue_edge(self, r1: str, r2: str) -> str:
        order = {"low": 0, "medium": 1, "high": 2}
        if order[r1] < order[r2]:
            return "+"
        if order[r1] > order[r2]:
            return "-"
        return "="

    def _decide(self, p1, p2, e1, e2, min1, min2, o1, o2, ue1, ue2, f1, f2, conf, ctx, weather_notes=None, quote_notes=None):
        args, risks, live = [], [], []
        weather_notes = weather_notes or []
        quote_notes = quote_notes or []

        best_edge = max(e1, e2)
        best_player = p1 if e1 >= e2 else p2
        best_odds = o1 if e1 >= e2 else o2
        best_min = min1 if e1 >= e2 else min2

        if ctx.get("path_diff"):
            args.append(ctx["path_diff"])
        for note in weather_notes[:2]:
            if note not in args:
                args.append(f"Weather: {note}")
        for note in quote_notes[:3]:
            if note not in args:
                args.append(note)
        for note_key in ("player1_error_profile_note", "player2_error_profile_note"):
            if ctx.get(note_key):
                args.append(ctx[note_key])
        for sig in ue1.signals + ue2.signals + f1.signals + f2.signals:
            if sig not in args:
                args.append(sig)

        if f1.collapse_risk == "high" or f2.collapse_risk == "high":
            risks.append("Fatigue/heat collapse possible in long match")
        if ctx.get("weather_heat_level") in ("high", "extreme"):
            risks.append(f"Heat stress ({ctx.get('temperature_c', '?')}°C) — longer rallies favor fitter player")
        if ctx.get("weather_wind_level") == "high":
            risks.append("Strong gusts — UE spikes likely on serve toss and net approaches")
        if ue1.ue_risk == "high" or ue2.ue_risk == "high":
            risks.append("Elevated UE risk — timing/pressure issue possible")
        for sig in ctx.get("player1_quote_signals", []) + ctx.get("player2_quote_signals", []):
            if sig == "injury_minor":
                risks.append("Player flagged minor injury in post-match press — verify in warm-up")
                break

        if best_edge < self.min_edge:
            return "NO BET", "0%", args or ["No edge above threshold"], risks, ["Wait for better price or live entry"]

        if best_odds < best_min:
            live.append(f"Market {best_odds:.2f} below minimum {best_min:.2f} — prefer live wait")
            action = "LIVE WAIT"
        elif conf == "low":
            action = "SMALL VALUE / LIVE WAIT"
        elif ctx.get("hedge_available"):
            action = "SMALL VALUE + LIVE HEDGE"
            live.append(ctx.get("hedge_note", "Cover stake on favourite if underdog bet taken"))
        else:
            action = "VALUE BET"

        stake = "0.25%-0.50%" if conf != "high" else "0.50%-0.75%"
        if best_edge > 0.10:
            args.append(f"{best_player} edge {best_edge:+.1%} vs market")

        return action, stake, args[:8], risks, live

    @staticmethod
    def _ue_to_dict(r):
        d = {k: v for k, v in r.__dict__.items() if k != "signals"}
        d["signals"] = r.signals
        return d

    @staticmethod
    def _fatigue_to_dict(r):
        return r.__dict__