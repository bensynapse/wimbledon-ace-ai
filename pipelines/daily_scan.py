"""Daily value scan with auto context, weather and optional Telegram."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

from ai.explanation import explain_from_payload
from config import MIN_VALUE_THRESHOLD
from data_sources.weather_source import WeatherSource
from intelligence import MatchIntelligenceAnalyzer, to_llm_payload
from models.probability_model import ProbabilityModelV1
from notifications.telegram import TelegramNotifier
from pipelines.context_builder import MatchContextBuilder

logger = logging.getLogger(__name__)


class DailyScanner:
    def __init__(self, context_builder: MatchContextBuilder, db=None):
        self.context = context_builder
        self.weather = WeatherSource()
        self.model = ProbabilityModelV1()
        self.analyzer = MatchIntelligenceAnalyzer(min_edge=MIN_VALUE_THRESHOLD)
        self.telegram = TelegramNotifier()
        self.db = db

    def scan_fixtures(
        self,
        fixtures: List[Dict],
        min_edge: float = MIN_VALUE_THRESHOLD,
        full_reports: bool = False,
    ) -> List[Dict]:
        hits = []
        for fix in fixtures:
            hit = self._analyze_fixture(fix, min_edge, full_reports)
            if hit:
                hits.append(hit)
        return sorted(hits, key=lambda x: x["edge"], reverse=True)

    def _analyze_fixture(
        self,
        fix: Dict,
        min_edge: float,
        full_reports: bool,
    ) -> Optional[Dict]:
        p1 = fix["player1"]
        p2 = fix["player2"]
        surface = fix.get("surface", "hard")
        o1 = fix.get("odds_player1", 0.0)
        o2 = fix.get("odds_player2", 0.0)
        if o1 <= 1 and o2 <= 1:
            return None

        ctx = self.context.build(
            p1, p2, surface=surface,
            match_date=fix.get("date"),
            tournament=fix.get("tournament", ""),
            extra={k: v for k, v in fix.items() if k not in ("player1", "player2")},
        )
        weather = self.weather.for_tournament(fix.get("tournament", ""))
        ctx.update(weather)

        price = self.model.predict(p1, p2, surface, ctx)
        implied_p1 = 1.0 / o1
        implied_p2 = 1.0 / o2
        edge_p1 = price.prob_p1 - implied_p1
        edge_p2 = price.prob_p2 - implied_p2

        if edge_p1 < min_edge and edge_p2 < min_edge:
            return None

        value_side = p1 if edge_p1 >= edge_p2 else p2
        edge = max(edge_p1, edge_p2)
        odds = o1 if value_side == p1 else o2
        model_prob = price.prob_p1 if value_side == p1 else price.prob_p2

        report = self.analyzer.analyze(
            player1=p1, player2=p2, surface=surface,
            model_prob_p1=price.prob_p1,
            market_odds_p1=o1, market_odds_p2=o2,
            tournament=fix.get("tournament", ""),
            context=ctx,
        )

        hit = {
            "match": f"{p1} vs {p2}",
            "value_side": value_side,
            "edge": edge,
            "odds": odds,
            "fair_odds": price.fair_odds_p1 if value_side == p1 else price.fair_odds_p2,
            "model_prob": model_prob,
            "tournament": fix.get("tournament", ""),
            "surface": surface,
            "action": report.recommended_action,
            "confidence": report.confidence,
            "stake": report.stake_pct_range,
            "date": fix.get("date", ""),
        }

        if full_reports:
            payload = to_llm_payload(report)
            hit["report"] = explain_from_payload(payload)
            hit["payload"] = payload

        if self.db:
            self.db.save_prediction(
                player_a=p1, player_b=p2, surface=surface,
                tournament=fix.get("tournament", ""),
                model_prob_a=price.prob_p1,
                fair_odds_a=price.fair_odds_p1,
                fair_odds_b=price.fair_odds_p2,
                market_odds_a=o1, market_odds_b=o2,
                edge_a=report.edge_p1, edge_b=report.edge_p2,
                confidence=report.confidence,
                recommended_action=report.recommended_action,
                minimum_odds_a=report.minimum_odds_p1,
                minimum_odds_b=report.minimum_odds_p2,
                stake_percent=report.stake_pct_range,
                payload=to_llm_payload(report),
            )

        return hit

    def run_and_notify(
        self,
        fixtures: List[Dict],
        tour: str = "atp",
        min_edge: float = MIN_VALUE_THRESHOLD,
    ) -> List[Dict]:
        hits = self.scan_fixtures(fixtures, min_edge=min_edge)
        self.telegram.send_value_scan(hits, tour=tour)
        return hits

    def daily_report(self, hits: List[Dict], tour: str = "atp") -> str:
        date_str = datetime.now().strftime("%Y-%m-%d")
        if not hits:
            return f"{date_str} | {tour.upper()} | Geen value bets vandaag."
        lines = [f"📊 Daily Report {date_str} ({tour.upper()})", f"{len(hits)} opportunities\n"]
        for h in hits[:15]:
            lines.append(
                f"• {h['match']}\n"
                f"  Pick: {h['value_side']} @ {h['odds']:.2f} | edge {h['edge']:+.1%}\n"
                f"  {h['action']} | stake {h['stake']} | conf {h['confidence']}"
            )
        return "\n".join(lines)