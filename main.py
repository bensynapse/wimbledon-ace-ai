"""
WimbledonAce AI — Grand Slam Tennis Betting Predictor 2026
==========================================================

AI-powered ATP/WTA predictions for Wimbledon, Roland Garros, US Open & more.

Usage:
    python main.py --mode train --tour atp
    python main.py --mode predict --tour atp --today
    python main.py --mode predict --tour atp --fixtures-file fixtures.json
    python main.py --mode analyze --tour atp
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from config import (
    BANKROLL,
    CONFIDENCE_THRESHOLD,
    KELLY_FRACTION,
    MATCH_HISTORY_WINDOW,
    MIN_VALUE_THRESHOLD,
    PROJECT_NAME,
    PROJECT_TAGLINE,
    TOURS,
    api_status,
)
from data_collector import HistoricalDataCollector, LiveFixtureCollector, get_demo_fixtures
from data_sources.weather_source import WeatherSource
from notifications.telegram import TelegramNotifier
from pipelines.backtest import OddsBacktester, format_backtest_report
from pipelines.context_builder import MatchContextBuilder
from pipelines.daily_scan import DailyScanner
from feature_engineering import FEATURE_NAMES, TennisFeatureEngine
from ai.explanation import explain_match
from database.db import TennisDatabase
from intelligence import MatchIntelligenceAnalyzer, format_report, to_llm_payload
from models.probability_model import ProbabilityModelV1
from models.underdog_adjustments import apply_underdog_adjustments
from prediction_model import BetRecommendation, EnsemblePredictor, ValueBetAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("WimbledonAceAI")


class WimbledonAceAI:
    """WimbledonAce AI — orchestrates Grand Slam tennis betting predictions."""

    def __init__(self, output_dir: str = "output/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.predictor = EnsemblePredictor(model_dir=str(self.output_dir / "models"))
        self.value_analyzer = ValueBetAnalyzer(
            min_value_threshold=MIN_VALUE_THRESHOLD,
            kelly_fraction=KELLY_FRACTION,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            bankroll=BANKROLL,
        )
        self._historical = None
        self._live = None
        self.feature_engine = TennisFeatureEngine(history_window=MATCH_HISTORY_WINDOW)
        self.probability_model = ProbabilityModelV1()
        self.db = TennisDatabase()
        self.context_builder = MatchContextBuilder()
        self.weather = WeatherSource()
        self.telegram = TelegramNotifier()
        self._context_warmed = False

    @property
    def historical(self) -> HistoricalDataCollector:
        if self._historical is None:
            self._historical = HistoricalDataCollector()
        return self._historical

    @property
    def live(self) -> LiveFixtureCollector:
        if self._live is None:
            self._live = LiveFixtureCollector()
        return self._live

    def warm_context(self, tour: str = "atp") -> None:
        """Load cached history into context builder for auto fatigue/form."""
        if self._context_warmed:
            return
        try:
            df = self.historical.download_tour(tour, start_year=datetime.now().year - 2)
            standings = []
            try:
                standings = self.live.tennis.get_standings(tour)
            except Exception:
                pass
            self.context_builder.warm(df, standings, tour=tour)
            self._context_warmed = True
            logger.info("Context builder warmed with %d matches", len(df))
        except Exception as exc:
            logger.warning("Could not warm context builder: %s", exc)

    @property
    def daily_scanner(self) -> DailyScanner:
        return DailyScanner(self.context_builder, db=self.db)

    def train(self, tour: str, start_year: int = 2018, cv_folds: int = 5) -> Dict:
        logger.info("=== TRAINING: %s from %s ===", tour.upper(), start_year)

        df = self.historical.download_tour(tour, start_year=start_year)
        logger.info("Loaded %d finished matches", len(df))

        X, y, feature_names = self.feature_engine.build_training_matrix(df)
        logger.info("Training matrix: %d samples, %d features", X.shape[0], X.shape[1])

        if X.shape[0] < 200:
            logger.warning("Small dataset (%d matches) — results may be unreliable", X.shape[0])

        results = self.predictor.train(X, y, feature_names=feature_names, cv_folds=cv_folds)
        self.predictor.save(filename=f"model_{tour.lower()}")

        report = {
            "tour": tour,
            "start_year": start_year,
            "total_matches": len(df),
            "training_samples": int(X.shape[0]),
            "features_used": feature_names,
            "model_results": {
                k: v for k, v in results.items()
                if k != "ensemble" or "classification_report" not in str(v)
            },
            "top_features": dict(list(self.predictor.feature_importances.items())[:10]),
            "trained_at": datetime.now().isoformat(),
        }
        report_path = self.output_dir / f"training_report_{tour}.json"
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)

        self._print_training_summary(results)
        logger.info("Training report saved to %s", report_path)
        return results

    def predict_matches(
        self,
        tour: str,
        fixtures: List[Dict],
        warm_history: bool = True,
    ) -> List[BetRecommendation]:
        logger.info("=== PREDICTING %d matches (%s) ===", len(fixtures), tour.upper())

        try:
            self.predictor.load(filename=f"model_{tour.lower()}")
        except FileNotFoundError:
            logger.error("Model not found. Train first: python main.py --mode train --tour %s", tour)
            return []

        if warm_history:
            df = self.historical.download_tour(tour, start_year=datetime.now().year - 2)
            self.feature_engine = TennisFeatureEngine(history_window=MATCH_HISTORY_WINDOW)
            self.feature_engine.ingest_history(df)
            try:
                standings = self.live.tennis.get_standings(tour)
                self.feature_engine.set_rankings(standings)
            except Exception as exc:
                logger.warning("Could not load rankings: %s", exc)

        recommendations = []
        for fixture in fixtures:
            player1 = fixture["player1"]
            player2 = fixture["player2"]
            logger.info("Analyzing: %s vs %s", player1, player2)

            try:
                features, context = self.feature_engine.build_match_features(
                    player1=player1,
                    player2=player2,
                    surface=fixture.get("surface", "hard"),
                    match_date=fixture.get("date"),
                )
                X = np.array([features], dtype=np.float32)
                prob_p1 = float(self.predictor.predict_proba(X)[0])

                odds = {
                    "player1": fixture.get("odds_player1", 0.0),
                    "player2": fixture.get("odds_player2", 0.0),
                }
                match_info = {
                    "match": f"{player1} vs {player2}",
                    "date": fixture.get("date", ""),
                    "tournament": fixture.get("tournament", ""),
                    "surface": fixture.get("surface", ""),
                    "player1": player1,
                    "player2": player2,
                }
                rec = self.value_analyzer.analyze_match(
                    prob_player1=prob_p1,
                    odds=odds,
                    match_info=match_info,
                    context_features=context,
                )
                recommendations.append(rec)
            except Exception as exc:
                logger.error("Error analyzing %s vs %s: %s", player1, player2, exc)

        value_bets = self.value_analyzer.filter_value_bets(recommendations)
        self._generate_prediction_report(recommendations, value_bets, tour)
        return recommendations

    def predict_today(self, tour: str) -> List[BetRecommendation]:
        fixtures = self.live.get_upcoming_fixtures(tour)
        if not fixtures:
            logger.warning("No upcoming %s fixtures found", tour.upper())
            return []
        fixtures = self.live.attach_odds(fixtures, tour)
        logger.info("Found %d upcoming matches", len(fixtures))
        return self.predict_matches(tour, fixtures)

    def analyze_tour(self, tour: str, start_year: int = 2018) -> Dict:
        logger.info("=== ANALYZING: %s ===", tour.upper())
        df = self.historical.download_tour(tour, start_year=start_year)

        analysis = {
            "tour": tour,
            "total_matches": len(df),
            "date_range": {
                "from": str(df["date"].min()),
                "to": str(df["date"].max()),
            },
            "surface_distribution": df["surface"].value_counts().to_dict(),
            "player1_win_rate": round(df["winner_is_player1"].mean(), 3),
            "unique_players": len(set(df["player1"]) | set(df["player2"])),
            "top_tournaments": df["tournament"].value_counts().head(10).to_dict(),
            "features": FEATURE_NAMES,
            "match_history_window": MATCH_HISTORY_WINDOW,
        }

        path = self.output_dir / f"analysis_{tour}.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(analysis, handle, indent=2, default=str)

        logger.info("Analysis saved to %s", path)
        print(f"\n{tour.upper()}: {analysis['total_matches']} matches, "
              f"{analysis['unique_players']} players")
        print(f"Surfaces: {analysis['surface_distribution']}")
        return analysis

    def _print_training_summary(self, results: Dict) -> None:
        print("\n" + "=" * 60)
        print(f"{PROJECT_NAME} — TRAINING SUMMARY")
        print("=" * 60)
        for model_name, metrics in results.items():
            if model_name == "ensemble":
                print(f"\nENSEMBLE accuracy: {metrics['train_accuracy']}")
            elif isinstance(metrics, dict) and "cv_accuracy_mean" in metrics:
                print(
                    f"  {model_name}: CV {metrics['cv_accuracy_mean']:.4f} "
                    f"+/- {metrics['cv_accuracy_std']:.4f}"
                )
        if self.predictor.feature_importances:
            print("\nTOP FEATURES:")
            for feat, imp in list(self.predictor.feature_importances.items())[:8]:
                print(f"  {feat:30s} {imp:.4f}")
        print("=" * 60)

    def _generate_prediction_report(
        self,
        all_recs: List[BetRecommendation],
        value_bets: List[BetRecommendation],
        tour: str,
    ) -> None:
        report = {
            "generated_at": datetime.now().isoformat(),
            "tour": tour,
            "total_matches": len(all_recs),
            "value_bets_found": len(value_bets),
            "predictions": [
                {
                    "match": rec.match,
                    "date": rec.date,
                    "tournament": rec.tournament,
                    "surface": rec.surface,
                    "probabilities": {
                        "player1": rec.prob_player1,
                        "player2": rec.prob_player2,
                    },
                    "odds": {
                        "player1": rec.odds_player1,
                        "player2": rec.odds_player2,
                    },
                    "best_bet": rec.best_bet,
                    "value": rec.best_value,
                    "confidence": rec.confidence,
                    "risk": rec.risk_level,
                    "key_factors": rec.key_factors,
                }
                for rec in all_recs
            ],
            "value_bets": [
                {
                    "match": rec.match,
                    "bet": rec.best_bet,
                    "value": rec.best_value,
                    "confidence": rec.confidence,
                    "kelly_stake": rec.kelly_stake_pct,
                    "suggested_stake": round(rec.kelly_stake_pct * BANKROLL, 2),
                    "risk": rec.risk_level,
                }
                for rec in value_bets
            ],
        }

        path = self.output_dir / f"predictions_{tour}_{datetime.now().strftime('%Y%m%d')}.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

        print("\n" + "=" * 60)
        print(f"{PROJECT_NAME} — PREDICTION REPORT")
        print("=" * 60)
        print(f"Matches analyzed: {len(all_recs)}")
        print(f"Value bets found: {len(value_bets)}")
        if value_bets:
            print("\nVALUE BETS:")
            for rec in value_bets:
                print(f"\n  {rec.match}")
                print(f"     Bet: {rec.best_bet}")
                print(f"     Value: {rec.best_value:+.1%}")
                print(f"     Kelly stake: EUR {rec.kelly_stake_pct * BANKROLL:.0f}")
        else:
            print("\n  No value bets meeting criteria found.")
        print(f"\nFull report: {path}")
        print("=" * 60)

    def analyze_intelligence(
        self,
        player1: str,
        player2: str,
        surface: str,
        model_prob_p1: float = None,
        odds_p1: float = 0.0,
        odds_p2: float = 0.0,
        context: Dict = None,
        tournament: str = "",
    ) -> Dict:
        """Full market intelligence report: model → fair odds → UE/fatigue → advice."""
        ctx = dict(context or {})
        if not ctx.get("model_override"):
            auto = self.context_builder.build(
                player1, player2, surface=surface,
                tournament=tournament, extra=ctx,
            )
            ctx = {**auto, **ctx}
        match_date = ctx.get("match_date")
        match_hour = ctx.get("match_hour")
        weather = self.weather.for_tournament(
            tournament or ctx.get("tournament", ""),
            match_date=match_date,
            match_hour=match_hour,
            surface=surface,
        )
        ctx.update({k: v for k, v in weather.items() if v is not None})
        for prefix in ("player1", "player2"):
            flags = list(ctx.get(f"{prefix}_context_flags", []))
            for flag in ctx.get("weather_context_flags", []):
                if flag not in flags:
                    flags.append(flag)
            ctx[f"{prefix}_context_flags"] = flags

        if odds_p1:
            ctx["odds_player1"] = odds_p1
            ctx["market_odds_p1"] = odds_p1
        if odds_p2:
            ctx["market_odds_p2"] = odds_p2

        if model_prob_p1 is None:
            price = self.probability_model.predict(player1, player2, surface, ctx)
            model_prob_p1 = price.prob_p1
            ctx["model_components"] = price.components
            adj = apply_underdog_adjustments(model_prob_p1, player1, player2, ctx)
            if adj.applied:
                model_prob_p1 = adj.adjusted_prob_p1
                ctx["underdog_adjustment"] = {
                    "base_prob_p1": adj.base_prob_p1,
                    "adjusted_prob_p1": adj.adjusted_prob_p1,
                    "signals": adj.signals,
                }
                logger.info(
                    "Underdog adjust: %s %.0f%% → %.0f%% | %s",
                    player1,
                    adj.base_prob_p1 * 100,
                    adj.adjusted_prob_p1 * 100,
                    "; ".join(adj.signals[:3]),
                )
            logger.info(
                "V1 model: %s %.0f%% / %s %.0f%%",
                player1, model_prob_p1 * 100, player2, (1 - model_prob_p1) * 100,
            )

        analyzer = MatchIntelligenceAnalyzer(min_edge=MIN_VALUE_THRESHOLD)
        report = analyzer.analyze(
            player1=player1,
            player2=player2,
            surface=surface,
            model_prob_p1=model_prob_p1,
            market_odds_p1=odds_p1,
            market_odds_p2=odds_p2,
            tournament=tournament,
            context=ctx,
        )

        print(format_report(report))
        print("\n" + "=" * 60)
        print("AI ANALYSIS")
        print("=" * 60)
        print(explain_match(report))

        payload = to_llm_payload(report)
        payload_path = self.output_dir / f"intelligence_{player1}_vs_{player2}.json".replace(" ", "_")
        with open(payload_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info("LLM-ready payload saved to %s", payload_path)

        self.db.save_prediction(
            player_a=player1,
            player_b=player2,
            surface=surface,
            tournament=tournament,
            model_prob_a=report.model_prob_p1,
            fair_odds_a=report.fair_odds_p1,
            fair_odds_b=report.fair_odds_p2,
            market_odds_a=odds_p1,
            market_odds_b=odds_p2,
            edge_a=report.edge_p1,
            edge_b=report.edge_p2,
            confidence=report.confidence,
            recommended_action=report.recommended_action,
            minimum_odds_a=report.minimum_odds_p1,
            minimum_odds_b=report.minimum_odds_p2,
            stake_percent=report.stake_pct_range,
            model_version="v1_tennis_abstract_elo+underdog",
            payload=payload,
        )
        return payload

    def scan_value(
        self,
        fixtures: List[Dict],
        min_edge: float = None,
        tour: str = "atp",
        full: bool = False,
        notify: bool = False,
    ) -> List[Dict]:
        """Scan fixtures with auto context, fatigue, weather and optional Telegram."""
        self.warm_context(tour)
        min_edge = min_edge or MIN_VALUE_THRESHOLD
        hits = self.daily_scanner.scan_fixtures(fixtures, min_edge=min_edge, full_reports=full)
        if notify:
            self.telegram.send_value_scan(hits, tour=tour)
        return hits

    def run_backtest(
        self,
        start_year: int = 2018,
        end_year: int = None,
        surface: str = None,
        min_edge: float = None,
        max_bets: int = None,
    ):
        """Backtest rank/points model vs historical bookmaker odds (dissfya dataset)."""
        backtester = OddsBacktester(
            min_edge=min_edge or MIN_VALUE_THRESHOLD,
            output_dir=str(self.output_dir),
        )
        result = backtester.run(
            start_year=start_year,
            end_year=end_year,
            surface=surface,
            max_bets=max_bets,
        )
        print(format_backtest_report(result))
        return result

    def run_daily(self, tour: str = "atp", min_edge: float = None) -> List[Dict]:
        """Full daily pipeline: fixtures → scan → report → Telegram."""
        self.warm_context(tour)
        fixtures = self.live.get_upcoming_fixtures(tour)
        fixtures = self.live.attach_odds(fixtures, tour)
        hits = self.scan_value(fixtures, min_edge=min_edge, tour=tour, full=True, notify=True)
        report = self.daily_scanner.daily_report(hits, tour=tour)
        print(report)
        report_path = self.output_dir / f"daily_report_{tour}_{datetime.now().strftime('%Y%m%d')}.txt"
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write(report)
        logger.info("Daily report saved to %s", report_path)
        return hits


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"{PROJECT_NAME} — {PROJECT_TAGLINE}",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "analyze", "intelligence", "scan-value", "backtest", "explain", "daily", "status"],
        required=True,
        help="Operating mode",
    )
    parser.add_argument("--player1", type=str, default=None, help="Player 1 name")
    parser.add_argument("--player2", type=str, default=None, help="Player 2 name")
    parser.add_argument(
        "--surface", type=str, default="hard",
        choices=["hard", "clay", "grass", "carpet"],
    )
    parser.add_argument("--odds-p1", type=float, default=0.0, help="Market odds player 1")
    parser.add_argument("--odds-p2", type=float, default=0.0, help="Market odds player 2")
    parser.add_argument(
        "--model-prob-p1", type=float, default=None,
        help="Model probability for player 1 (0-1). If omitted, uses feature engine.",
    )
    parser.add_argument(
        "--context-file", type=str, default=None,
        help="JSON with UE/fatigue/weather context for intelligence mode",
    )
    parser.add_argument("--tournament", type=str, default="", help="Tournament name")
    parser.add_argument(
        "--tour",
        type=str,
        default="atp",
        choices=list(TOURS.keys()),
        help="Tennis tour (atp or wta)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2018,
        help="Start year for historical data",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="End year for backtest mode (default: current year)",
    )
    parser.add_argument(
        "--max-bets",
        type=int,
        default=None,
        help="Cap simulated bets in backtest mode",
    )
    parser.add_argument(
        "--backtest-surface",
        type=str,
        default=None,
        choices=["hard", "clay", "grass", "carpet"],
        help="Filter backtest to one surface (default: all)",
    )
    parser.add_argument(
        "--fixtures-file",
        type=str,
        default=None,
        help="JSON file with match fixtures to predict",
    )
    parser.add_argument(
        "--today",
        action="store_true",
        help="Predict upcoming fixtures (fetches live data + odds)",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Force re-download of historical match data",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=MIN_VALUE_THRESHOLD,
        help="Minimum edge for scan-value mode",
    )
    parser.add_argument(
        "--payload-file",
        type=str,
        default=None,
        help="JSON payload for explain mode",
    )
    parser.add_argument(
        "--match-id",
        type=str,
        default=None,
        help="Match identifier for explain mode",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full intelligence reports in scan-value mode",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Send Telegram alert after scan-value/daily",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use demo fixtures (no API keys needed)",
    )
    args = parser.parse_args()

    system = WimbledonAceAI()

    if args.refresh_data:
        system.historical.download_tour(args.tour, start_year=args.start_year, refresh=True)

    if args.mode == "train":
        system.train(args.tour, start_year=args.start_year)
    elif args.mode == "analyze":
        system.analyze_tour(args.tour, start_year=args.start_year)
    elif args.mode == "predict":
        if args.today:
            system.predict_today(args.tour)
        elif args.fixtures_file:
            with open(args.fixtures_file, encoding="utf-8") as handle:
                fixtures = json.load(handle)
            system.predict_matches(args.tour, fixtures)
        else:
            logger.error("Use --today or --fixtures-file for predictions")
            sys.exit(1)
    elif args.mode == "intelligence":
        if not args.player1 or not args.player2:
            logger.error("intelligence mode requires --player1 and --player2")
            sys.exit(1)

        system.warm_context(args.tour)
        context = {}
        if args.context_file:
            with open(args.context_file, encoding="utf-8") as handle:
                context = json.load(handle)

        if args.model_prob_p1 is not None:
            context["model_override"] = True

        system.analyze_intelligence(
            player1=args.player1,
            player2=args.player2,
            surface=args.surface,
            model_prob_p1=args.model_prob_p1,
            odds_p1=args.odds_p1,
            odds_p2=args.odds_p2,
            context=context,
            tournament=args.tournament,
        )
    elif args.mode == "scan-value":
        if args.demo:
            fixtures = get_demo_fixtures(args.tour)
        elif args.fixtures_file:
            with open(args.fixtures_file, encoding="utf-8") as handle:
                fixtures = json.load(handle)
        else:
            fixtures = system.live.get_upcoming_fixtures(args.tour)
            fixtures = system.live.attach_odds(fixtures, args.tour)
        hits = system.scan_value(
            fixtures, min_edge=args.min_edge, tour=args.tour,
            full=args.full, notify=args.notify,
        )
        if not hits:
            print(f"No value above {args.min_edge:.0%} edge.")
        else:
            print(f"Found {len(hits)} value opportunities:\n")
            for hit in hits:
                print(
                    f"  {hit['match']} → {hit['value_side']} "
                    f"edge {hit['edge']:+.1%} @ {hit['odds']} (fair {hit['fair_odds']:.2f})\n"
                    f"    {hit['action']} | stake {hit['stake']} | conf {hit['confidence']}"
                )
                if args.full and hit.get("report"):
                    print(f"\n{hit['report']}\n")
    elif args.mode == "backtest":
        system.run_backtest(
            start_year=args.start_year,
            end_year=args.end_year,
            surface=args.backtest_surface,
            min_edge=args.min_edge,
            max_bets=args.max_bets,
        )
    elif args.mode == "daily":
        system.run_daily(args.tour, min_edge=args.min_edge)
    elif args.mode == "status":
        status = api_status()
        print(f"\n{PROJECT_NAME} — Data Source Status\n")
        for name, ok in status.items():
            if name == "data_source":
                print(f"  → active source: {ok}")
                continue
            icon = "✓" if ok else "✗"
            print(f"  {icon} {name}")
        print("\n  GitHub ATP: github.com/Tennismylife/TML-Database")
        print("  GitHub Charting: github.com/JeffSackmann/tennis_MatchChartingProject")
        if not status.get("kaggle"):
            print("  → Kaggle WTA: bash scripts/download_kaggle.sh")
        if not status.get("kaggle_odds"):
            print("  → Kaggle odds backtest: bash scripts/download_kaggle_odds.sh")
        if not status.get("tennis_abstract_elo"):
            print("  → Tennis Abstract Elo: python3 cli.py update --elo")
        if not status["odds_api"]:
            print("  → ODDS_API_KEY (fixtures+odds): https://the-odds-api.com/")
        if not status["tennis_api"]:
            print("  → TENNIS_API_KEY (optional/WTA): https://api-tennis.com/")
        if not status["telegram"]:
            print("  → TELEGRAM: optional, set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID")
        print()
    elif args.mode == "explain":
        from ai.explanation import explain_from_payload
        payload = None
        if args.payload_file:
            with open(args.payload_file, encoding="utf-8") as handle:
                payload = json.load(handle)
        elif args.match_id:
            path = system.output_dir / f"intelligence_{args.match_id}.json"
            if path.exists():
                with open(path, encoding="utf-8") as handle:
                    payload = json.load(handle)
        if not payload:
            logger.error("Provide --payload-file or --match-id with existing output")
            sys.exit(1)
        print(explain_from_payload(payload))


if __name__ == "__main__":
    main()