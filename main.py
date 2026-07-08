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
)
from data_collector import HistoricalDataCollector, LiveFixtureCollector
from feature_engineering import FEATURE_NAMES, TennisFeatureEngine
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
        self.historical = HistoricalDataCollector()
        self.live = LiveFixtureCollector()
        self.feature_engine = TennisFeatureEngine(history_window=MATCH_HISTORY_WINDOW)

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"{PROJECT_NAME} — {PROJECT_TAGLINE}",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "analyze"],
        required=True,
        help="Operating mode",
    )
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


if __name__ == "__main__":
    main()