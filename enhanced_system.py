"""
Enhanced Betting System - Main Orchestrator
===========================================

Ties together:
  1. Data collection (historical + live API)
  2. Feature engineering (ProphitBet-style + enhanced)
  3. Model training (ensemble)
  4. Value bet detection (Kelly Criterion)
  5. Report generation
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from enhanced_config import (
    API_FOOTBALL_KEY,
    BANKROLL,
    CONFIDENCE_THRESHOLD,
    GOAL_DIFF_MARGIN,
    KELLY_FRACTION,
    MATCH_HISTORY_WINDOW,
    MIN_VALUE_THRESHOLD,
    NEWS_API_KEY,
    ODDS_API_KEY,
)
from data_collector import MatchDataAggregator, HistoricalDataCollector
from feature_engineering import EnhancedStatisticsEngine, FeatureVectorBuilder
from prediction_model import BetRecommendation, EnsemblePredictor, ValueBetAnalyzer

logger = logging.getLogger("EnhancedBettingSystem")


# ============================================================
# LEAGUE CODES (football-data.co.uk)
# ============================================================

LEAGUE_CODES = {
    # Top 5 leagues
    "premier_league": "E0",
    "championship": "E1",
    "serie_a": "I1",
    "bundesliga": "D1",
    "la_liga": "SP1",
    "ligue_1": "F1",
    # Netherlands
    "eredivisie": "N1",
    # Turkey
    "super_lig": "T1",
    # Other
    "primeira_liga": "P1",
    "belgian_pro": "B1",
    "scottish_prem": "SC0",
}

# API-Football league IDs (for live data)
API_LEAGUE_IDS = {
    "E0": 39,  # Premier League
    "I1": 135,  # Serie A
    "D1": 78,  # Bundesliga
    "SP1": 140,  # La Liga
    "F1": 61,  # Ligue 1
    "N1": 88,  # Eredivisie
    "T1": 203,  # Süper Lig
}


class EnhancedBettingSystem:
    """
    Main orchestrator for the enhanced betting system.
    """

    def __init__(self, output_dir: str = "output/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.data_aggregator = MatchDataAggregator(
            api_football_key=API_FOOTBALL_KEY,
            odds_api_key=ODDS_API_KEY,
            news_api_key=NEWS_API_KEY,
        )
        self.stats_engine = EnhancedStatisticsEngine(
            match_history_window=MATCH_HISTORY_WINDOW,
            goal_diff_margin=GOAL_DIFF_MARGIN,
        )
        self.feature_builder = FeatureVectorBuilder()
        self.predictor = EnsemblePredictor(model_dir=str(self.output_dir / "models"))
        self.value_analyzer = ValueBetAnalyzer(
            min_value_threshold=MIN_VALUE_THRESHOLD,
            kelly_fraction=KELLY_FRACTION,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            bankroll=BANKROLL,
        )
        self.historical = HistoricalDataCollector()

    # ============================================================
    # TRAINING PIPELINE
    # ============================================================

    def train(self, league_code: str, start_year: int = 2018, cv_folds: int = 5) -> Dict:
        """
        Full training pipeline:
        1. Download historical data
        2. Compute all features
        3. Train ensemble model
        4. Save model and report
        """
        logger.info("=== TRAINING: %s from %s ===", league_code, start_year)

        # Step 1: Download historical data
        logger.info("Step 1: Downloading historical data...")
        df = self.historical.download_league(league_code, "", start_year=start_year)
        if df is None or df.empty:
            logger.error("No data downloaded. Check league code.")
            return {"error": "No data available"}

        logger.info("  Downloaded %s matches", len(df))

        # Step 2: Compute ProphitBet-style rolling stats
        logger.info("Step 2: Computing rolling statistics...")
        df = self.stats_engine.compute_rolling_stats(df)

        # Step 3: Compute enhanced rolling stats
        logger.info("Step 3: Computing enhanced statistics...")
        df = self.stats_engine.compute_enhanced_rolling_stats(df)

        # Step 4: Compute referee features
        logger.info("Step 4: Computing referee features...")
        df = self.stats_engine.compute_referee_features(df)

        # Step 5: Build training matrix
        logger.info("Step 5: Building feature matrix...")
        feature_cols = [
            c for c in self.feature_builder.get_all_feature_names() if c in df.columns
        ]
        logger.info("  Using %s features: %s...", len(feature_cols), feature_cols[:10])

        X, y = self.feature_builder.build_training_matrix(
            df, feature_columns=feature_cols
        )
        logger.info(
            "  Training matrix: %s samples, %s features", X.shape[0], X.shape[1]
        )

        if X.shape[0] < 100:
            logger.warning("Very small dataset - results may be unreliable")

        # Step 6: Train ensemble
        logger.info("Step 6: Training ensemble model...")
        results = self.predictor.train(
            X, y, feature_names=feature_cols, cv_folds=cv_folds
        )

        # Step 7: Save
        logger.info("Step 7: Saving model...")
        self.predictor.save(filename=f"model_{league_code}")

        # Save training report
        report = {
            "league": league_code,
            "start_year": start_year,
            "total_matches": len(df),
            "training_samples": int(X.shape[0]),
            "features_used": feature_cols,
            "model_results": {
                k: v
                for k, v in results.items()
                if k != "ensemble" or "classification_report" not in str(v)
            },
            "top_features": dict(list(self.predictor.feature_importances.items())[:15]),
            "trained_at": datetime.now().isoformat(),
        }

        report_path = self.output_dir / f"training_report_{league_code}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info("Training complete! Report saved to %s", report_path)
        self._print_training_summary(results)

        return results

    # ============================================================
    # PREDICTION PIPELINE
    # ============================================================

    def predict_matches(
        self, league_code: str, fixtures: List[Dict], start_year: int = 2018
    ) -> List[BetRecommendation]:
        """
        Predict upcoming matches and generate bet recommendations.
        """
        logger.info("=== PREDICTING %s matches ===", len(fixtures))

        # Load model
        try:
            self.predictor.load(filename=f"model_{league_code}")
        except FileNotFoundError:
            logger.error("Model not found. Train first with --mode train")
            return []

        # Download fresh historical data for rolling stats
        df = self.historical.download_league(
            league_code, "", start_year=start_year
        )
        if df is None:
            return []

        # Compute features
        df = self.stats_engine.compute_rolling_stats(df)
        df = self.stats_engine.compute_enhanced_rolling_stats(df)
        df = self.stats_engine.compute_referee_features(df)

        recommendations = []
        api_league_id = API_LEAGUE_IDS.get(league_code, 0)
        current_year = datetime.now().year

        for fixture in fixtures:
            logger.info(
                "Analyzing: %s vs %s",
                fixture["home_team"],
                fixture["away_team"],
            )

            try:
                # Get match context from live APIs
                ctx = self.data_aggregator.build_match_context(
                    fixture_id=fixture.get("fixture_id", 0),
                    home_team=fixture["home_team"],
                    away_team=fixture["away_team"],
                    home_team_id=fixture.get("home_team_id", 0),
                    away_team_id=fixture.get("away_team_id", 0),
                    league_id=api_league_id,
                    season=current_year,
                    date_str=fixture.get("date", ""),
                    league_name=league_code,
                )

                # Context features
                ctx_features = EnhancedStatisticsEngine.context_to_features(ctx)

                # Get latest rolling stats for both teams
                home_row = self._get_latest_team_row(
                    df, fixture["home_team"], "Home"
                )
                away_row = self._get_latest_team_row(
                    df, fixture["away_team"], "Away"
                )

                # Compute Poisson features
                league_avg = df["HG"].mean() if "HG" in df.columns else 1.35
                h_attack = (
                    home_row.get("HGF", league_avg * self.stats_engine.n)
                    / max(self.stats_engine.n * league_avg, 0.1)
                )
                h_defense = (
                    home_row.get("HGA", league_avg * self.stats_engine.n)
                    / max(self.stats_engine.n * league_avg, 0.1)
                )
                a_attack = (
                    away_row.get("AGF", league_avg * self.stats_engine.n)
                    / max(self.stats_engine.n * league_avg, 0.1)
                )
                a_defense = (
                    away_row.get("AGA", league_avg * self.stats_engine.n)
                    / max(self.stats_engine.n * league_avg, 0.1)
                )

                poisson = EnhancedStatisticsEngine.compute_poisson_features(
                    h_attack, h_defense, a_attack, a_defense, league_avg
                )

                # Build feature vector
                combined_row = {**home_row.to_dict(), **away_row.to_dict()}
                combined_row.update(ctx_features)
                combined_row.update(poisson)

                feature_vector = []
                for f_name in self.predictor.feature_names:
                    feature_vector.append(float(combined_row.get(f_name, 0.0)))
                X = np.array([feature_vector], dtype=np.float32)

                # Predict
                probs = self.predictor.predict_proba(X)[0]

                # Build recommendation
                odds = {
                    "home": ctx.odds_home,
                    "draw": ctx.odds_draw,
                    "away": ctx.odds_away,
                    "over25": ctx.odds_over25,
                    "under25": ctx.odds_under25,
                }
                match_info = {
                    "match": f"{fixture['home_team']} vs {fixture['away_team']}",
                    "date": fixture.get("date", ""),
                    "league": league_code,
                }

                rec = self.value_analyzer.analyze_match(
                    model_probs=probs,
                    odds=odds,
                    match_info=match_info,
                    context_features=ctx_features,
                    poisson_probs=poisson,
                )
                recommendations.append(rec)

            except Exception as exc:
                logger.error(
                    "Error analyzing %s vs %s: %s",
                    fixture["home_team"],
                    fixture["away_team"],
                    exc,
                )

        # Filter and sort value bets
        value_bets = self.value_analyzer.filter_value_bets(recommendations)

        # Generate report
        self._generate_prediction_report(
            recommendations, value_bets, league_code
        )

        return recommendations

    # ============================================================
    # ANALYSIS MODE
    # ============================================================

    def analyze_league(self, league_code: str, start_year: int = 2018):
        """
        Analyze a league's data and generate insights report.
        """
        logger.info("=== ANALYZING: %s ===", league_code)

        df = self.historical.download_league(
            league_code, "", start_year=start_year
        )
        if df is None:
            return None

        df = self.stats_engine.compute_rolling_stats(df)
        df = self.stats_engine.compute_enhanced_rolling_stats(df)
        df = self.stats_engine.compute_referee_features(df)

        # Generate analysis
        analysis = {
            "league": league_code,
            "total_matches": len(df),
            "seasons": sorted(df["Season"].unique().tolist()),
            "home_win_rate": round((df["Result"] == "H").mean(), 3),
            "draw_rate": round((df["Result"] == "D").mean(), 3),
            "away_win_rate": round((df["Result"] == "A").mean(), 3),
            "avg_goals": round((df["HG"] + df["AG"]).mean(), 2),
            "over25_rate": round(((df["HG"] + df["AG"]) > 2.5).mean(), 3),
            "btts_rate": round(
                ((df["HG"] > 0) & (df["AG"] > 0)).mean(), 3
            ),
        }

        if "HY" in df.columns:
            analysis["avg_yellows"] = round((df["HY"] + df["AY"]).mean(), 2)
        if "Referee" in df.columns:
            analysis["top_referees"] = df["Referee"].value_counts().head(10).to_dict()

        # Save
        path = self.output_dir / f"analysis_{league_code}.json"
        with open(path, "w") as f:
            json.dump(analysis, f, indent=2, default=str)

        logger.info("Analysis saved to %s", path)
        return analysis

    # ============================================================
    # HELPERS
    # ============================================================

    def _get_latest_team_row(
        self, df: pd.DataFrame, team_name: str, role: str
    ) -> pd.Series:
        """Get the most recent data row for a team."""
        team_col = "Home" if role == "Home" else "Away"
        team_df = df[df[team_col] == team_name].sort_values(
            "Date", ascending=False
        )
        if team_df.empty:
            # Try fuzzy match
            matches = df[df[team_col].str.contains(team_name[:5], case=False, na=False)]
            if not matches.empty:
                team_df = matches.sort_values("Date", ascending=False)

        if team_df.empty:
            logger.warning("No data found for %s as %s", team_name, role)
            return pd.Series(dtype=float)

        return team_df.iloc[0]

    def _print_training_summary(self, results: Dict):
        """Print training results summary."""
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        for model_name, metrics in results.items():
            if model_name == "ensemble":
                print(f"\n🏆 ENSEMBLE: Accuracy = {metrics['train_accuracy']}")
            elif isinstance(metrics, dict) and "cv_accuracy_mean" in metrics:
                print(
                    f"  {model_name}: "
                    f"CV = {metrics['cv_accuracy_mean']:.4f} "
                    f"± {metrics['cv_accuracy_std']:.4f}"
                )

        if self.predictor.feature_importances:
            print("\n📊 TOP 10 FEATURES:")
            for i, (feat, imp) in enumerate(
                list(self.predictor.feature_importances.items())[:10]
            ):
                bar = "█" * int(imp * 100)
                print(f"  {i+1}. {feat:30s} {imp:.4f} {bar}")
        print("=" * 60)

    def _generate_prediction_report(
        self,
        all_recs: List[BetRecommendation],
        value_bets: List[BetRecommendation],
        league_code: str,
    ):
        """Generate prediction report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "league": league_code,
            "total_matches": len(all_recs),
            "value_bets_found": len(value_bets),
            "predictions": [],
            "value_bets": [],
        }

        for rec in all_recs:
            report["predictions"].append(
                {
                    "match": rec.match,
                    "date": rec.date,
                    "probabilities": {
                        "home": rec.prob_home,
                        "draw": rec.prob_draw,
                        "away": rec.prob_away,
                        "over25": rec.prob_over25,
                    },
                    "best_bet": rec.best_bet,
                    "value": rec.best_value,
                    "confidence": rec.confidence,
                    "risk": rec.risk_level,
                    "key_factors": rec.key_factors,
                }
            )

        for rec in value_bets:
            report["value_bets"].append(
                {
                    "match": rec.match,
                    "bet": rec.best_bet,
                    "value": rec.best_value,
                    "confidence": rec.confidence,
                    "kelly_stake": rec.kelly_stake_pct,
                    "suggested_stake": round(rec.kelly_stake_pct * BANKROLL, 2),
                    "risk": rec.risk_level,
                    "key_factors": rec.key_factors,
                }
            )

        path = self.output_dir / f"predictions_{league_code}_{datetime.now().strftime('%Y%m%d')}.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Print summary
        print("\n" + "=" * 60)
        print("PREDICTION REPORT")
        print("=" * 60)
        print(f"Matches analyzed: {len(all_recs)}")
        print(f"Value bets found: {len(value_bets)}")

        if value_bets:
            print("\n🎯 VALUE BETS:")
            for rec in value_bets:
                print(f"\n  📌 {rec.match}")
                print(f"     Bet: {rec.best_bet}")
                print(f"     Value: {rec.best_value:+.1%}")
                print(f"     Confidence: {rec.confidence:.0%}")
                print(f"     Kelly Stake: €{rec.kelly_stake_pct * BANKROLL:.0f}")
                print(f"     Risk: {rec.risk_level}")
                print(f"     Factors: {'; '.join(rec.key_factors[:3])}")
        else:
            print("\n  No value bets meeting criteria found.")

        print(f"\nFull report: {path}")
        print("=" * 60)
