"""
Enhanced Feature Engineering Module
Combines ProphitBet-style rolling statistics with:
  - Player-level impact features
  - Referee features
  - Injury/squad strength features
  - xG-based features
  - Momentum/news features
  - Head-to-head features
  - Card discipline features
  - Odds-implied probability features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict

from data_collector import MatchContext, TeamSquadStatus, RefereeProfile


class EnhancedStatisticsEngine:
    """
    Extended version of ProphitBet's StatisticsEngine.
    Adds ~40 new features on top of the original ~27.
    """

    def __init__(self, match_history_window: int = 10,
                 goal_diff_margin: int = 2):
        self.n = match_history_window
        self.gd_margin = goal_diff_margin

    # ============================================================
    # 1. PROPHITBET-STYLE ROLLING FEATURES (kept for compatibility)
    # ============================================================

    def compute_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all ProphitBet-style rolling window statistics.
        Input df must have: Date, Season, Home, Away, HG, AG, Result
        and optionally: HST, AST, HC, AC, HS, AS, HF, AF, HY, AY, HR, AR
        """
        df = df.sort_values("Date").copy()

        # --- Basic rolling stats (from ProphitBet) ---
        for team_col, prefix in [("Home", "H"), ("Away", "A")]:
            # Wins
            if prefix == "H":
                df[f"{prefix}W"] = self._rolling_sum(
                    df, team_col, df["Result"].eq("H").astype(int))
                df[f"{prefix}L"] = self._rolling_sum(
                    df, team_col, df["Result"].eq("A").astype(int))
            else:
                df[f"{prefix}W"] = self._rolling_sum(
                    df, team_col, df["Result"].eq("A").astype(int))
                df[f"{prefix}L"] = self._rolling_sum(
                    df, team_col, df["Result"].eq("H").astype(int))

            # Goals for/against
            gf_col = f"{prefix}G"
            ga_col = "AG" if prefix == "H" else "HG"
            df[f"{prefix}GF"] = self._rolling_sum(df, team_col, df[gf_col])
            df[f"{prefix}GA"] = self._rolling_sum(df, team_col, df[ga_col])
            df[f"{prefix}GD"] = df[f"{prefix}GF"] - df[f"{prefix}GA"]

        # Cross-comparisons
        df["HAGF"] = df["HGF"] - df["AGF"]
        df["HAGA"] = df["HGA"] - df["AGA"]
        df["HAGD"] = df["HGD"] - df["AGD"]

        return df

    # ============================================================
    # 2. NEW: ENHANCED ROLLING FEATURES
    #    (what ProphitBet misses)
    # ============================================================

    def compute_enhanced_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling stats for shots, fouls, cards, referee patterns.
        These require extra columns from football-data.co.uk that
        ProphitBet partially ignores.
        """
        df = df.copy()

        # --- Total Shots Rolling (not just on target) ---
        if "HS" in df.columns and "AS" in df.columns:
            df["H_Shots_Roll"] = self._rolling_sum(df, "Home", df["HS"])
            df["A_Shots_Roll"] = self._rolling_sum(df, "Away", df["AS"])
            df["H_Shot_Accuracy"] = (
                df.get("HST", 0) / df["HS"].replace(0, np.nan)
            ).fillna(0)
            df["A_Shot_Accuracy"] = (
                df.get("AST", 0) / df["AS"].replace(0, np.nan)
            ).fillna(0)

        # --- Yellow & Red Cards Rolling ---
        if "HY" in df.columns:
            df["H_Yellows_Roll"] = self._rolling_sum(df, "Home", df["HY"])
            df["A_Yellows_Roll"] = self._rolling_sum(df, "Away", df["AY"])
        if "HR" in df.columns:
            df["H_Reds_Roll"] = self._rolling_sum(df, "Home", df["HR"])
            df["A_Reds_Roll"] = self._rolling_sum(df, "Away", df["AR"])

        # --- Fouls Rolling ---
        if "HF" in df.columns:
            df["H_Fouls_Roll"] = self._rolling_sum(df, "Home", df["HF"])
            df["A_Fouls_Roll"] = self._rolling_sum(df, "Away", df["AF"])

        # --- Card Discipline Score ---
        # Higher = more undisciplined
        if "HY" in df.columns and "HR" in df.columns:
            df["H_Discipline"] = (
                df.get("H_Yellows_Roll", 0) + 3 * df.get("H_Reds_Roll", 0)
            )
            df["A_Discipline"] = (
                df.get("A_Yellows_Roll", 0) + 3 * df.get("A_Reds_Roll", 0)
            )

        # --- Shots-to-Goals Conversion Rate ---
        if "HS" in df.columns:
            df["H_Conversion"] = (
                df.get("HGF", 0) /
                df.get("H_Shots_Roll", pd.Series(1)).replace(0, np.nan)
            ).fillna(0)
            df["A_Conversion"] = (
                df.get("AGF", 0) /
                df.get("A_Shots_Roll", pd.Series(1)).replace(0, np.nan)
            ).fillna(0)

        # --- Draw rate ---
        df["H_Draw_Rate"] = self._rolling_sum(
            df, "Home", df["Result"].eq("D").astype(int))
        df["A_Draw_Rate"] = self._rolling_sum(
            df, "Away", df["Result"].eq("D").astype(int))

        # --- Clean Sheet Rate ---
        df["H_CleanSheet"] = self._rolling_sum(
            df, "Home", (df["AG"] == 0).astype(int))
        df["A_CleanSheet"] = self._rolling_sum(
            df, "Away", (df["HG"] == 0).astype(int))

        # --- BTTS (Both Teams to Score) Rate ---
        btts = ((df["HG"] > 0) & (df["AG"] > 0)).astype(int)
        df["H_BTTS_Rate"] = self._rolling_sum(df, "Home", btts)
        df["A_BTTS_Rate"] = self._rolling_sum(df, "Away", btts)

        # --- Over 2.5 Goals Rate ---
        over25 = (df["HG"] + df["AG"] > 2.5).astype(int)
        df["H_Over25_Rate"] = self._rolling_sum(df, "Home", over25)
        df["A_Over25_Rate"] = self._rolling_sum(df, "Away", over25)

        # --- Form Points (W=3, D=1, L=0) last N ---
        df["_H_Points"] = df["Result"].map({"H": 3, "D": 1, "A": 0})
        df["_A_Points"] = df["Result"].map({"A": 3, "D": 1, "H": 0})
        df["H_FormPts"] = self._rolling_sum(
            df, "Home", df["_H_Points"].fillna(0))
        df["A_FormPts"] = self._rolling_sum(
            df, "Away", df["_A_Points"].fillna(0))
        df.drop(columns=["_H_Points", "_A_Points"], inplace=True,
                errors="ignore")

        # --- Scoring First Rate (approximated: scored at halftime) ---
        # This requires HT data which isn't always available

        return df

    # ============================================================
    # 3. REFEREE FEATURES
    # ============================================================

    def compute_referee_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-referee aggregated statistics.
        Requires 'Referee' column in the dataframe.
        """
        if "Referee" not in df.columns:
            return df

        df = df.copy()

        # Group by referee and compute historical stats
        ref_stats = df.groupby("Referee").agg(
            ref_matches=("HG", "count"),
            ref_avg_goals=("HG", lambda x: (x + df.loc[x.index, "AG"]).mean()),
            ref_avg_yellows=("HY", lambda x: (
                x + df.loc[x.index, "AY"]).mean()
                if "HY" in df.columns else 0),
            ref_home_wins=("Result", lambda x: (x == "H").mean()),
        ).reset_index()

        # Merge back (each match gets the referee's historical profile)
        df = df.merge(ref_stats, on="Referee", how="left")

        # Referee strictness score
        if "HY" in df.columns:
            overall_avg_cards = (df["HY"] + df["AY"]).mean()
            df["Ref_Strictness"] = (
                df["ref_avg_yellows"] / max(overall_avg_cards, 1)
            )
        else:
            df["Ref_Strictness"] = 0.5

        return df

    # ============================================================
    # 4. MATCH CONTEXT FEATURES (from live API data)
    # ============================================================

    @staticmethod
    def context_to_features(ctx: MatchContext) -> Dict[str, float]:
        """
        Convert a MatchContext object into a flat feature dictionary.
        These features supplement the historical rolling stats.
        """
        features = {}

        # --- Squad Strength ---
        if ctx.home_squad:
            features["home_squad_strength"] = ctx.home_squad.squad_strength_score
            features["home_injured_count"] = len(ctx.home_squad.injured_players)
            features["home_suspended_count"] = len(ctx.home_squad.suspended_players)
            features["home_missing_key_count"] = len(ctx.home_squad.missing_key_players)
            features["home_available_key_count"] = len(ctx.home_squad.available_key_players)
        else:
            features.update({
                "home_squad_strength": 1.0,
                "home_injured_count": 0,
                "home_suspended_count": 0,
                "home_missing_key_count": 0,
                "home_available_key_count": 0
            })

        if ctx.away_squad:
            features["away_squad_strength"] = ctx.away_squad.squad_strength_score
            features["away_injured_count"] = len(ctx.away_squad.injured_players)
            features["away_suspended_count"] = len(ctx.away_squad.suspended_players)
            features["away_missing_key_count"] = len(ctx.away_squad.missing_key_players)
            features["away_available_key_count"] = len(ctx.away_squad.available_key_players)
        else:
            features.update({
                "away_squad_strength": 1.0,
                "away_injured_count": 0,
                "away_suspended_count": 0,
                "away_missing_key_count": 0,
                "away_available_key_count": 0
            })

        # Squad strength differential
        features["squad_strength_diff"] = (
            features["home_squad_strength"] - features["away_squad_strength"]
        )

        # --- Referee Features ---
        if ctx.referee:
            features["ref_avg_yellows"] = ctx.referee.avg_yellows_per_match
            features["ref_avg_goals"] = ctx.referee.avg_goals_per_match
            features["ref_home_bias"] = ctx.referee.home_win_pct
            features["ref_strictness"] = ctx.referee.card_strictness_score
        else:
            features.update({
                "ref_avg_yellows": 0,
                "ref_avg_goals": 0,
                "ref_home_bias": 0.5,
                "ref_strictness": 0.5
            })

        # --- H2H Features ---
        h2h_total = max(ctx.h2h_matches, 1)
        features["h2h_home_win_rate"] = ctx.h2h_home_wins / h2h_total
        features["h2h_away_win_rate"] = ctx.h2h_away_wins / h2h_total
        features["h2h_draw_rate"] = ctx.h2h_draws / h2h_total
        features["h2h_avg_goals"] = ctx.h2h_avg_goals
        features["h2h_dominance"] = (
            (ctx.h2h_home_wins - ctx.h2h_away_wins) / h2h_total
        )

        # --- xG Features ---
        features["home_xg_for"] = ctx.home_xg_for
        features["home_xg_against"] = ctx.home_xg_against
        features["away_xg_for"] = ctx.away_xg_for
        features["away_xg_against"] = ctx.away_xg_against
        features["home_xg_diff"] = ctx.home_xg_for - ctx.home_xg_against
        features["away_xg_diff"] = ctx.away_xg_for - ctx.away_xg_against
        features["xg_superiority"] = (
            (ctx.home_xg_for - ctx.home_xg_against) -
            (ctx.away_xg_for - ctx.away_xg_against)
        )

        # --- Odds-Implied Probabilities ---
        if ctx.odds_home > 0:
            total_implied = (
                1/ctx.odds_home + 1/ctx.odds_draw + 1/ctx.odds_away
            )
            features["implied_prob_home"] = (
                (1/ctx.odds_home) / total_implied if total_implied > 0 else 0.33
            )
            features["implied_prob_draw"] = (
                (1/ctx.odds_draw) / total_implied if total_implied > 0 else 0.33
            )
            features["implied_prob_away"] = (
                (1/ctx.odds_away) / total_implied if total_implied > 0 else 0.33
            )
            features["odds_home"] = ctx.odds_home
            features["odds_draw"] = ctx.odds_draw
            features["odds_away"] = ctx.odds_away
        else:
            features.update({
                "implied_prob_home": 0.33,
                "implied_prob_draw": 0.33,
                "implied_prob_away": 0.33,
                "odds_home": 0, "odds_draw": 0, "odds_away": 0
            })

        # --- News Sentiment ---
        features["home_sentiment"] = ctx.home_news_sentiment
        features["away_sentiment"] = ctx.away_news_sentiment
        features["sentiment_diff"] = (
            ctx.home_news_sentiment - ctx.away_news_sentiment
        )

        return features

    # ============================================================
    # 5. POISSON MODEL FEATURES
    # ============================================================

    @staticmethod
    def compute_poisson_features(
        home_attack: float, home_defense: float,
        away_attack: float, away_defense: float,
        league_avg_goals: float = 1.35
    ) -> Dict[str, float]:
        """
        Compute Poisson-based probability features.
        Uses attack/defense strengths relative to league average.

        Parameters:
            home_attack: Home team's avg goals scored / league avg
            home_defense: Home team's avg goals conceded / league avg
            away_attack: Away team's avg goals scored / league avg
            away_defense: Away team's avg goals conceded / league avg
            league_avg_goals: League average goals per team per match
        """
        from scipy.stats import poisson

        # Expected goals
        home_lambda = home_attack * away_defense * league_avg_goals
        away_lambda = away_attack * home_defense * league_avg_goals

        # Clamp to reasonable range
        home_lambda = np.clip(home_lambda, 0.1, 5.0)
        away_lambda = np.clip(away_lambda, 0.1, 5.0)

        # Build probability matrix (0-6 goals each)
        max_goals = 7
        prob_matrix = np.zeros((max_goals, max_goals))
        for i in range(max_goals):
            for j in range(max_goals):
                prob_matrix[i, j] = (
                    poisson.pmf(i, home_lambda) *
                    poisson.pmf(j, away_lambda)
                )

        # Extract probabilities
        home_win = np.sum(np.tril(prob_matrix, -1))
        draw = np.sum(np.diag(prob_matrix))
        away_win = np.sum(np.triu(prob_matrix, 1))

        over25 = 1 - sum(
            prob_matrix[i, j]
            for i in range(max_goals)
            for j in range(max_goals)
            if i + j <= 2
        )
        btts = 1 - sum(
            prob_matrix[i, j]
            for i in range(max_goals)
            for j in range(max_goals)
            if i == 0 or j == 0
        )

        return {
            "poisson_home_win": round(home_win, 4),
            "poisson_draw": round(draw, 4),
            "poisson_away_win": round(away_win, 4),
            "poisson_over25": round(over25, 4),
            "poisson_btts": round(btts, 4),
            "poisson_home_lambda": round(home_lambda, 3),
            "poisson_away_lambda": round(away_lambda, 3),
        }

    # ============================================================
    # HELPER: Rolling aggregation
    # ============================================================

    def _rolling_sum(self, df: pd.DataFrame, group_col: str,
                     values: pd.Series) -> pd.Series:
        """
        ProphitBet-style rolling sum: shift 1, roll N, sum.
        """
        temp = df[[group_col]].copy()
        temp["_val"] = values.values
        result = temp.groupby(group_col)["_val"].transform(
            lambda x: x.shift(1).rolling(window=self.n,
                                         min_periods=self.n).sum()
        )
        return result


# ============================================================
# FEATURE VECTOR BUILDER
# ============================================================

class FeatureVectorBuilder:
    """
    Combines all feature sources into a single feature vector
    ready for ML model input.
    """

    # Feature groups for easy toggling
    ROLLING_FEATURES = [
        "HW", "AW", "HL", "AL", "HGF", "AGF", "HGA", "AGA",
        "HGD", "AGD", "HAGF", "HAGA", "HAGD",
    ]
    ENHANCED_ROLLING = [
        "H_Shots_Roll", "A_Shots_Roll", "H_Shot_Accuracy", "A_Shot_Accuracy",
        "H_Yellows_Roll", "A_Yellows_Roll", "H_Reds_Roll", "A_Reds_Roll",
        "H_Fouls_Roll", "A_Fouls_Roll", "H_Discipline", "A_Discipline",
        "H_Conversion", "A_Conversion", "H_Draw_Rate", "A_Draw_Rate",
        "H_CleanSheet", "A_CleanSheet", "H_BTTS_Rate", "A_BTTS_Rate",
        "H_Over25_Rate", "A_Over25_Rate", "H_FormPts", "A_FormPts",
    ]
    CONTEXT_FEATURES = [
        "home_squad_strength", "away_squad_strength", "squad_strength_diff",
        "home_injured_count", "away_injured_count",
        "home_missing_key_count", "away_missing_key_count",
        "ref_strictness", "ref_home_bias",
        "h2h_home_win_rate", "h2h_away_win_rate", "h2h_dominance",
        "h2h_avg_goals",
        "home_xg_for", "home_xg_against", "away_xg_for", "away_xg_against",
        "xg_superiority",
        "implied_prob_home", "implied_prob_draw", "implied_prob_away",
        "home_sentiment", "away_sentiment", "sentiment_diff",
    ]
    POISSON_FEATURES = [
        "poisson_home_win", "poisson_draw", "poisson_away_win",
        "poisson_over25", "poisson_btts",
    ]

    def get_all_feature_names(self) -> List[str]:
        """Return all possible feature names."""
        return (self.ROLLING_FEATURES + self.ENHANCED_ROLLING +
                self.CONTEXT_FEATURES + self.POISSON_FEATURES)

    def build_feature_vector(
        self,
        rolling_row: pd.Series,
        context_features: Dict[str, float],
        poisson_features: Dict[str, float]
    ) -> np.ndarray:
        """
        Build a single feature vector for one match.
        """
        features = []

        # Rolling features from dataframe row
        for col in self.ROLLING_FEATURES + self.ENHANCED_ROLLING:
            features.append(float(rolling_row.get(col, 0.0)))

        # Context features from API data
        for col in self.CONTEXT_FEATURES:
            features.append(float(context_features.get(col, 0.0)))

        # Poisson features
        for col in self.POISSON_FEATURES:
            features.append(float(poisson_features.get(col, 0.0)))

        return np.array(features, dtype=np.float32)

    def build_training_matrix(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build X, y matrices from a fully-featured dataframe.
        For training, context and Poisson features should already be
        merged into the dataframe columns.
        """
        if feature_columns is None:
            feature_columns = [c for c in self.get_all_feature_names()
                              if c in df.columns]

        df_clean = df.dropna(subset=feature_columns + ["Result"])

        X = df_clean[feature_columns].values.astype(np.float32)
        y = df_clean["Result"].map({"H": 0, "D": 1, "A": 2}).values

        return X, y
