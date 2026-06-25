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

        # eddwebster/football_analytics inspired: player similarity + squad valuation (TM-style + clustering ideas)
        # Use available player stats from squad for contrib, add valuation if attached to ctx
        home_players = []
        if ctx.home_squad and ctx.home_squad.available_key_players:
            home_players = [ {"goals_season": p.goals_season, "assists_season": p.assists_season, "xg_per_90": p.xg_per_90, "xa_per_90": p.xa_per_90 } for p in ctx.home_squad.available_key_players ]
        away_players = []
        if ctx.away_squad and ctx.away_squad.available_key_players:
            away_players = [ {"goals_season": p.goals_season, "assists_season": p.assists_season, "xg_per_90": p.xg_per_90, "xa_per_90": p.xa_per_90 } for p in ctx.away_squad.available_key_players ]

        home_val = getattr(ctx, "home_market_values", None)  # dict name->value if collected (e.g. TM)
        away_val = getattr(ctx, "away_market_values", None)

        home_sim_val = compute_player_similarity_and_valuation_features(home_players, home_val)
        away_sim_val = compute_player_similarity_and_valuation_features(away_players, away_val)

        for k, v in home_sim_val.items():
            features[f"home_{k}"] = v
        for k, v in away_sim_val.items():
            features[f"away_{k}"] = v

        # ALL priority gap features wired (workload, setpieces, motivation, CLV, WC patterns, etc.)
        gap_kwargs = {
            "travel_days": getattr(ctx, "travel_days", 0),
            "rest_days": getattr(ctx, "rest_days", 7),
            "is_dead_rubber": getattr(ctx, "is_dead_rubber", False),
            "model_prob": features.get("implied_prob_home", 0.6),
            "live_odds": getattr(ctx, "live_odds", 2.5),
            "closing_line": getattr(ctx, "closing_line", 2.3),
            "dead_rubber": getattr(ctx, "is_dead_rubber", False),
            "home_coach": getattr(ctx, "home_coach", None),
            "away_coach": getattr(ctx, "away_coach", None),
            "wc_history": getattr(ctx, "wc_history_detailed", None),
            "home_team": getattr(ctx, "home_team", ""),
            "away_team": getattr(ctx, "away_team", ""),
        }
        all_gaps = compute_all_priority_features(**gap_kwargs)
        for k, v in all_gaps.items():
            features[k] = v

        # Coach / trainer style (how they let the team play) + historical WC (jfjelstul multi-year + recent this-year)
        home_coach = getattr(ctx, "home_coach", None) or getattr(ctx, "coach", None)
        away_coach = getattr(ctx, "away_coach", None)
        if home_coach or away_coach:
            coach_feats = compute_coach_tactical_features(home_coach or {}, away_coach or {})
            for k, v in coach_feats.items():
                features[k] = v
            # also add matchup directly if present on ctx
            matchup = getattr(ctx, "coach_matchup", None)
            if matchup and isinstance(matchup, dict) and "style_clash_score" in matchup:
                features["coach_matchup"] = matchup.get("style_clash_score", features.get("coach_matchup", 0.0))

        hist = getattr(ctx, "wc_history_detailed", None) or getattr(ctx, "home_wc_history", None) or getattr(ctx, "wc_historical", None)
        if isinstance(hist, dict):
            r = hist.get("recent_wcs", hist)
            if isinstance(r, dict):
                features["wc_hist_win_rate"] = r.get("win_rate", 0.0)
            elif "win_rate" in hist:
                features["wc_hist_win_rate"] = hist.get("win_rate", 0.0)

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

        # --- WC / ENVIRONMENTAL CONTEXT FEATURES (new for WK bot) ---
        features["is_wc"] = 1.0 if getattr(ctx, "is_national_team", False) else 0.0
        features["altitude_effect"] = 0.92 if getattr(ctx, "venue", "") and "azteca" in getattr(ctx, "venue", "").lower() else 1.0
        features["crowd_factor"] = getattr(ctx, "crowd_factor", 1.0)
        features["climate_adapt"] = getattr(ctx, "climate_adapt_score", 1.0)
        features["travel_fatigue"] = max(0.85, 1.0 - (getattr(ctx, "travel_days", 0) * 0.015))
        # Stage importance: group stage dead rubbers lower variance
        stage = getattr(ctx, "stage", "group").lower()
        features["stage_importance"] = 1.15 if "final" in stage or "qf" in stage or "sf" in stage else (0.95 if "group" in stage else 1.0)
        features["expected_crowd"] = float(getattr(ctx, "expected_crowd", 45000) or 45000) / 80000.0  # normalized

        # Weather summary
        ws = getattr(ctx, "weather_summary", {}) or {}
        features["heat_risk"] = 0.88 if ws.get("altitude_m", 0) > 1500 else 0.95
        features["indoor_controlled"] = 1.02 if ws.get("roof") else 1.0

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

        # --- soccerdata WhoScored + FBref enrichment (when present in context) ---
        # WhoScored (etc.) provides possession, shots, ratings
        ws_sample = getattr(ctx, "whoscored_player_sample", None) or getattr(ctx, "whoscored_schedule", None)
        if ws_sample or getattr(ctx, "whoscored_schedule", None):
            try:
                ws_dict = {"player_match_stats": pd.DataFrame(getattr(ctx, "whoscored_player_sample", [])) if getattr(ctx, "whoscored_player_sample", None) else pd.DataFrame()}
                if getattr(ctx, "whoscored_schedule", None) is not None:
                    ws_dict["schedule"] = getattr(ctx, "whoscored_schedule")
                add_whoscored_features_to_context(features, ws_dict, getattr(ctx, "home_team", ""), getattr(ctx, "away_team", ""))
            except Exception:
                pass

        # FBref soccerdata xG if a df sample exists on context (backward)
        sd_sample = getattr(ctx, "sd_team_xg_sample", None)
        if sd_sample:
            try:
                sd_df = pd.DataFrame(sd_sample) if not isinstance(sd_sample, pd.DataFrame) else sd_sample
                add_soccerdata_features_to_context(features, sd_df, getattr(ctx, "home_team", ""), getattr(ctx, "away_team", ""))
            except Exception:
                pass

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
        # WC additions
        "is_wc", "altitude_effect", "crowd_factor", "climate_adapt",
        "travel_fatigue", "stage_importance", "expected_crowd",
        "heat_risk", "indoor_controlled",
        # eddwebster-inspired squad valuation & similarity
        "home_squad_market_value_eur", "away_squad_market_value_eur",
        "home_squad_contrib_score", "away_squad_contrib_score",
        "home_squad_contrib_std", "away_squad_contrib_std",
        # All gap priorities (workload, setpiece, motivation, CLV, WC patterns, tactical)
        "fatigue_modifier", "recovery_score", "travel_impact", "jetlag_modifier",
        "setpiece_value", "style_matchup_adv", "tactical_evolution",
        "motivation_mult", "public_bias_adj", "fan_pressure",
        "clv", "live_edge",
        "dead_rubber_boost", "generational_adj", "prep_quality",
        # Coach/trainer style + multi-year WC/EC historical (past + this year qualifiers)
        "coach_matchup", "home_press", "away_press", "home_poss_pref", "away_poss_pref",
        "home_direct", "away_direct", "home_setpiece_focus", "away_setpiece_focus",
        "coach_formation_match", "wc_hist_win_rate",
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


# ============================================================
# SOCCERDATA HELPERS (PRIMARY source integration)
# ============================================================

def compute_soccerdata_xg_features(team_stats_df: pd.DataFrame, 
                                    home_team: str, away_team: str) -> Dict[str, float]:
    """
    Extract xG superiority, per90 and efficiency from soccerdata FBref-style DF.
    Call with result of SoccerDataWrapper.get_fbref_team_xg() or read_team_season_stats("shooting").
    """
    feats = {
        "sd_xg_diff": 0.0,
        "sd_xg_home": 0.0,
        "sd_xg_away": 0.0,
        "sd_xg_superiority": 0.0,
    }
    if team_stats_df is None or team_stats_df.empty or "team" not in team_stats_df.columns:
        return feats
    try:
        df = team_stats_df.copy()
        # Heuristics for column names coming from FBref (xG, Gls, etc.)
        xg_col = None
        for c in df.columns:
            cl = str(c).lower()
            if cl in ("xg", "x_g", "expected_goals"):
                xg_col = c
                break
        if xg_col is None:
            for c in df.columns:
                if "xg" in str(c).lower():
                    xg_col = c
                    break
        gls_col = "Gls" if "Gls" in df.columns else None
        mins_col = "90s" if "90s" in df.columns else None

        def _find_row(name: str):
            m = df["team"].str.contains(name, case=False, na=False)
            return df[m].iloc[0] if m.any() else None

        hrow = _find_row(home_team)
        arow = _find_row(away_team)

        if hrow is not None:
            feats["sd_xg_home"] = float(hrow.get(xg_col, hrow.get(gls_col, 0)) or 0)
            if mins_col and float(hrow.get(mins_col, 1) or 1) > 0:
                feats["sd_xg_home"] /= float(hrow.get(mins_col, 1))
        if arow is not None:
            feats["sd_xg_away"] = float(arow.get(xg_col, arow.get(gls_col, 0)) or 0)
            if mins_col and float(arow.get(mins_col, 1) or 1) > 0:
                feats["sd_xg_away"] /= float(arow.get(mins_col, 1))

        feats["sd_xg_diff"] = feats["sd_xg_home"] - feats["sd_xg_away"]
        # positive = home expected to dominate on xG
        feats["sd_xg_superiority"] = feats["sd_xg_diff"]
    except Exception:
        pass
    return feats


def add_soccerdata_features_to_context(context_features: Dict, sd_df: pd.DataFrame,
                                        home: str, away: str) -> Dict:
    """Merge soccerdata xG features into existing context feature dict."""
    xg = compute_soccerdata_xg_features(sd_df, home, away)
    # avoid double "sd_" if compute already prefixed in future
    for k, v in xg.items():
        key = k if k.startswith("sd_") else f"sd_{k}"
        context_features[key] = v
    return context_features


def compute_whoscored_features(ws_data: Dict, home_team: str, away_team: str) -> Dict[str, float]:
    """Extract betting-relevant features from WhoScored data (via soccerdata).
    Includes possession, shot volume, card tendencies, average player ratings if present.
    WhoScored often gives different angles than FBref (great for spotting mispricings)."""
    feats = {
        "ws_home_poss": 50.0,
        "ws_away_poss": 50.0,
        "ws_home_shots": 0.0,
        "ws_away_shots": 0.0,
        "ws_home_rating": 6.5,
        "ws_away_rating": 6.5,
        "ws_card_diff": 0.0,
    }
    if not ws_data or "error" in ws_data:
        return feats
    try:
        # Team match stats if present
        tms = ws_data.get("team_match_stats")
        pms = ws_data.get("player_match_stats")
        # Heuristic matching
        def _match_team(df, name):
            if df is None or df.empty or "team" not in df.columns:
                return None
            m = df["team"].astype(str).str.contains(name, case=False, na=False)
            return df[m].iloc[0] if m.any() else None

        if isinstance(tms, pd.DataFrame) and not tms.empty:
            hr = _match_team(tms, home_team)
            ar = _match_team(tms, away_team)
            if hr is not None:
                feats["ws_home_poss"] = float(hr.get("Poss", hr.get("possession", 50)) or 50)
                feats["ws_home_shots"] = float(hr.get("Sh", hr.get("shots", 0)) or 0)
            if ar is not None:
                feats["ws_away_poss"] = float(ar.get("Poss", ar.get("possession", 50)) or 50)
                feats["ws_away_shots"] = float(ar.get("Sh", ar.get("shots", 0)) or 0)

        if isinstance(pms, pd.DataFrame) and not pms.empty:
            # Average rating per team if 'rating' or 'player_rating' columns exist
            cols = [c for c in pms.columns if "rating" in str(c).lower() or "ratin" in str(c).lower()]
            if cols and "team" in pms.columns:
                for side, team in [("home", home_team), ("away", away_team)]:
                    tm = pms[pms["team"].astype(str).str.contains(team, case=False, na=False)]
                    if not tm.empty:
                        rcol = cols[0]
                        avg_r = float(tm[rcol].mean()) if pd.api.types.is_numeric_dtype(tm[rcol]) else 6.6
                        feats[f"ws_{side}_rating"] = round(avg_r, 2)
            # Cards proxy
            card_cols = [c for c in pms.columns if "card" in str(c).lower() or "yc" in str(c).lower() or "rc" in str(c).lower()]
            if card_cols:
                feats["ws_card_diff"] = float(pms[card_cols[0]].sum() if pd.api.types.is_numeric_dtype(pms[card_cols[0]]) else 0) * 0.1
    except Exception:
        pass
    return feats


def add_whoscored_features_to_context(context_features: Dict, ws_data: Dict, home: str, away: str) -> Dict:
    ws = compute_whoscored_features(ws_data, home, away)
    for k, v in ws.items():
        context_features[k] = v
    return context_features


def xg_from_shot_location(distance: float, angle: float = None, use_simple: bool = True) -> float:
    """
    Simple xG model from shot location, inspired by SoccermaticsForPython (FoTD).
    Uses distance (meters) and optionally angle (radians).
    This is the classic first xG model taught in the book/lectures.
    
    Can be used with StatsBomb shot data or approximated from other sources.
    """
    if distance <= 0:
        return 0.0
    
    # Classic distance-only logistic style (approximation from early models)
    if use_simple or angle is None:
        # Rough logistic: higher distance = lower xG
        # Tuned roughly from typical open data models
        xg = 1 / (1 + np.exp(0.15 * (distance - 12)))   # peaks near goal
        return max(0.01, min(0.95, xg))
    
    # Better: distance + angle (from 3xGModel.py logic)
    # Angle term increases xG for central shots
    angle_term = np.sin(angle) if angle else 1.0
    base = 1 / (1 + np.exp(0.12 * (distance - 10)))
    xg = base * (0.6 + 0.4 * angle_term)
    return max(0.01, min(0.98, xg))


def batch_xg_from_events(shots_df: pd.DataFrame) -> pd.Series:
    """
    Given a DataFrame of shots (with 'Distance' or location columns), compute xG.
    Compatible with StatsBombWCImporter event data or Wyscout-style.
    """
    if shots_df.empty:
        return pd.Series(dtype=float)
    
    df = shots_df.copy()
    if "Distance" not in df.columns and "location" in df.columns:
        # Approximate from StatsBomb [x,y] (x from own goal 0-120, convert)
        try:
            df["Distance"] = df["location"].apply(
                lambda loc: np.sqrt((120 - loc[0])**2 + (40 - loc[1])**2) * 105/120 if isinstance(loc, (list, tuple)) else 20
            )
        except Exception:
            df["Distance"] = 20.0
    
    distances = df.get("Distance", pd.Series([20]*len(df)))
    angles = df.get("Angle", None)
    
    if angles is not None:
        return distances.combine(angles, lambda d, a: xg_from_shot_location(d, a, use_simple=False))
    else:
        return distances.apply(lambda d: xg_from_shot_location(d))


def compute_vaep_style_features(team_action_values: Dict[str, float]) -> Dict[str, float]:
    """
    Convert socceraction VAEP outputs (or proxies) into features.
    VAEP = offensive_value + defensive_value per action.
    High team/player VAEP indicates strong contribution to positive game states.
    """
    feats = {}
    if not team_action_values:
        return feats
    feats["vaep_total_actions"] = team_action_values.get("total_actions", 0) or team_action_values.get("num_actions", 0)
    feats["vaep_passes"] = team_action_values.get("passes", 0)
    # In real use after full VAEP rating: feats["vaep_offensive"] = sum(ov for ov in ...)
    feats["vaep_quality_proxy"] = min(1.0, feats["vaep_total_actions"] / 800.0)  # rough normalization
    return feats


def compute_grf_synthetic_features(sim_results: Dict) -> Dict[str, float]:
    """
    Features from Google Research Football RL sims (synthetic data/RL agents).
    Use structured obs or episode stats for "RL-based" threat, possession, adaptation proxies.
    Augment real data for WC "what-if" (e.g., squad changes, condition variants).
    """
    feats = {}
    if not sim_results or "error" in sim_results:
        return feats
    feats["grf_sim_steps"] = sim_results.get("steps", 0)
    feats["grf_sim_reward"] = sim_results.get("total_reward", 0)
    feats["grf_estimated_xg_proxy"] = sim_results.get("sim_estimated_xg_proxy", 0)
    # Extend with obs parsing: e.g., avg player positioning variance as "tactics diversity"
    return feats


# ============================================================
# ALL PRIORITY GAP FEATURES (implemented for every missing high-value piece)
# ============================================================

def compute_workload_recovery_features(travel_days: int = 0, rest_days: int = 7, club_minutes: int = 400, nt_minutes: int = 200, jetlag_hours: float = 0.0) -> Dict[str, float]:
    """Detailed player workload & recovery (club+NT, travel, jetlag, fatigue)."""
    fatigue = max(0.7, 1.0 - (club_minutes + nt_minutes) / 1200)
    recovery = min(1.0, rest_days / 5.0)
    travel_impact = max(0.8, 1.0 - travel_days / 10)
    jetlag_impact = max(0.85, 1.0 - abs(jetlag_hours) / 12)
    return {"fatigue_modifier": round(fatigue, 3), "recovery_score": round(recovery, 3), "travel_impact": round(travel_impact, 3), "jetlag_modifier": round(jetlag_impact, 3)}

def compute_setpiece_tactical_features(setpiece_xg: float = 0.1, corner_conv: float = 0.08, pressing_index: float = 0.5, formation_evolution: float = 0.0) -> Dict[str, float]:
    """Set-piece metrics + tactical/style matchups (pressing, formation)."""
    sp_value = setpiece_xg * 1.2 + corner_conv * 0.8
    style_advantage = (1 - pressing_index) if pressing_index > 0.6 else pressing_index
    return {"setpiece_value": round(sp_value, 4), "style_matchup_adv": round(style_advantage, 3), "tactical_evolution": round(formation_evolution, 2)}


def compute_coach_tactical_features(home_coach: Dict, away_coach: Dict) -> Dict[str, float]:
    """Trainer/coach playing style features + matchup (very important for WC).
    Data comes from CoachTacticalLoader (formation, press, possession preference, directness, setpiece emphasis).
    Supports both current "this year" coaches and historical coach impact across WC/EK.
    """
    feats = {
        "coach_matchup": 0.0,
        "home_press": 0.55, "away_press": 0.55,
        "home_poss_pref": 0.55, "away_poss_pref": 0.55,
        "home_direct": 0.40, "away_direct": 0.40,
        "home_setpiece_focus": 0.50, "away_setpiece_focus": 0.50,
        "coach_formation_match": 0.0,
    }
    if not home_coach and not away_coach:
        return feats
    try:
        hp = float(home_coach.get("press", home_coach.get("press_intensity", 0.55)))
        ap = float(away_coach.get("press", away_coach.get("press_intensity", 0.55)))
        hposs = float(home_coach.get("poss_pref", home_coach.get("possession_preference", 0.55)))
        aposs = float(away_coach.get("poss_pref", away_coach.get("possession_preference", 0.55)))
        hd = float(home_coach.get("direct", home_coach.get("direct_play", 0.40)))
        ad = float(away_coach.get("direct", away_coach.get("direct_play", 0.40)))
        hs = float(home_coach.get("setpiece", home_coach.get("setpiece_emphasis", 0.50)))
        as_ = float(away_coach.get("setpiece", away_coach.get("setpiece_emphasis", 0.50)))

        feats["home_press"] = round(hp, 3)
        feats["away_press"] = round(ap, 3)
        feats["home_poss_pref"] = round(hposs, 3)
        feats["away_poss_pref"] = round(aposs, 3)
        feats["home_direct"] = round(hd, 3)
        feats["away_direct"] = round(ad, 3)
        feats["home_setpiece_focus"] = round(hs, 3)
        feats["away_setpiece_focus"] = round(as_, 3)

        # matchup: positive favors the home side's style
        feats["coach_matchup"] = round((hp - ap) * 0.4 + (hposs - aposs) * 0.3 + (hd - ad) * 0.2 + (hs - as_) * 0.1, 4)

        # crude formation similarity
        hf = str(home_coach.get("formation", ""))
        af = str(away_coach.get("formation", ""))
        feats["coach_formation_match"] = 0.08 if hf[:3] == af[:3] else -0.04
    except Exception:
        pass
    return feats

def compute_motivation_bias_features(is_dead_rubber: bool = False, public_bet_pct: float = 60.0, fan_pressure: float = 1.0, must_win: bool = False) -> Dict[str, float]:
    """Psychological/motivational + public bias."""
    motivation = 1.15 if must_win else (0.9 if is_dead_rubber else 1.0)
    bias = 1.0 + (public_bet_pct - 50) / 200   # fade heavy public money
    return {"motivation_mult": round(motivation, 3), "public_bias_adj": round(bias, 3), "fan_pressure": round(fan_pressure, 2)}

def compute_clv_live_features(model_prob: float, live_odds: float = 2.5, closing_line: float = 2.3) -> Dict[str, float]:
    """Live/closing odds + real CLV."""
    clv = (model_prob * closing_line) - 1
    live_value = (model_prob * live_odds) - 1
    return {"clv": round(clv, 4), "live_edge": round(live_value, 4)}

def compute_wc_historical_patterns(dead_rubber: bool = False, generational_cycle: float = 1.0, prep_quality: float = 1.0) -> Dict[str, float]:
    """WC-specific historical patterns (dead rubber, generational, preparation)."""
    dr_boost = 1.08 if dead_rubber else 1.0
    return {"dead_rubber_boost": round(dr_boost, 3), "generational_adj": round(generational_cycle, 3), "prep_quality": round(prep_quality, 3)}

def compute_all_priority_features(**kwargs) -> Dict[str, float]:
    """One-call aggregator for every priority gap feature + soccerdata/WhoScored etc when passed in."""
    feats = {}
    feats.update(compute_workload_recovery_features(**{k: kwargs.get(k, v) for k, v in {"travel_days":0,"rest_days":7,"club_minutes":400,"nt_minutes":200,"jetlag_hours":0}.items()}))
    feats.update(compute_setpiece_tactical_features(**{k: kwargs.get(k, v) for k, v in {"setpiece_xg":0.1,"corner_conv":0.08,"pressing_index":0.5,"formation_evolution":0.0}.items()}))
    feats.update(compute_motivation_bias_features(**{k: kwargs.get(k, v) for k, v in {"is_dead_rubber":False,"public_bet_pct":60.0,"fan_pressure":1.0,"must_win":False}.items()}))
    feats.update(compute_clv_live_features(**{k: kwargs.get(k, v) for k, v in {"model_prob":0.6,"live_odds":2.5,"closing_line":2.3}.items()}))
    feats.update(compute_wc_historical_patterns(**{k: kwargs.get(k, v) for k, v in {"dead_rubber":False,"generational_cycle":1.0,"prep_quality":1.0}.items()}))

    # WhoScored / soccerdata "etc" sources directly if dicts passed in
    ws = kwargs.get("whoscored_data")
    if ws:
        ws_f = compute_whoscored_features(ws, kwargs.get("home_team", ""), kwargs.get("away_team", ""))
        for k, v in ws_f.items():
            feats[k] = v
    sd_df = kwargs.get("soccerdata_team_xg")
    if sd_df is not None:
        try:
            import pandas as pd
            if isinstance(sd_df, list):
                sd_df = pd.DataFrame(sd_df)
            xg_f = compute_soccerdata_xg_features(sd_df, kwargs.get("home_team", ""), kwargs.get("away_team", ""))
            for k, v in xg_f.items():
                feats[k] = v
        except Exception:
            pass

    # Coach / trainer tactical (how they let play) + historical WC/EC multi-year
    coach_h = kwargs.get("home_coach") or kwargs.get("coach_home")
    coach_a = kwargs.get("away_coach") or kwargs.get("coach_away")
    if coach_h or coach_a:
        ch = compute_coach_tactical_features(coach_h or {}, coach_a or {})
        for k, v in ch.items():
            feats[k] = v
    hist = kwargs.get("wc_history") or kwargs.get("wc_historical_multi")
    if hist:
        # simple extraction
        feats["wc_hist_win_rate"] = hist.get("recent_wcs", {}).get("win_rate", hist.get("all_time", {}).get("win_rate", 0.0)) if isinstance(hist, dict) else 0.0
    return feats


def compute_xt_features(actions_df: pd.DataFrame) -> Dict[str, float]:
    """
    xT (Expected Threat) features from federicorabanos/futbol-data-visualizacion style.
    Uses socceraction.xthreat on SPADL-like data (from StatsBombWCImporter or socceraction).
    Computes team xT for moves (passes/dribbles) — measures progressive threat/creation value.
    Great for WC features: team "danger" creation, especially in open play.
    Reference: xT grids (l=12,w=8), fit on events, predict per action.
    """
    feats = {"team_xt": 0.0, "xt_per_action": 0.0}
    if actions_df is None or actions_df.empty:
        return feats
    try:
        import socceraction.xthreat as xthreat
        # Assume actions_df is SPADL format with start/end x/y (0-100 or scaled)
        # Scale if needed (socceraction expects ~105x68 normalized often)
        mov_actions = xthreat.get_successful_move_actions(actions_df)
        if mov_actions.empty:
            return feats
        # Fit or load simple model (in practice prefit gridxT.json)
        xt_model = xthreat.ExpectedThreat(l=12, w=8)
        # For demo, assume prefit or fit on sample; in real use load from previous
        xt_model.fit(mov_actions)  # placeholder; cache in practice
        mov_actions["xT_value"] = xt_model.predict(mov_actions)
        team_xt = mov_actions["xT_value"].sum()
        feats["team_xt"] = round(team_xt, 4)
        feats["xt_per_action"] = round(team_xt / max(len(mov_actions), 1), 5)
        feats["xt_positive_actions"] = len(mov_actions[mov_actions["xT_value"] > 0])
    except Exception as e:
        feats["xt_error"] = str(e)[:50]
    return feats


def compute_player_similarity_and_valuation_features(player_stats: List[Dict], squad_market_values: Dict[str, float] = None) -> Dict[str, float]:
    """
    Inspired by eddwebster/football_analytics:
    - player_similarity_and_clustering notebooks (PCA + K-Means on per90 stats like goals, assists, xG, progressive passes, etc. to find "Piqué-like" or similar players).
    - TM valuation notebooks (market values for squad depth, expected contribution "gaan doen").
    Features for WC squad analysis: valuation aggregates + similarity/diversity scores to assess bench strength and find comparable players.
    Extend with full clustering (use sklearn PCA/KMeans on normalized stats from soccerdata/StatsBomb).
    """
    feats = {}
    if not player_stats:
        return feats

    n = max(len(player_stats), 1)

    # Squad contribution (goals + assists as proxy; enhance with xG, xa from stats)
    total_contrib = sum(
        p.get('goals_season', 0) + p.get('assists_season', 0) +
        p.get('xg_per_90', 0) * 90 + p.get('xa_per_90', 0) * 90   # rough per season if per90
        for p in player_stats
    )
    feats["squad_contrib_score"] = total_contrib / n

    # Valuation features (plug in real TM data via scraper or API)
    if squad_market_values:
        total_val = sum(squad_market_values.values())
        feats["squad_market_value_eur"] = total_val
        feats["avg_player_value_eur"] = total_val / len(squad_market_values)
        feats["top_player_value_ratio"] = max(squad_market_values.values()) / (total_val / len(squad_market_values)) if squad_market_values else 0

    # Similarity / diversity proxy (variance in contrib as "depth"; in full notebook style use stats vectors)
    contribs = [
        p.get('goals_season', 0) + p.get('assists_season', 0) +
        p.get('xg_per_90', 0) * 90 + p.get('xa_per_90', 0) * 90
        for p in player_stats
    ]
    if contribs and len(contribs) > 1:
        feats["squad_contrib_std"] = float(np.std(contribs))
        feats["squad_contrib_diversity"] = feats["squad_contrib_std"] / (np.mean(contribs) + 1e-6)  # CV as diversity

    # Placeholder for player similarity score (e.g., for a key player vs squad avg)
    if len(player_stats) > 0:
        avg_contrib = sum(contribs) / n
        feats["key_player_similarity_to_avg"] = min(1.0, avg_contrib / (max(contribs) + 1))  # rough

    return feats


def compute_tracking_workload_features(tracking_data: Dict) -> Dict[str, float]:
    """
    From roboflow/sports tracking output (or equivalent position data over time):
    Compute physical workload features for player/team form and adaptation.
    Critical for WC: fatigue, recovery, heat/altitude effects on movement.
    """
    feats = {}
    if not tracking_data or "error" in tracking_data:
        return feats
    # Examples (populate from real extract)
    feats["est_distance_covered_m"] = tracking_data.get("estimated_player_distance_per_90", 8500)
    feats["avg_speed_kmh"] = tracking_data.get("avg_team_speed_kmh", 7.0)
    feats["high_intensity_ratio"] = tracking_data.get("high_intensity_minutes", 10) / 90.0
    # Can feed into adaptation modifiers or player availability
    return feats


def compute_event_features_from_statsbomb(event_feats: Dict[str, float]) -> Dict[str, float]:
    """
    Convert raw event-derived stats (from StatsBombWCImporter / handbook style)
    into model-ready features. Examples of soccer analytics techniques:
    - Shot volume + rough quality (distance)
    - Pass success and progression rates (proxy for build-up / threat)
    These can supplement xG from soccerdata or be used to derive "team style".
    """
    out = {}
    if not event_feats:
        return out
    out["sb_shots"] = float(event_feats.get("shots", 0))
    out["sb_pass_success"] = float(event_feats.get("pass_success_rate", 0.0))
    out["sb_forward_pass_rate"] = float(event_feats.get("forward_pass_rate", 0.0))
    # Simple composite "creation" score
    creation = (out["sb_pass_success"] * 0.4 + out["sb_forward_pass_rate"] * 0.6) if out.get("sb_forward_pass_rate") else out["sb_pass_success"]
    out["sb_creation_score"] = round(creation, 4)
    if "avg_shot_dist_sq" in event_feats and event_feats["avg_shot_dist_sq"] > 0:
        # smaller distance = better quality shots (inverse proxy)
        out["sb_shot_quality"] = round(1.0 / (1 + event_feats["avg_shot_dist_sq"] / 10000), 3)
    return out
