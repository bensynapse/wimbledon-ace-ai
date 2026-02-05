"""
Enhanced Prediction Model Module
Ensemble of multiple classifiers with value bet detection.

Models included:
  - XGBoost (gradient boosting)
  - Random Forest
  - Logistic Regression (calibrated probabilities)
  - Neural Network (optional)

Inspired by ProphitBet's multi-model approach but with:
  - Ensemble averaging for more robust predictions
  - Proper probability calibration
  - Value bet detection with Kelly Criterion
  - Confidence scoring
"""

import numpy as np
import pandas as pd
import pickle
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    log_loss, brier_score_loss
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


# ============================================================
# BET RECOMMENDATION
# ============================================================

@dataclass
class BetRecommendation:
    """Complete bet recommendation with reasoning."""
    match: str
    date: str
    league: str
    # Probabilities
    prob_home: float
    prob_draw: float
    prob_away: float
    prob_over25: float
    prob_btts: float
    # Odds
    odds_home: float
    odds_draw: float
    odds_away: float
    odds_over25: float
    odds_under25: float
    # Value analysis
    value_home: float  # edge = prob * odds - 1
    value_draw: float
    value_away: float
    value_over25: float
    # Best bet
    best_bet: str
    best_value: float
    confidence: float  # 0-1 model confidence
    kelly_stake_pct: float
    # Context
    key_factors: List[str]
    risk_level: str  # "low", "medium", "high"


# ============================================================
# ENSEMBLE PREDICTOR
# ============================================================

class EnsemblePredictor:
    """
    Ensemble model combining multiple classifiers.
    Uses soft voting with calibrated probabilities.
    """

    def __init__(self, model_dir: str = "models/"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.scaler = StandardScaler()
        self.models: Dict[str, Any] = {}
        self.ensemble = None
        self.is_trained = False
        self.feature_names: List[str] = []
        self.feature_importances: Dict[str, float] = {}

    def build_models(self) -> Dict[str, Any]:
        """Build the individual classifiers."""
        models = {
            "xgboost": XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_lambda=1.0,
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),
            "logistic": LogisticRegression(
                C=1.0,
                max_iter=1000,
                multi_class="multinomial",
                class_weight="balanced",
                random_state=42
            )
        }
        self.models = models
        return models

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        cv_folds: int = 5,
        calibrate: bool = True
    ) -> Dict[str, Any]:
        """
        Train the ensemble with cross-validation evaluation.

        Returns dict with per-model and ensemble metrics.
        """
        self.feature_names = feature_names or [
            f"f_{i}" for i in range(X.shape[1])
        ]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Build models
        self.build_models()

        # Train and evaluate each model
        results = {}
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        trained_models = {}
        for name, model in self.models.items():
            logger.info(f"Training {name}...")

            # Cross-validation scores
            cv_scores = cross_val_score(
                model, X_scaled, y, cv=cv, scoring="accuracy"
            )

            # Fit on full data
            model.fit(X_scaled, y)

            # Calibrate probabilities
            if calibrate:
                cal_model = CalibratedClassifierCV(
                    estimator=model, method="isotonic",
                    cv=cv_folds
                )
                cal_model.fit(X_scaled, y)
                trained_models[name] = cal_model
            else:
                trained_models[name] = model

            results[name] = {
                "cv_accuracy_mean": round(cv_scores.mean(), 4),
                "cv_accuracy_std": round(cv_scores.std(), 4),
                "train_accuracy": round(
                    accuracy_score(y, model.predict(X_scaled)), 4
                )
            }
            logger.info(
                f"  {name}: CV={cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
            )

        # Build ensemble (soft voting)
        self.ensemble = VotingClassifier(
            estimators=list(trained_models.items()),
            voting="soft"
        )
        # VotingClassifier needs to be fit, but our sub-models are already fit
        # So we manually set estimators
        self.ensemble.estimators_ = list(trained_models.values())
        self.ensemble.le_ = self.ensemble.estimators_[0].classes_ if hasattr(
            self.ensemble.estimators_[0], 'classes_') else np.unique(y)

        # Ensemble CV score
        ensemble_preds = self.predict_proba(X)
        ensemble_pred_classes = np.argmax(ensemble_preds, axis=1)
        results["ensemble"] = {
            "train_accuracy": round(
                accuracy_score(y, ensemble_pred_classes), 4
            ),
            "classification_report": classification_report(
                y, ensemble_pred_classes, output_dict=True
            )
        }

        # Feature importances (from tree-based models)
        self._compute_feature_importances()

        self.is_trained = True
        self.models = trained_models
        return results

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using ensemble averaging.
        Returns array of shape (n_samples, 3) for [Home, Draw, Away].
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")

        X_scaled = self.scaler.transform(X)

        # Average probabilities across models
        all_probs = []
        for name, model in self.models.items():
            try:
                probs = model.predict_proba(X_scaled)
                all_probs.append(probs)
            except Exception as e:
                logger.warning(f"Model {name} prediction failed: {e}")

        if not all_probs:
            raise RuntimeError("All models failed to predict.")

        # Weighted average (can be customized based on CV performance)
        avg_probs = np.mean(all_probs, axis=0)

        # Normalize to sum to 1
        row_sums = avg_probs.sum(axis=1, keepdims=True)
        avg_probs = avg_probs / row_sums

        return avg_probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def _compute_feature_importances(self):
        """Extract and average feature importances from tree models."""
        importances = np.zeros(len(self.feature_names))
        count = 0

        for name, model in self.models.items():
            base = model
            # Unwrap CalibratedClassifierCV if needed
            if hasattr(model, 'estimator'):
                base = model.estimator
            if hasattr(base, 'feature_importances_'):
                importances += base.feature_importances_
                count += 1

        if count > 0:
            importances /= count
            self.feature_importances = dict(
                zip(self.feature_names, importances)
            )
            # Sort by importance
            self.feature_importances = dict(
                sorted(self.feature_importances.items(),
                       key=lambda x: x[1], reverse=True)
            )

    def save(self, filename: str = "ensemble_model"):
        """Save model to disk."""
        path = self.model_dir / f"{filename}.pkl"
        state = {
            "scaler": self.scaler,
            "models": self.models,
            "feature_names": self.feature_names,
            "feature_importances": self.feature_importances,
            "is_trained": self.is_trained
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"Model saved to {path}")

    def load(self, filename: str = "ensemble_model"):
        """Load model from disk."""
        path = self.model_dir / f"{filename}.pkl"
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.scaler = state["scaler"]
        self.models = state["models"]
        self.feature_names = state["feature_names"]
        self.feature_importances = state["feature_importances"]
        self.is_trained = state["is_trained"]
        logger.info(f"Model loaded from {path}")


# ============================================================
# VALUE BET ANALYZER
# ============================================================

class ValueBetAnalyzer:
    """
    Detects value bets by comparing model probabilities to
    bookmaker odds-implied probabilities.

    Uses Kelly Criterion for optimal stake sizing.
    """

    def __init__(
        self,
        min_value_threshold: float = 0.05,
        kelly_fraction: float = 0.25,
        confidence_threshold: float = 0.55,
        bankroll: float = 1000.0
    ):
        self.min_value = min_value_threshold
        self.kelly_frac = kelly_fraction
        self.conf_threshold = confidence_threshold
        self.bankroll = bankroll

    def analyze_match(
        self,
        model_probs: np.ndarray,
        odds: Dict[str, float],
        match_info: Dict[str, str],
        context_features: Dict[str, float] = None,
        poisson_probs: Dict[str, float] = None
    ) -> BetRecommendation:
        """
        Analyze a single match for value betting opportunities.

        Parameters:
            model_probs: [prob_home, prob_draw, prob_away]
            odds: {"home": x, "draw": x, "away": x, "over25": x, "under25": x}
            match_info: {"match": str, "date": str, "league": str}
            context_features: dict from feature engineering
            poisson_probs: dict from Poisson model
        """
        prob_h, prob_d, prob_a = model_probs

        # Combine with Poisson if available (weighted blend)
        if poisson_probs:
            poisson_weight = 0.3  # 30% Poisson, 70% ML
            prob_h = (1 - poisson_weight) * prob_h + poisson_weight * poisson_probs.get("poisson_home_win", prob_h)
            prob_d = (1 - poisson_weight) * prob_d + poisson_weight * poisson_probs.get("poisson_draw", prob_d)
            prob_a = (1 - poisson_weight) * prob_a + poisson_weight * poisson_probs.get("poisson_away_win", prob_a)

            # Renormalize
            total = prob_h + prob_d + prob_a
            prob_h, prob_d, prob_a = prob_h/total, prob_d/total, prob_a/total

        # Value = (probability * odds) - 1
        odds_h = odds.get("home", 0)
        odds_d = odds.get("draw", 0)
        odds_a = odds.get("away", 0)
        odds_o25 = odds.get("over25", 0)
        odds_u25 = odds.get("under25", 0)

        value_h = (prob_h * odds_h - 1) if odds_h > 0 else -1
        value_d = (prob_d * odds_d - 1) if odds_d > 0 else -1
        value_a = (prob_a * odds_a - 1) if odds_a > 0 else -1

        # Over 2.5 value (from Poisson if available)
        prob_o25 = poisson_probs.get("poisson_over25", 0.5) if poisson_probs else 0.5
        prob_btts = poisson_probs.get("poisson_btts", 0.5) if poisson_probs else 0.5
        value_o25 = (prob_o25 * odds_o25 - 1) if odds_o25 > 0 else -1

        # Find best value bet
        values = {
            "Home Win": (value_h, prob_h, odds_h),
            "Draw": (value_d, prob_d, odds_d),
            "Away Win": (value_a, prob_a, odds_a),
            "Over 2.5": (value_o25, prob_o25, odds_o25),
        }

        best_bet = max(values, key=lambda k: values[k][0])
        best_value, best_prob, best_odds = values[best_bet]

        # Kelly Criterion stake
        kelly = self._kelly_criterion(best_prob, best_odds)

        # Confidence = max probability (higher = more certain)
        confidence = max(prob_h, prob_d, prob_a)

        # Key factors analysis
        key_factors = self._analyze_key_factors(
            context_features or {}, model_probs, odds
        )

        # Risk level
        risk = self._assess_risk(confidence, best_value, context_features or {})

        return BetRecommendation(
            match=match_info.get("match", "Unknown"),
            date=match_info.get("date", ""),
            league=match_info.get("league", ""),
            prob_home=round(prob_h, 4),
            prob_draw=round(prob_d, 4),
            prob_away=round(prob_a, 4),
            prob_over25=round(prob_o25, 4),
            prob_btts=round(prob_btts, 4),
            odds_home=odds_h,
            odds_draw=odds_d,
            odds_away=odds_a,
            odds_over25=odds_o25,
            odds_under25=odds_u25,
            value_home=round(value_h, 4),
            value_draw=round(value_d, 4),
            value_away=round(value_a, 4),
            value_over25=round(value_o25, 4),
            best_bet=best_bet,
            best_value=round(best_value, 4),
            confidence=round(confidence, 4),
            kelly_stake_pct=round(kelly, 4),
            key_factors=key_factors,
            risk_level=risk
        )

    def _kelly_criterion(self, prob: float, odds: float) -> float:
        """
        Fractional Kelly Criterion.
        Returns recommended stake as % of bankroll.
        """
        if odds <= 1 or prob <= 0:
            return 0.0

        b = odds - 1  # Net odds
        q = 1 - prob  # Probability of losing

        kelly = (b * prob - q) / b

        # Apply fraction and cap
        kelly = max(0, kelly * self.kelly_frac)
        kelly = min(kelly, 0.10)  # Never more than 10% of bankroll

        return kelly

    def _analyze_key_factors(
        self,
        ctx: Dict[str, float],
        probs: np.ndarray,
        odds: Dict[str, float]
    ) -> List[str]:
        """Generate human-readable key factors for the bet."""
        factors = []

        # Squad strength
        strength_diff = ctx.get("squad_strength_diff", 0)
        if abs(strength_diff) > 0.1:
            stronger = "Home" if strength_diff > 0 else "Away"
            factors.append(
                f"{stronger} team has stronger available squad "
                f"(diff: {strength_diff:+.2f})"
            )

        # Missing key players
        h_missing = ctx.get("home_missing_key_count", 0)
        a_missing = ctx.get("away_missing_key_count", 0)
        if h_missing > 0:
            factors.append(f"Home missing {int(h_missing)} key player(s)")
        if a_missing > 0:
            factors.append(f"Away missing {int(a_missing)} key player(s)")

        # xG advantage
        xg_sup = ctx.get("xg_superiority", 0)
        if abs(xg_sup) > 0.3:
            better = "Home" if xg_sup > 0 else "Away"
            factors.append(
                f"{better} has xG superiority ({xg_sup:+.2f})"
            )

        # H2H dominance
        h2h_dom = ctx.get("h2h_dominance", 0)
        if abs(h2h_dom) > 0.2:
            dom = "Home" if h2h_dom > 0 else "Away"
            factors.append(f"{dom} dominates H2H record")

        # Referee factor
        ref_strict = ctx.get("ref_strictness", 0.5)
        if ref_strict > 0.7:
            factors.append("Strict referee: expect more cards")
        elif ref_strict < 0.3:
            factors.append("Lenient referee: fewer stoppages expected")

        # Sentiment
        sent_diff = ctx.get("sentiment_diff", 0)
        if abs(sent_diff) > 0.3:
            positive_team = "Home" if sent_diff > 0 else "Away"
            factors.append(f"{positive_team} has positive news momentum")

        # Model vs Market disagreement
        if odds.get("home", 0) > 0:
            implied_h = 1 / odds["home"]
            if probs[0] > implied_h + 0.08:
                factors.append(
                    f"Market underestimates Home "
                    f"(model: {probs[0]:.0%} vs market: {implied_h:.0%})"
                )
            if probs[2] > 1/odds.get("away", 99) + 0.08:
                factors.append(
                    f"Market underestimates Away "
                    f"(model: {probs[2]:.0%} vs market: {1/odds['away']:.0%})"
                )

        if not factors:
            factors.append("No standout factors - standard form-based prediction")

        return factors

    def _assess_risk(
        self,
        confidence: float,
        value: float,
        ctx: Dict[str, float]
    ) -> str:
        """Assess bet risk level."""
        score = 0

        if confidence > 0.65:
            score -= 1
        elif confidence < 0.45:
            score += 2

        if value > 0.15:
            score -= 1
        elif value < 0.03:
            score += 1

        # High injury count = unpredictable
        total_missing = (
            ctx.get("home_missing_key_count", 0) +
            ctx.get("away_missing_key_count", 0)
        )
        if total_missing >= 3:
            score += 1

        if score <= 0:
            return "low"
        elif score <= 2:
            return "medium"
        else:
            return "high"

    def filter_value_bets(
        self,
        recommendations: List[BetRecommendation]
    ) -> List[BetRecommendation]:
        """
        Filter recommendations to only include genuine value bets.
        """
        value_bets = []
        for rec in recommendations:
            if (rec.best_value >= self.min_value and
                    rec.confidence >= self.conf_threshold):
                value_bets.append(rec)

        # Sort by value (highest first)
        value_bets.sort(key=lambda r: r.best_value, reverse=True)
        return value_bets
