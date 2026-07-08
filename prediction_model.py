"""
WimbledonAce AI — Prediction Engine
Ensemble ML classifier with Kelly Criterion value bet detection.
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class BetRecommendation:
    """Complete bet recommendation for a tennis match."""
    match: str
    date: str
    tournament: str
    surface: str
    player1: str
    player2: str
    prob_player1: float
    prob_player2: float
    odds_player1: float
    odds_player2: float
    value_player1: float
    value_player2: float
    best_bet: str
    best_value: float
    confidence: float
    kelly_stake_pct: float
    key_factors: List[str]
    risk_level: str


class EnsemblePredictor:
    """Binary ensemble: predicts probability that player1 wins."""

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
        models = {
            "gradient_boost": GradientBoostingClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=10,
                random_state=42,
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            "logistic": LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        }
        self.models = models
        return models

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        cv_folds: int = 5,
        calibrate: bool = True,
    ) -> Dict[str, Any]:
        self.feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]
        X_scaled = self.scaler.fit_transform(X)
        self.build_models()

        results = {}
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        trained_models = {}

        for name, model in self.models.items():
            logger.info("Training %s...", name)
            cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy")
            model.fit(X_scaled, y)

            if calibrate:
                cal_model = CalibratedClassifierCV(
                    estimator=model, method="isotonic", cv=cv_folds
                )
                cal_model.fit(X_scaled, y)
                trained_models[name] = cal_model
            else:
                trained_models[name] = model

            results[name] = {
                "cv_accuracy_mean": round(cv_scores.mean(), 4),
                "cv_accuracy_std": round(cv_scores.std(), 4),
                "train_accuracy": round(accuracy_score(y, model.predict(X_scaled)), 4),
            }

        self.ensemble = VotingClassifier(
            estimators=list(trained_models.items()),
            voting="soft",
        )
        self.ensemble.estimators_ = list(trained_models.values())
        self.ensemble.le_ = (
            self.ensemble.estimators_[0].classes_
            if hasattr(self.ensemble.estimators_[0], "classes_")
            else np.unique(y)
        )

        ensemble_preds = self.predict_proba(X)
        ensemble_pred_classes = (ensemble_preds >= 0.5).astype(int)
        results["ensemble"] = {
            "train_accuracy": round(accuracy_score(y, ensemble_pred_classes), 4),
            "classification_report": classification_report(
                y, ensemble_pred_classes, output_dict=True
            ),
        }

        self._compute_feature_importances()
        self.is_trained = True
        self.models = trained_models
        return results

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(player1 wins) for each row."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")

        X_scaled = self.scaler.transform(X)
        all_probs = []
        for name, model in self.models.items():
            try:
                probs = model.predict_proba(X_scaled)
                p1 = probs[:, 1] if probs.shape[1] == 2 else probs[:, 0]
                all_probs.append(p1)
            except Exception as exc:
                logger.warning("Model %s prediction failed: %s", name, exc)

        if not all_probs:
            raise RuntimeError("All models failed to predict.")

        return np.clip(np.mean(all_probs, axis=0), 0.01, 0.99)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def _compute_feature_importances(self):
        importances = np.zeros(len(self.feature_names))
        count = 0
        for model in self.models.values():
            base = model.estimator if hasattr(model, "estimator") else model
            if hasattr(base, "feature_importances_"):
                importances += base.feature_importances_
                count += 1

        if count:
            importances /= count
            self.feature_importances = dict(
                sorted(
                    zip(self.feature_names, importances),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )

    def save(self, filename: str = "wimbledon_ace_model"):
        path = self.model_dir / f"{filename}.pkl"
        state = {
            "scaler": self.scaler,
            "models": self.models,
            "feature_names": self.feature_names,
            "feature_importances": self.feature_importances,
            "is_trained": self.is_trained,
        }
        with open(path, "wb") as handle:
            pickle.dump(state, handle)
        logger.info("Model saved to %s", path)

    def load(self, filename: str = "wimbledon_ace_model"):
        path = self.model_dir / f"{filename}.pkl"
        with open(path, "rb") as handle:
            state = pickle.load(handle)
        self.scaler = state["scaler"]
        self.models = state["models"]
        self.feature_names = state["feature_names"]
        self.feature_importances = state["feature_importances"]
        self.is_trained = state["is_trained"]
        logger.info("Model loaded from %s", path)


class ValueBetAnalyzer:
    """Compare model probabilities to bookmaker odds."""

    def __init__(
        self,
        min_value_threshold: float = 0.05,
        kelly_fraction: float = 0.25,
        confidence_threshold: float = 0.55,
        bankroll: float = 1000.0,
    ):
        self.min_value = min_value_threshold
        self.kelly_frac = kelly_fraction
        self.conf_threshold = confidence_threshold
        self.bankroll = bankroll

    def analyze_match(
        self,
        prob_player1: float,
        odds: Dict[str, float],
        match_info: Dict[str, str],
        context_features: Optional[Dict[str, float]] = None,
    ) -> BetRecommendation:
        prob_p1 = float(prob_player1)
        prob_p2 = 1.0 - prob_p1

        odds_p1 = odds.get("player1", 0.0)
        odds_p2 = odds.get("player2", 0.0)

        value_p1 = (prob_p1 * odds_p1 - 1) if odds_p1 > 0 else -1
        value_p2 = (prob_p2 * odds_p2 - 1) if odds_p2 > 0 else -1

        values = {
            f"{match_info.get('player1', 'Player 1')} ML": (value_p1, prob_p1, odds_p1),
            f"{match_info.get('player2', 'Player 2')} ML": (value_p2, prob_p2, odds_p2),
        }
        best_bet, (best_value, best_prob, best_odds) = max(
            values.items(), key=lambda item: item[1][0]
        )

        kelly = self._kelly_criterion(best_prob, best_odds)
        confidence = max(prob_p1, prob_p2)
        key_factors = self._analyze_key_factors(
            context_features or {},
            prob_p1,
            odds,
            match_info,
        )
        risk = self._assess_risk(confidence, best_value, context_features or {})

        return BetRecommendation(
            match=match_info.get("match", "Unknown"),
            date=match_info.get("date", ""),
            tournament=match_info.get("tournament", ""),
            surface=match_info.get("surface", ""),
            player1=match_info.get("player1", ""),
            player2=match_info.get("player2", ""),
            prob_player1=round(prob_p1, 4),
            prob_player2=round(prob_p2, 4),
            odds_player1=odds_p1,
            odds_player2=odds_p2,
            value_player1=round(value_p1, 4),
            value_player2=round(value_p2, 4),
            best_bet=best_bet,
            best_value=round(best_value, 4),
            confidence=round(confidence, 4),
            kelly_stake_pct=round(kelly, 4),
            key_factors=key_factors,
            risk_level=risk,
        )

    def _kelly_criterion(self, prob: float, odds: float) -> float:
        if odds <= 1 or prob <= 0:
            return 0.0
        b = odds - 1
        q = 1 - prob
        kelly = max(0.0, (b * prob - q) / b)
        return min(kelly * self.kelly_frac, 0.10)

    def _analyze_key_factors(
        self,
        ctx: Dict[str, float],
        prob_p1: float,
        odds: Dict[str, float],
        match_info: Dict[str, str],
    ) -> List[str]:
        factors = []
        p1 = match_info.get("player1", "Player 1")
        p2 = match_info.get("player2", "Player 2")

        elo_diff = ctx.get("elo_diff", 0.0)
        if abs(elo_diff) > 50:
            fav = p1 if elo_diff > 0 else p2
            factors.append(f"{fav} has Elo edge ({elo_diff:+.0f})")

        surface_diff = ctx.get("surface_elo_diff", 0.0)
        surface = match_info.get("surface", "")
        if abs(surface_diff) > 40 and surface:
            fav = p1 if surface_diff > 0 else p2
            factors.append(f"{fav} stronger on {surface}")

        h2h_edge = ctx.get("h2h_win_rate_diff", 0.0)
        if abs(h2h_edge) > 0.2:
            fav = p1 if h2h_edge > 0 else p2
            factors.append(f"{fav} leads head-to-head")

        fatigue = ctx.get("days_since_last_match_diff", 0.0)
        if abs(fatigue) > 3:
            rested = p1 if fatigue > 0 else p2
            factors.append(f"{rested} has more rest")

        if odds.get("player1", 0) > 0:
            implied_p1 = 1 / odds["player1"]
            if prob_p1 > implied_p1 + 0.06:
                factors.append(
                    f"Market underestimates {p1} "
                    f"(model {prob_p1:.0%} vs market {implied_p1:.0%})"
                )
        if odds.get("player2", 0) > 0:
            implied_p2 = 1 / odds["player2"]
            if (1 - prob_p1) > implied_p2 + 0.06:
                factors.append(
                    f"Market underestimates {p2} "
                    f"(model {1 - prob_p1:.0%} vs market {implied_p2:.0%})"
                )

        return factors or ["Standard form-based prediction"]

    def _assess_risk(
        self,
        confidence: float,
        value: float,
        ctx: Dict[str, float],
    ) -> str:
        score = 0
        if confidence > 0.68:
            score -= 1
        elif confidence < 0.52:
            score += 2
        if value > 0.12:
            score -= 1
        elif value < 0.03:
            score += 1
        if ctx.get("injury_flag", 0) > 0:
            score += 1
        if score <= 0:
            return "low"
        if score <= 2:
            return "medium"
        return "high"

    def filter_value_bets(
        self,
        recommendations: List[BetRecommendation],
    ) -> List[BetRecommendation]:
        value_bets = [
            rec
            for rec in recommendations
            if rec.best_value >= self.min_value
            and rec.confidence >= self.conf_threshold
        ]
        value_bets.sort(key=lambda rec: rec.best_value, reverse=True)
        return value_bets