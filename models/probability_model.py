"""
V1 probability model — weighted blend per master spec.

Base model =
  35% overall Elo
  30% surface Elo
  15% recent form
  10% serve/return
   5% fatigue
   5% matchup/weather
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


DEFAULT_ELO = 1500.0

COMPONENT_WEIGHTS = {
    "overall_elo": 0.35,
    "surface_elo": 0.30,
    "recent_form": 0.15,
    "serve_return": 0.10,
    "fatigue": 0.05,
    "conditions": 0.05,
}


def elo_to_probability(elo_a: float, elo_b: float) -> float:
    """Logistic win probability from Elo difference."""
    diff = elo_a - elo_b
    return 1.0 / (1.0 + 10 ** (-diff / 400))


def _clamp_prob(p: float) -> float:
    return max(0.02, min(0.98, p))


@dataclass
class ModelPrice:
    """Fair price output for one match."""
    player1: str
    player2: str
    surface: str
    prob_p1: float
    prob_p2: float
    fair_odds_p1: float
    fair_odds_p2: float
    components: Dict[str, float]
    model_version: str = "v1_weighted"


class ProbabilityModelV1:
    """Transparent weighted probability model — not a black-box guess."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or COMPONENT_WEIGHTS

    def predict(
        self,
        player1: str,
        player2: str,
        surface: str,
        context: Optional[Dict] = None,
    ) -> ModelPrice:
        ctx = context or {}
        components = self._component_probs(player1, player2, surface, ctx)
        prob_p1 = sum(
            self.weights[k] * components[k] for k in self.weights
        )
        prob_p1 = _clamp_prob(prob_p1)
        prob_p2 = 1.0 - prob_p1

        return ModelPrice(
            player1=player1,
            player2=player2,
            surface=surface,
            prob_p1=round(prob_p1, 4),
            prob_p2=round(prob_p2, 4),
            fair_odds_p1=round(1.0 / prob_p1, 2),
            fair_odds_p2=round(1.0 / prob_p2, 2),
            components={k: round(v, 4) for k, v in components.items()},
        )

    def _component_probs(self, p1: str, p2: str, surface: str, ctx: Dict) -> Dict[str, float]:
        elo_p1 = ctx.get("player1_elo", ctx.get("elo_p1", DEFAULT_ELO))
        elo_p2 = ctx.get("player2_elo", ctx.get("elo_p2", DEFAULT_ELO))
        surf_p1 = ctx.get("player1_surface_elo", ctx.get(f"elo_{surface}_p1", elo_p1))
        surf_p2 = ctx.get("player2_surface_elo", ctx.get(f"elo_{surface}_p2", elo_p2))

        form_p1 = ctx.get("player1_form", ctx.get("form_p1", 0.5))
        form_p2 = ctx.get("player2_form", ctx.get("form_p2", 0.5))
        serve_p1 = ctx.get("player1_serve_return", ctx.get("serve_return_p1", 0.5))
        serve_p2 = ctx.get("player2_serve_return", ctx.get("serve_return_p2", 0.5))

        fatigue_p1 = 1.0 - ctx.get("player1_fatigue_score", ctx.get("fatigue_score_p1", 0.0))
        fatigue_p2 = 1.0 - ctx.get("player2_fatigue_score", ctx.get("fatigue_score_p2", 0.0))

        cond_p1 = ctx.get("player1_conditions", ctx.get("conditions_p1", 0.5))
        cond_p2 = ctx.get("player2_conditions", ctx.get("conditions_p2", 0.5))

        if "surface_elo_diff" in ctx:
            surf_diff = ctx["surface_elo_diff"]
            surf_p1 = DEFAULT_ELO + surf_diff / 2
            surf_p2 = DEFAULT_ELO - surf_diff / 2

        if "combined_strength_diff" in ctx:
            strength = ctx["combined_strength_diff"]
            elo_p1 = DEFAULT_ELO + strength * 200
            elo_p2 = DEFAULT_ELO - strength * 200

        return {
            "overall_elo": elo_to_probability(elo_p1, elo_p2),
            "surface_elo": elo_to_probability(surf_p1, surf_p2),
            "recent_form": _normalize_pair(form_p1, form_p2),
            "serve_return": _normalize_pair(serve_p1, serve_p2),
            "fatigue": _normalize_pair(fatigue_p1, fatigue_p2),
            "conditions": _normalize_pair(cond_p1, cond_p2),
        }


def _normalize_pair(a: float, b: float) -> float:
    total = a + b
    if total <= 0:
        return 0.5
    return a / total