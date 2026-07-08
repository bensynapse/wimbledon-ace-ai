"""SQLite persistence for predictions and bet_log — Supabase-ready schema."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class TennisDatabase:
    def __init__(self, db_path: str = "data/tennis_intelligence.db"):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT,
                    player_a TEXT NOT NULL,
                    player_b TEXT NOT NULL,
                    surface TEXT,
                    tournament TEXT,
                    model_probability_a REAL,
                    model_probability_b REAL,
                    fair_odds_a REAL,
                    fair_odds_b REAL,
                    market_odds_a REAL,
                    market_odds_b REAL,
                    edge_a REAL,
                    edge_b REAL,
                    confidence TEXT,
                    recommended_action TEXT,
                    minimum_odds_a REAL,
                    minimum_odds_b REAL,
                    stake_percent TEXT,
                    model_version TEXT,
                    payload_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS bet_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    match TEXT NOT NULL,
                    pick TEXT,
                    odds_taken REAL,
                    closing_odds REAL,
                    stake_percent REAL,
                    result TEXT,
                    profit_loss REAL,
                    clv REAL,
                    reason TEXT,
                    model_version TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_predictions_created
                    ON predictions(created_at);
                CREATE INDEX IF NOT EXISTS idx_bet_log_date
                    ON bet_log(date);
            """)

    def save_prediction(
        self,
        player_a: str,
        player_b: str,
        surface: str,
        model_prob_a: float,
        fair_odds_a: float,
        fair_odds_b: float,
        market_odds_a: float = 0.0,
        market_odds_b: float = 0.0,
        edge_a: float = 0.0,
        edge_b: float = 0.0,
        confidence: str = "low",
        recommended_action: str = "NO BET",
        minimum_odds_a: float = 0.0,
        minimum_odds_b: float = 0.0,
        stake_percent: str = "0%",
        model_version: str = "v1",
        tournament: str = "",
        match_id: str = "",
        payload: Optional[Dict] = None,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO predictions (
                    match_id, player_a, player_b, surface, tournament,
                    model_probability_a, model_probability_b,
                    fair_odds_a, fair_odds_b,
                    market_odds_a, market_odds_b,
                    edge_a, edge_b, confidence, recommended_action,
                    minimum_odds_a, minimum_odds_b, stake_percent,
                    model_version, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    match_id or f"{player_a}_vs_{player_b}_{datetime.now().strftime('%Y%m%d')}",
                    player_a, player_b, surface, tournament,
                    model_prob_a, 1.0 - model_prob_a,
                    fair_odds_a, fair_odds_b,
                    market_odds_a, market_odds_b,
                    edge_a, edge_b, confidence, recommended_action,
                    minimum_odds_a, minimum_odds_b, stake_percent,
                    model_version,
                    json.dumps(payload) if payload else None,
                ),
            )
            return cur.lastrowid

    def log_bet(
        self,
        date: str,
        match: str,
        pick: str,
        odds_taken: float,
        stake_percent: float,
        reason: str = "",
        model_version: str = "v1",
        result: str = "pending",
        closing_odds: float = 0.0,
        profit_loss: float = 0.0,
        clv: float = 0.0,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO bet_log (
                    date, match, pick, odds_taken, closing_odds,
                    stake_percent, result, profit_loss, clv, reason, model_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    date, match, pick, odds_taken, closing_odds,
                    stake_percent, result, profit_loss, clv, reason, model_version,
                ),
            )
            return cur.lastrowid

    def recent_predictions(self, limit: int = 20) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def value_predictions(self, min_edge: float = 0.05) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM predictions
                WHERE edge_a >= ? OR edge_b >= ?
                ORDER BY CASE WHEN edge_a > edge_b THEN edge_a ELSE edge_b END DESC
                """,
                (min_edge, min_edge),
            ).fetchall()
        return [dict(r) for r in rows]