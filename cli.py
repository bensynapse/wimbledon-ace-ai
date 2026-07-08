#!/usr/bin/env python3
"""
Tennis Market Intelligence CLI — tennisbot commands.

Usage:
    python cli.py update --all
    python cli.py matches --today
    python cli.py predict "Player A" "Player B" --surface grass
    python cli.py scan-value --min-edge 0.05
    python cli.py explain --match-id MATCH_ID
    python cli.py intelligence --player1 "A" --player2 "B" --context-file ctx.json
    python cli.py elo rebuild
    python cli.py elo list --top 30 --surface grass --compare-ta
    python cli.py elo sim "Player A" "Player B" --surface grass --tournament Wimbledon --round SF
    python cli.py elo export
    python cli.py elo matchups --file matchups.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from config import MIN_VALUE_THRESHOLD, PROJECT_NAME, PROJECT_TAGLINE, TOURS, api_status
from data_collector import get_demo_fixtures
from main import WimbledonAceAI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tennisbot",
        description=f"{PROJECT_NAME} — {PROJECT_TAGLINE}",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_update = sub.add_parser("update", help="Refresh historical data")
    p_update.add_argument("--all", action="store_true", help="Update all tours")
    p_update.add_argument("--tour", default="atp", choices=list(TOURS.keys()))
    p_update.add_argument("--start-year", type=int, default=2018)
    p_update.add_argument("--kaggle", action="store_true", help="Download Kaggle tennis dataset")
    p_update.add_argument("--kaggle-odds", action="store_true", help="Download Kaggle ATP odds dataset")
    p_update.add_argument("--elo", action="store_true", help="Refresh Tennis Abstract Elo ratings")

    p_backtest = sub.add_parser("backtest", help="Backtest model vs historical bookmaker odds")
    p_backtest.add_argument("--start-year", type=int, default=2018)
    p_backtest.add_argument("--end-year", type=int, default=None)
    p_backtest.add_argument("--min-edge", type=float, default=MIN_VALUE_THRESHOLD)
    p_backtest.add_argument("--surface", default=None, choices=["hard", "clay", "grass", "carpet"])
    p_backtest.add_argument("--max-bets", type=int, default=None)

    p_matches = sub.add_parser("matches", help="List upcoming matches")
    p_matches.add_argument("--today", action="store_true")
    p_matches.add_argument("--tour", default="atp", choices=list(TOURS.keys()))

    p_predict = sub.add_parser("predict", help="Predict a head-to-head match")
    p_predict.add_argument("player1", type=str)
    p_predict.add_argument("player2", type=str)
    p_predict.add_argument("--surface", default="hard", choices=["hard", "clay", "grass", "carpet"])
    p_predict.add_argument("--tour", default="atp", choices=list(TOURS.keys()))
    p_predict.add_argument("--odds-p1", type=float, default=0.0)
    p_predict.add_argument("--odds-p2", type=float, default=0.0)
    p_predict.add_argument("--context-file", type=str, default=None)
    p_predict.add_argument("--tournament", type=str, default="")

    p_scan = sub.add_parser("scan-value", help="Scan fixtures for value bets")
    p_scan.add_argument("--min-edge", type=float, default=MIN_VALUE_THRESHOLD)
    p_scan.add_argument("--tour", default="atp", choices=list(TOURS.keys()))
    p_scan.add_argument("--fixtures-file", type=str, default=None)
    p_scan.add_argument("--full", action="store_true")
    p_scan.add_argument("--notify", action="store_true")
    p_scan.add_argument("--demo", action="store_true")

    p_explain = sub.add_parser("explain", help="Explain a saved intelligence report")
    p_explain.add_argument("--match-id", type=str, default=None)
    p_explain.add_argument("--payload-file", type=str, default=None)

    p_intel = sub.add_parser("intelligence", help="Full market intelligence report")
    p_intel.add_argument("--player1", required=True)
    p_intel.add_argument("--player2", required=True)
    p_intel.add_argument("--surface", default="hard")
    p_intel.add_argument("--odds-p1", type=float, default=0.0)
    p_intel.add_argument("--odds-p2", type=float, default=0.0)
    p_intel.add_argument("--model-prob-p1", type=float, default=None)
    p_intel.add_argument("--context-file", type=str, default=None)
    p_intel.add_argument("--tournament", type=str, default="")

    p_train = sub.add_parser("train", help="Train ML ensemble")
    p_train.add_argument("--tour", default="atp", choices=list(TOURS.keys()))
    p_train.add_argument("--start-year", type=int, default=2018)

    p_daily = sub.add_parser("daily", help="Full daily scan + report + Telegram")
    p_daily.add_argument("--tour", default="atp", choices=list(TOURS.keys()))
    p_daily.add_argument("--min-edge", type=float, default=MIN_VALUE_THRESHOLD)
    p_daily.add_argument("--demo", action="store_true")

    p_status = sub.add_parser("status", help="Check API configuration")

    p_elo = sub.add_parser("elo", help="WimbledonAce Elo leaderboard + puntensimulatie")
    elo_sub = p_elo.add_subparsers(dest="elo_command", required=True)

    p_elo_rebuild = elo_sub.add_parser("rebuild", help="Herbereken Elo uit matchgeschiedenis")
    p_elo_rebuild.add_argument("--tour", default="atp", choices=list(TOURS.keys()))
    p_elo_rebuild.add_argument("--start-year", type=int, default=2018)

    p_elo_list = elo_sub.add_parser("list", help="Toon Elo-leaderboard")
    p_elo_list.add_argument("--tour", default="atp", choices=list(TOURS.keys()))
    p_elo_list.add_argument("--top", type=int, default=25)
    p_elo_list.add_argument("--surface", default=None, choices=["hard", "clay", "grass", "carpet"])
    p_elo_list.add_argument("--compare-ta", action="store_true", help="Vergelijk met Tennis Abstract")

    p_elo_sim = elo_sub.add_parser("sim", help="Simuleer Elo-punten bij winst/verlies")
    p_elo_sim.add_argument("player1", type=str)
    p_elo_sim.add_argument("player2", type=str)
    p_elo_sim.add_argument("--tour", default="atp", choices=list(TOURS.keys()))
    p_elo_sim.add_argument("--surface", default="hard", choices=["hard", "clay", "grass", "carpet"])
    p_elo_sim.add_argument("--tournament", default="")
    p_elo_sim.add_argument("--round", default="")
    p_elo_sim.add_argument("--best-of", type=int, default=3)

    p_elo_export = elo_sub.add_parser("export", help="Exporteer volledige Elo-lijst naar CSV")
    p_elo_export.add_argument("--tour", default="atp", choices=list(TOURS.keys()))
    p_elo_export.add_argument("--surface", default=None, choices=["hard", "clay", "grass", "carpet"])

    p_elo_matchups = elo_sub.add_parser("matchups", help="Export win/verlies deltas voor matchups")
    p_elo_matchups.add_argument("--tour", default="atp", choices=list(TOURS.keys()))
    p_elo_matchups.add_argument("--file", required=True, help="JSON met matchups [{player1, player2, surface, ...}]")
    p_elo_matchups.add_argument("--output", default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    system = WimbledonAceAI()

    if args.command == "update":
        if getattr(args, "kaggle", False):
            from data_sources.kaggle_tennis import KaggleTennisSource
            src = KaggleTennisSource()
            if src.download():
                print("Kaggle dataset downloaded to data/kaggle_tennis/")
            else:
                print("Kaggle download failed — check ~/.kaggle/kaggle.json")
        if getattr(args, "kaggle_odds", False):
            from data_sources.kaggle_odds import KaggleOddsSource
            src = KaggleOddsSource()
            if src.download():
                print("Kaggle odds dataset downloaded to data/kaggle_odds/")
            else:
                print("Kaggle odds download failed — check ~/.kaggle/kaggle.json")
        if getattr(args, "elo", False) or getattr(args, "all", False):
            from data_sources.tennis_abstract_elo import TennisAbstractEloSource
            src = TennisAbstractEloSource()
            df = src.download(tour="atp", refresh=True)
            print(f"Tennis Abstract ATP Elo updated: {len(df)} players → data/tennis_abstract/")
        if getattr(args, "all", False):
            from pipelines.elo_leaderboard import rebuild_elo_ratings
            df = rebuild_elo_ratings(tour="atp", start_year=args.start_year)
            print(f"WimbledonAce Elo rebuilt: {len(df)} players → data/wimbledon_ace_elo/")
        tours = list(TOURS.keys()) if getattr(args, "all", False) else [args.tour]
        for tour in tours:
            system.historical.download_tour(tour, start_year=args.start_year, refresh=True)
            print(f"Updated {tour.upper()} data")

    elif args.command == "matches":
        fixtures = system.live.get_upcoming_fixtures(args.tour)
        fixtures = system.live.attach_odds(fixtures, args.tour)
        if not fixtures:
            print("No upcoming matches found.")
            return
        for f in fixtures:
            print(
                f"{f.get('date', '')} | {f['player1']} vs {f['player2']} "
                f"| {f.get('tournament', '')} | odds {f.get('odds_player1', '-')} / {f.get('odds_player2', '-')}"
            )

    elif args.command == "predict":
        system.warm_context(args.tour)
        context = {}
        if args.context_file:
            with open(args.context_file, encoding="utf-8") as handle:
                context = json.load(handle)
        system.analyze_intelligence(
            player1=args.player1,
            player2=args.player2,
            surface=args.surface,
            model_prob_p1=None,
            odds_p1=args.odds_p1,
            odds_p2=args.odds_p2,
            context=context,
            tournament=args.tournament,
        )

    elif args.command == "scan-value":
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
            return
        print(f"Found {len(hits)} value opportunities:\n")
        for hit in hits:
            print(
                f"  {hit['match']} → {hit['value_side']} edge {hit['edge']:+.1%} @ {hit['odds']}\n"
                f"    {hit['action']} | stake {hit['stake']}"
            )

    elif args.command == "daily":
        if args.demo:
            fixtures = get_demo_fixtures(args.tour)
            hits = system.scan_value(fixtures, min_edge=args.min_edge, tour=args.tour, full=True)
            report = system.daily_scanner.daily_report(hits, tour=args.tour)
            print(report)
        else:
            system.run_daily(args.tour, min_edge=args.min_edge)

    elif args.command == "status":
        status = api_status()
        print(f"\n{PROJECT_NAME} — Data Source Status\n")
        for name, ok in status.items():
            if name == "data_source":
                print(f"  → active source: {ok}")
            else:
                print(f"  {'✓' if ok else '✗'} {name}")
        print("\n  GitHub ATP: github.com/Tennismylife/TML-Database (free)")

    elif args.command == "explain":
        payload = _load_payload(args.match_id, args.payload_file)
        if not payload:
            sys.exit(1)
        from ai.explanation import explain_from_payload
        print(explain_from_payload(payload))

    elif args.command == "intelligence":
        system.warm_context("atp")
        context = {}
        if args.context_file:
            with open(args.context_file, encoding="utf-8") as handle:
                context = json.load(handle)
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

    elif args.command == "train":
        system.train(args.tour, start_year=args.start_year)

    elif args.command == "backtest":
        system.run_backtest(
            start_year=args.start_year,
            end_year=args.end_year,
            surface=args.surface,
            min_edge=args.min_edge,
            max_bets=args.max_bets,
        )

    elif args.command == "elo":
        _run_elo_command(args)


def _run_elo_command(args) -> None:
    from models.elo_tracker import WimbledonAceEloTracker
    from pipelines.elo_leaderboard import (
        export_matchup_sheet,
        format_exchange_report,
        format_leaderboard_table,
        rebuild_elo_ratings,
    )

    if args.elo_command == "rebuild":
        df = rebuild_elo_ratings(tour=args.tour, start_year=args.start_year)
        print(f"WimbledonAce Elo rebuilt: {len(df)} players")
        print(format_leaderboard_table(df, surface="grass", top_n=15))

    elif args.elo_command == "list":
        tracker = WimbledonAceEloTracker()
        df = tracker.load(args.tour)
        print(format_leaderboard_table(
            df, surface=args.surface, top_n=args.top, compare_ta=args.compare_ta,
        ))

    elif args.elo_command == "sim":
        tracker = WimbledonAceEloTracker()
        tracker.load(args.tour)
        exchange = tracker.compute_exchange(
            args.player1, args.player2,
            surface=args.surface,
            tournament=args.tournament,
            round_name=args.round,
            best_of=args.best_of,
        )
        print(format_exchange_report(exchange, tournament=args.tournament, round_name=args.round))

    elif args.elo_command == "export":
        tracker = WimbledonAceEloTracker()
        out = tracker.export_csv(tour=args.tour, surface=args.surface)
        print(f"Exported → {out}")

    elif args.elo_command == "matchups":
        with open(args.file, encoding="utf-8") as handle:
            matchups = json.load(handle)
        out = export_matchup_sheet(
            matchups, tour=args.tour,
            output_path=Path(args.output) if args.output else None,
        )
        print(f"Matchup deltas exported → {out}")


def _load_payload(match_id: str | None, payload_file: str | None) -> dict | None:
    if payload_file:
        with open(payload_file, encoding="utf-8") as handle:
            return json.load(handle)

    if match_id:
        path = Path("output") / f"intelligence_{match_id}.json"
        if path.exists():
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
        # Search recent outputs
        for p in sorted(Path("output").glob("intelligence_*.json"), reverse=True):
            with open(p, encoding="utf-8") as handle:
                data = json.load(handle)
            if match_id.lower() in data.get("match", "").lower():
                return data
        print(f"No payload found for match-id: {match_id}")
        return None

    print("Provide --match-id or --payload-file")
    return None


if __name__ == "__main__":
    main()