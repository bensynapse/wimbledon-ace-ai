# 🎾 WimbledonAce AI — Tennis Market Intelligence

> **Detect mispriced tennis markets using Elo, surface data, contextual UE errors, fatigue, weather and odds movement.**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![ML](https://img.shields.io/badge/ML-Ensemble-green.svg)]()
[![Tennis](https://img.shields.io/badge/Wimbledon-2026-purple.svg)]()

**WimbledonAce AI** finds **value bets** on tennis matches using machine learning, Elo ratings, surface analysis, and the **Kelly Criterion** — built for the 2026 Grand Slam season.

---

## ⚡ Features

- 🏆 **Grand Slam ready** — Wimbledon grass, Roland Garros clay, US Open & Australian Open hard courts
- 🤖 **ML Ensemble** — Gradient Boosting + Random Forest + Logistic Regression with calibrated probabilities
- 📊 **8 smart features** — Elo diff, surface Elo, form, H2H, rest days, ATP/WTA rankings
- 💰 **Value bet detection** — Kelly Criterion stake sizing, edge calculation vs bookmaker odds
- 🔴 **Live predictions** — Today's ATP/WTA fixtures + real-time odds via api-tennis & The Odds API
- 📈 **Train on history** — Download & cache years of match data, retrain anytime

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
```

### Recommended data stack

| Source | Role | Cost |
|--------|------|------|
| [TML-Database (GitHub)](https://github.com/Tennismylife/TML-Database) | ATP matches, results, rankings, serve stats | **Free** |
| [Match Charting Project](https://github.com/JeffSackmann/tennis_MatchChartingProject) | UE/winners FH/BH, rally profiles (jullie edge) | **Free** |
| [The Odds API](https://the-odds-api.com/) | Upcoming fixtures + bookmaker odds | Free tier 500 req/mo |
| [Open-Meteo](https://open-meteo.com/) | Weather (heat, wind) | Free |
| [Kaggle guillemservera/tennis](https://www.kaggle.com/datasets/guillemservera/tennis) | **WTA** + ATP history | Free (API token) |
| [api-tennis.com](https://api-tennis.com/) | Optional paid fallback | ~$15–30/mo |
| Telegram Bot API | Daily alerts | Free |

> De klassieke **Jeff Sackmann `tennis_atp`** repo is offline. **TML-Database** is de actieve opvolger met live updates tot 2026.

Copy `.env.example` → `.env` and fill in keys:

```bash
cp .env.example .env
# Live fixtures + odds:
export ODDS_API_KEY="your-odds-api-key"

# WTA data (eenmalig):
pip install kaggle
# Kaggle → Account → API Token → ~/.kaggle/kaggle.json
bash scripts/download_kaggle.sh
# of: python3 cli.py update --kaggle --all
```

Check configuration:

```bash
python main.py --mode status
```

### Train the model

```bash
python main.py --mode train --tour atp --start-year 2022
```

### Scan for value (with demo data, no API keys)

```bash
python main.py --mode scan-value --demo --full
```

### Daily pipeline (scan + report + Telegram)

```bash
python main.py --mode daily --tour atp
```

### Full intelligence report (ADM vs Cobolli example)

```bash
python main.py --mode intelligence \
  --player1 "Alex de Minaur" --player2 "Flavio Cobolli" \
  --surface grass --odds-p1 1.25 --odds-p2 4.10 --model-prob-p1 0.66 \
  --context-file data/examples/cobolli_adm_wimbledon.json
```

### Predict today's matches

```bash
python main.py --mode predict --tour atp --today
```

### Analyze the dataset

```bash
python main.py --mode analyze --tour wta
```

---

## 🧠 How It Works

```
api-tennis.com ──► Historical matches ──► Elo + Form + H2H features
The Odds API   ──► Live bookmaker odds ──► Value = model prob × odds - 1
                        │
                        ▼
              WimbledonAce AI Ensemble
                        │
                        ▼
              Kelly stake + value bet report
```

---

## 📁 Project Structure

| File | Role |
|------|------|
| `main.py` | WimbledonAce AI orchestrator (train / predict / analyze) |
| `config.py` | API keys, Kelly settings, tour config |
| `data_collector.py` | api-tennis + Odds API data pipeline |
| `feature_engineering.py` | Elo, surface form, H2H feature builder |
| `prediction_model.py` | ML ensemble + value bet analyzer |

---

## 🏷️ Topics

`wimbledon` `wimbledon-2026` `tennis` `tennis-betting` `sports-betting` `machine-learning` `ai` `atp` `wta` `grand-slam` `value-betting` `kelly-criterion` `prediction` `roland-garros` `us-open` `australian-open` `grass-court` `clay-court`

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. Sports betting involves financial risk. Never bet more than you can afford to lose. Past performance does not guarantee future results.

---

**WimbledonAce AI** · Grand Slam Tennis Betting Predictor 2026 · Built with Python & scikit-learn