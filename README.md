# 🎾 WimbledonAce AI — Grand Slam Tennis Betting Predictor 2026

> **AI-powered ATP & WTA match predictions for Wimbledon, Roland Garros, US Open, Australian Open & every Grand Slam.**

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

Add your API keys to `config.py` or as environment variables:

```bash
export TENNIS_API_KEY="your-api-tennis-key"
export ODDS_API_KEY="your-odds-api-key"
```

### Train the model

```bash
python main.py --mode train --tour atp --start-year 2022
```

### Predict today's matches (Wimbledon season!)

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