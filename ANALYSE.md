# ProphitBet Analyse & Enhanced Betting Module

## Wat ProphitBet goed doet (en wat we overnemen)

ProphitBet heeft een solide basis met rolling window statistieken over historische wedstrijddata.
Het berekent ~27 features per wedstrijd, waaronder thuis/uit winsten, doelsaldo's, schoten op doel
en hoekschoppen over de laatste N wedstrijden. De multi-model aanpak (XGBoost, Random Forest,
SVM, KNN, Neural Net) met Optuna hyperparameter tuning en cross-validatie is professioneel.

**Overgenomen van ProphitBet:**
- Rolling window statistieken (HW, AW, HL, AL, HGF, AGF, etc.)
- Multi-model ensemble approach (verbeterd met soft voting)
- Probability calibration (CalibratedClassifierCV)
- Football-data.co.uk als data source
- Train/test split op chronologische volgorde

---

## Wat ProphitBet MIST (en wat wij toevoegen)

### 1. Speler-niveau data
ProphitBet werkt alleen op team-niveau. Wij voegen toe:
- **Blessures**: Welke spelers zijn geblesseerd en hoe belangrijk zijn ze
- **Schorsingen**: Gele/rode kaarten accumulatie
- **Key players**: Top 5 spelers per team (goals + assists), en of ze beschikbaar zijn
- **Shots per 90 min**: Individuele schotkracht
- **Assists & key passes**: Creatieve output per speler
- **xG & xA per speler**: Verwachte output

### 2. Scheidsrechter analyse
ProphitBet negeert de scheidsrechter volledig. Wij voegen toe:
- Gemiddeld aantal gele/rode kaarten per wedstrijd
- Gemiddeld aantal doelpunten onder deze scheidsrechter
- Thuisvoordeel-bias per scheidsrechter
- "Strictness score" (streng vs soepel)

### 3. Kaarten & discipline
- Rolling totaal gele kaarten (thuis/uit)
- Rolling totaal rode kaarten
- Discipline score = yellows + 3 * reds
- Impact op schorsingsrisico

### 4. Uitgebreide wedstrijdstatistieken
ProphitBet gebruikt alleen shots on target en corners. Wij voegen toe:
- **Totale schoten** (niet alleen op doel)
- **Schotnauwkeurigheid** (on target / totaal)
- **Conversie ratio** (goals / shots)
- **Overtredingen** (fouls committed)
- **Clean sheet rate**
- **BTTS (Both Teams to Score) rate**
- **Over 2.5 goals rate**
- **Form punten** (W=3, D=1, L=0)

### 5. Head-to-Head
ProphitBet doet geen H2H analyse. Wij voegen toe:
- H2H winrate thuis/uit
- H2H gemiddelde doelpunten
- H2H dominantie score

### 6. xG (Expected Goals)
- Team xG for/against (rolling gemiddelde)
- xG superiority (verschil tussen teams)
- Vergelijking xG vs werkelijke goals (over/under performing)

### 7. Odds-analyse
ProphitBet gebruikt odds alleen als context. Wij voegen toe:
- Implied probabilities (genormaliseerd)
- Model vs markt vergelijking
- Value bet detectie (edge = model_prob * odds - 1)
- Kelly Criterion voor stake sizing

### 8. Nieuws & momentum
- Sentiment analyse van recente nieuwsartikelen per team
- Blessurenieuws impact
- Transfer/managerwisseling detectie

### 9. Poisson Model
- Aanvals/verdedigingssterkte berekening
- Exacte doelpuntenverdeling
- BTTS en Over/Under kansen

---

## Module Structuur

```
enhanced_betting_module/
├── config.py              # API keys, model settings
├── data_collector.py      # Alle data bronnen (API-Football, odds, nieuws, historisch)
├── feature_engineering.py # Feature berekeningen (rolling + enhanced + context + Poisson)
├── prediction_model.py    # Ensemble model + value bet detectie
├── main.py               # Orchestrator: train / predict / analyze
└── requirements.txt      # Python dependencies
```

## Gebruik

```bash
# Stap 1: Vul je API keys in config.py

# Stap 2: Train het model
python main.py --mode train --league E0 --start-year 2018

# Stap 3: Analyseer een competitie
python main.py --mode analyze --league N1

# Stap 4: Voorspel wedstrijden
python main.py --mode predict --league T1 --fixtures-file fixtures.json
```

## Feature Overzicht (67 features totaal)

| Categorie | # Features | Bron |
|-----------|-----------|------|
| ProphitBet rolling stats | 13 | football-data.co.uk |
| Enhanced rolling (shots, cards, fouls) | 24 | football-data.co.uk |
| Squad strength & injuries | 11 | api-football.com |
| Referee | 4 | api-football.com |
| H2H | 5 | api-football.com |
| xG | 5 | api-football.com |
| Odds-implied | 6 | the-odds-api.com |
| News sentiment | 3 | newsapi.org |
| Poisson | 5 | Berekend |

---

## Volgende stap

Upload je Codex-code zodat ik deze module direct kan integreren met jouw bestaande systeem.
Specifiek wil ik zien:
- Je huidige Poisson model implementatie
- Je odds comparison logica
- Je API integraties (The Odds API etc.)
- Je value bet detectie code

Dan merge ik alles tot één werkend systeem.
