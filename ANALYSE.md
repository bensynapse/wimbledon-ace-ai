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
## Analytics Handbook (devinpleuler) — Key Inspirations for WK Bot

Bron: https://github.com/devinpleuler/analytics-handbook (2023 update)

**Belangrijke concepts & technieken overgenomen/aan te nemen:**
- **Event Data vs Aggregates**: StatsBomb open-data repo (https://github.com/statsbomb/open-data) is de canonical bron. WC data (comp 43) incl. 2022 (met 360 freeze frames!), 2018, en historische toernooien (1958+). Volledige events + lineups + 360 (player posities op moment van event voor pressure/positions). Gebruik direct JSON (lokaal clone) of via statsbombpy.
- **Match Results & Fixtures**: openfootball/football.json (https://github.com/openfootball/football.json) biedt gratis JSON met historische fixtures/results voor 100+ leagues (EPL, Bundesliga, La Liga etc., seizoenen terug tot ~2010). Eenvoudige structuur voor form, H2H, rolling stats. Publiek domein, direct raw GitHub fetch. Gebruik voor club data om NT-modellen te verbeteren (club form -> internationale prestaties, ELO updates, training data). Complementair aan soccerdata (stats), StatsBomb (events), jfjelstul (WC history).
- **StatsBomb + mplsoccer + kloppy** als moderne Python stack voor soccer data (naast onze soccerdata FBref).
- **WhoScored + "etc" via soccerdata**: volledig ondersteund! `SoccerDataWrapper` bevat nu expliciete methodes voor `get_whoscored_data`, `get_whoscored_player_stats`, `get_espn_data`, `get_fotmob_data`, `get_sofascore_data` (SoFIFA) + `get_all_soccerdata_sources`.  
- **WC/EK historische data (meerdere jaren) + dit jaar + trainers**: 
  - jfjelstul/worldcup = beste bron voor gestructureerde historische WC data over vele jaren (matches, player_appearances, win rates, knockout progress). Gebruikt via `get_multi_year_wc_stats` / `get_team_wc_history`.
  - soccerdata (FBref + WhoScored) = beste voor recente vorm, ratings, qualifiers "this year" (2024/2025/2026 windows).
  - rezar + openfootball = beste voor exacte 2026 actuele squads, groups, venues, fixtures.
  - Nieuwe `CoachTacticalLoader` + `compute_coach_tactical_features`: coach naam, formatie, press_intensity, possession pref, direct play, setpiece focus, matchup score. Wordt gebruikt in context, all_priority_features en sims. Zeer belangrijk voor "hoe de trainer ze laat spelen".
  - Alle data gecombineerd in `da.get_all_gap_data()`, `da.get_historical_wc_and_coach()`, MatchContext en feature pipeline.
  - WhoScored levert gedetailleerde match stats (possession, shots, ratings, cards) die vaak andere signalen geven dan FBref — nuttig voor value.
  - ESPN/FotMob/SoFIFA vullen ratings + schedule aan. Gebruikt in data_collector (context verrijking) + feature_engineering (`compute_whoscored_features`, `add_whoscored_features_to_context`, integratie in `compute_all_priority_features` + `context_to_features`).
  - Voor WC: INT-World Cup + club league data voor speler vorm/waardering (club → NT carry-over).
- **Praktische features uit events**:
  - Shot locations / distance / angle → betere xG proxies en shot quality.
  - Pass success + forward/progressive passes → buildup style, creation threat.
  - Per-player action counts voor workload en key contribution (beter dan goals/assists alleen).
- **WC-specifiek**: Volledige 2022 World Cup events beschikbaar → perfecte historische training set voor international / national team modellen (complementair aan jfjelstul).
- **Filosofie**: Diep domeinkennis + scepsis + Python (pandas + sklearn + xgboost). Past perfect bij onze "value where bookies miss data" focus (e.g. combine event features met altitude/heat adaptation).
- **Aanbevolen research** (uit handbook): 
  - VAEP / "Actions speak louder than goals" (Decroos et al.) — action values i.p.v. alleen goals.
  - Spearman xG / off-ball, Fernandez-Bornn expected threat (xT), pass probability models.
  - Player vectors / style clustering.

**Huidige integratie:**
- `scrapers.StatsBombWCImporter` (nu met directe JSON support van open-data clone + 360 data)
- `get_match_360`, pressure proxies in extract features
- `feature_engineering.compute_event_features_from_statsbomb` + spatial features
- Primary free bron voor gedetailleerde WC events + 360 (beter dan aggregates alleen)
- `FootballJsonLoader` + `get_league_data` in DataAggregator voor club league results/fixtures (form, H2H, training).
- `data_collector` enrichment + `DataAggregator` summary
- Gebruik 2022 WC events als rijke bron voor backtesting en feature engineering voor 2026.

**Volgende stappen mogelijk**: 
- Volledige pipeline: clone open-data → StatsBombWCImporter(local) → socceraction VAEP op WC matches → features per player/action.
- 360 freeze frames gebruiken voor betere pressure / positioning features in xG/VAEP.
- Combineer met roboflow/sports tracking data voor fysieke + event-based multi-modal player ratings.
- Historische WC (1958+) gebruiken voor robuuste backtesting van nationale team modellen.

---
## federicorabanos/futbol-data-visualizacion — xT, Tournament Sims & Viz

Bron: https://github.com/federicorabanos/futbol-data-visualizacion (LanusStats)

**Kernwaarde voor WK bot:**
- **Expected Threat (xT)**: Detailed notebooks on generating xT grids (using socceraction.xthreat on SPADL data), applying to matches for possession value/threat. Great for advanced "creation" features beyond xG/VAEP.
- **Tournament simulation**: Full sims of tournaments using ELO ratings + Transfermarkt team valuations, scaled/combined, Poisson modeling for score prediction, Monte Carlo for group stage positions, advancement probs, knockout paths.
- Visualizations: Pass maps, pitch plots, tables, xT heatmaps — useful for analysis/debugging or bet reports.
- Practical Python for data viz/analysis, with Argentine league focus but general (StatsBomb, etc.).
- Streams/videos folder with live examples.

**Integratie:**
- xT: Enhance feature_engineering with xT computation (via socceraction on events from StatsBombWCImporter or soccerdata). Use for team/player threat metrics in WC matches.
- Tournament sim: Add/extend in backtest.py or new wc_simulator: use our ELO + market values (from eddwebster/TM), Poisson for expected goals, sim full 48-team WC 2026 (groups + knockouts) for probs on "reach QF", "group winner", outrights — key for value where bookies misprice dynamics.
- Viz: Optional matplotlib/seaborn helpers for xT or sim outputs.
- Fits perfectly with existing: socceraction (xT/VAEP), ELO, Poisson, WC context features, player vals.

This enables Monte Carlo sims for WC betting strategy (e.g., correlated bets across matches, bankroll allocation on paths). Combine with historical WC data from jfjelstul/StatsBomb for better ELO calibration.

---
## SoccermaticsForPython (Friends of Tracking Data) — xG + Poisson Simulation

Bron: https://github.com/Friends-of-Tracking-Data-FoTD/SoccermaticsForPython

**Belangrijkste bijdragen voor onze WK bot:**
- **Eenvoudige maar krachtige xG modellen** (3xGModel.py, 5xGModelFit.py): Schoten omzetten naar Distance (meters) + Angle (radians). 2D heatmaps + logistic/probability fitting. Direct toepasbaar op StatsBomb events (we hebben nu StatsBombWCImporter).
- **Match simulatie met Poisson** (11SimulateMatches.py): GLM (statsmodels) voor team attack/defense strengths (`goals ~ home + team + opponent`). Volledige score probability matrix + P(win/draw). Perfect voor:
  - Betere expected goals per national team matchup.
  - Over/Under, BTTS, exact scores met echte verdelingen.
  - Monte Carlo simulaties voor groep + knock-out (WC-specifiek zeer waardevol).
- **FCPython.py**: Pitch plotting helpers (kan later voor debugging of rapporten).
- Algemene aanpak: van raw events → features (distance/angle, possession chains, pass heatmaps) → model → simulation.

**Huidige integratie (2026-06):**
- Uitbreiding van `PoissonModel` in model.py met:
  - `simulate_match(home_xg, away_xg)` → volledige prob matrix + markten.
  - `fit_team_strength_poisson(matches_df)` (GLM als statsmodels aanwezig).
  - `predict_from_strength_model`.
- `feature_engineering.py`: `xg_from_shot_location(distance, angle)` + `batch_xg_from_events` (werkt met onze StatsBomb shots).
- Past perfect bovenop bestaande Poisson (xG blending, home advantage) en de StatsBomb + soccerdata data laag.

Dit vult de analytics-handbook aan met expliciete Poisson team modelling en simulatietools — cruciaal voor value in een 48-team World Cup (veel onzekerheid in groepen, herstel, etc.).

---
## socceraction (ML-KULeuven) — VAEP & xT Action Valuation

Bron: https://github.com/ML-KULeuven/socceraction

**Kernwaarde voor WK bot:**
- **SPADL**: Standaardiseert events (StatsBomb, Wyscout...) tot uniforme acties (pass, dribble, shot... met start/end posities).
- **VAEP** (Valuing Actions by Estimating Probabilities): Modelleert voor elke actie hoezeer die de kans op scoren / tegentreffer verandert (offensive + defensive value). Som per speler = veel betere "impact rating" dan goals of xG alleen.
- **xT** (Expected Threat): Meet dreiging/progressie van balbezit.
- Papers: "Actions speak louder than goals" (Decroos et al. 2019) + vergelijking xT vs VAEP.
- Direct bruikbaar met onze StatsBombWCImporter → convert events → compute values → features.

**Integratie:**
- `scrapers.SoccerActionProcessor`: convert_statsbomb_events + basis VAEP/xT helpers.
- `feature_engineering.compute_vaep_style_features`.
- Uitbreiding van player/team features met action values (naast bestaande xG, rolling stats).
- Gebruik 2022 WC data voor historische player ratings / team creation value.
- Combineer met jfjelstul player appearances voor workload + impact.

Dit geeft de bot écht geavanceerde "player selections" analyse en betere features voor nationale teams waar traditionele stats schaars zijn.

---
## roboflow/sports — Computer Vision voor Sports Analytics

Bron: https://github.com/roboflow/sports

**Kernwaarde voor WK bot:**
- YOLO-based detection: players (incl. keepers/ref), ball, pitch keypoints.
- Tracking: ByteTrack + team classification (embeddings + clustering).
- Radar visualization & ViewTransformer: bird's-eye player positions (real-world coords via calibration).
- Physical metrics: distance traveled, speed, high-intensity efforts, formations, possession proxies.
- Challenges addressed: ball tracking, re-ID, jersey OCR (future), camera calibration.

**Integratie:**
- `scrapers.RoboflowSportsAnalyzer`: extract_tracking_features, workload computation.
- `feature_engineering.compute_tracking_workload_features` (distance, speed → fatigue/adaptation input).
- Gebruik met video (qualifiers, friendlies, historical broadcasts) of pre-extracted positions.
- Combineert perfect met socceraction (events + VAEP values) + StatsBomb + onze context (heat/altitude beïnvloeden bewegingspatronen).
- Voor 2026: verwerk publieke footage voor unpriced fysieke data (werkbelasting, pressing intensity, herstel).

Roadmap in repo + datasets op Roboflow Universe maken het makkelijk om custom WC-modellen te trainen.

---
## Belangrijke ontbrekende elementen (gaps) die extra edge geven voor de WK bot

We hebben nu een sterke basis met data (soccerdata + StatsBomb + openfootball + jfjelstul + football.json + socceraction), features (xG/VAEP/xT, workload/tracking, player similarity/valuation, ELO, context zoals altitude/heat/crowd/adaptation), en modellen (ensembles, Poisson, sims, value betting).

**Top missing pieces die écht extra helpen** (prioriteit op basis van impact voor value betting + "waar bookies data missen"):

### 1. Real-time / verse squad & injury data (hoogste prioriteit)
- **Waarom cruciaal**: Voor "player selections" (hoe ze het gaan doen). Blessures, schorsingen, rotatie, vorm in laatste clubwedstrijden en friendlies zijn vaak laat bekend en slecht geprijsd.
- **Wat we missen**: Live scraping van betrouwbare bronnen (Transfermarkt updates, officiële NT accounts, nieuws), historische injury impact per speler/positie, "availability score" per wedstrijd.
- **Extra edge**: Value op teams met "verborgen" sleutspelers die net fit zijn of juist missen. Combineer met onze squad_strength en market value features.
- **Hoe toevoegen**: Nieuwe scraper (bijv. in scrapers.py) voor TM injury pages of API-Football + news sentiment. Voeg toe aan PlayerInfo / TeamSquadStatus + context_to_features.

### 2. Volledige tournament simulation (Monte Carlo met paths + correlations)
- **Waarom cruciaal**: Voor 48-team WC is group stage + bracket dynamiek enorm (derde plaatsen, "must win" vs dead rubber, pad naar finale). Single-match probs volstaan niet voor "reach QF/SF" of outright value.
- **Wat we missen**: Simuleer hele toernooien (groepen + knock-out) met correlations (moeheid, motivatie, bracket luck). Gebruik ELO + xG/VAEP + adaptation factors.
- **Extra edge**: Bookies prijzen "to reach QF" of "group winner" vaak inefficient (vooral bij underdogs of favorieten in makkelijke groepen).
- **Hoe toevoegen**: Uitbreiden van bestaande sims (backtest.py / prediction_model.py + federicorabanos-style + Google Football RL). Monte Carlo met 10k+ runs. Output: prob per team per stage + value spots.

### 3. Gedetailleerde player workload & recovery (club + NT gecombineerd)
- **Waarom cruciaal**: "Gewend" + herstel is key in expanded format (veel wedstrijden, reizen, klimaat). Club + internationale minuten, high-intensity sprints, travel days, jetlag.
- **Wat we missen**: Accurate combined workload (niet alleen laatste 5 wedstrijden), fatigue curves per positie/leeftijd, recovery models (days rest vs performance drop).
- **Extra edge**: Ondergewaardeerde teams met veel rust of juist "overcooked" favorieten. Koppel aan roboflow tracking + eddwebster similarity.
- **Hoe toevoegen**: Nieuwe features in feature_engineering.py (workload_per_player, fatigue_modifier). Data uit jfjelstul player_appearances + soccerdata club stats + TM minutes.

### 4. Set-piece specifieke metrics + tactical style matchups
- **Waarom cruciaal**: Set pieces (corners, free-kicks, penalties) zijn stabiel en vaak slecht geprijsd. Stijl clash (high press vs counter, direct vs build-up) bepaalt veel meer dan pure xG.
- **Wat we missen**: % goals from set pieces, corner conversion, xG from set pieces, pressing intensity, formation evolution.
- **Extra edge**: WC teams die goed zijn in set pieces of een stijl hebben die niet past bij de tegenstander (denk aan "compact" vs "open" in hitte).
- **Hoe toevoegen**: StatsBomb events + socceraction voor set-piece VAEP. Features zoals "set_piece_xg_superiority", "pressing_index". Gebruik football.json voor historische set-piece data.

### 5. Psychologische / motivationele factoren + public bias
- **Waarom cruciaal**: "Must-win" vs "nothing to play for", coach druk, fan verwachting, historische "giant killing" in WC, overreactie op friendlies/kwalificatie.
- **Wat we missen**: Dead-rubber modeling per groep, motivation score (inspired by ProphitBet), betting market sentiment (public % on big teams), line movement.
- **Extra edge**: Fade the public op favorieten in makkelijke groepen; value op "under the radar" teams met hoge motivatie.
- **Hoe toevoegen**: MOTIVATION_FACTORS uitbreiden met WC-specifieke logica (group stage MD3, "revenge" matches). Odds data integration (The Odds API) voor public bias proxy. LLM voor news sentiment (als we hybrid reasoning toevoegen).

### 6. Live / closing odds + echte CLV tracking
- **Waarom cruciaal**: Value = model prob vs sharp closing line. Zonder live odds en CLV meten, weet je niet of je echt edge hebt.
- **Wat we missen**: Real-time odds van meerdere books (Pinnacle focus), line movement, closing line value (CLV) per bet.
- **Extra edge**: Alleen wedden als we closing line value hebben (tracker.py is er al, maar geen live feed).
- **Hoe toevoegen**: The Odds API integratie uitbreiden in scrapers.py. Automatische CLV logging in backtest + live predict flow.

### 7. National team specifieke historische patronen (WC-only)
- **Waarom cruciaal**: Club stats vertalen niet 1-op-1 naar NT (andere coach, andere stijl, pressure). WC heeft unieke dynamiek (groepsfase, knock-out variance, voorbereiding).
- **Wat we missen**: WC-specifieke form (alleen interlands), manager cycles, "pre-tournament form" vs actual WC, penalty shootout / extra time modeling.
- **Extra edge**: Historische "surprises" en underperformance van favorieten in knock-out.
- **Hoe toevoegen**: jfjelstul + StatsBomb WC data filteren op "only WC matches". Extra features in feature_engineering (wc_only_form, manager_tenure).

### Overige nuttige toevoegingen (minder urgent maar waardevol)
- Real-time weer + pitch condities (beter dan statische VENUES).
- Broadcast tracking (SkillCorner-achtige data voor pressing/distances).
- Market liquidity + sharp vs public money splits.
- LLM-hybrid reasoning (van sports-betting-toolbox) voor natuurlijke verklaringen + gap-filling met nieuws.

**Prioriteit advies**: Begin met 1 (injuries/squad) + 2 (full tournament sim) + 3 (workload). Dat geeft de grootste sprong in "value waar bookies data missen" voor selections en path betting in een 48-team WK.

Deze gaps sluiten perfect aan bij de originele vraag (player past/future, adaptation, crowd, referee, pitfalls). Met de huidige stack (data loaders + features + ensembles) kunnen we ze relatief snel toevoegen zonder alles overhoop te halen.

Bron: https://github.com/eddwebster/football_analytics

**Kernwaarde voor WK bot:**
- **Massive curated resources list**: Data sources (FBref, Understat, Transfermarkt, StatsBomb, Wyscout, Opta, physical, odds, etc.), libraries (soccerdata, socceraction, statsbombpy, worldfootballR, mplsoccer, kloppy, etc. – many already integrated), tutorials (Python/R/Tableau), papers, GitHub repos, concepts (xG modeling, VAEP, xT, player similarity/clustering, player valuation, tactics, set pieces, tracking data).
- **Author's notebooks**: Structured pipeline – 1) scraping (FBref team/player stats, TM valuations, etc.), 2) parsing, 3) engineering, 4) unification (multi-source), 5) analysis/projects (xG with LR/XGBoost, VAEP models, player similarity via PCA+KMeans, tracking data, England Euro 2020 tournament analysis).
- Specific gems for WC: Player similarity/style analysis (identify comparable players for "gaan doen" expectations), squad valuation from TM, tournament-specific modeling, data unification best practices for national teams (club + int'l data).
- Historical + club data focus that translates well to international/WC (e.g., relative league strength, player ratings).

**Integratie:**
- Referenced as primary "awesome list" in resources/docs for ongoing research.
- Ideas from player_similarity notebooks → potential `feature_engineering.player_similarity_features` (cluster players by style/stats for WC squad depth analysis).
- TM scraping/valuation notebooks → extend PlayerInfo/TeamSquadStatus with market value for "squad strength" and value betting (e.g., under/over-valued players in selections).
- xG/VAEP notebooks reinforce/complement our Soccermatics + socceraction work.
- Best practices for multi-source data unification (soccerdata + StatsBomb + openfootball/jfjelstul + TM).
- England Euro 2020 notebook as template for 2026 tournament stage modeling (group dynamics, dead rubbers).

Gebruik deze repo als "go-to" gids bij uitbreiding van features, nieuwe data sources (TM valuations!), of analyse van WC squads. Combineer met onze bestaande modular pipeline voor productie-grade WK bot.

**Volgende mogelijke uitbreiding**: pipeline om highlights → tracking data → workload features te halen en te mergen in MatchContext voor betere "squad depth" en "adaptation" scores.

---
## google-research/football (GRF) — RL Simulator voor Football

Bron: https://github.com/google-research/football

**Kernwaarde voor WK bot:**
- Volledige 3D physics-based RL environment (Gym-like API) voor soccer: single/multi-agent, academy scenarios tot full 11v11.
- Observations: structured (player/ball positions, velocities, ownership) of raw pixels.
- Acties: discrete (pass, shoot, move, sprint, etc.) of continuous.
- Pre-trained agents (PPO), built-in AI, self-play.
- Logging/replays/dumps: genereer synthetic match data, events, traces voor training/augmentation (possession, shots, tactics).
- Multi-agent RL: model team dynamics, player interactions, tactics – ideaal voor "squad depth", "team adaptation", "player selections".
- Research focus: RL for sports, imitation learning. Kaggle comp, papers.
- Scenarios: customize voor "what-if" (e.g., fatigue/heat door reward shaping of physics mods voor adaptation modeling).

**Integratie:**
- Optionele sim wrapper in scrapers (GoogleFootballSimulator): run episodes, extract synthetic events/stats (xG-like, possession value, player contrib) om echte data (StatsBomb, jfjelstul) te augmenteren voor zeldzame WC scenarios.
- Feature engineering: gebruik structured obs voor advanced features (positioning, pressure, "RL threat" als complement op xT/VAEP).
- Backtest/prediction: hybrid sims – combineer met ELO/Poisson (van eerdere repos) voor Monte Carlo WC tournament sims met realistische agent behaviors i.p.v. pure stats. Genereer "optimal play" baselines voor value detection (waar bookies menselijke biases hebben).
- WC-specifiek: sim "adaptation" door env params (snelheid in heat) of agent policies (vermoeidheid). Genereer data voor squad depth analyse (sub agents voor bench impact).
- Synergie: feed ELO/strengths als agent init; gebruik replays voor extra VAEP/xT training; combineer met roboflow tracking (real + sim) en eddwebster sim/vals.
- Voor backtesting/strategy: sim duizenden WC paths onder verschillende condities (crowd, altitude) voor EV op "reach QF", groups, outrights – waar data schaars is.

Install: `pip install gfootball` (of build from source; Docker aanbevolen). Optioneel voor advanced RL sims – core blijft op open data + stats/ML.

Dit voegt "agent-based" modeling toe voor diepere inzicht in "hoe teams spelen" en value waar pure historische data tekortschiet. Gebruik pre-trained agents voor snelle sims zonder full training.

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

---

## WK / World Cup 2026 Uitbreiding (2026-06)
Toegevoegd ter ondersteuning van "onze wk bot":
- config.py: VENUES_2026 (16 stadions + alt, roof, coords, heat), WC_LEAGUE=1, factor dicts (HEAT/CROWD/ADAPT).
- scrapers: uitgebreide Weather + WCOpenData (openfootball GitHub raw squads/fixtures).
- data_collector: WC squad/teams/fixtures calls (API-F + open), MatchContext WC velden (venue, crowd, adapt, travel, stage).
- features: nieuwe WC context features (altitude, crowd, heat, stage, adapt).
- prediction: WC key factors in value analyzer (altitude, crowd, dead-rubber etc.).
- main: --venue / --stage support + WC predict branch.
- Reusable: alles bouwt voort op bestaande PlayerInfo, aggregator, ensemble, kelly.

Gebruik voorbeeld:
  python main.py --mode predict --league WC --venue "Mexico City" --stage group --fixtures-file wc_fixtures.json

Belangrijke valkuilen die we nu beter kunnen raken: altitude-effect (niet volledig in odds), grote crowds voor hosts, group-stage motivation asymmetry, heat impact op unders, squad depth bij 48 teams. 

Data die bookies missen of traag prijzen: gedetailleerde NT player workload, venue-specifieke acclimatisatie, open squad updates vs live odds.

## AlphaPy (SportFlow) als extra inspiratie (2026-06)
Gebruiker wees op https://github.com/ScottfreeLLC/AlphaPy

AlphaPy is een AutoML framework met specifieke **SportFlow** voor het voorspellen van sportevenementen (naast MarketFlow voor trading).

Belangrijke elementen die relevant zijn voor onze WK bot:
- Volledige model pipeline: data → features → modellen (scikit-learn, XGBoost, LightGBM, CatBoost, Keras) → ensembles (blended/stacked).
- Geautomatiseerde feature engineering en modelselectie.
- Specifieke support voor sports betting / event prediction.
- Backtesting en portfolio analyse (via pyfolio integratie).

Huidige status:
- Directe installatie van legacy AlphaPy heeft dependency issues (imbalanced-learn API veranderingen).
- Aanbeveling: Gebruik als **architecturale inspiratie** i.p.v. harde dependency.
  - Emuleer SportFlow pipeline met onze WC data loaders (openfootball, jfjelstul, rezar) + bestaande feature_engineering.
  - Overweeg ensembles en AutoML elementen toe te voegen (we hebben al Optuna in sommige code).
  - Bekijk alphapy-pro voor modernere versie.

Onze modulaire structuur (data_collector + feature_engineering + prediction_model + value analyzer) past goed bij AlphaPy's model pipeline. We kunnen de sterke punten (SportFlow voor sports, ensembles) combineren met onze WC-specifieke data en Kelly/EV logica.

Gebruik in toekomstige iteraties:
  - Experimenteer met AlphaPy SportFlow op sample data uit onze loaders.
  - Voeg stacked ensembles toe voor betere WC voorspellingen.
