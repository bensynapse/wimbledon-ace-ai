# 🎾 Tennis-Bet V3 - Volledige Analyse Tool

## Wat dit is
Volledige replicatie van BSP methode + onze V3 upgrades (Elo surface, Monte Carlo, UE/fatigue tracker, value detection, live hedge).

## Hoe te gebruiken

### 1. CLI
```bash
python tennis_v3/cli.py analyze --p1 "Taylor Fritz" --p2 "Alexander Zverev" --odds1 1.90 --odds2 1.90
```

### 2. xlsx
Genereert `Tennis_Analysis_V3.xlsx` met alle tabs.

### 3. X post
`--post-x` flag maakt thread met analyse.

## Structuur
- `tennis_v3/` - core code
- `data/` - screenshots & images
- `PLAN.md` - dagelijkse workflow
- `BACKTEST.md` - resultaten

**We gaan dit nu vullen.**