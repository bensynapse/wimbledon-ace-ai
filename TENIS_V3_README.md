# WimbledonAce AI — Market Intelligence

## Wat dit is
Tennis wedstrijd-analyse met Elo, UE/charting, weer, perscitaten, vermoeidheid en value detection — geen black-box odds, alles rule-based + model.

## CLI
```bash
python cli.py intelligence --player1 "Arthur Fery" --player2 "Alexander Zverev" \
  --surface grass --tournament Wimbledon --odds-p1 5.50 --odds-p2 1.14 \
  --context-file ctx.json
```

Context JSON voor wedstrijddag-weer:
```json
{"match_date": "2026-07-10", "match_hour": 15}
```

## Structuur
- `cli.py` — hoofd-CLI
- `intelligence/` — UE, fatigue, rapporten
- `data_sources/` — weer, charting, quotes, Tennis Abstract Elo
- `pipelines/` — context builder, daily scan, backtest
- `models/` — probability model, eigen Elo tracker