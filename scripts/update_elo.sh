#!/usr/bin/env bash
# Refresh Tennis Abstract ATP Elo ratings (weekly source)
# Run: bash scripts/update_elo.sh

set -euo pipefail
cd "$(dirname "$0")/.."
python3 cli.py update --elo
python3 -c "
from data_sources.tennis_abstract_elo import TennisAbstractEloSource
s = TennisAbstractEloSource()
for name in ['Arthur Fery','Alexander Zverev','Flavio Cobolli','Taylor Fritz']:
    r = s.lookup(name)
    if r:
        print(f\"  {r['player']}: Elo {r.get('elo')} | gElo {r.get('gelo')} | ATP #{int(r.get('atp_rank',0))}\")
"