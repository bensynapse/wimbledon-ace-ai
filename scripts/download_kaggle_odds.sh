#!/usr/bin/env bash
# Download dissfya/atp-tennis-2000-2023daily-pull (ATP + historical bookmaker odds)
#
# Setup:
#   pip install kaggle
#   # Kaggle → Account → Create API Token → save as ~/.kaggle/kaggle.json
#   chmod 600 ~/.kaggle/kaggle.json
#
# Run:
#   bash scripts/download_kaggle_odds.sh

set -euo pipefail
cd "$(dirname "$0")/.."

OUT="data/kaggle_odds"
mkdir -p "$OUT"

if ! command -v kaggle &>/dev/null; then
  echo "Installing kaggle CLI..."
  pip3 install kaggle
fi

kaggle datasets download -d dissfya/atp-tennis-2000-2023daily-pull -p "$OUT" --unzip

echo ""
echo "Done. CSV at: $OUT/atp_tennis.csv"
echo "Backtest: python3 main.py --mode backtest --start-year 2018 --min-edge 0.05"