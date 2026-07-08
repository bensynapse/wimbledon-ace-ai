#!/bin/bash
# Download guillemservera/tennis from Kaggle (ATP + WTA)
#
# Setup (once):
#   pip install kaggle
#   # Kaggle → Account → Create API Token → save as ~/.kaggle/kaggle.json
#   chmod 600 ~/.kaggle/kaggle.json
#
# Run:
#   bash scripts/download_kaggle.sh

set -euo pipefail
cd "$(dirname "$0")/.."

OUT="data/kaggle_tennis"
mkdir -p "$OUT"

if ! command -v kaggle &>/dev/null; then
  echo "Installing kaggle CLI..."
  pip install kaggle
fi

echo "Downloading guillemservera/tennis → $OUT"
kaggle datasets download -d guillemservera/tennis -p "$OUT" --unzip

echo "Done. Files:"
ls -la "$OUT" | head -20
echo ""
echo "Test: python3 -c \"from data_sources.kaggle_tennis import KaggleTennisSource; s=KaggleTennisSource(); print(len(s.download_tour('wta', 2022)))\""