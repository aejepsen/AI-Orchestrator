#!/usr/bin/env bash
# Empacota o projeto pro Colab (sem venv, git, node_modules, resultados).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-$HOME/ai-orchestrator.zip}"
rm -f "$OUT"
zip -qr "$OUT" gateway services evals train \
  -x "*/__pycache__/*" -x "*.pyc" -x "evals/results/*" -x "train/dataset/*" \
  -x "gateway/tests/*" -x "*/node_modules/*"
echo "OK: $OUT ($(du -h "$OUT" | cut -f1)) — faça upload pro MyDrive/"
