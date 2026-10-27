#!/usr/bin/env bash
# One command for every number and figure in the write-up.
#
#   Full run (needs the dataset, ~1.5-2 h on a Colab T4):   DATA_DIR=/path/to/data bash run_all.sh
#   Tables + figures only, from committed results (no data, no GPU, ~1 min):  bash run_all.sh --report-only
#   Synthetic smoke test of the whole pipeline (~3 min on CPU):               bash run_all.sh --smoke
#
# Committed checkpoints in checkpoints/ are reused; add --retrain to train from scratch.
set -euo pipefail
cd "$(dirname "$0")"

MODE=full
EXTRA=()
for a in "$@"; do
  case "$a" in
    --report-only) MODE=report ;;
    --smoke) MODE=smoke ;;
    --retrain) EXTRA+=(--retrain) ;;
    *) echo "unknown option $a"; exit 1 ;;
  esac
done

if [ "$MODE" = report ]; then
  python -m src.report
  exit 0
fi

if [ "$MODE" = smoke ]; then
  # Separate output folders so the smoke test never touches the real results
  export WORK=work/smoke DATA_DIR=work/smoke/data
  EXTRA+=(--smoke --results work/smoke/results --checkpoints work/smoke/checkpoints --figures work/smoke/figures)
  [ -d "$DATA_DIR" ] || python -m src.synthetic --out "$DATA_DIR" --n 60
fi

: "${DATA_DIR:?Set DATA_DIR to the dataset root (bonafide/ and spoof/<material>/), see README}"

python -m src.data     --data "$DATA_DIR" "${EXTRA[@]}"   # 1. fixed split (results/split.csv)
python -m src.extract  --data "$DATA_DIR" "${EXTRA[@]}"   # 2. minutiae for every image (cached in work/)
python -m src.train    --data "$DATA_DIR" "${EXTRA[@]}"   # 3. MobileNet-v1 per training seed
python -m src.evaluate --data "$DATA_DIR" "${EXTRA[@]}"   # 4. scores for all patch-selection settings
python -m src.timing   --data "$DATA_DIR" "${EXTRA[@]}"   # 5. single-thread CPU latency by stage
python -m src.report   "${EXTRA[@]}"                      # 6. tables, figures, headline.json
