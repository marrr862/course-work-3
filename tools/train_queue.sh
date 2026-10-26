#!/bin/bash
# Stage 4 training queue (runs sequentially; log: reports/v4/queue.log)
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
run() { echo "=== $(date '+%H:%M') start $*"; $PY -m fakenews.train_transformer "$@" 2>&1 | grep -v -i warn; echo "=== $(date '+%H:%M') done $1 $2"; }
run --dataset fakenewsnet --model roberta-base --input url_title --tag rb_url
run --dataset liar --model roberta-base --input liar_meta --tag rb_meta --max-len 128
run --dataset fakenewsnet --model roberta-large --input title --tag rl_title --batch 16 --lr 1e-5 --epochs 3
#un --dataset fakenewsnet --model roberta-large --input url_title --tag rl_url --batch 16 --lr 1e-5 --epochs 3
echo "=== ALL DONE $(date '+%H:%M')"
