#!/bin/bash
# Stage 4: metrics (CPU) + blind judge (Claude CLI) + results page. See eval/README.md for fields.
set -e; cd "$(dirname "$0")/.."; . scripts/env.sh
PY=${PY:-eval/venv/bin/python}; V=${1:-turbo_lora_ema}
$PY eval/metrics.py --track fast --variant $V --all --workers 6
$PY eval/judge.py   --track fast --variant $V --all --workers 4
$PY eval/build_page.py        # -> $OUT_DIR/render2real-poc/index.html
