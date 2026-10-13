#!/usr/bin/env bash
set -euo pipefail
cd /home/ibb3/ouro-jlens
exec /home/ibb3/ouro-jlens/.venv/bin/python -u \
  /home/ibb3/ouro-jlens/continue_deception_run3.py \
  >> /home/ibb3/ouro-jlens/results/deception-run3-20261009/continuation.log 2>&1
