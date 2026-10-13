#!/usr/bin/env bash
set -euo pipefail
cd /home/ibb3/ouro-jlens
export OURO_PROJECT_ROOT=/home/ibb3/ouro-jlens
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
exec /home/ibb3/ouro-jlens/.venv/bin/python -u \
  /home/ibb3/ouro-jlens/results/deception-run3-20261009/code/run_deception_run3.py \
  --output /home/ibb3/ouro-jlens/results/deception-run3-20261009 \
  --phase discovery \
  >> /home/ibb3/ouro-jlens/results/deception-run3-20261009/run.log 2>&1
