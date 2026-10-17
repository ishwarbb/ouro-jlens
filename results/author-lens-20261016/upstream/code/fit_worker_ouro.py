"""Fit the Jacobian lens on Ouro-2.6B over the unrolled 192 virtual layers.

Usage: python fit_worker_ouro.py WORKER N_WORKERS [--limit N] [--dim-batch B]
Worker i takes prompts[i::N_WORKERS]; merge with merge_lenses.py afterwards.
"""

import argparse
import json
import logging

import torch
import transformers

import jlens
from ouro_lens import from_ouro

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

parser = argparse.ArgumentParser()
parser.add_argument("worker", type=int)
parser.add_argument("n_workers", type=int)
parser.add_argument("--prompts", default="prompts_1000.json")
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--prefix", default="ouro_w")
parser.add_argument("--dim-batch", type=int, default=64)
args = parser.parse_args()

with open(args.prompts) as f:
    prompts = json.load(f)
if args.limit:
    prompts = prompts[: args.limit]
prompts = prompts[args.worker :: args.n_workers]
print(f"worker {args.worker}/{args.n_workers}: {len(prompts)} prompts", flush=True)

from ouro_lens import load_ouro
hf, tok = load_ouro()
hf = hf.cuda()
model = from_ouro(hf, tok)
print(model, flush=True)

lens = jlens.fit(
    model,
    prompts,
    dim_batch=args.dim_batch,
    max_seq_len=128,
    checkpoint_path=f"ckpt_{args.prefix}{args.worker}.pt",
    checkpoint_every=10,
)
lens.save(f"lens_{args.prefix}{args.worker}.pt")
print(f"peak GPU mem: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
print("DONE_FIT", flush=True)
