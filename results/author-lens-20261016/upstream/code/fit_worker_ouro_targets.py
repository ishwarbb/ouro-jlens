"""Fit per-loop-target Jacobian lenses on Ouro-2.6B: J_{v->t} for each loop-end
target t in {47, 95, 143} (the loop-4 target 191 is the existing lens).

Usage: python fit_worker_ouro_targets.py WORKER N_WORKERS [--dim-batch B]
"""

import argparse
import json
import logging

import torch

import jlens
from ouro_lens import from_ouro, load_ouro

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

parser = argparse.ArgumentParser()
parser.add_argument("worker", type=int)
parser.add_argument("n_workers", type=int)
parser.add_argument("--prompts", default="prompts_1000.json")
parser.add_argument("--dim-batch", type=int, default=64)
parser.add_argument("--targets", type=int, nargs="+", default=[47, 95, 143])
args = parser.parse_args()

with open(args.prompts) as f:
    prompts = json.load(f)[args.worker :: args.n_workers]
print(f"worker {args.worker}/{args.n_workers}: {len(prompts)} prompts, "
      f"targets {args.targets}", flush=True)

hf, tok = load_ouro()
hf = hf.cuda()
model = from_ouro(hf, tok)
print(model, flush=True)

for t in args.targets:
    print(f"=== target v{t} ===", flush=True)
    lens = jlens.fit(
        model,
        prompts,
        target_layer=t,
        dim_batch=args.dim_batch,
        max_seq_len=128,
        checkpoint_path=f"ckpt_ouro_t{t}_w{args.worker}.pt",
        checkpoint_every=25,
    )
    lens.save(f"lens_ouro_t{t}_w{args.worker}.pt")
    print(f"target v{t} done, peak GPU mem "
          f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)

print("DONE_FIT_TARGETS", flush=True)
