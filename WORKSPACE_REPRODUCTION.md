# Full Ouro fit and workspace verification

The completed original run used GPUs **0–3** for both fitting and evaluation. Eight logical
row shards are scheduled on these four physical GPUs. The supervisor survives a terminal disconnect and proceeds from fitting
to merging, evaluation, and analysis. It does not send messages or use paid APIs.

Archived state: [fit status](results/jlens-family-1000/status.json).
Dashboard: `tmux attach -t ouro-workspace`. Window 1 shows live progress,
saved checkpoints, an approximate fit ETA, and GPU telemetry; window 2 shows
supervisor and fit logs; window 3 follows evaluation logs. Use `Ctrl-b`, then
`1`, `2`, or `3` to switch windows, and `Ctrl-b`, then `d` to detach. Closing
the dashboard does not stop the supervisor. Recreate it with
`.venv/bin/python create_ouro_tmux.py` if needed.
Frozen experiment specification: [workspace_protocol.json](workspace_protocol.json).
Local freeze receipt: [protocol-freeze-receipt.json](results/workspace-full/protocol-freeze-receipt.json).

## Fit

The fit estimates **476** source-to-target matrices: all upstream sources for
each loop-end target 47, 95, 143, and 191. Every matrix averages the same 1,000
WikiText-103 prompts. Each prompt has 128 tokens including BOS; positions 0–15
and 127 are excluded from the estimator reductions. The full-run corpus is
`fitting_data/wikitext103-1000-128.json`. Twenty short records in the earlier
corpus were replaced by continuing the same ordered selection until 1,000
qualifying records were obtained. Dataset revision, row indices, model weights,
and implementation hashes are recorded.

Each logical shard computes a disjoint block of 256 output dimensions on **every prompt**;
merging concatenates rows rather than averaging independent shards. The
estimator keeps row results on the GPU during backpropagation and transfers
them together, avoiding repeated synchronization. Its cotangents, source/target
position reduction, and FP32 prompt average match the reference. Validation:

- Exact equality with the reference on the actual tiny Ouro architecture,
  including partitioning at a row boundary not aligned with the derivative batch.
- Exact full-checkpoint agreement with the reference at source sites 0 and 190
  for 256 dimensions of the final-target lens; see
  [benchmark-191-b8.json](results/jlens-family-1000/benchmark-191-b8.json).
- The production derivative batch is 16, with eight independent row shards and
  at most four active workers. The original eight-shard manifest is retained so
  checkpoints remain compatible when a shard moves to another GPU.
  Every worker checks deterministic forwards and all four native loop logits.
- Atomic checkpoints every 50 prompts; additional 500-prompt snapshots support
  convergence comparisons. Model weights remain frozen; KV caching and gradient
  checkpointing are disabled.

The portable family is approximately 7.44 GiB, plus worker checkpoints and
half-run snapshots. Large tensor files are Git-ignored. `family.json` is written
only after all workers have completed and every matrix is assembled. A separate
save/load comparison checks exact FP32 preservation before evaluation.

## Experiments

All prompt data come from the pinned reference repository's
`jacobian-lens/data/experiments/`. No evaluation prompt is used for fitting or
to select a favorable intervention strength. The primary physical band is
zero-based 24–40, with edits maintained through loops 2–4. The primary swap is
the original study's pseudoinverse coordinate permutation between normalized
concept directions. For two unit directions, this equals a reflection in their
difference; tests verify this even when directions are strongly correlated.

The plan covers 399 independently resumable items:

| Family | Items | Main observations |
|---|---:|---|
| Flexible generalization | 64 prompts / 192 swaps | Does swapping an argument redirect different downstream functions? |
| Two-hop reasoning | 90 | Can a swap of an unspoken intermediate redirect the answer? Are argument ablations more disruptive than unrelated-token ablations? |
| Verbal report | 14 categories | Does the model name the swapped-in concept? |
| Language selectivity | 8 paired passages | Is the language label more readable when the task asks for it? |
| Directed modulation | 22 topics × 24 instructions | Does topic readout differ under focus, control, and suppression while copying identical text? |
| Introspection | 101 candidate concepts | Does injecting previously absent content make it reportable? |

Tokenization exclusions remain in the records. Complete single-token forms
are required; the evaluator never silently substitutes a first subtoken.
Verbal-report source directions use the model's actual emitted token. Output
matching includes exact case-insensitive whole-token variants, and actual
argmax/top-5 membership is distinguished from optimistic ranks under ties.

Controls include unedited baselines, random edits with the same local update
norm, raw-logit-lens directions, final-target directions, an early-layer band,
and edits confined to individual loops. Projection transfer, band 24–36, and
double strength are predefined sensitivities. They do not replace a failed
primary result. Loop-end rank trajectories are recorded to inspect restoration
after an early edit.

The analysis reports raw counts, baseline task accuracy, baseline-correct
subsets, and paired bootstrap differences clustered by semantic unit. All 10,000
resamples use seed 42. Ordinary 95% intervals and more conservative intervals
are reported. Small samples, especially the eight language passages, may leave
an effect unestablished even if its point estimate is positive.

This is a **qualitative reproduction**, not a promise to match the paper's
percentages. The authors' source code, exact fitting sample, prompt formatting,
and complete causal implementation were not available. The tests address
several core workspace properties; capacity, ignition, dual-task interference,
and the sparse concept inventory are not assessed.

## Files and commands

```bash
cd ouro-jlens

# Read current progress without touching the running processes.
cat results/jlens-family-1000/status.json
tail -n 5 results/jlens-family-1000/workers/worker-0/worker.log
tmux attach -t ouro-workspace

# Correctness tests, no full checkpoint loaded.
OMP_NUM_THREADS=2 .venv/bin/python -m unittest -v \
  test_ouro_lens test_ouro_family test_ouro_full_fit \
  test_workspace_interventions test_workspace_analysis test_ouro_supervisor

# Only if the supervisor has stopped: resume its preserved checkpoints.
OMP_NUM_THREADS=2 .venv/bin/python run_ouro_workspace.py --gpus 0,1,2,3

# Recompute the completed report from recorded trials, without GPU inference.
OMP_NUM_THREADS=2 .venv/bin/python analyze_workspace.py --output results/workspace-full
```

An exclusive run lock prevents duplicate supervisors. Its PID
is stored in `results/jlens-family-1000/supervisor.pid`. After a supervisor
restart, live fitting workers are adopted by verified PID and process identity;
queued shards resume their most recent atomic checkpoint. A worker failure leaves
checkpoints intact and writes `supervisor-error.json`; it is not recorded as a
successful fit.

Expected completed outputs:

- `results/jlens-family-1000/family.json`, `target-*.pt`: the full fitted family.
- `results/jlens-family-1000/convergence.json`: 500-to-1,000-prompt matrix shifts
  and cosines at every source and target.
- `results/workspace-full/records/`: one JSON file per evaluation item, retaining
  all conditions, prompts, exclusions, loop outputs, and measurements.
- `results/workspace-full/summary.json` and `REPORT.md`: recomputed results.
- `results/workspace-full/code/`: exact evaluator/analysis source snapshots.

`results/workspace-debug*` use the earlier ten-prompt smoke family solely for
testing the harness. They must not be reported as full-fit reproduction results.
