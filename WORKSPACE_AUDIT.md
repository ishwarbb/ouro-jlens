# Audit of the Ouro workspace reproduction

This audit responds to the inconclusive results of `results/workspace-full/`.
It preserves that run and reuses its fitted 1,000-prompt lens family. The new
run is `results/workspace-audit/`. It covers every released prompt for flexible
generalization, two-hop steering and latent readout, verbal report, and language
selectivity, plus the previously omitted line-count selectivity task.

The audit is informed by the previous results, including the previous
projection-transfer sensitivity. It is not a fresh preregistration or an
independent sample. The finite grid is fixed before this new run; no setting is
chosen because it produces the paper's number. All outcomes, including failed
settings, are retained.

## What was wrong with the earlier comparison

1. **Operator mismatch.** The old primary used a reflection that exchanges two
   coordinates. This is a mathematically valid operator but not the same
   operation as source-projection transfer. The looped-model paper describes
   clamp-transfer. We now use `h' = h + (h·a)(b−a)` as the primary interpretation,
   with normalized source `a` and destination `b`. The paper does not specify
   enough algebra to prove this is the exact author implementation. We therefore
   also run coordinate exchange, sequential destination overwrite, and a
   minimum-norm clamp enforcing `h'·a=0` and `h'·b=h·a`.
   The [original study's methods](https://transformer-circuits.pub/2026/workspace/index.html)
   explicitly give the pseudoinverse coordinate exchange, while its verbal
   explanation also describes subtracting a source projection and adding it
   to the target. Thus the old operator was not invented or mathematically
   broken. There is an unresolved specification difference; transfer is a
   plausible reading of the looped paper, not a verified recovery of its code.
2. **Verbal-report metric mismatch.** Our former headline criterion required
   top-1 success. The paper's headline is top-5. Both metrics are now explicit.
3. **Unnecessary intervention exclusions.** The former evaluator required the
   *original answer* to have a whole-token spelling even when evaluating a swap
   to a different, eligible answer. That changes the test denominator and
   suppresses valid interventions. The new evaluator requires eligible source
   and destination directions, scores the replacement separately, and retains
   every attempted trial in the denominator.
4. **Conflation of significance and reproduction.** The original language
   selectivity effect was +25 percentage points, the same point estimate the
   paper reports. Our cluster interval included zero on eight passages. That
   is limited precision, not disagreement with the reported point estimate.
5. **Overstrong latent-readout criterion.** The old criterion demanded own-loop
   readout outperform raw and final-target readout after taking a union across
   loops 2–4. Loop 4's own and final lenses are identical. This obscures the
   earlier-loop distinction and adds a criterion the paper does not require.
   We now report each loop and matched physical sites, with loops 2–3 isolated
   for the comparison against final-target readout.
6. **Incomplete selectivity coverage.** The first run included language but not
   line-count selectivity. Both released tasks are now evaluated.
7. **Documentation error.** `WORKSPACE_REPRODUCTION.md` stated 399 items; the
   frozen original manifest and actual run contained 299. The new manifest
   enumerates every item explicitly and the analyzer checks exact ID coverage.

Consequently, the original “not established” labels should be read as outcomes
of that specific operator, metric, and statistical criterion, not as evidence
that the paper's corresponding result failed to reproduce.

## Fit and adapter audit

The existing fit has 476 matrices, covering all upstream sources for targets
47, 95, 143 and 191. It uses the same 1,000 examples for every matrix, 128 input
tokens, skips positions 0–15, and uses the reference estimator's exclusion of
the final prediction position. The source/output-position reduction is copied
from the reference estimator and covered by exact agreement checks. Output-row
shards are concatenated, not averaged as separate prompt estimates.

The virtual adapter preserves the native tied model computation, including
between-loop normalization. Each layer firing has a distinct tap. Native
loop-end logits, stopped-prefix equivalence and repeated forwards are checked
exactly. Parameters are frozen; gradient checkpointing and KV caching are
disabled. Existing tests also check differentiation through recurrent blocks.
The full checkpoint validation is repeated by each audit worker.

What cannot be claimed identical: the authors' exact corpus rows, tokenizer
selection details, model revision, attention backend, and code were not located.
Our corpus selection is recorded and deterministic, but that does not establish
identity with theirs. The original fit's 500-to-1,000-example convergence file
remains available; this audit does not use evaluation performance to refit or
select new matrices. The paper uses a different Transformers version and GPU.
Native equivalence and estimator tests reduce implementation risk but do not
prove that all numerical differences are immaterial.

The independent `audit_ouro_fit.py` check verified all 3,808 saved row blocks
(476 complete matrices) against the worker sums with exact equality. Its
float64 convergence reductions put own-band cosine minima at 0.99857, 0.99817,
0.99818 and 0.99826 for the four targets. Maximum relative 500-to-1,000 changes
in those bands are 5.42%, 6.05%, 6.09% and 5.96%. These are half-versus-full
differences, not the final single-prompt running-mean shifts used elsewhere.
The old FP32 cosine reduction occasionally exceeded one by approximately
0.0001; the new validation artifact fixes that reporting precision issue
without changing any matrix. See `results/workspace-audit/fit-audit.json`.

## Fixed rerun design

Primary tasks:

- All 64 flexible prompts and all 192 within-category swaps.
- All 90 two-hop prompts, including full physical-depth latent readout.
- All 14 verbal-report categories; first ten released candidates, excluding
  the actual baseline answer as in the released instructions.
- All eight paired language passages.
- All 11 line-count passages and all four released conditions.

Primary edits cover physical layers 24–40 inclusive, loops 2–4, all prompt
positions, with each loop's own target lens. The following complete declared
grid is run on every primary causal item:

- Every nonempty subset of loops 2, 3, 4, plus all four loops.
- Strengths 0.25, 0.5, 0.75, 1, 1.25, 1.5 and 2.
- Physical bands 24–40, 20–40, 23–39, 24–36, 28–40, 32–40 and 0–16.
- Four operator definitions, raw-lens and final-target directions, and a
  final-position-only intervention.
- Three independent random-direction controls, each scaled to the local
  transfer update norm. Controls use their own evolving state at each hook;
  they do not replay a stored primary trajectory.

This is one-factor-at-a-time coverage around the primary setting, **not** a
Cartesian product of every setting. Loop subsets are exhaustively enumerated.
The sensitivity family is exploratory and no “best” setting substitutes for the
primary comparison.

Additional full-corpus primary-transfer reruns check BOS removal, direction
tokens without preferred leading spaces, lowercase direction selection,
raw/chat formatting, and trailing-whitespace removal. Verbal report also checks
colon-only, `Answer:`, and empty assistant prefills. These address specific
unknowns in the public description; they do not establish the authors' wrapper.

Latent readouts retain ranks and top-1/5/8 hits at each physical source layer,
including matched own/raw/final comparisons in the band and all four loops.
Only prompt-absent concepts enter the latent analysis. Concept foils are logged.
Language readout uses the post-passage question span. Line-count readout covers
non-BOS prompt positions and separately tracks general two-digit/number-word
tokens and the actual count. As in the released instructions, the first line's
character count is computed after wrapping at the supplied width.

## Scoring and statistics

Whole-token case-insensitive matches and literal top-k membership are the main
metrics. Multi-token words cannot receive a whole-word next-token success.
Separately labelled prefix scores diagnose the impact of that restriction;
they must not be reported as complete-word answers. Direction failures and
unscorable whole-token targets remain visible, including in attempted-trial
denominators. Baseline accuracy, valid-category report sources, and exact token
IDs are recorded.

The paper reports raw counts and Wilson intervals. We report those counts and
Wilson intervals for descriptive comparison, alongside paired bootstrap
intervals clustered by semantic unit. Reusing a concept across prompts or
sweeping its strength does not add independent samples. The old seven-family
tail adjustment is retained only as a sensitivity measure; it is not a
multiple-comparison guarantee for this expanded grid.

Per-condition loop-end answer ranks are saved to distinguish an early edit
being reversed from a late edit failing to displace the original answer.
This is a test of predicted trajectories, not a proof that a particular
normalization operation causes them.

## Provenance and execution

The paper HTML and parsed text were downloaded on 2026-09-28. Its HTML contains
no authors' repository or artifact link; focused web searches did not locate an
author release. The paper refers to released verification artifacts, so absence
of a located link is not proof that no release exists. The original study's
public prompt files are available in the pinned `jacobian-lens` checkout.

Sources:

- [Looped-model paper, v1](https://arxiv.org/html/2609.01924v1), especially
  §§3, 7 and Appendices B, G and H.
- [Local released task instructions](jacobian-lens/data/experiments/README.md).
- [Original run protocol](results/workspace-full/protocol-frozen.json).

```bash
cd ouro-jlens
.venv/bin/python workspace_audit.py init
OMP_NUM_THREADS=2 .venv/bin/python -u workspace_audit_queue.py run
.venv/bin/python workspace_audit_analysis.py --output results/workspace-audit
```

The supervisor uses GPUs 0–3 only, checkpoints each completed item atomically,
checks immutable code/data hashes, and produces the final report only when
every expected record exists. An exclusive lock prevents duplicate supervisors.
Logs, live progress, code snapshots and exact prompts stay under the run folder.
The initial scheduler was interrupted after 558 completed items because
round-robin assignment concentrated expensive primary items on two workers.
The replacement scheduler shares a locked queue across the same four GPUs and
skips those records. `execution-plan.json` records this scheduling change and
its driver hash; the scientific manifest and numerical code remain unchanged.

The audit does not run the entire eleven-family paper: capacity, ignition,
dual-task interference, summoning, and the concept inventory are separate from
the previously inconclusive tests. It does not test Huginn, Qwen, or the
Thinking checkpoint. These limits must remain explicit in any conclusion.

## Audit of the previously positive tests

Those tests were not requested for rerunning, but their positive local results
also must not be described as exact numerical replications:

- Our original single-direction ablations used the two-hop task's eligible,
  baseline-correct subset (48 trials). The paper reports an ablation population
  of 42 for Ouro. Its precise population has not been verified against ours.
  We did not implement its ten-atom inventory-subspace ablation. Our positive
  necessity result establishes a local causal effect under our protocol.
- Our original injected-content top-1 rate at the largest strength was 70.3%,
  versus the paper's 31%. That is qualitative evidence of reportability, not a
  reproduced percentage. Exact token selection, prefix and scale conventions
  remain possible sources of difference.
- Our topic modulation rates were 85.5% / 37.9% / 14.3% for focus / neutral /
  suppression. They have the reported ordering but not identical values. We
  used all 24 released phrasings and one deterministically assigned carrier
  per topic, and did not evaluate the math or line-width modulation arms.

The earlier final answer therefore overstated matters if “supported” was read
as “implemented and numerically reproduced every corresponding paper test.”
Its intended statistical meaning applied only to the local tests. The new
comparison reports raw agreement and inferential evidence separately.

## Follow-up diagnostics

Two additional runs are explicitly exploratory and separate from the fixed
1,077-item grid:

- `results/workspace-report-tokens/`: all 14 report categories, testing
  punctuation consumption and greedy-word identification before choosing a
  source direction. Missing whole-token source concepts remain unresolved.
- `results/workspace-strength-audit/`: all 192 flexible swaps under four
  precise interpretations of increased intervention strength. The frozen
  primary definition is replayed as an exact regression check.

The figure and regression finalizer is `finalize_workspace_audit.py`. It also
reports every loop-subset comparison, including a combination that performs
better than the paper's preferred loops 2–4. That result must not be omitted
when discussing whether *every* remaining loop is necessary.
