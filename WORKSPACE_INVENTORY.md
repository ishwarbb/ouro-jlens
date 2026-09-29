# Validated scoring and sparse workspace inventory

This follow-up implements the next two improvements identified by the completed
[paper audit](WORKSPACE_AUDIT.md): explicit prompt/token/scoring contracts, and
the sparse inventory and frozen ten-direction ablation in
[Appendix E of the looped-transformer paper](https://arxiv.org/html/2609.01924v1).
It reuses the fitted 1,000-prompt family and preserves all earlier runs.

The frozen run is `results/workspace-inventory-v2/`. Its `manifest.json` pins
code, data, tokenizer/model configuration, and fit manifests. `protocol.json`
specifies every choice before this run. This is a development follow-up on
previously seen prompts, not an independent confirmation study.

The run is complete: 80 capture tasks, 188 inventory layers, and all 64
ablation trials. All 36 relevant tests pass. Independent reconstruction checked
75,608 pursuit cells, 244,088 coefficient snapshots, and 29,172 frozen bases;
all 64 baseline logits match the native model exactly. See the
[findings](results/workspace-inventory-v2/FINDINGS.md),
[full report](results/workspace-inventory-v2/REPORT.md), and
[independent validation](results/workspace-inventory-v2/VALIDATION.md).

## Prompt and score contracts

`workspace_validated.py` stores exact prompt text, wrapper version, token IDs,
and fingerprints. Encoding adds at most one BOS, rejects overlong inputs, and
never silently truncates. Only EOS and chat turn-end tokens stop generation; thought markers remain in the trace and are excluded from completed-answer credit. Greedy and teacher-forced continuation append actual
IDs to the original input; decoded text is never re-tokenized into model input.

Three distinct measurements remain separate:

- Whole-token top-1 and top-5 over all exact case-insensitive token spellings.
- A completed first word from greedy generation, requiring a boundary or stop.
  A partial multi-token prefix does not count as a complete word.
- Full teacher-forced continuation log probability for `encode(' ' + answer)`.
  This is a likelihood diagnostic, never reported as answer accuracy.

Verbal report uses one fixed native-chat `Answer:` prefill across all 14
released categories. This was selected using the preceding audit; it is not a
new blind discovery. The source direction remains the actual greedy next token.
Sources that are punctuation, partial words, or outside the category are
flagged and retained in the main denominator. A valid-source subset is
descriptive. Continuation interventions act at the original prefill positions.

## Sparse inventory

`workspace_inventory.py` builds the normalized dictionary `a = J.T @ u /
norm(J.T @ u)`, using raw unembedding rows and FP32 arithmetic with TF32
disabled. It does not insert RMSNorm into the dictionary. The native top-8
readout does use the model's normal unembedding, including normalization.

The allowed dictionary contains non-special tokens whose stripped text is
alphabetic, allowing internal apostrophes and hyphens. This is an explicit
interpretation of the paper's unspecified non-word filter; exact author-code
identity is not established. The complete token list is saved.

Positive greedy pursuit selects distinct atoms from the current residual and
refits nonnegative coefficients on the entire selected support. Accelerated
projected gradient stops at a normalized projected KKT residual of at most
`2e-6`; unconverged fits abort the run. Objective increases beyond floating-point
tolerance also abort. Snapshots at 8, 16, 25 and 50 atoms retain coefficients,
selection IDs, state norms and squared residual fractions. Thresholds 0, 0.01
and 0.05 are offline slices of the same artifact. At zero threshold, selected
atoms with coefficients refitted to zero count under the paper's `>=` rule.

The readout run covers the animal and hidden-sport prompts quoted verbatim in
the paper, all prefill and up to 128 generated tokens, and all 188 fitted
within-loop source sites. Four loop-end targets are not source sites in the
fitted family. This is not the paper's complete eight-prompt/whole-depth dump,
and its mean explained variance must not be equated with the paper's eight-
prompt aggregate. Native J-lens and raw top-8 results use the same saved cells.

## Frozen subspace ablation

All 64 released flexible-generalization prompts are executed. Every ablation
uses the same subset that was correct before editing; all-attempt counts are
also retained. No condition selects its own denominator.

A clean pass supplies the first ten pursuit atoms at every intervention cell.
Rank-aware modified QR with reorthogonalization produces a frozen basis. A
dependent column becomes zero, avoiding removal of arbitrary extra directions.
The second forward applies `h <- h - Q @ Q.T @ h` across the physical band
24–40 (zero based), at all prefill positions including BOS.

Conditions cover all loops 2–4, loop 2 alone, loop 4 alone, and three seeded
random-vocabulary controls. Each random span uses ten atoms fixed per layer,
shared over positions and trials. Single argument, answer, and unrelated-token
ablations use the same interface and score. An all-zero sham basis must preserve
the complete next-token logits exactly. Baselines must also replay exactly
between the capture and ablation stages.

## Running and reviewing

```bash
.venv/bin/python -m unittest test_workspace_inventory test_workspace_interventions \
  test_workspace_audit test_workspace_analysis test_ouro_family test_ouro_full_fit
.venv/bin/python run_workspace_inventory.py init --output results/new-inventory-run
.venv/bin/python run_workspace_inventory.py run --output results/new-inventory-run
```

The supervisor restricts workers to GPUs 0–3 and uses a dynamically claimed
queue. Initialization rejects an existing manifest. A stopped run can resume
with `run` only if all frozen inputs still match; completed records are checked
and retained. Scientific-code changes require a new run directory.

The active tmux session is `ouro-workspace`, windows `inventory-run` and
`inventory-status`. The output directory contains:

- `status.json`, worker logs, and per-worker progress.
- `capture/`: exact prompts, scores, generated IDs, and clean residuals.
- `inventory/`: selections, coefficient snapshots, residual fractions,
  native top-8 comparisons, frozen bases, random atoms, and integrity hashes.
- `ablation/`: all 64 trials and every condition, including baseline failures.
- `summary.json`, `REPORT.md`, the full `occupancy-grid.csv`, concept counts,
  and standalone PNG/PDF figures after completion.
- `VALIDATION.json` and `VALIDATION.md`: independent reconstruction, native
  baseline replay, historical score replay, and explicit comparability limits.
- `animal-concept-profiles.csv` and PNG/PDF: descriptive profiles whose named
  versus unspoken classes come from the actual saved transcript. The model
  says "trees", so tree belongs in the explicitly named class for this run.

The numerical test suite checks NNLS against an independent exhaustive active-
set solution, implicit versus explicit dictionary scores, exact sparse
reconstruction, scalar versus batched pursuit, rank-deficient projectors,
frozen-position hooks, matched random intervention norms, and split-token
generation. The actual checkpoint smoke test is in
`results/workspace-inventory-smoke.json`; it checks native loop logits,
determinism, stopped-prefix equivalence and a real 50-step pursuit.

After the main run, reproduce the independent verification and descriptive
concept profiles with:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python verify_workspace_inventory.py \
  --output results/workspace-inventory-v2
.venv/bin/python profile_workspace_inventory.py \
  --output results/workspace-inventory-v2
```

Remaining work for a paper-wide replication includes the other six demo
transcripts, exact endpoint/filter/wrapper conventions, pursuit-based capacity,
and an independent frozen confirmation set. These do not prevent assessing the
specific new implementations, but they prevent claiming complete procedural
identity with the paper.

The interrupted `workspace-inventory-v1` artifacts preserve a development bug: all special tokens were initially treated as terminal, so the sport prompt stopped at `<think>`. Version 2 fixes this, adds a regression test, and reruns every stage under a new frozen manifest.
