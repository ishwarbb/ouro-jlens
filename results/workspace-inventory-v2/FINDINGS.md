# Completed follow-up: what improved and what remains different

Implemented validated prompt/token/scoring contracts, the paper's positive
sparse-pursuit construction, the complete 12-setting coefficient grid, and
frozen clean-state ten-direction ablation. The run used only GPUs 0–3 and
preserved earlier results. All 36 relevant unit tests pass.

Independent verification reconstructed 75,608 pursuit cells, 244,088 coefficient
snapshots and 29,172 ablation bases. All 64 baseline logits match native model
outputs exactly. The largest reconstruction-fraction discrepancy was 6.6e-7;
orthogonality and span errors were below 3.1e-6.

## Results against the paper

| Measurement | This run | Paper |
|---|---:|---:|
| Baseline correct flexible prompts | 41/64 | 42/64 |
| Answers retained: pursuit removal, loops 2–4 | 0/41 | 0/42 |
| Answers retained: pursuit removal, loop 2 only | 12/41 | 6/42 |
| Answers retained: pursuit removal, loop 4 only | 0/41 | 0/42 |
| Answers retained: random span, primary seed | 28/41 | 37/42 |
| Random span, two additional seeds | 33/41 each | Not reported |
| Verbal report top-5, fixed `Answer:` wrapper | 110/131 (84.0%) | 106/133 (79.7%) |

Paper reference: [Wang & Reid, Appendix E and causal evaluation](https://arxiv.org/html/2609.01924v1).
The full/last-loop pursuit ablation effect agrees qualitatively. Early-loop
and random-control results differ; the random controls are more disruptive
here. They sample the declared 47,506-word atom dictionary, whose exact
relationship to the authors' filter is unresolved. Nothing was selected to
make those controls match the paper.

The report score is unchanged from the previous fixed `Answer:` audit. All 257
available baseline/transfer score records and five missing-direction exclusions
replay exactly. Whole-token top-1 and completed-word accuracy are both 101/131.
The wrapper was selected using earlier results, so this is improved measurement
reliability, not independent confirmation or a new model improvement.

## What prevents a complete replication claim

- Only 9/14 report categories produce a category-valid first-token source.
  Five target trials also lack a whole-token intervention direction. All remain
  in the 131-attempt denominator.
- Inventory covers two quoted prompts and 188 fitted source sites, not all
  eight demonstrations and all 192 sites. The animal and sport traces explain
  19.1% and 19.5% of state energy at 50 atoms; those are not the paper's
  eight-prompt 21.5% aggregate.
- The sport prompt spends its entire 128-token budget inside a thought block,
  producing no final answer. It cannot establish the paper's hidden-sport
  final-answer result. The animal answer explicitly names trees, so its concept
  classes must differ from the paper's transcript-based classes.
- Exact author wrappers, token filtering and numerical settings remain
  unresolved. Pursuit-based capacity and independent confirmation are still
  separate work.

The first development run exposed an overly broad stop rule that stopped at
`<think>`. The corrected run stops only at EOS or chat turn-end, retains thought
tokens, and excludes them from completed-answer credit. The interrupted run is
preserved, and a regression test covers the fix.

See [full results](REPORT.md), [independent validation](VALIDATION.md),
[all-cell inventory and ablation figure](inventory-and-ablation.png), and
[transcript-correct concept profiles](animal-concept-profiles.png).
