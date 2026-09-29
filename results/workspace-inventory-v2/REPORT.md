# Validated scoring, sparse inventory and subspace ablation

Development follow-up to the completed paper audit. The report wrapper was chosen using previously observed outcomes; this is not independent confirmation. Earlier results are preserved.

## Verbal report

Fixed native-chat `Answer:` wrapper; category-valid first-token sources: 9/14. Invalid sources remain in the all-attempt denominator. All generation extends exact token IDs; no truncation or text re-tokenization.

| Metric | Transfer | Random delta | Baseline |
|---|---:|---:|---:|
| top1 | 101/131 (77.1%) | 3/131 (2.3%) | 0/131 (0.0%) |
| top5 | 110/131 (84.0%) | 13/131 (9.9%) | 20/131 (15.3%) |
| completed_word | 101/131 (77.1%) | 5/131 (3.8%) | 2/131 (1.5%) |

Teacher-forced sequence log probabilities are stored separately and never treated as answer accuracy. Continuation edits act only at the original prompt positions. A completed first word requires a boundary or termination, so a split prefix is never credited as a full word.

## Ten-direction subspace ablation

Baseline correct: 41/64 (64.1%). Every condition uses that same baseline-correct set; all 64 prompts were executed. All 64 sham passes reproduce baseline logits exactly.

| Condition | Answers kept here | Paper Ouro answers kept |
|---|---:|---:|
| pursuit_all | 0/41 (0.0%) | 0/42 |
| pursuit_loop2 | 12/41 (29.3%) | 6/42 |
| pursuit_loop4 | 0/41 (0.0%) | 0/42 |
| random_42 | 28/41 (68.3%) | 37/42 |
| random_43 | 33/41 (80.5%) | — |
| random_44 | 33/41 (80.5%) | — |
| argument_all | 31/41 (75.6%) | 30/42 |
| argument_loop2 | 40/41 (97.6%) | 42/42 |
| argument_loop4 | 32/41 (78.0%) | 32/42 |
| answer_all | 11/41 (26.8%) | 8/42 |
| unrelated_all | 40/41 (97.6%) | 42/42 |

Paper reference: [Appendix E, Table 11 and Appendix B, Table 8](https://arxiv.org/html/2609.01924v1). These comparisons use the released flexible-generalization prompts and our frozen physical-layer indexing; unresolved author-code details prevent claiming exact procedural identity.

The pursuit basis is selected from clean states, refitted with nonnegative coefficients, and frozen before the intervention pass. Random controls use ten vocabulary atoms shared over positions at each layer, with three fixed seeds. Paired cluster-bootstrap intervals and all-attempt results are in `summary.json`. Intervals are exploratory, not corrected for the full study.

## Sparse inventory

Two exact paper-quoted prompts (animal and hidden sport), all prefill and up to 128 greedy continuation tokens, all 188 fitted source sites. Dictionary: 47,506 allowed word tokens. All 12 (k, threshold) combinations are in `occupancy-grid.csv`; per-cell selections, coefficient snapshots and residual fractions are in the layer artifacts.

| Prompt | Cells | Variance explained at k=25 | At k=50 | Primary support/top8 Jaccard |
|---|---:|---:|---:|---:|
| demo-animal | 13,160 | 15.4% | 19.1% | 0.034 |
| demo-sport | 33,276 | 15.9% | 19.5% | 0.039 |

Largest normalized NNLS KKT residual: 2e-06; maximum refit iterations: 1950; largest basis orthogonality error: 2.5e-06. Solver failures abort a run. Rank-deficient spans are recorded without inventing extra directions.

The paper’s 21.5% explained-variance result averages eight demonstrations. Our two-prompt values have a different population and are not an exact replication of that aggregate. Top8 uses native unembedding including normalization; pursuit uses the stated linear J-transpose atom dictionary.

## Remaining gaps

Exact author prompt wrappers, word filtering, solver settings and all eight demonstration transcripts remain unresolved. The word filter and explicit plural merges are disclosed here. Capacity under pursuit and a larger frozen independent confirmation set remain separate work. These results characterize the fitted instrument and causal response, not consciousness or a complete inventory of all stored concepts.
