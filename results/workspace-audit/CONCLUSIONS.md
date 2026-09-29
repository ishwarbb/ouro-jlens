# Completed Ouro workspace audit

The earlier “not established” summary mixed protocol differences, metric choices
and statistical uncertainty with replication outcomes. It was not an adequate
comparison with the paper. The new run supports several effects that looked
inconclusive before, while leaving specific discrepancies and a counterexample
to a stronger recurrence claim.

Completed on GPUs 0–3: the fixed 1,077-item audit, a 14-category report-token
diagnostic, and a 64-prompt strength diagnostic covering all 192 flexible swaps.
Every worker exited successfully. The fixed run covers 29 intervention
conditions on every primary causal item, all declared prompt variants, all
released affected prompts, and both released selectivity tasks. This is an
exhaustive run of the declared finite grid, not every possible implementation
or a Cartesian product of all choices.

## Comparison

Paper values: [Wang and Reid, v1, Tables 5–7](https://arxiv.org/html/2609.01924v1#A2).
Local denominators are retained; unknown author implementation details prevent
a claim of exact end-to-end reproduction.

| Question | Paper | This audit | Assessment |
|---|---|---|---|
| Flexible steering, loops 2–4 | 73/192 | 73/192 | Same total; strong paired effect against baseline and all three random controls |
| Loop 2 only / loop 4 only | 40/192 / 48/192 | 40/192 / 48/192 | Same totals; loops 2–4 outperform each single-loop condition |
| Two-hop steering | 26/90 | 32/90 | Qualitative effect reproduced; counts differ |
| Verbal-report top-5 | 106/133, about 80% | 68/136, 50%, with `One word:`; 110/131, 84%, with the fixed `Answer:` variant | Strong prompt dependence; no exact numerical replication claim |
| Language selectivity | +25 percentage points | +25 percentage points | Point estimate reproduced; eight passages leave the cluster interval at [0, 50] points |
| Line-count readout, top-1 | Direct 100%, unasked 9% | Direct 9%, unasked 9% | Not reproduced under our wrapper and whole-token readout |
| Increased strength | 14/192 | Whole-update ×2: 71/192; destination-only ×2: 13/192 | Algebra matters; the latter is an exploratory close match, not verified author code |

For latent readout, own-loop lenses recover prompt-absent intermediate concepts
on 82/90 loop-2 and 84/90 loop-3 items. Final-target lenses recover 22/90 and
23/90 at those same physical sites. The matched earlier-loop contrast supports
the lens-family readout effect. The old requirement that own-loop lenses beat
final-target lenses after taking a union including loop 4 was misleading:
those lenses are identical in loop 4. Superiority over raw-logit readout is
smaller and not robust to the conservative interval check in loops 2–3.

## What the exhaustive loop sweep changes

Loops 2–4 beat loop 2, loop 3 and loop 4 individually on flexible steering.
However, **loops 2+3 alone score 84/192**, versus 73/192 for loops 2+3+4.
The exploratory paired semantic-cluster interval for that difference is
+1.6 to +10.9 percentage points. The full subset table is retained in
[VALIDATION.md](VALIDATION.md).

Thus the narrower observation “multiple-loop edits can outperform single-loop
edits” reproduces. These data do **not** justify a universal requirement to
edit every remaining loop. This sweep is exploratory and its many comparisons
do not constitute independently controlled confirmatory tests.

## Why the previous run underperformed

- Its primary edit exchanged two coordinates. Source-projection transfer gives
  very different behavior. The original study explicitly specifies the former;
  the looped paper describes clamp-transfer without sufficient algebra to
  identify its exact implementation. Both operations are valid, so this is
  an unresolved specification difference, not a proven arithmetic bug.
- Verbal report was judged primarily at top-1 although the paper's headline is
  top-5. The `One word:` wrapper also produced punctuation or non-category
  answers in 10/14 categories, making many purported concept swaps ill-posed.
- The old evaluator skipped otherwise possible swaps when the original answer
  lacked a whole-token form. The new run keeps all attempts and separates
  direction eligibility, whole-word success and prefix-only diagnostics.
- A nonpositive confidence bound was treated too readily as non-reproduction.
  Language selectivity already had the paper's point estimate.
- The latent-readout criterion mixed earlier loops with an identical-lens
  comparison at the final loop and imposed additional superiority requirements.

The report-token follow-up consumed formatting tokens and separately decoded
the greedy concept before selecting its direction. It did not uniformly solve
the problem: many outputs were split-token or outside the released candidate
list. Those cases are retained. This is evidence of wrapper/measurement
fragility, not a reason to discard difficult categories.

At top-5, the added line-count test reads a general number token in 11/11 direct
conditions and 2/11 unasked conditions. At top-1 it reads one in 1/11 under
either condition. General-number readability is not correctness of the count;
whole-token exact-count scoring has tokenizer coverage limitations.

## Verification and remaining uncertainty

- 26 unit tests passed across the audit, intervention, analysis, adapter,
  family and fit checks; separate strength-algebra checks also passed.
- Native logits at all four loop ends and deterministic repeated forwards
  matched exactly on each worker.
- All 476 fitted matrices exactly match their 3,808 constituent saved row
  blocks. Float64 half/full convergence checks are in `fit-audit.json`.
- 1,170 overlapping archived condition records reproduce exactly at all four
  loop ends. All 192 repeated doubled-update trials also match exactly.
- Code/data hashes, source IDs, prompts, exclusions and all outcomes are saved.
  The original run was preserved.

The exact author fitting rows, prompt wrappers, token-selection conventions,
causal code and raw records were not located. We matched the stated fit
geometry and verified our estimator/adapter, but did not refit against an
unavailable corpus. Matching several totals does not establish implementation
identity. The original positive ablation, introspection and modulation tests
also differed numerically or in coverage; their audit is in
[WORKSPACE_AUDIT.md](../../WORKSPACE_AUDIT.md).

This is not the full eleven-family paper: capacity, ignition, dual-task
interference, summoning, concept inventory and other checkpoints remain outside
this rerun's scope.

## Artifacts

- [Main report and full fixed grid](REPORT.md)
- [Methods audit](../../WORKSPACE_AUDIT.md)
- [Regression checks, loop subsets and report subsets](VALIDATION.md)
- [Report-token diagnostic](../workspace-report-tokens/REPORT.md)
- [Strength-definition diagnostic](../workspace-strength-audit/REPORT.md)
- [Comparison figure](paper-comparison.png) and [PDF](paper-comparison.pdf)
- [Sensitivity/readout figure](sensitivity-and-readout.png)
- [Machine-readable results](summary.json) and [CSV](rates.csv)
