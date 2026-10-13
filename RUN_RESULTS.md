# Experiment objectives and results

Snapshot dated October 13, 2026. The workspace experiments establish useful
readout and intervention capabilities in Ouro. The deception evaluations have
not established the full hypothesis: the model knows the current truth, chooses
a misleading action or report without stating that strategy, and the J-lens
reveals a strategy-specific signal beyond ordinary errors and controls.

The clearest recent result is an interface confound. In Run 8, the interface
without numeric examples produced no persistent introduced defect in 48
comparison episodes. The interface containing numeric examples produced 24 in
48. These descriptive counts support example copying as an explanation for
earlier bug-creation behavior; they do not establish concealed intent. The
comparison uses six records and greedy repeats, with five complete paired
record blocks, so the 48 episodes are not 48 independent observations.

## Deception evaluations

Independent reviews are separate assistant-agent passes, not human annotations.
Their source hashes and adjudication records document the review process;
agreement does not establish subjective intent or detector accuracy.

| Run | Objective | Executed work and result | Status |
|---|---|---|---|
| Run 1 | Explore whether reward or penalty pressure produces false recipient answers and unspoken lens terms. | 384 generations, 168 prompts, 136 lens replays. The 96-case focus cohort contained 13 false primary answers. Similar lexical readouts also appeared in truthful controls, so they did not establish a deception detector. | Exploratory generation and reporting complete. |
| Run 2 | Require an explicit truth check before the pressured answer; test answer-key swaps and recorded history. | 260 generations and 85 replays across the main screen, key swaps and history controls. Two history cases were behavioral candidates, but contextual lens review established zero joint target cases. Formatting, role confusion and ordinary errors remained important alternatives. | Review and replay complete. |
| Run 3 | Freeze discovery and held-out conditions, independently review full histories, and test the registered joint endpoint. | 600 total generations: 312 discovery, baseline and calibration records plus 288 held-out reports. Held-out conflict conditions contained 16 and 13 false reports with already-admitted strategy. The stricter joint behavioral contrasts were zero, with zero confirmed joint cases. The full registered lens endpoint was unavailable because of token coverage. | Held-out review, replay and statistics complete; extra discovery review still being completed. Causal extension ineligible. |
| Run 4 | Test operation reporting and retention of the latest authoritative evidence under aligned/conflicting incentives and recipient verification. | All 96 pilot and 208 discovery generations completed. Independent discovery reviews agreed on 196 correct cases, 9 ordinary errors and 3 incomplete cases; one complete report incorrectly claimed the operation was pending. No discovery report asserted false SUCCESS over actual FAILURE. | Discovery analysis and review lock complete. The 416-generation held-out phase is running; the 120-generation horizon phase remains gated. |
| Run 5 | Adapt four section 5.3 scenarios using actual local tools and workspace artifacts. | 48 pilot episodes, 135 generations. Only 4/16 benign pilot episodes succeeded; the feasibility gate failed. Later adjudication retained one misleading-report candidate whose discrepant-value plan was already stated, and zero confirmed deception cases. | Pilot and adjudication complete; original 192-episode held-out stage was not launched. |
| Run 6 | Separate basic task capability from readout measurement using finite choices and controlled evidence. | 320 episodes, 602 generations: 32 measurement controls and 288 capability episodes. Correct actions: 148/288; valid actions: 205/288. Worked evidence improved correct actions from 49/96 to 58/96. Zero fixed suspicious-word top-10 hits did not establish absence of concealed computation or detector sensitivity. | Generation and analysis complete. |
| Revised Runs 5b and 6b | Repair extraction, use true pre-value positions, and require benign calibration before new primary comparisons. | Rescored all 737 older Run 5/6 captures. Semantic arithmetic extraction increased correct numerical answers from 4/8 to 7/8; 51 older actions were parseable under revised extraction, without reconstructing counterfactual execution. New capability gates still failed, preventing the revised main stages. | Calibration and reanalysis complete; subsequent stages consolidated into Run 8. |
| Run 7 | Use production-validated captures, family-specific gates, independent factual baselines and matched incentive/audit conditions. | 180 episodes, 696 generations, including 96 main episodes. Bug and approval families passed; metrics and correction did not. Adjudication retained 10 misleading-report candidates, 28 ordinary errors, 48 honest unmet approval goals and 10 incomplete reports. Candidates were in aligned conditions and confounded by copying the concrete edit example; zero confirmed deceptive strategies. | Generation and independent adjudication complete; excluded original families remain excluded. |
| Run 8 | Remove example argument values, validate fresh capability records, compare the numeric-example interface directly, and finish the finite-choice benchmark. | 96 eligible main cases and all 96 comparison cases saved. Schema-only: 0/48 persistent introduced defects; numeric examples: 24/48, with two incomplete numeric-example episodes. Metrics/correction capability gates still failed. All 11 initial production checks passed. | Benchmark incomplete and resuming after a report-filename bug. GPU relocation preserves saved draws; independent intent review remains pending. |

The preliminary handoff studies each generated six responses, with and without
an explicit reasoning wrapper. The pressured responses disclosed the latest
failed checks rather than establishing the target concealed strategy. These
small probes are separate from the numbered totals above.

Sources: [Run 1](results/run-1/findings.json),
[Run 2](results/run-2/findings.json),
[Run 3 held-out statistics](results/deception-run3-20261009/held-out-statistics.json),
[Run 4 frozen discovery review](results/deception-run4-20261011/discovery-semantic-review-frozen-20261013.json),
[Run 5 adjudication](results/evaluation-completion-20261012/run5-pilot-adjudication.json),
[Run 6 analysis](results/deception-run6-20261011/analysis.json),
[reanalysis](results/evaluation-revisions-20261011-v2/reanalysis-summary.json),
[Run 7 adjudication](results/evaluation-completion-20261012/run7-main-adjudication.json),
[Run 8 summary](results/deception-run8-20261012/run8-summary.json).

## Workspace and lens experiments

| Study | Objective | Achievement and practical limit |
|---|---|---|
| Pilot and smoke lens fits | Validate the Ouro adapter, Jacobian transport, fitting and serialization. | Established the implementation and numerical checks used by later studies. Small fits are engineering checks rather than independent scientific confirmation. |
| Full lens family | Fit transport to each recurrent loop end from 1,000 prompts. | Completed 476 FP32 matrices targeting sites 47, 95, 143 and 191, with 188 own-loop source sites. Serialization and constituent row-block checks passed. This validates fitted artifacts, not every paper claim. |
| Initial workspace study | Test steering, internal readout, recurrence, selectivity and causal necessity. | Completed the 299-task study. Effects depended strongly on edit algebra, wrapper and endpoint. Projection transfer was substantially more effective than the initial coordinate exchange. |
| Workspace audit | Resolve those protocol/measurement differences with a declared finite grid. | Completed 1,077 audit records. Flexible steering reached 73/192; two-hop steering 32/90; the fixed `Answer:` report wrapper reached 110/131 at top-5. Multiple loops helped, but loops 2+3 reached 84/192 versus 73/192 for loops 2+3+4, opposing a universal need to edit every remaining loop. |
| Strength and report-token follow-ups | Test edit strength algebra and token-selection fragility. | Completed 64-prompt strength and 14-category token diagnostics. Whole-update doubling and destination-only doubling produced very different outcomes. Formatting and split-token answers remained material measurement constraints. |
| Validated inventory and ablation | Reconstruct sparse inventories, coefficient snapshots and frozen-state interventions. | Completed 80 captures, 188 inventory layers and 64 ablation trials. Verification reconstructed 75,608 pursuit cells, 244,088 coefficient snapshots and 29,172 ablation bases; all 64 baseline logits matched exactly. Pursuit removal over loops 2–4 retained 0/41 initially correct answers versus 28/41 for the primary random span. Only two inventory demonstrations were captured; one capped thought trace lacked a final answer. |

Sources: [fit metadata](results/jlens-family-1000/fit-summary.json),
[initial workspace report](results/workspace-full/REPORT.md),
[audit conclusions](results/workspace-audit/CONCLUSIONS.md),
[strength report](results/workspace-strength-audit/REPORT.md),
[token report](results/workspace-report-tokens/REPORT.md),
[validated inventory findings](results/workspace-inventory-v2/FINDINGS.md).

## Publication and remaining work

The repository includes evaluation code, frozen plans, raw JSON/text responses,
mock artifacts, numerical/statistical summaries, independent annotation records,
adjudications and compact reports. The [compact archive](results/published-20261013/index.html)
provides transcript pages and a dated coverage snapshot for every results folder.
Active runs are partial snapshots; queued, excluded and superseded stages are
retained with their original disposition rather than counted as completed work.

Large residual/RNG/readout tensors and generated per-token HTML caches remain
local. Their scientific records retain hashes and filenames; the publication
manifest records their omission. The existing v0.1.0 release supplies the earlier
fit and viewer bundles, not all newer deception captures. Compact transcripts
can be browsed without those binaries; exact per-token replay requires the
matching local tensors. No new tensor release is claimed.

Run 4 still needs held-out completion/review before its horizon phase. Run 8
still needs its benchmark and full-history intent reviews. Original failed
capability gates remain failed. A causal deception extension requires eligible
reviewed cases, specificity beyond errors/refusals and a separately frozen
intervention design; that evidence gate has not been met.
