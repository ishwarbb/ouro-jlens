# Run 10 plan for confirming Ouro loop interventions

Plan dated October 13, 2026. Run 10 will test the specific observation that editing loops 2+3 can steer Ouro more effectively than editing loops 2+3+4. It uses new prompts and a fixed contrast, separate from [Run 9's cross-model audit](RUN9_PLAN.md). Either study can run first; Run 10 does not depend on finding deceptive behavior or on the completion of earlier active runs.

The machine-readable design is [plans/run10/protocol.json](plans/run10/protocol.json). [The server handoff](plans/README.md) describes restoration and implementation. This plan is ready for implementation; the new runner, prompt records and execution manifest still need to be produced and validated on the destination server. No Run 10 inference has started.

## Question and prior evidence

The exploratory [workspace audit](results/workspace-audit/CONCLUSIONS.md) found 84/192 steering successes for loops 2+3 and 73/192 for loops 2+3+4. Its semantic-cluster interval for the difference was +1.6 to +10.9 percentage points. The prompts had already been used during development, and the loop contrast emerged from a wider sweep. Run 10 tests the direction of this one contrast on fresh examples; the old interval is not a power calculation.

The primary hypothesis is that the mean paired steering-success difference, loops 2+3 minus loops 2+3+4, is positive under the declared operator and prompt distribution. A positive result would qualify claims about the necessity of editing every remaining loop. A negative or inconclusive result would limit the generality of the earlier observation. Neither outcome establishes a universal property of recurrence.

## Checkpoint and fitted lenses

Use `ByteDance/Ouro-2.6B` revision `1ed04250da1a9936042725d302e81c8fa2ab5abd`, the existing 48-layer, four-loop configuration, and the 1,000-prompt lens family from release `v0.1.0`. The expected model weight SHA-256 is `2fdf9805a3510b7dc82959a2db4872ef767a5bdc55520c6a1b54fc3e463c1bba`. Restore the four fitted target tensors using [artifacts.json](artifacts.json) and its exact checksums. Use own-loop targets 47, 95, 143 and 191, with zero-based virtual-site numbering.

Keep the model weights, loop count, lens family, numerical precision and intervention algebra fixed. Match the audit's BF16 model forward with FP32 fitted matrices, normalized directions and update arithmetic, casting edited states back to the model dtype. Use native-versus-adapter checks and cache-free prompt forwards. Resolve and record the actual PyTorch, Transformers, attention backend and hardware versions on the new host. Any necessary environment change requires preflight numerical validation; it does not authorize tuning against the new main outcomes. Do not refit lenses on Run 10 prompts.

## Fresh prompt population

Materialize 64 prompts across eight semantic function templates, eight prompts per template, each with three destination concepts. Each record specifies the exact rendered prompt, source concept, three destination concepts, baseline answer, and three expected counterfactual answers. For a function `f`, a source-to-destination concept edit is scored against `f(destination)`. For example, a France-to-China edit in a capital question targets the answer Beijing, not the word China.

Use new prompt strings and source/destination combinations relative to the archived workspace studies and Run 9. Preserve a similarity inventory against prior prompts and classify template reuse honestly: new instances of a known function do not demonstrate unseen-function generalization. Choose all 64 prompts before any model outputs, from a larger deterministic candidate pool using only duplicate checks, unambiguous factual mappings and tokenizer eligibility. Freeze the candidate pool, selection order and rejected-candidate reasons. No replacement based on baseline accuracy or intervention success is allowed.

For the primary metric, prefer source and destination concepts with distinct whole-token directions and expected answers with whole-token forms. Select direction tokens using the existing leading-space preference in `Vocabulary.direction_token`. Save every chosen token ID and all accepted answer-token variants. If an unexpected direction or scoring failure occurs after freezing, retain its trial in the attempted denominator and flag measurement eligibility separately. Never silently replace that item.

Use the existing raw flexible-task prompt convention, explicit BOS handling and a 2,048-token input ceiling. Save the exact input IDs. Do not add the report experiment's `Answer:` wrapper, a new chat wrapper or a reasoning instruction. Reject overlong candidates before freezing rather than truncating the input.

## Intervention definition

Normalize the pulled-back source and destination directions at each site:

`a = normalize(J_site.T @ W[source_token])`

`b = normalize(J_site.T @ W[destination_token])`

Apply the previously audited projection transfer at strength 1:

`delta(h) = (h · a) * (b - a)`

`h_edited = h + delta(h)`

Use physical layers 24 through 40 inclusive, indexed from zero, at all original prompt positions. Loops are numbered 1 through 4. The edited virtual sites are 72–88 for loop 2, 120–136 for loop 3, and 168–184 for loop 4. They are block outputs before loop-end normalization, as in the validated adapter. No edits apply to loop 1 in any active condition.

Use [workspace_audit.py](workspace_audit.py) and [workspace_interventions.py](workspace_interventions.py) as implementation references, with a new output root. Do not run the old audit initializer as a substitute for Run 10: it selects old prompts and a different condition grid. Also, the generic `random_matched` operator in `workspace_interventions.py` uses the coordinate-exchange norm. Run 10's random controls require the **projection-transfer** norm; the audit's `random_transfer` implementation is the relevant reference.

## Nine fixed conditions

| Condition | Edit |
|---|---|
| `baseline` | No intervention hooks |
| `loops_2` | Projection transfer in loop 2 |
| `loops_3` | Projection transfer in loop 3 |
| `loops_4` | Projection transfer in loop 4 |
| `loops_23` | Projection transfer in loops 2 and 3 |
| `loops_234` | Projection transfer in loops 2, 3 and 4 |
| `sham` | Same selected-site hook machinery, zero update |
| `loops_23_random_4` | Projection transfer in loops 2 and 3; a norm-matched random update in loop 4 |
| `random_234` | Norm-matched random updates in loops 2, 3 and 4 |

At each random-control hook and token, calculate the norm of `delta(h)` on that arm's incoming state, then apply that norm along a frozen random unit vector. Draw vectors from an isotropic Gaussian and normalize them. Derive seeds from the fixed seed 2026101310 plus record ID, destination and virtual site; share the loop-4 vectors across the two random conditions. Save the derived seeds, vectors or reconstructible generator specification, and actual displacement norms.

The random directions are independent of observed outcomes. The controls match the local hypothetical transfer norm, not an entire stored trajectory's perturbation energy. Record aggregate norms so the remaining dose difference is inspectable. The added random loop-4 condition helps determine whether an extra edit is generally disruptive or whether the effect depends on its semantic direction.

## Outcomes and inference

The **primary endpoint** preserves the previous audit's measurement: the final-loop next-token argmax belongs to the predeclared whole-word token set for the expected counterfactual answer. The primary contrast is `loops_23 - loops_234`. This avoids changing the endpoint while claiming to confirm the old 84/192 versus 73/192 result.

The **secondary endpoint** is the first completed answer under greedy continuation, at most 32 new tokens, allowing split-token words. Freeze a deterministic answer extractor and accepted strings before generation; retain raw completions. Completed answers must end at a frozen delimiter or natural EOS/turn end. A capped or incomplete answer is a failure for this endpoint, even if its prefix looks correct. Do not require that every correct short answer use the entire token budget.

During secondary continuation, reapply the same edits only at the original prompt positions on each cache-free forward; never expand the edit mask to generated tokens. This defines a prompt-state intervention consistent with the first-step endpoint. Keep generation settings identical across conditions. Normalize whole-word aliases consistently and report disagreements between first-token and completed-answer scoring.

For each of the 64 source prompts, average the three destination-level primary differences. Estimate the population mean with 50,000 bootstrap draws stratified by the eight templates, resampling the eight prompt records within each template. Use seed 2026101310 and percentile 95% intervals. Run a two-sided paired sign-flip test of the 64 prompt differences with 100,000 Monte Carlo draws and the plus-one correction; state its symmetry/exchangeability assumption. The single primary test uses alpha 0.05. Call the specified positive hypothesis supported only if the mean is positive, the lower interval bound exceeds zero, and the test rejects.

All other loop contrasts, random-control comparisons, template subsets and completed-answer analyses are secondary and descriptive. Keep the six Run 9 multiplicity slots separate: these studies answer distinct questions. If a later report combines their tests into one confirmatory claim, declare and account for that broader family.

Show baseline knowledge, direction eligibility, missingness and all attempted denominators. Report the baseline-correct subset only as a prespecified sensitivity analysis; do not drop baseline-incorrect items from the primary analysis. No infrastructure or instrumentation failure is scored as a scientific negative: retry infrastructure failures at the identical frozen trial ID/configuration, preserve attempts, and stop if reproducibility cannot be established. Wrong answers and capped model outputs retain their declared scores. Never reroll an unfavorable valid output.

## Preflight and verification

Before main generation, use disjoint engineering prompts to verify all 192 virtual invocations, the selected sites, own-loop routing, native/adapter logits, deterministic repeated forwards, and the exact transfer algebra. Verify baseline and sham agreement at the declared numerical tolerance. Check that random-update norms equal the hypothetical transfer norms at each hook and that hooks are removed after every arm, including failures.

Instrument original prompt positions and confirm the mask remains unchanged during secondary continuation. Test expected-answer mappings and split-token extraction with explicit examples, including the France-to-China capital example. Test resumptions and duplicate-ID rejection. Bind the full prompt manifest, model and lens checksums, runtime versions, scorer, seeds and condition definitions in a new immutable execution manifest before opening main results.

The existing CPU adapter and operator tests are reusable. New tests must cover the mixed loop-2+3/loop-4-random arm and the prompt-only continuation mask; passing only the historical tests would not validate those additions.

## Budget and saved artifacts

The design has `64 prompts × 3 destinations × 9 conditions = 1,728` logical trial settings. Each has one primary prompt evaluation and a secondary continuation of at most 32 generated tokens, at most 55,296 generated tokens in total. An implementation can reuse the primary prompt evaluation as the continuation's first step. Baselines may be cached across destinations only when the input and configuration hashes are identical; logical denominators still retain every trial.

For predictable scheduling, budget up to eight full forward passes per runtime-check prompt across eight disjoint engineering prompts: at most 64 preflight forwards, separate from the main count. Numerical unit tests and artifact verification are separate CPU work. Measure per-forward latency and memory on the destination server before setting worker concurrency. Run 10 needs the original Ouro checkpoint and existing lenses, not the larger Run 9 model panel.

Store `execution.json`, `plan/records.json`, `plan/conditions.json`, `preflight.json`, append-only event records, per-trial next-token scores and completions, intervention norms, exact prompt/token IDs, and a summary containing paired record differences. Save final logits or the lossless quantities needed to verify exact argmax/set membership; do not rely only on a top-10 screenshot. Use `results/workspace-run10-{run_id}/` as a new output directory, resolved from the checkout or an explicit output-root override.

Write an account of the primary result even if it fails. Do not promote a better-performing exploratory loop subset or change the intervention strength after viewing the main data.
