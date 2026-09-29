# Ouro workspace replication audit

Completed the entire fixed grid over all released prompts for the previously inconclusive families. This is a method audit informed by earlier results, not an independent preregistered replication. The original run remains in `../workspace-full/`.

## Comparison with the paper

Reference: [Wang and Reid, v1, Tables 6–7](https://arxiv.org/html/2609.01924v1#A2). The transfer formula is an explicit interpretation of its clamp-transfer description; author code was not located. Denominators and tokenization differences are retained, so these are not guaranteed identical experiments.

| Measurement | Paper | Audit primary, whole-token metric |
|---|---:|---:|
| Flexible steering | 73/192 (38%) | 73/192 (38.0%) |
| Loop 2 only | 40/192 (21%) | 40/192 (20.8%) |
| Loop 3 only | 29% | 57/192 (29.7%) |
| Loop 4 only | 48/192 (25%) | 48/192 (25.0%) |
| Double strength | 14/192 (7%) | 71/192 (37.0%) |
| Two-hop steering | 26/90 (29%) | 32/90 (35.6%) |
| Verbal report top-5 | 106/133 (80%) | 68/136 (50.0%) |
| Verbal report top-1 | 67% | 47/136 (34.6%) |
| Language selectivity difference | +25 pp | +25.0 pp |
| Line-count label readout, direct / unasked | 100% / 9% | 9.1% / 9.1% |

## Method corrections and interpretation

- The original primary used a coordinate-exchange reflection. This audit uses source-projection transfer and retains reflection plus two alternative clamp interpretations as sensitivities.
- Verbal report uses top-5 as the paper headline does, and also retains top-1. Every original-answer token is recorded; punctuation sources and invalid-category answers are visible.
- Interventions no longer depend on the original answer having a complete single-token form. All attempts remain in denominators, with eligibility counts and separately labelled prefix scores.
- The previous language effect already equalled the reported +25 points. A confidence interval including zero is uncertainty, not numerical disagreement.
- Latent readout is examined per loop at matched physical positions. The former all-loops criterion included loop 4, where own-loop and final-target lenses are identical, and demanded superiority not required by the paper.
- Line-count selectivity was absent from the first run and is included here. Exact author prompt wrapping remains unknown.
- The adapter and existing fit use all four loops, distinct virtual sites, native loop-end logits, disabled cache/checkpointing, and the reference Jacobian estimator. Real-checkpoint verification is repeated on each worker.
- The fit shares corpus size, token length, and skip geometry with the stated recipe. Exact author corpus rows and code are unavailable. Software and GPU differ. The existing fitted matrices are reused, not re-estimated.

## Paired evidence

Semantic-cluster bootstrap intervals (10,000 resamples). The wider intervals reuse the earlier seven-family tail adjustment as a sensitivity check; they do not control all comparisons in this expanded grid. Repeated conditions do not create new independent observations.

| Contrast | Difference | 95% cluster interval | Conservative interval | Clusters |
|---|---:|---|---|---:|
| flexible: transfer - baseline | +38.0 pp | [24.5, 51.0] pp | [19.8, 55.2] pp | 16 |
| flexible: transfer - random_42 | +38.0 pp | [24.5, 51.0] pp | [19.8, 55.2] pp | 16 |
| flexible: transfer - random_43 | +38.0 pp | [24.5, 51.0] pp | [19.8, 55.2] pp | 16 |
| flexible: transfer - random_44 | +38.0 pp | [24.5, 51.0] pp | [19.8, 55.2] pp | 16 |
| flexible: transfer - loops_2 | +17.2 pp | [9.4, 25.0] pp | [6.8, 27.6] pp | 16 |
| flexible: transfer - loops_3 | +8.3 pp | [2.6, 14.6] pp | [1.0, 17.2] pp | 16 |
| flexible: transfer - loops_4 | +13.0 pp | [5.7, 21.4] pp | [3.6, 25.0] pp | 16 |
| flexible: transfer - swap | +34.4 pp | [22.4, 46.4] pp | [18.2, 50.5] pp | 16 |
| flexible: transfer - raw | +32.3 pp | [18.2, 46.4] pp | [13.5, 51.2] pp | 16 |
| flexible: transfer - final | +17.7 pp | [9.9, 26.0] pp | [6.8, 29.7] pp | 16 |
| probe: transfer - baseline | +31.1 pp | [20.9, 41.6] pp | [17.0, 46.2] pp | 70 |
| probe: transfer - random_42 | +32.2 pp | [22.5, 42.3] pp | [18.9, 46.9] pp | 70 |
| probe: transfer - random_43 | +34.4 pp | [25.0, 44.2] pp | [21.5, 47.9] pp | 70 |
| probe: transfer - random_44 | +32.2 pp | [22.1, 42.7] pp | [18.5, 47.0] pp | 70 |
| probe: transfer - loops_2 | +20.0 pp | [10.3, 30.0] pp | [6.9, 34.4] pp | 70 |
| probe: transfer - loops_3 | +11.1 pp | [3.7, 19.1] pp | [1.2, 22.4] pp | 70 |
| probe: transfer - loops_4 | +12.2 pp | [4.6, 20.0] pp | [2.3, 23.0] pp | 70 |
| probe: transfer - swap | +26.7 pp | [16.1, 37.6] pp | [12.3, 41.9] pp | 70 |
| probe: transfer - raw | +25.6 pp | [15.7, 35.7] pp | [11.8, 39.6] pp | 70 |
| probe: transfer - final | +12.2 pp | [3.6, 20.4] pp | [1.1, 23.5] pp | 70 |
| report: transfer - baseline | +36.0 pp | [17.3, 56.4] pp | [11.4, 64.6] pp | 14 |
| report: transfer - random_42 | +40.4 pp | [20.1, 62.2] pp | [13.7, 70.8] pp | 14 |
| report: transfer - random_43 | +37.5 pp | [17.4, 59.0] pp | [11.5, 67.2] pp | 14 |
| report: transfer - random_44 | +38.2 pp | [18.7, 59.1] pp | [12.2, 68.3] pp | 14 |
| report: transfer - loops_2 | +33.8 pp | [15.2, 53.4] pp | [10.1, 61.4] pp | 14 |
| report: transfer - loops_3 | +26.5 pp | [11.6, 42.5] pp | [7.9, 49.3] pp | 14 |
| report: transfer - loops_4 | -2.9 pp | [-7.2, 1.4] pp | [-8.8, 2.9] pp | 14 |
| report: transfer - swap | +35.3 pp | [23.9, 47.4] pp | [20.1, 52.6] pp | 14 |
| report: transfer - raw | -5.1 pp | [-24.3, 16.1] pp | [-29.7, 24.7] pp | 14 |
| report: transfer - final | +11.0 pp | [4.4, 19.3] pp | [2.3, 22.3] pp | 14 |
| latent loop1: own - foil | +78.8 pp | [71.3, 85.4] pp | [68.4, 87.6] pp | 70 |
| latent loop1: own - raw | +13.3 pp | [5.9, 21.7] pp | [3.3, 25.3] pp | 70 |
| latent loop1: own - final | +81.1 pp | [71.6, 89.6] pp | [67.3, 92.1] pp | 70 |
| latent loop2: own - foil | +85.8 pp | [78.7, 92.2] pp | [75.8, 94.3] pp | 70 |
| latent loop2: own - raw | +10.0 pp | [1.1, 19.8] pp | [-2.3, 23.3] pp | 70 |
| latent loop2: own - final | +66.7 pp | [54.3, 78.3] pp | [49.5, 82.4] pp | 70 |
| latent loop3: own - foil | +90.0 pp | [83.8, 95.1] pp | [81.0, 96.6] pp | 70 |
| latent loop3: own - raw | +10.0 pp | [1.2, 18.9] pp | [-1.2, 22.1] pp | 70 |
| latent loop3: own - final | +67.8 pp | [55.7, 79.5] pp | [50.8, 83.8] pp | 70 |
| latent loop4: own - foil | +90.0 pp | [83.7, 95.0] pp | [80.7, 96.5] pp | 70 |
| latent loop4: own - raw | +8.9 pp | [1.0, 17.8] pp | [-2.3, 21.2] pp | 70 |
| latent loop4: own - final | +0.0 pp | [0.0, 0.0] pp | [0.0, 0.0] pp | 70 |
| latent loops23: own - foil | +87.2 pp | [80.7, 92.7] pp | [78.1, 94.6] pp | 70 |
| latent loops23: own - raw | +5.6 pp | [-2.2, 13.8] pp | [-4.7, 17.1] pp | 70 |
| latent loops23: own - final | +63.3 pp | [51.1, 75.6] pp | [46.4, 79.5] pp | 70 |
| latent loops234: own - foil | +87.5 pp | [81.3, 92.7] pp | [78.8, 94.5] pp | 70 |
| latent loops234: own - raw | +6.7 pp | [0.0, 14.6] pp | [-2.7, 17.7] pp | 70 |
| latent loops234: own - final | +1.1 pp | [0.0, 3.6] pp | [0.0, 5.2] pp | 70 |
| language: explicit - automatic | +25.0 pp | [0.0, 50.0] pp | [0.0, 75.0] pp | 8 |
| linecount hit1: direct - none | +0.0 pp | [0.0, 0.0] pp | [0.0, 0.0] pp | 11 |
| linecount hit5: direct - none | +81.8 pp | [54.5, 100.0] pp | [45.5, 100.0] pp | 11 |

## Complete intervention grid

All settings were fixed before this rerun. These rows describe sensitivity; no winning setting replaces the primary.

| Condition | Flexible top-1 | Two-hop top-1 | Report top-5 |
|---|---:|---:|---:|
| band_0_16 | 17/192 (8.9%) | 5/90 (5.6%) | 20/136 (14.7%) |
| band_20_40 | 62/192 (32.3%) | 25/90 (27.8%) | 71/136 (52.2%) |
| band_23_39 | 68/192 (35.4%) | 29/90 (32.2%) | 60/136 (44.1%) |
| band_24_36 | 74/192 (38.5%) | 31/90 (34.4%) | 59/136 (43.4%) |
| band_28_40 | 75/192 (39.1%) | 31/90 (34.4%) | 66/136 (48.5%) |
| band_32_40 | 73/192 (38.0%) | 30/90 (33.3%) | 66/136 (48.5%) |
| baseline | 0/192 (0.0%) | 4/90 (4.4%) | 19/136 (14.0%) |
| dual_clamp | 6/192 (3.1%) | 1/90 (1.1%) | 1/136 (0.7%) |
| final | 39/192 (20.3%) | 21/90 (23.3%) | 53/136 (39.0%) |
| last_position | 23/192 (12.0%) | 22/90 (24.4%) | 88/136 (64.7%) |
| loops_1234 | 70/192 (36.5%) | 28/90 (31.1%) | 68/136 (50.0%) |
| loops_2 | 40/192 (20.8%) | 14/90 (15.6%) | 22/136 (16.2%) |
| loops_23 | 84/192 (43.8%) | 25/90 (27.8%) | 34/136 (25.0%) |
| loops_24 | 59/192 (30.7%) | 24/90 (26.7%) | 67/136 (49.3%) |
| loops_3 | 57/192 (29.7%) | 22/90 (24.4%) | 32/136 (23.5%) |
| loops_34 | 63/192 (32.8%) | 31/90 (34.4%) | 67/136 (49.3%) |
| loops_4 | 48/192 (25.0%) | 21/90 (23.3%) | 72/136 (52.9%) |
| random_42 | 0/192 (0.0%) | 3/90 (3.3%) | 13/136 (9.6%) |
| random_43 | 0/192 (0.0%) | 1/90 (1.1%) | 17/136 (12.5%) |
| random_44 | 0/192 (0.0%) | 3/90 (3.3%) | 16/136 (11.8%) |
| raw | 11/192 (5.7%) | 9/90 (10.0%) | 75/136 (55.1%) |
| replace_target | 2/192 (1.0%) | 2/90 (2.2%) | 1/136 (0.7%) |
| strength_0.25 | 70/192 (36.5%) | 32/90 (35.6%) | 61/136 (44.9%) |
| strength_0.5 | 76/192 (39.6%) | 33/90 (36.7%) | 60/136 (44.1%) |
| strength_0.75 | 74/192 (38.5%) | 32/90 (35.6%) | 63/136 (46.3%) |
| strength_1.25 | 73/192 (38.0%) | 31/90 (34.4%) | 71/136 (52.2%) |
| strength_1.5 | 72/192 (37.5%) | 30/90 (33.3%) | 74/136 (54.4%) |
| strength_2 | 71/192 (37.0%) | 29/90 (32.2%) | 63/136 (46.3%) |
| swap | 7/192 (3.6%) | 8/90 (8.9%) | 20/136 (14.7%) |
| transfer | 73/192 (38.0%) | 32/90 (35.6%) | 68/136 (50.0%) |

## Prompt and direction-token sensitivities

| Profile | Flexible top-1 | Two-hop top-1 | Report top-5 |
|---|---:|---:|---:|
| alternate_format | 44/192 (22.9%) | 11/90 (12.2%) | 41/139 (29.5%) |
| answer_prefill | not applicable | not applicable | 110/131 (84.0%) |
| colon_prefill | not applicable | not applicable | 88/134 (65.7%) |
| lower_direction | 60/192 (31.2%) | 28/90 (31.1%) | 66/136 (48.5%) |
| no_bos | 69/192 (35.9%) | 29/90 (32.2%) | 76/135 (56.3%) |
| no_prefill | not applicable | not applicable | 65/133 (48.9%) |
| no_space_direction | 74/192 (38.5%) | 27/90 (30.0%) | 63/136 (46.3%) |
| primary | 73/192 (38.0%) | 32/90 (35.6%) | 68/136 (50.0%) |
| rstrip | 73/192 (38.0%) | 31/90 (34.4%) | 68/136 (50.0%) |

## Eligibility and denominator audit

| Primary family | Attempts | Valid directions | Whole-token target answer |
|---|---:|---:|---:|
| flexible | 192 | 192 | 183 |
| probe | 90 | 90 | 85 |
| report | 136 | 131 | 131 |

Prefix-only outcomes, per-category effects, matched-loop latent readout, baseline accuracy, and loop-end rank trajectories are in `summary.json`. Prefix hits are not complete-word success. Every prompt, token ID, condition and score is retained in `records/`.

## Remaining limits

This audit covers the previously inconclusive families and both released selectivity tasks. It does not execute all eleven suites, re-fit on the unreleased author sample, test other checkpoints, or independently establish the proposed boundary mechanism. Capacity, ignition, dual-task interference, summoning, and the sparse inventory remain outside this run. An exhaustive finite grid is not an exhaustive search over every possible implementation.
