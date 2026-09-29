# Ouro workspace reproduction

Full 1,000-prompt fitting and predeclared tests of the base Ouro-2.6B checkpoint. Results below are operational tests of workspace properties; the authors’ exact code and fitting sample were unavailable.

## Primary evidence

| Property | Paired 95% intervals positive | Conservative intervals positive |
|---|---|---|
| Flexible steering | Not established | Not established |
| Internal reasoning | Yes | Not established |
| Verbal report | Not established | Not established |
| Sustained recurrence interface | Not established | Not established |
| Causal necessity | Yes | Yes |
| Latent readout | Not established | Not established |
| Task selectivity | Not established | Not established |
| Directed modulation | Yes | Yes |
| Injected-content report | Yes | Yes |

Conservative intervals use percentile tails 0.05/(2×7), covering seven primary hypothesis families. This is a conservative descriptive check, not a simultaneous guarantee for every individual contrast or a substitute for independent replication.

## Steering counts

| Experiment | Condition | Successes / trials | Rate |
|---|---|---:|---:|
| flexible | baseline | 0 / 176 | 0.0% |
| flexible | sustained | 7 / 176 | 4.0% |
| flexible | loop2 | 0 / 176 | 0.0% |
| flexible | loop3 | 2 / 176 | 1.1% |
| flexible | loop4 | 10 / 176 | 5.7% |
| flexible | random_matched | 0 / 176 | 0.0% |
| flexible | raw_lens | 0 / 176 | 0.0% |
| flexible | early_band | 0 / 176 | 0.0% |
| flexible | projection_transfer | 68 / 176 | 38.6% |
| flexible | band24_36 | 12 / 176 | 6.8% |
| flexible | double_strength | 1 / 176 | 0.6% |
| flexible | final_target | 9 / 176 | 5.1% |
| probe | baseline | 4 / 83 | 4.8% |
| probe | sustained | 8 / 83 | 9.6% |
| probe | loop2 | 6 / 83 | 7.2% |
| probe | loop3 | 6 / 83 | 7.2% |
| probe | loop4 | 6 / 83 | 7.2% |
| probe | random_matched | 4 / 83 | 4.8% |
| probe | raw_lens | 5 / 83 | 6.0% |
| probe | early_band | 4 / 83 | 4.8% |
| probe | projection_transfer | 32 / 83 | 38.6% |
| probe | band24_36 | 11 / 83 | 13.3% |
| probe | double_strength | 0 / 83 | 0.0% |
| probe | final_target | 5 / 83 | 6.0% |
| report | baseline | 0 / 131 | 0.0% |
| report | sustained | 5 / 131 | 3.8% |
| report | loop2 | 1 / 131 | 0.8% |
| report | loop3 | 0 / 131 | 0.0% |
| report | loop4 | 12 / 131 | 9.2% |
| report | random_matched | 1 / 131 | 0.8% |
| report | raw_lens | 12 / 131 | 9.2% |
| report | early_band | 1 / 131 | 0.8% |
| report | projection_transfer | 47 / 131 | 35.9% |
| report | band24_36 | 2 / 131 | 1.5% |
| report | double_strength | 22 / 131 | 16.8% |
| report | final_target | 3 / 131 | 2.3% |
| report_top5 | baseline | 19 / 131 | 14.5% |
| report_top5 | sustained | 20 / 131 | 15.3% |
| report_top5 | loop2 | 18 / 131 | 13.7% |
| report_top5 | loop3 | 19 / 131 | 14.5% |
| report_top5 | loop4 | 26 / 131 | 19.8% |
| report_top5 | random_matched | 17 / 131 | 13.0% |
| report_top5 | raw_lens | 60 / 131 | 45.8% |
| report_top5 | early_band | 19 / 131 | 14.5% |
| report_top5 | projection_transfer | 68 / 131 | 51.9% |
| report_top5 | band24_36 | 14 / 131 | 10.7% |
| report_top5 | double_strength | 22 / 131 | 16.8% |
| report_top5 | final_target | 14 / 131 | 10.7% |
| ablation_retention | argument | 15 / 48 | 31.2% |
| ablation_retention | answer | 4 / 48 | 8.3% |
| ablation_retention | unrelated | 45 / 48 | 93.8% |
| ablation_retention | argument_loop2 | 47 / 48 | 97.9% |
| ablation_retention | argument_loop4 | 14 / 48 | 29.2% |
| introspection | 0 | 0 / 101 | 0.0% |
| introspection | 0.5 | 36 / 101 | 35.6% |
| introspection | 1.0 | 50 / 101 | 49.5% |
| introspection | 2.0 | 61 / 101 | 60.4% |
| introspection | 4.0 | 67 / 101 | 66.3% |
| introspection | 8.0 | 71 / 101 | 70.3% |
| introspection | random8 | 0 / 101 | 0.0% |

## Paired differences

| Contrast | Difference (percentage points) | 95% cluster interval | Clusters |
|---|---:|---|---:|
| flexible: sustained - baseline | 4.0 | [0.0, 8.8] | 16 |
| flexible: sustained - random_matched | 4.0 | [0.0, 8.8] | 16 |
| flexible: sustained - loop2 | 4.0 | [0.0, 8.8] | 16 |
| flexible: sustained - loop3 | 2.8 | [0.0, 6.6] | 16 |
| flexible: sustained - loop4 | -1.7 | [-5.1, 1.2] | 16 |
| flexible: sustained - raw_lens | 4.0 | [0.0, 8.8] | 16 |
| flexible: sustained - final_target | -1.1 | [-4.8, 2.2] | 16 |
| flexible: sustained - early_band | 4.0 | [0.0, 8.8] | 16 |
| probe: sustained - baseline | 4.8 | [1.1, 10.0] | 65 |
| probe: sustained - random_matched | 4.8 | [1.1, 10.0] | 65 |
| probe: sustained - loop2 | 2.4 | [0.0, 6.1] | 65 |
| probe: sustained - loop3 | 2.4 | [0.0, 6.1] | 65 |
| probe: sustained - loop4 | 2.4 | [0.0, 6.1] | 65 |
| probe: sustained - raw_lens | 3.6 | [0.0, 8.2] | 65 |
| probe: sustained - final_target | 3.6 | [0.0, 8.0] | 65 |
| probe: sustained - early_band | 4.8 | [1.1, 10.0] | 65 |
| report: sustained - baseline | 3.8 | [0.0, 8.3] | 14 |
| report: sustained - random_matched | 3.1 | [-0.8, 7.7] | 14 |
| report: sustained - loop2 | 3.1 | [-0.8, 7.7] | 14 |
| report: sustained - loop3 | 3.8 | [0.0, 8.3] | 14 |
| report: sustained - loop4 | -5.3 | [-10.2, -1.5] | 14 |
| report: sustained - raw_lens | -5.3 | [-14.3, 3.0] | 14 |
| report: sustained - final_target | 1.5 | [-1.6, 5.3] | 14 |
| report: sustained - early_band | 3.1 | [-0.8, 7.7] | 14 |
| necessity: unrelated - argument retention | 62.5 | [46.5, 77.1] | 41 |
| latent: own - foil | 87.2 | [81.1, 92.5] | 70 |
| latent: own - raw | 6.7 | [0.0, 14.6] | 70 |
| latent: own - final | 1.1 | [0.0, 3.6] | 70 |
| language: explicit - automatic | 25.0 | [0.0, 50.0] | 8 |
| modulation: focus - control | 47.6 | [36.1, 59.1] | 22 |
| modulation: control - suppress | 23.5 | [16.6, 30.5] | 22 |
| introspection: 8 - 0 | 70.3 | [61.4, 79.2] | 101 |
| introspection: 8 - random8 | 70.3 | [61.4, 79.2] | 101 |

## Interpretation limits

The protocol and sensitivities were fixed before full-fit outcomes. Bootstrap units are semantic clusters, not individual repeated swaps. Language selectivity has only eight passages. Tokenization exclusions and baseline-correct subsets are retained in summary.json. Readout hits alone do not establish causal use. Sensitivity results are shown but never replace failed primary conditions.

Not assessed: all eleven suites, capacity, ignition, dual-task interference, or the sparse concept inventory. Differences in the undisclosed causal-band and prompt-formatting details limit exact comparison with the paper’s percentages.

## Example successful interventions

- 'The capital of France is the city of': swap 'France' → 'China'; top output ' Paris' → ' Beijing'; expected 'Beijing'.
- 'The capital of Canada is the city of': swap 'Canada' → 'China'; top output ' Ottawa' → ' Beijing'; expected 'Beijing'.
- 'Most people in France speak': swap 'France' → 'China'; top output ' French' → ' Chinese'; expected 'Chinese'.
- 'France is a country on the continent of': swap 'France' → 'China'; top output ' Europe' → ' Asia'; expected 'Asia'.
- 'Canada is a country on the continent of': swap 'Canada' → 'China'; top output ' North' → ' Asia'; expected 'Asia'.
- 'The single-word name for the currency now used in Canada is the': swap 'Canada' → 'France'; top output ' Canadian' → ' Euro'; expected 'Euro'.
- 'In the northern hemisphere, February is in the season of': swap 'February' → 'April'; top output ' winter' → ' spring'; expected 'spring'.
- 'Fact: The number of players per side in the sport invented in Springfield, Massachusetts is ': swap 'basketball' → 'baseball'; top output '1' → '9'; expected '9'.
- 'Fact: The continent of the country where Toronto is located is': swap 'Canada' → 'France'; top output ' North' → ' Europe'; expected 'Europe'.
- 'Fact: The chemical symbol for the element with atomic number 11 is': swap 'sodium' → 'iron'; top output ' Na' → ' Fe'; expected 'Fe'.
- 'Fact: The capital of the country where Munich is located is': swap 'Germany' → 'Japan'; top output ' Berlin' → ' Tokyo'; expected 'Tokyo'.
- '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nThink of a color. Answer in one word.<|im_end|>\n<|im_start|>assistant\nOne word:': swap 'Blue' → 'Red'; top output ' Blue' → ' Red'; expected 'Red'.
