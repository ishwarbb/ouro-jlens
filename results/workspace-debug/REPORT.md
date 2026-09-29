# Ouro workspace reproduction

Full 1,000-prompt fitting and predeclared tests of the base Ouro-2.6B checkpoint. Results below are operational tests of workspace properties; the authors’ exact code and fitting sample were unavailable.

## Primary evidence

| Property | Paired 95% intervals positive | Conservative intervals positive |
|---|---|---|
| Flexible steering | Not established | Not established |
| Internal reasoning | Not established | Not established |
| Verbal report | Not established | Not established |
| Sustained recurrence interface | Not established | Not established |
| Causal necessity | Not established | Not established |
| Latent readout | Not established | Not established |
| Task selectivity | Not established | Not established |
| Directed modulation | Not established | Not established |
| Injected-content report | Not established | Not established |

Conservative intervals use percentile tails 0.05/(2×7), covering seven primary hypothesis families. This is a conservative descriptive check, not a simultaneous guarantee for every individual contrast or a substitute for independent replication.

## Steering counts

| Experiment | Condition | Successes / trials | Rate |
|---|---|---:|---:|
| flexible | baseline | 0 / 3 | 0.0% |
| flexible | sustained | 1 / 3 | 33.3% |
| flexible | loop2 | 0 / 3 | 0.0% |
| flexible | loop3 | 0 / 3 | 0.0% |
| flexible | loop4 | 1 / 3 | 33.3% |
| flexible | random_matched | 0 / 3 | 0.0% |
| flexible | raw_lens | 0 / 3 | 0.0% |
| flexible | early_band | 0 / 3 | 0.0% |
| flexible | projection_transfer | 3 / 3 | 100.0% |
| flexible | band24_36 | 1 / 3 | 33.3% |
| flexible | double_strength | 0 / 3 | 0.0% |
| probe | baseline | 0 / 1 | 0.0% |
| probe | sustained | 0 / 1 | 0.0% |
| probe | loop2 | 0 / 1 | 0.0% |
| probe | loop3 | 0 / 1 | 0.0% |
| probe | loop4 | 0 / 1 | 0.0% |
| probe | random_matched | 0 / 1 | 0.0% |
| probe | raw_lens | 0 / 1 | 0.0% |
| probe | early_band | 0 / 1 | 0.0% |
| probe | projection_transfer | 0 / 1 | 0.0% |
| probe | band24_36 | 0 / 1 | 0.0% |
| probe | double_strength | 0 / 1 | 0.0% |
| report | baseline | 0 / 10 | 0.0% |
| report | sustained | 0 / 10 | 0.0% |
| report | loop2 | 0 / 10 | 0.0% |
| report | loop3 | 0 / 10 | 0.0% |
| report | loop4 | 0 / 10 | 0.0% |
| report | random_matched | 1 / 10 | 10.0% |
| report | raw_lens | 0 / 10 | 0.0% |
| report | early_band | 0 / 10 | 0.0% |
| report | projection_transfer | 1 / 10 | 10.0% |
| report | band24_36 | 0 / 10 | 0.0% |
| report | double_strength | 0 / 10 | 0.0% |
| report_top5 | baseline | 2 / 10 | 20.0% |
| report_top5 | sustained | 0 / 10 | 0.0% |
| report_top5 | loop2 | 2 / 10 | 20.0% |
| report_top5 | loop3 | 2 / 10 | 20.0% |
| report_top5 | loop4 | 0 / 10 | 0.0% |
| report_top5 | random_matched | 4 / 10 | 40.0% |
| report_top5 | raw_lens | 6 / 10 | 60.0% |
| report_top5 | early_band | 0 / 10 | 0.0% |
| report_top5 | projection_transfer | 2 / 10 | 20.0% |
| report_top5 | band24_36 | 0 / 10 | 0.0% |
| report_top5 | double_strength | 0 / 10 | 0.0% |
| introspection | 0 | 0 / 1 | 0.0% |
| introspection | 0.5 | 0 / 1 | 0.0% |
| introspection | 1.0 | 0 / 1 | 0.0% |
| introspection | 2.0 | 0 / 1 | 0.0% |
| introspection | 4.0 | 0 / 1 | 0.0% |
| introspection | 8.0 | 0 / 1 | 0.0% |
| introspection | random8 | 0 / 1 | 0.0% |

## Paired differences

| Contrast | Difference (percentage points) | 95% cluster interval | Clusters |
|---|---:|---|---:|
| flexible: sustained - baseline | 33.3 | Unavailable | 1 |
| flexible: sustained - random_matched | 33.3 | Unavailable | 1 |
| flexible: sustained - loop2 | 33.3 | Unavailable | 1 |
| flexible: sustained - loop3 | 33.3 | Unavailable | 1 |
| flexible: sustained - loop4 | 0.0 | Unavailable | 1 |
| flexible: sustained - raw_lens | 33.3 | Unavailable | 1 |
| flexible: sustained - early_band | 33.3 | Unavailable | 1 |
| probe: sustained - baseline | 0.0 | Unavailable | 1 |
| probe: sustained - random_matched | 0.0 | Unavailable | 1 |
| probe: sustained - loop2 | 0.0 | Unavailable | 1 |
| probe: sustained - loop3 | 0.0 | Unavailable | 1 |
| probe: sustained - loop4 | 0.0 | Unavailable | 1 |
| probe: sustained - raw_lens | 0.0 | Unavailable | 1 |
| probe: sustained - early_band | 0.0 | Unavailable | 1 |
| report: sustained - baseline | 0.0 | Unavailable | 1 |
| report: sustained - random_matched | -10.0 | Unavailable | 1 |
| report: sustained - loop2 | 0.0 | Unavailable | 1 |
| report: sustained - loop3 | 0.0 | Unavailable | 1 |
| report: sustained - loop4 | 0.0 | Unavailable | 1 |
| report: sustained - raw_lens | 0.0 | Unavailable | 1 |
| report: sustained - early_band | 0.0 | Unavailable | 1 |
| latent: own - foil | 0.0 | Unavailable | 1 |
| latent: own - raw | 0.0 | Unavailable | 1 |
| latent: own - final | 0.0 | Unavailable | 1 |
| language: explicit - automatic | 0.0 | Unavailable | 1 |
| modulation: focus - control | 26.7 | Unavailable | 1 |
| modulation: control - suppress | 33.3 | Unavailable | 1 |
| introspection: 8 - 0 | 0.0 | Unavailable | 1 |
| introspection: 8 - random8 | 0.0 | Unavailable | 1 |

## Interpretation limits

The protocol and sensitivities were fixed before full-fit outcomes. Bootstrap units are semantic clusters, not individual repeated swaps. Language selectivity has only eight passages. Tokenization exclusions and baseline-correct subsets are retained in summary.json. Readout hits alone do not establish causal use. Sensitivity results are shown but never replace failed primary conditions.

Not assessed: all eleven suites, capacity, ignition, dual-task interference, or the sparse concept inventory. Differences in the undisclosed causal-band and prompt-formatting details limit exact comparison with the paper’s percentages.

## Example successful interventions

- 'The capital of France is the city of': swap 'France' → 'China'; top output ' Paris' → ' Beijing'; expected 'Beijing'.
