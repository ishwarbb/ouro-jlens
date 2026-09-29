# Verbal-report token diagnostic

Exploratory follow-up after the fixed audit exposed punctuation as the source in many categories. Only formatting tokens are consumed before scoring. A second variant identifies the greedy reported concept, then uses a complete-token concept direction if one exists. Unresolved concepts remain failures. This changes the prompt/scored position and cannot be claimed identical to the paper. It does not replace the frozen main result.

| Variant / score | Transfer | Unedited | Coordinate exchange | Random |
|---|---:|---:|---:|---:|
| skip_formatting/top1 | 70/136 (51.5%) | 0/136 (0.0%) | 5/136 (3.7%) | 2/136 (1.5%) |
| skip_formatting/top5 | 86/136 (63.2%) | 14/136 (10.3%) | 23/136 (16.9%) | 17/136 (12.5%) |
| decode_concept/top1 | 43/132 (32.6%) | 0/132 (0.0%) | 8/132 (6.1%) | 2/132 (1.5%) |
| decode_concept/top5 | 54/132 (40.9%) | 14/132 (10.6%) | 23/132 (17.4%) | 9/132 (6.8%) |

Valid category-source counts: {'skip_formatting': 4, 'decode_concept': 8}

Complete prefixes, emitted token sequences, unresolved cases, and all seven conditions are in the per-category records.
