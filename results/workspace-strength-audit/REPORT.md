# Strength semantics diagnostic

Exploratory follow-up to a discrepancy with the paper’s doubled-strength row. The primary definition remains unchanged; the authors’ exact strength implementation is unknown. Every interpretation is tested on all 192 swaps.

| Interpretation | Flexible top-1 |
|---|---:|
| Double the whole residual update (frozen main definition) | 71/192 (37.0%) |
| Remove source once; add twice its magnitude in destination direction | 13/192 (6.8%) |
| Scale both unnormalized direction vectors by two (four times the update) | 19/192 (9.9%) |
| Transfer only positive source projections; double the update | 73/192 (38.0%) |

Paper reference: 14/192 under its doubled-strength setting ([Table 7](https://arxiv.org/html/2609.01924v1#A2)). A close number under an alternative is not evidence that the author implementation has been recovered.
