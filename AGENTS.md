# Ouro lens selection

Use the author's released base-Ouro lens for all new analyses, experiments,
readouts, and interventions, as requested by the user on 16 October 2026.
The default is `ouro_lens_config.DEFAULT_FAMILY`:
`results/author-lens-20261016/family`.

The lens repository is `zigzag9939/jlens-looped-transformers`, pinned to
`dc9e1b928a2d14bbb09e914525e39c258a489b05`, for `ByteDance/Ouro-2.6B` revision
`1ed04250da1a9936042725d302e81c8fa2ab5abd`. Use own-loop targets 47, 95, 143,
and 191 unless the experiment explicitly studies another target.

Preserve historical experiments and comparisons with their recorded lens.
The original fit remains at `results/jlens-family-1000`. Historical numbered
run scripts, frozen inventory/audit runners, fitting supervisors, and audit
reconstruction code intentionally retain that pin. Before adapting any of
those scripts for a new study, use DEFAULT_FAMILY, record its identity, and
choose a new output directory. Do not relabel old results as author-lens runs
or reuse their cached readouts with another family.

# Target-word evaluation

The current research question is conditional emergence after a deception
incentive is added: among benign responses with no target-word hits in the
specified J-lens window, how many matched incentive responses gain hits?
The user explicitly permits target words in other benign responses. Do not
impose zero false alarms across the entire benign population as a success
criterion. Report the eligible baseline denominator, the absent-to-present
count/rate, and which words appear. Also distinguish absence of all monitored
words from absence of just the individual word being counted.

Establish behavioral labels independently of lens hits. Keep reasoning/answer
mismatches, ordinary errors, and explicit instruction following separate from
verified scheming. Incentive-associated lexical emergence can be measured
without verified scheming labels; label it accordingly. Document how pairs
are formed and whether seeds match. For independent samples, compare against
benign resampling and avoid claiming a single-response causal counterfactual.
Freeze words, token variants, sites, windows, and thresholds before fresh
evaluation. Label any split of already-inspected data retrospective.
