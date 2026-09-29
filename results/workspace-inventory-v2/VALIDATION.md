# Independent reconstruction and quality review

Passed: 188 layer artifacts, 75,608 pursuit cells, 244,088 coefficient snapshots, and 29,172 frozen projector cells.

This separate verifier reconstructs atoms from the LM head and transposed saved Jacobians. It recomputes residual fractions and nonnegative-fit KKT conditions from stored coefficients, and verifies each frozen basis is orthogonal and spans its selected atoms. It does not invoke the pursuit solver or basis constructor.

All 64 baseline forwards match the native model's logits exactly. All 257 baseline/transfer report score records and 5 whole-token direction exclusions match the earlier frozen `Answer:` audit exactly. Excluded interventions remain failures in the all-attempt denominator. This is measurement hardening, not a newly discovered accuracy improvement.

## Numerical errors

- residual_fraction_error: 6.56e-07
- normalized_kkt: 2.8e-06
- orthogonality_error: 2.5e-06
- span_error: 3.02e-06
- native_logits_error: 0

## Interpretation limits

The animal completion differs from the paper transcript and explicitly says **trees**. Tree therefore cannot be labeled latent merely by importing the paper’s classification. Class labels must be derived from our own saved prompt and completion.

The hidden-sport prompt emits `<think>` and uses all 128 tokens in that thought block. Its inventory is a valid measurement of that trace, but it is not evidence for a concept hidden from the final answer: no final answer was produced. The complete raw text and cap flag are retained.

Report has 9/14 category-valid first-token sources. The 131-attempt rates retain the five invalid-source categories; the valid-source subset is descriptive. The fixed `Answer:` wrapper was chosen using the preceding audit, so improved report scores are not an independent confirmation.

The fitted-source grid omits the four loop-end targets and uses two of the paper’s eight demonstration prompts. The paper’s 21.5% explained-variance aggregate has a different population. Capacity under pursuit and fresh confirmation examples remain outstanding.

The corrected pipeline supersedes the preserved v1 development run, whose overly broad special-token stopping rule was caught during preflight. EOS and turn-end stop generation; thought markers do not.
