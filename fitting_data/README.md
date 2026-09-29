# Pilot fitting corpus

Ten short neutral passages authored for this fitting pilot, not sampled from
DolusChat or MASK. They require no honesty labels. These are smoke-test data,
not a representative pretraining corpus; a larger fit should use diverse text.
The runner adds one BOS token, truncates to 64 tokens, and excludes the first
16 and final token positions from the reference Jacobian estimator.

The three separate readout prompts are held out from this fitting set. They
support a qualitative apply check, not a quantitative lens-quality claim.
