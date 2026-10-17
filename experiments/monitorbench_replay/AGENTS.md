# MonitorBench replay scope

This experiment uses the author's separate Ouro-2.6B-Thinking lens. Its model
revision is `f1edd81e7ac41355db670500ceaf204e0f73af68`, and its author-lens revision
is `dc9e1b928a2d14bbb09e914525e39c258a489b05`, folder `lenses/ouro_thinking`.
Do not substitute the repository's base-Ouro default family here.

The initial trial was CPU-only. The user subsequently authorized GPU 8 only
"very quickly." CPU execution must hide CUDA devices. GPU 8 remains excluded
from ordinary runs; --brief-gpu8 permits at most two early-window responses,
four rollout loops, 512 input tokens and a 60-second deadline. Preserve these bounds.

The brief GPU FP32 trial passed; BF16 full-prefix/chunked validation failed.
FP32 is the validated default. Explicit BF16 runs require validation before
measurement. Do not relax tolerances to conceal the recorded discrepancy.

Replay the saved actual prompt and completion without generation. Keep source
results read-only. Retain unmatched, unsupported, incomplete and oversized
attempts with explicit status; missing readouts are not negative hits.

Follow the repository's conditional-emergence criterion. Record checkpoint,
lens, code, tokenization, window and source identities. Do not describe these
text reconstructions as exact historical traces, or lexical hits as verified
scheming. Preserve existing inference artifacts when updating aggregation.
