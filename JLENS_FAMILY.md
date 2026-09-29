# Ouro loop-end Jacobian lens family

The 1,000-prompt fit is complete. Its 476 FP32 matrices are published in the
release `lenses` bundle; see [README.md](README.md). The smoke-fit measurements
below describe the earlier ten-prompt validation stage.

Implementation: [ouro_lens_family.py](ouro_lens_family.py).
Runner: [fit_ouro_family.py](fit_ouro_family.py).
The earlier four-site, final-target pilot remains documented in [JLENS.md](JLENS.md).

## Paper and release search

[Wang and Reid, arXiv:2609.01924v1](https://arxiv.org/html/2609.01924v1)
find that a final-target lens obscures Ouro's early-loop content. Their remedy
is to fit separate transports to each loop end. They also find that narrow
causal edits generally need repeating across later loops. These observations
concern the lens's reach; an unreadable state need not lack the concept.
The architecture comparison confounds supervision, scale, and training, so it
does not isolate the effect of deep supervision experimentally.

Checked on 2026-09-28: the paper mentions released artifacts in Appendix A
but supplies no repository or weights URL. Its external links, the first
author's [public repositories](https://github.com/realwenlongwang?tab=repositories),
GitHub repository search, and Hugging Face model listings did not locate an
author release. This is a search result, not proof that none exists.

[VykosMolt/JLens-Ouro](https://github.com/VykosMolt/JLens-Ouro) is an independent
study, with code and evaluation records. Its README explicitly excludes large
lens banks from GitHub; it is not the linked paper's release.
[Anthropic's reference implementation](https://github.com/anthropics/jacobian-lens)
is already pinned in this repository and supplies the unchanged fitter.

## What this implementation does

- The adapter exposes 192 distinct block outputs before loop-end normalization.
  An intermediate fit stops after its target hook, preserving the exact upstream
  graph, including earlier loops and between-loop normalization.
- Each family contains targets **47, 95, 143, 191**, using zero-based indices.
  Readout chooses the source's own loop end; explicitly selecting another target
  is supported only if that transport was fitted.
- Fitting uses frozen weights, activation gradients, cache-free SDPA forwards,
  and disabled gradient checkpointing. The estimator sums cotangents over valid
  target positions, averages gradients over valid source positions, and averages
  those matrices equally over prompts. It is not a token-diagonal Jacobian.
- Transport is `h @ J.T`. Unembedding applies the model norm once, then its LM
  head. Token directions are normalized `J.T @ lm_head.weight[token_id]`.
- Injection and projection ablation support specified token positions and
  matched source offsets in every remaining loop.
- Resumption checks the corpus, prompt selection, model weights, code hashes,
  estimator, geometry, and numerical settings. Each target has its own atomic
  checkpoint. Portable matrices use FP32 to avoid FP16 overflow and preserve
  the fitted means exactly.

The existing Transformers **4.57.6** environment is retained. The paper's
Transformers-5 shims are unnecessary here; this implementation does not claim
compatibility with arbitrary Transformers 5 versions. Downloaded Ouro model
code is not modified.

## Run

Use the existing `.venv` and downloaded `model/` from the main README. Preparing
the corpus requires `huggingface_hub` and `pyarrow`, already installed here.

```bash
cd ouro-jlens
OMP_NUM_THREADS=2 .venv/bin/python -m unittest -v test_ouro_lens test_ouro_family

# Prepare once; an existing corpus is preserved.
.venv/bin/python fit_ouro_family.py --phase prepare --num-prompts 1000

# Completed smoke configuration: every within-loop source, ten prompts.
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python fit_ouro_family.py --phase fit \
  --scope within-loop --num-prompts 10 --dim-batch 8 --checkpoint-every 5 \
  --output-dir results/jlens-family-smoke

# Apply the saved family to a different prompt.
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 \
  .venv/bin/python fit_ouro_family.py --phase apply \
  --output-dir results/jlens-family-smoke \
  --prompt 'The capital city of the country shaped like a boot is'

# Full upstream-source geometry and 1,000-prompt protocol (long GPU job).
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python fit_ouro_family.py --phase fit \
  --num-prompts 1000 --scope all --dim-batch 4 \
  --output-dir results/jlens-family-1000
```

GPU commands need GPU visibility; this workspace's restricted sandbox hides
CUDA. Repeating an unchanged fit command resumes its per-target checkpoints.
Use a new output directory when changing corpus size, geometry, code, or settings.
`--offsets 1,11,23` offers a smaller fit at matched distances in every loop.

The prepared corpus pins WikiText-103-raw-v1 revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, takes the first 1,000 training
records with at least 600 stripped characters, and records the original shard
and row. This follows the upstream example selector. Exact identity with the
authors' undisclosed prompt list cannot be established. Encoding prepends BOS,
truncates to 128 tokens, and excludes positions 0–15 and the final position
from both estimator reductions.

`within-loop` fits **188** matrices, covering each nonterminal source in its
own loop. `all` fits **476** matrices, including earlier-loop sources for each
target, enabling direct cross-boundary comparisons. Loop endpoints themselves
use identity transport. FP32 portable storage is about 2.94 GiB or 7.44 GiB,
respectively; resumable sums need approximately the same space again. These
large files are Git-ignored and supplied by the release `lenses` bundle; the
full-fit manifests and summaries are tracked in Git. Smoke-fit tensors can be
regenerated with the earlier smoke command.

## Python use

```python
from ouro_lens import OuroLensModel
from ouro_lens_family import OuroLensFamily

# hf and tokenizer are the exact checkpoint used for fitting.
model = OuroLensModel(hf, tokenizer)
family = OuroLensFamily.load('results/jlens-family-smoke')
try:
    reads, loop_logits, ids = family.apply(
        model, 'The currency used in the country shaped like a boot is',
        sources=[24, 36, 72, 84, 120, 132, 168, 180],
        positions=[-1],
    )
    # Use explicit IDs: many concept strings tokenize to more than one token.
    token_ids = tokenizer.encode(' Italy', add_special_tokens=False)
    assert len(token_ids) == 1
    sites = family.sites_at_offsets([11, 23], first_loop=1)
    with family.intervene(model, token_ids[0], sources=sites,
                          positions=[-1], mode='inject', strength=1.0):
        edited_reads, edited_loop_logits, _ = family.apply(model, 'A short prompt')
finally:
    model.close()
```

The Python API checks architecture geometry; callers must supply the correct
weights. The CLI additionally checks saved weight hashes when applying a lens.
For all-token interventions as in the paper's main causal protocols, pass
`positions=None`; the example above deliberately edits only the last position.
Injection strengths here are absolute residual coefficients, not multiples of
a measured activation standard deviation.
For exact chat replay, use recorded token IDs with `model.forward` and an
`ActivationRecorder`, then `family.transport` and `model.unembed`.

## Evidence and limits

The full-checkpoint smoke fit completed in **708.7 seconds** (about 11.8 minutes)
with **9.14 GiB** peak allocated GPU memory on an L40S. All 188 matrices are
finite and nonzero, and FP32 save/load preserved every entry exactly. The
last-prompt maximum relative mean shifts were 0.0878, 0.0651, 0.0597, and 0.0578
for loops 1–4; this small fit should not be treated as converged.

The 14 CPU tests instantiate the actual Ouro architecture with small random
weights. They cover separate recurrent activations, loop logits, deterministic
forwards, intermediate/full graph agreement, finite differences across a loop
boundary, dimension batching with a partial final batch, matrix orientation,
own-loop routing, serialization, and intervention cleanup/position scope.
The fitting runner is also checked for exact resume without recomputation and
rejection of a changed corpus.

Full-checkpoint evidence is written under `results/jlens-family-smoke/`:

- `verification.json`: 192 distinct sites, exact logits at all four loop ends,
  intermediate/full-forward identity, and deterministic repeated forwards.
- `fit-manifest.json`: corpus, checkpoint, implementation, and estimator hashes.
- `family.json` and `target-*.pt`: explicit targets and the portable matrices.
- `fit-summary.json`: fitted coverage, prompt count, elapsed time, memory,
  finite/nonzero checks, and exact save/load checks.
- `readout.json`: own-loop top tokens, raw-logit-lens comparison, and actual
  loop-end baselines for a held-out prompt. Each apply command replaces this
  small readout file.

This section's artifact is the earlier **ten-prompt smoke fit**. The later
1,000-prompt fit, controlled semantic probes, sparse inventories, and causal
evaluation are described in [WORKSPACE_REPRODUCTION.md](WORKSPACE_REPRODUCTION.md),
[WORKSPACE_AUDIT.md](WORKSPACE_AUDIT.md), and [WORKSPACE_INVENTORY.md](WORKSPACE_INVENTORY.md).
Those follow-ups retain explicit limitations and do not constitute all eleven
paper suites or a complete replication. Finite matrices, vocabulary top tokens,
and low bulk KL alone do not demonstrate semantic validity.
