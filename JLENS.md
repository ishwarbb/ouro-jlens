# Ouro Jacobian lens pilot

For the loop-end lens family from arXiv:2609.01924, see
[JLENS_FAMILY.md](JLENS_FAMILY.md). It adds separate per-loop targets and
WikiText fitting; the artifact described below is the earlier final-target pilot.

The adapter in `ouro_lens.py` exposes all **192 virtual sites**: 48 physical
blocks invoked on each of four recurrent passes. Identity taps preserve the
original residual tensors and computation, while allowing the reference
recorder to distinguish individual invocations. A site index is
`(pass - 1) * 48 + (physical_layer - 1)` for one-based pass/layer labels.

This initial fit selects four sites: **47, 95, 143, 190** (pass 1/layer 48,
pass 2/layer 48, pass 3/layer 48, pass 4/layer 47). Each is transported into
site 191, the last block of pass 4, **before** its final normalization.
Unembedding applies that normalization once and then the model's LM head.
This is a four-site pilot, not a full 192-site fit.

## Completed run

Ten passages fitted in 272.19 seconds across the initial one-passage run and
nine-passage resume, with 16.51 GiB peak allocated GPU memory.
The portable lens is 32 MiB. All three held-out readouts are finite; the final
model readout matches the checkpoint exactly when both project only the last
token (`logits_to_keep=1`). Projecting the entire sequence instead introduced
a maximum BF16 rounding difference of 0.0625 in the first attempted check.

## What was verified

- All recurrent invocations have distinct activation records, in execution order.
- The final readout exactly matches the actual checkpoint's default logits and
  its explicitly selected final-pass logits on the verification sequence.
- A cross-pass gradient agrees with finite differences on a small instance of
  the actual Ouro architecture; frozen model weights receive no gradients.
- The reference Jacobian estimator agrees between dimension batch sizes 1 and 2
  on that small architecture.
- Full-checkpoint benchmark gradients are finite and nonzero at every selected site.
- Fitted matrices are finite before and after saving/loading the FP16 lens file.

The fit uses the reference estimator: cotangents summed over valid target
positions and gradients averaged over valid source positions, then averaged
across prompts. With a 64-token sequence, the first 16 positions and the last
position are excluded, leaving 47 source/target positions. Model weights are
frozen, activation gradients are enabled, and KV caching is disabled. Fitting
does not use the inference cache workaround because it never creates a cache.

## Reproduce

The J-lens code is vendored under `jacobian-lens/` at
`98f59779f89e092c418615e5e67c52fa5fd590cd`. Its package metadata declares
Transformers >=5.5. This adapter uses its model-independent protocol and fitter,
which were tested here with Ouro's existing Transformers 4.57.6 environment.
Install the pinned code without resolving that broader dependency constraint:

```bash
.venv/bin/pip install packaging==26.3
.venv/bin/pip install --no-deps --no-build-isolation -e ./jacobian-lens
OMP_NUM_THREADS=2 .venv/bin/python -m unittest -v test_ouro_lens.py
```

Model weights/code must already be present (`.venv/bin/python download_model.py`).
Run GPU commands outside the Codex sandbox, which hides GPU devices:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 \
  .venv/bin/python fit_ouro_lens.py --phase benchmark --dim-batch 16
CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 \
  .venv/bin/python fit_ouro_lens.py --phase fit --dim-batch 16 --num-prompts 10
CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 \
  .venv/bin/python fit_ouro_lens.py --phase apply --dim-batch 16
.venv/bin/python render_jlens.py
```

`results/jlens/fit-checkpoint.pt` stores local resumable sums after each prompt.
It is excluded from Git. `results/jlens/lens.pt` is the portable fitted artifact (release `pilot` bundle);
loading it does not require the training checkpoint. Restore it with
`python3 download_artifacts.py --bundle pilot`, or refit from scratch. The manifest guards model, adapter, corpus and fitting settings when
resuming locally. Use a new `--output-dir` when changing the configuration.

The ten fitting passages are authored neutral text in `fitting_data/neutral_10.json`.
They are separate from DolusChat, MASK, and the three held-out readout prompts.
They require no honesty labels or paid judge API. This small synthetic corpus
is for a smoke fit; it does not establish general-purpose lens quality. A larger
fit should use diverse pretraining-like text and held-out lens evaluations.

## View and apply

The standalone interactive viewer is documented in [OURO_VIEWER.md](OURO_VIEWER.md).
The historical pilot readouts, timings and verification are under `results/jlens/`.
Running `render_jlens.py` also writes the legacy summary to `site/jlens.html`;
serve that directory with `python3 -m http.server --bind 127.0.0.1 --directory site 8768`.

```python
import jlens
from ouro_lens import OuroLensModel

# hf and tokenizer are the loaded Ouro checkpoint and its tokenizer.
model = OuroLensModel(hf, tokenizer)
lens = jlens.JacobianLens.load('results/jlens/lens.pt')
lens_logits, model_logits, ids = lens.apply(
    model, 'A short example passage to inspect with the lens.', positions=[-1]
)
model.close()
```

Raw-text encoding prepends one BOS token. For replaying saved evaluation token
IDs, pass those IDs directly to `model.forward()` with an `ActivationRecorder`;
this preserves the exact chat input. Transport recorded float activations with
`lens.transport()`, then call `model.unembed()` to obtain token readouts.
The result is a vocabulary readout, not a scalar honesty score.
