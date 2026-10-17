---
license: apache-2.0
library_name: jlens
tags:
  - interpretability
  - jacobian-lens
  - global-workspace
  - looped-transformer
  - depth-recurrence
base_model:
  - Qwen/Qwen3.6-27B
  - ByteDance/Ouro-2.6B
  - ByteDance/Ouro-2.6B-Thinking
  - tomg-group-umd/huginn-0125
---

> 📄 **Paper:** [Looped Transformers under the Jacobian Lens: Does the Global Workspace Survive Recurrence?](https://arxiv.org/abs/2609.01924) (Wenlong Wang and Fergal Reid, arXiv:2609.01924)

# Jacobian lenses for looped and recurrent transformers

Fitted Jacobian lenses, experiment outputs and fitting code for [our paper](https://arxiv.org/abs/2609.01924) extending Anthropic's J-space / Global Workspace analysis to looped (Ouro) and depth-recurrent (Huginn) transformers, with Qwen3.6-27B as the standard-transformer control.

**Repository layout**

| Path | Contents |
|---|---|
| `lenses/qwen/`, `lenses/ouro/`, `lenses/ouro_thinking/`, `lenses/huginn/` | Fitted lenses (`.pt`), 43 GB total |
| `MODEL_REVISIONS.md` | Exact Hugging Face revisions the lenses were fitted against |
| `results/` | Causal-suite outputs (`ws_*.json`), pursuit occupancy dumps, top-k readout dumps, analysis JSON, figures |
| `results/thinking/` | The same for Ouro-2.6B-Thinking |
| `results/logs/` | The run logs that `code/verify_ws.py` parses |
| `code/` | Fitting workers, merge scripts, looped/recurrent model adapters, experiment scripts, verification scripts. Requires Anthropic's `jacobian-lens` package |
| `requirements.txt` | Exact package versions of the run environment |

## What these files are

Every `jacobian_lens_*.pt` / `lens_*.pt` file here is a **fitted Jacobian lens**:
a set of matrices that describe how a model's residual-stream state at one
layer influences its state at a later target layer. They were produced with
Anthropic's open-source `jacobian-lens` toolkit
(https://github.com/anthropics/jacobian-lens, Apache-2.0), which accompanies
their Global Workspace / "J-space" paper, and extended by us to looped and
recurrent transformers.

For a source layer `v` and a target layer `t`, the lens stores

    J_v = E_prompts[ d h_t / d h_v ]        (a d_model x d_model matrix)

averaged over 1000 prompts drawn from a pretraining-like corpus, following
the protocol described in Anthropic's reference implementation. The prompts
themselves are not redistributed here; `code/prep_prompts.py` regenerates them
with the toolkit's own prompt loader. Transporting a state through `J_v` and
then through the model's own unembedding gives a *readout* of what a
mid-computation state means in the target layer's coordinates. The
"J-space" of Anthropic's paper is the subspace of the residual stream that
these matrices carry into the readout: the low-dimensional slice encoding
nameable concepts, which is where the workspace signature is measured.

## What these files are NOT

- **They are not model weights.** No parameter of Qwen, Ouro or Huginn is
  stored here. A lens cannot generate text, cannot be loaded with
  `from_pretrained`, and is useless without the exact model it was fitted on.
- **They cannot be used alone.** Reading out or intervening requires (a) the
  original model weights at the pinned revision below, (b) a forward pass to
  produce the residual states `h_v`, and (c) the model's own unembedding
  matrix. The lens only supplies the linear map `h_v -> J_v h_v`.
- **They are model-revision specific.** A lens is a function of the frozen
  weights. If the model checkpoint changes (new safetensors, changed
  tokenizer, edited `trust_remote_code` files) the lens is stale and must be
  refitted. The Hugging Face revisions used are in `MODEL_REVISIONS.md`.
- **They are linearised, mean summaries.** Each `J_v` is an average over
  prompts of a per-prompt Jacobian. It is only informative where individual
  prompts' Jacobians agree, which in practice means within one loop or
  within about two recurrences of the target. See the paper for the
  validated horizons.

## File format

Each file is a `torch.save` dict written by `jlens.lens.JacobianLens.save`:

    {
      "J":             {source_layer: Tensor[d_model, d_model] fp16, ...},
      "n_prompts":     1000,
      "source_layers": [0, 1, ..., t-1],
      "d_model":       int,
    }

Load with `JacobianLens.load(path)` from the `jacobian-lens` checkout in
https://github.com/anthropics/jacobian-lens. Size scales as
`n_source_layers x d_model^2`, which is why deep-target lenses are large.

Files named `lens_<x>_w0_<t>.pt` are complete single-worker fits, identical
in kind to `jacobian_lens_*.pt`; the naming is an artefact of the
multi-worker fitting harness. All worker shards and fit checkpoints have
been deleted; every file here is a merged, full-1000-prompt lens.

## Inventory

Layer indices are *virtual* layers: for looped/recurrent models the shared
weights are unrolled so that the model presents as a plain deep stack and the
unmodified `jlens` fit/apply code works (adapters: `code/ouro_lens.py`,
`code/huginn_lens.py`).

### Qwen3.6-27B (standard transformer, 64 layers, d_model 5120)

| File | Sources | Target | Role |
|---|---|---|---|
| `jacobian_lens_qwen3.6-27b.pt` | L0-62 | L63 (final) | Main lens; reproduces Anthropic's L24-48 workspace band |
| `jacobian_lens_qwen_t24.pt` | L0-23 | L24 | Band-entry lens; factorisation and below-band tests |

### ByteDance Ouro-2.6B (looped, 48 physical layers x 4 loops = 192 virtual, d_model 2048)

Virtual layer `v = 48*loop + layer`. Loop ends are v47, v95, v143, v191.

| File | Sources | Target | Role |
|---|---|---|---|
| `jacobian_lens_ouro-2.6b.pt` | v0-190 | v191 (final) | Full-depth lens |
| `jacobian_lens_ouro-2.6b_t47.pt` | v0-46 | v47 | Loop-1 end (checkpoint) lens |
| `jacobian_lens_ouro-2.6b_t95.pt` | v0-94 | v95 | Loop-2 end lens |
| `jacobian_lens_ouro-2.6b_t143.pt` | v0-142 | v143 | Loop-3 end lens |
| `jacobian_lens_ouro_mid_t120.pt` | v0-119 | v120 | Mid-loop-3 target (band-interior control) |
| `jacobian_lens_ouro_mid_t132.pt` | v0-131 | v132 | Mid-loop-3 target (band-interior control) |

### ByteDance Ouro-2.6B-Thinking (`lenses/ouro_thinking/`)

Same layout as the base model, fitted on the reasoning-SFT checkpoint for the
robustness section: `lenses/ouro_thinking/jacobian_lens_ouro-2.6b{,_t47,_t95,_t143}.pt`.
Despite the identical filenames these are **different matrices** from the
base-model lenses above and must be used with the Thinking weights.

### Huginn-0125 (depth-recurrent; prelude 2 + 4 layers x 16 recurrences + coda 2 = 68 virtual, d_model 5280)

| File | Sources | Target | Role |
|---|---|---|---|
| `jacobian_lens_huginn-0125_r16.pt` | v0-66 | v67 (final) | Full-depth lens at r=16 |
| `jacobian_lens_huginn_ts{S}_t{T}.pt`, `lens_huginn_ts{S}_w0_t{T}.pt` | v0-(T-1) | vT | Recurrence-end tiling: one lens per recurrence S=0..14, target T = 4S+5 |

The 15 tiling lenses (t5, t9, ..., t61) exist because each Huginn lens reads
reliably only within about two recurrences of its own target; together they
tile the depth axis at constant lens distance.

### Other `.pt` files

`results/ouro_jsim.pt` (and `results/thinking/ouro_jsim.pt`) are small analysis outputs
(pairwise cosine similarities between Ouro `J_v` matrices), not lenses.

## Reproducing

There are three levels, from cheapest to most expensive: check the released
numbers (CPU, minutes), re-run the experiments with the released lenses
(GPU), or refit the lenses from scratch (many GPU-hours).

### 1. Environment

```bash
git clone https://github.com/anthropics/jacobian-lens
git -C jacobian-lens checkout 581d398613e5602a5af361e1c34d3a92ea82ba8e
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt --no-deps
```

`requirements.txt` is the exact environment the runs used (Python 3.12.13,
torch 2.13.0 + CUDA 13.0, transformers 5.14.1). `--no-deps` is deliberate:
`causal-conv1d`'s metadata pins an old torch. The experiment scripts also read
prompt sets from the checkout's `data/experiments/`, so keep the checkout.

Every model load in `code/` passes the revision in `code/revisions.py`
(identical to `MODEL_REVISIONS.md`), so the right checkpoints and
`trust_remote_code` files are downloaded automatically.

### 2. Check the released numbers (no GPU)

```bash
python code/verify_ws.py         # 232 checks: causal suite, top-k readouts, logs
python code/verify_pursuit.py    # 132 checks: pursuit occupancy, capacity, ablation
```

Both recompute every headline number from the raw files in `results/` and
compare it with the reported value; both currently pass in full.
`results/pursuit/pursuit_analysis_{model}.json` and `NUMBERS_PACK.md` are
regenerated from the dumps with:

```bash
python code/ws_pursuit_analyze.py results/pursuit results/topk qwen
python code/ws_pursuit_analyze.py results/pursuit results/topk ouro
python code/ws_pursuit_analyze.py results/pursuit results/topk_gapfill16 huginn
python code/ws_pursuit_numbers.py > results/pursuit/NUMBERS_PACK.md
```

### 3. Re-run the experiments with the released lenses

The scripts load lenses, prompts and outputs by bare filename from the
current directory (the flat layout they were run in). Build a run directory
that symlinks the lenses and code into that layout:

```bash
python code/setup_workdir.py run --jlens jacobian-lens
cd run && python prep_prompts.py      # regenerates prompts_1000.json exactly
```

Add `--with-results` to also copy in the released results (for the
analysis/plot steps alone). Results are copied, not linked, so a re-run
never overwrites the released files.

Each command writes the named files into the run directory; compare them
with `results/`. `{m}` is one of `qwen`, `ouro`, `huginn`.

| Command | Writes (see `results/`) |
|---|---|
| `python ws_run.py {m} all` | `ws_{m}_{probe_swap,introspection,summoning,selectivity_language,selectivity_linecount,capacity,dual_task,modulation,ignition}.json` |
| `python ws_extra.py {m} all` | `ws_{m}_verbal_report.json`, `ws_{m}_flexgen.json` |
| `python ws_ablate.py {m}` / `ws_ablate10.py {m}` | `ws_{m}_ablate.json` / `ws_{m}_ablate10.json` |
| `python ws_capacity_pursuit.py {m}` | `ws_{m}_capacity_pursuit.json`, `ws_{m}_capacity_pursuit_dump.json.gz` |
| `python ws_below_band.py all` | `ws_qwen_below_band_{sanity,kurtosis,verbal,ignition}.json` |
| `python swap_ouro_v2.py` | `ws_ouro_flexgen_v2.json` |
| `python verbal_ouro.py`, `python swap_ouro.py` | stdout only; saved as `results/logs/verbal_ouro.log`, `swap_ouro.log` |
| `mkdir topk && python ws_topk_dump.py {m} topk` | `topk/topk_{m}_{prompt}.json.gz` |
| `python ws_pursuit_dump.py {m} topk pursuit` (after the top-k dump) | `pursuit/pursuit_{m}_{prompt}.json.gz` |
| `python analyze_ouro_lens.py`, `python probes_logitlens.py`, `python plot_ouro.py` | `ouro_analysis.json`, `ouro_jsim.pt`, `ouro_probes_ll.json`, `ouro_lens_curves.png`, `ouro_jsim_heatmap.png` |
| `python analyze_ouro_targets.py`, `python plot_targets.py` | `ouro_targets_analysis.json`, `ouro_targets_curves.png` |
| `python analyze_jcompare.py` / `analyze_huginn_jcompare.py` | `jcompare_analysis.json` / `huginn_jcompare.json` |
| `python analyze_gap.py` / `analyze_qwen_gap.py` / `qwen_overlay.py` | `gap_analysis.json` / `qwen_t24_gap.json` / `qwen_overlay.json` |
| `python delib_trace.py`, then `python delib_causal.py` | `delib_trace.json`, `delib_causal.json` |
| `python inject_huginn.py` | `inject_huginn.json` |

Huginn top-k dumps: `ws_topk_dump.py` uses the complete 16-lens tiling,
matching `results/topk_gapfill16/`, which is the canonical Huginn dump (the
verify scripts read it). `results/topk/topk_huginn_*` is the earlier 4-lens
dump and `results/topk_gapfill/` an intermediate one from a partial tiling;
both are kept for reference.

**Ouro-2.6B-Thinking** (`results/thinking/`): its lenses share filenames
with the base Ouro lenses, so use a separate run directory and select the
model by environment variable:

```bash
python code/setup_workdir.py run_thinking --jlens jacobian-lens --thinking
cd run_thinking && export OURO_MODEL_ID=ByteDance/Ouro-2.6B-Thinking
python prep_prompts.py
```

then run the Ouro commands above (`ws_run.py ouro ...`, `ws_ablate.py ouro`,
`analyze_ouro_lens.py`, `verbal_ouro.py`, ...). The `thinking_*.py` scripts
are identical copies of their base scripts, except that
`thinking_smoke_ouro.py` loads the Thinking model directly.

### 4. Refit the lenses from scratch

Use `setup_workdir.py --no-lenses`: the merge scripts save lenses by bare
filename and would otherwise write through the symlinks into `lenses/`.
Every fit worker takes `WORKER N_WORKERS` and fits its share of
`prompts_1000.json`; run one per GPU, then merge the shards.

| Lens | Fit (one per worker `W`) | Merge |
|---|---|---|
| `jacobian_lens_qwen3.6-27b.pt` | `fit_worker.py W 6` | `merge_lenses.py` |
| `jacobian_lens_qwen_t24.pt` | `fit_worker_qwen_t24.py W 3` | `merge_qwen_t24.py` |
| `jacobian_lens_ouro-2.6b.pt` | `fit_worker_ouro.py W 6` | `merge_lenses_ouro.py` |
| `jacobian_lens_ouro-2.6b_t{47,95,143}.pt` | `fit_worker_ouro_targets.py W 6` | `merge_targets.py` |
| `jacobian_lens_ouro_mid_t{120,132}.pt` | `fit_worker_midband.py W 5` | `merge_midband.py` |
| `jacobian_lens_huginn-0125_r16.pt` | `fit_worker_huginn.py W 3` | `JacobianLens.merge` over `lens_huginn_w*.pt` |
| `lens_huginn_ts{3,7,11}_w0_t*.pt` | `fit_worker_huginn.py 0 1 --target-step S --prefix huginn_tsS_w` | none (single worker) |
| `jacobian_lens_huginn_ts{1,5,9,13}_t*.pt` | `fit_worker_huginn.py W N --target-step S --prefix huginn_tsS_w` (N = 1, 2, 2, 3) | `merge_huginn_gapfill.py` |
| `jacobian_lens_huginn_ts{0,2,4,...,14}_t*.pt` | `fit_worker_huginn.py W N --target-step S --dim-batch 64 --prefix huginn_tsS_w` (N = 1 for S=0, else 2) | `merge_one_huginn.py S 4S+5` |

Approximate cost on B200s: Qwen final ~3 h on 6 GPUs, Qwen t24 ~2 h on 3,
Ouro and Huginn full-depth of the same order. `--dim-batch` only trades
memory for speed. Expect a refit to match the released lens up to
floating-point accumulation differences, not bit for bit.

The paper describes what was found with these lenses. Anthropic's original toolkit and paper: https://github.com/anthropics/jacobian-lens.

## Citation

If you use these lenses or results, please cite:

```bibtex
@article{wang2026looped,
  title={Looped Transformers under the Jacobian Lens: Does the Global Workspace Survive Recurrence?},
  author={Wang, Wenlong and Reid, Fergal},
  journal={arXiv preprint arXiv:2609.01924},
  year={2026}
}
```
