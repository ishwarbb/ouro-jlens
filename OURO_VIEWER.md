# Ouro workspace explorer

A local, interactive viewer for the existing Ouro-2.6B checkpoint and the
1,000-prompt Jacobian lens family. It serves recorded experiments immediately
and loads the model when the first live analysis is requested. The frontend
uses native HTML, CSS, JavaScript, Canvas and SVG; no Node, build step, CDN,
external model API or new inference dependencies are required.

## Open it

From the repository root, start the recorded viewer with Python 3.10+:

```bash
python3 ouro_viewer.py --read-only --port 8769
```

Open **http://127.0.0.1:8769/**. The five bundled examples work without model
weights, a GPU, or extra Python packages. For remote access, run this on your
**local computer**, replacing the SSH connection details:

```bash
ssh -N -o ExitOnForwardFailure=yes -p SSH_PORT \
  -L 127.0.0.1:18769:127.0.0.1:8769 USER@HOST
```

Then open **http://127.0.0.1:18769/** locally and keep the tunnel running.

For live analysis, follow the dependency and model setup in [README.md](README.md),
then restore the fitted family and saved example states:

```bash
python3 download_artifacts.py --bundle lenses --bundle viewer
.venv/bin/python ouro_viewer.py --port 8769 --gpu 0
```

`--gpu` accepts 0–3 and 8. Automatic mode falls back to CPU when the selected
GPU has less than 10 GiB free and moves to that GPU before a later job if
capacity becomes available. CPU inference uses eight threads. `--device cpu`
forces CPU; `--device cuda` requires GPU capacity. The model and lenses stay
loaded between jobs. A single worker serializes model operations.

`--data PATH` selects another viewer data folder. New analysis requires the
checkpoint and fitted lenses. Exact traces and inventories for a saved run
also require that run's residual states. Existing JSON readouts are browsable
without any tensor artifacts.

To regenerate the two research viewer exports, restore the `inventories`
bundle and run `.venv/bin/python export_ouro_viewer.py`. Existing exports are
preserved unless `--refresh-research` is provided.

## Explore

- Use **Chat** for conversational instructions, or **Raw text** to continue a passage. The native chat template with an optional system
  message and assistant prefill. Generate up to 128 tokens, or choose zero
  to inspect only the prompt. The prompt plus continuation must fit in 512
  tokens. Generation is greedy and preserves the exact token IDs.
- Select a token and inspect the layer × token grid with the J-lens, raw
  logit lens, or sparse inventory. Filter loops, the workspace band, and
  prompt/continuation positions. Arrow keys move the selected cell.
- Pin up to four complete-token concepts. The chart compares their ranks
  across all four loops at the selected position. Clicking a chart point
  selects that layer; clicking a readout or frequency card pins its word.
- Compute a sparse inventory for a live cell. The recorded animal and sport
  experiments already include inventories throughout their captured grids.
  Choose 8, 16, 25, or 50 atoms and a normalized coefficient threshold.
- Use **Intervene** to transfer a source projection to another concept or
  remove a source direction. Choose loops and physical layers. The viewer
  regenerates the same prompt and saves a separate run, with the baseline
  continuation alongside it and a link back to the original run.
- Copy a view link or download JSON. Links restore the run, selected cell,
  lens, filters, pins, and inventory settings on the same viewer server.
  Downloads include metadata, token IDs, readouts, computed inventories,
  exact traces, and view settings. Residual tensors remain on the server.

The currency example has a saved live readout ready to open. Research captures
retain the original generation limits and thought/final-answer flags.

## What the values mean

Ouro repeats 48 physical layers four times. Physical layers are numbered
0–47, loops 1–4, and virtual sites 0–191. J-lens transport uses each source
site's own loop-end target. New live runs include all 192 sites; the four
loop ends use identity transport. The imported research captures contain
188 fitted source sites, so their missing loop ends are explicitly blank.

J-lens and logit-lens cells store the top eight vocabulary probabilities.
Pinning a concept computes its **full-vocabulary rank** from saved residuals,
using all whole-token case variants, strict-greater-than ties, and summed
variant probability. Before that computation completes, saved top-eight
matches are shown as bounded information: an absent match means `>8`.
Multi-token concepts do not have an exact single-token trace.

Inventories reuse the validated positive sparse pursuit and coefficient
refitting in `workspace_inventory.py`. Atoms are normalized J-transpose
unembedding directions. Displayed coefficients are divided by the original
state norm, and reconstruction reports one minus the squared residual
fraction. Coefficients are not probabilities. Atom order is pursuit selection
order; refitted coefficients can be zero. Thresholding follows the existing
`>=` rule. Frequency cards count matching word slots in the current selection;
they do not merge synonyms or establish a causal effect.

Transfers apply `h + (h·source)(destination − source)` using unit directions;
erasure removes the source projection. Edits apply only at original prompt
positions in the chosen loops/layers, both during generation and when
recording the resulting run. This is an interactive causal probe; a successful
example is not a statistical reproduction of the paper's headline results.

The checkpoint is BF16, transport is FP32, and inference uses SDPA with no KV
cache. Runtime device, model revision, lens-family digest, exact inputs, and
stop flags are recorded. CPU/GPU numerical differences are possible. The
existing research artifacts and their verification manifests are unchanged.
See [WORKSPACE_INVENTORY.md](WORKSPACE_INVENTORY.md) for methodological limits.

## Files and verification

- `ouro_viewer.py`: localhost server, persistent store, validation and job queue.
- `ouro_viewer_engine.py`: generation, readouts, exact traces and interventions.
- `export_ouro_viewer.py`: imports the immutable research captures.
- `ouro-viewer/`: browser interface and local assets.
- `results/ouro-viewer/runs/`: saved JSON, residuals, traces and cell inventories.
- `results/ouro-viewer/server.log`: running server log.
- `results/ouro-viewer/verification/`: browser report and screenshots.

The five published viewer examples are tracked; new viewer sessions, residual tensors and server logs are ignored by Git. Re-running the importer
preserves existing exports; `--refresh-research` replaces only its two named
research exports. Live runs remain intact. Cancelled jobs stop between model
operations; model loading and an individual forward pass finish first.

```bash
# Validation, persistence, serialized cancellation, HTTP scope and read-only mode.
.venv/bin/python -m unittest -v test_ouro_viewer.py

# Browser testing uses a separate environment; do not change model dependencies.
python3 -m venv /tmp/ouro-browser-env
/tmp/ouro-browser-env/bin/pip install playwright
/tmp/ouro-browser-env/bin/playwright install chromium
/tmp/ouro-browser-env/bin/python test_ouro_viewer_browser.py --live
```

The browser acceptance test requires the exported research captures and a
saved live currency example. It exercises both live computation and saved
data: full-vocabulary tracing, sparse pursuit, share/export persistence,
missing sites, capped thought output, native final-site generation alignment,
loop-end identity transport, France→China intervention, cancellation, desktop
and 390px mobile layouts. Omit `--live` for the browsing-only checks.

## References

The interaction design draws on the [J-Lens Visualizer](https://j-lens-visualizer.vercel.app/),
[Neuronpedia's shared J-lens view](https://www.neuronpedia.org/qwen3.6-27b/jlens?shareId=cmr2kx72r000mpt2x71l23e0z),
and [WeZZard/jlens-qwen36](https://github.com/WeZZard/jlens-qwen36).
The implementation adapts those interaction ideas to Ouro's repeated layers,
local fit and existing validated analysis code. Scientific context is in
[Wang and Reid, arXiv:2609.01924v1](https://arxiv.org/html/2609.01924v1).
