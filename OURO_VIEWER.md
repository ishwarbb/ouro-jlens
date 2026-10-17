# Ouro workspace explorer

A local, interactive viewer for the existing Ouro-2.6B checkpoint and the
author's published 1,000-prompt Jacobian lens family for new runs. It serves recorded experiments immediately
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
python3 download_author_lens.py
.venv/bin/python ouro_viewer.py --port 8769 --gpu 0
```

`--gpu` accepts 0–8. Automatic mode falls back to CPU when the selected
GPU has less than 10 GiB free and moves to that GPU before a later job if
capacity becomes available. CPU inference uses eight threads. `--device cpu`
forces CPU; `--device cuda` requires GPU capacity. The model and lenses stay
loaded between jobs. A single worker serializes model operations.

New analyses and interventions use the pinned author family in
`results/author-lens-20261016/family`. Each new run records its lens repository,
revision, family path and manifest hash. Additional traces, inventories and
windows for an existing run use that run's recorded family, preserving its
original results. The old local fit remains available for these operations.

`--data PATH` selects another viewer data folder. New analysis requires the
checkpoint and fitted lenses. Exact traces and inventories for a saved run
also require that run's residual states. Existing JSON readouts are browsable
without any tensor artifacts.

To regenerate the two research viewer exports, restore the `inventories`
bundle and run `.venv/bin/python export_ouro_viewer.py`. Existing exports are
preserved unless `--refresh-research` is provided.

## Professor presentation collection

Open **http://127.0.0.1:8769/professor** (or the same `/professor` path on your forwarded port). The sidebar's **Research examples** link opens saved cases with exact responses or explicitly labeled next-token predictions, prompts, takeaways, interpretation limits, and source hashes. Interactive top-25 readouts provide recorded position, loop, layer, and readout-method selectors; other cases show recorded concept-readout or causal-control tables. Raw logits are labeled separately from probabilities; missing cells remain missing.

The collection covers openly stated false reporting, truthful lexical flags, reasoning/answer disagreement, ambiguous word meaning, absent monitored hits, within-response correction, token caps, intermediate concepts, causal transfer, and controlled span ablation. Selection is illustrative, not a detector ranking or effect estimate.

Saved data: `results/ouro-viewer/professor-examples/`. Rebuild only these presentation exports with `.venv/bin/python export_professor_examples.py`; original experiment files remain unchanged. Source-selection review is in `results/ouro-viewer/professor-example-review.json`.

Examples 11–13 use the author's released lens: two fresh multiplication reasoning/answer mismatches and an archived battery contradiction. Their full truthful controls, timing, and interpretation limits are included. See [the follow-up report](reports/professor-20261013/additional-author-lens-examples.html) and `results/professor-followups-20261016/analysis.json`. All 32 fresh attempts are retained, including disclosed strategies and errors. Rebuild the analysis with `.venv/bin/python analyze_professor_followups.py` and the new exports with `.venv/bin/python export_professor_followups.py`; the historical exporter preserves the resulting `supplemental-examples.json` entries.

Examples 14–21 present eight distinct tasks from the [1,024-response author-lens study](reports/professor-20261013/author-lens-scale-20261016.html). All 1,024 responses have fixed-window author-lens and raw-lens screening; 30 candidates and controls have full generated-trajectory replays. Eleven reviewed clean mismatches include three duplicate response forms. Early lexical flags also occur in aligned controls, so these are behavioral examples, not validated hidden schemes. The filterable report retains every outcome, and `results/professor-scale-20261016/responses.csv` exports every response. Rebuild its analysis with `.venv/bin/python summarize_professor_scale.py` and its presentation with `.venv/bin/python export_professor_scale.py`. Both supplement exporters preserve the other study's entries.

The current [incentive-transition report](reports/professor-20261013/incentive-transitions-20261016.html)
answers the user's clarified question: among correct-answer controls with no
target-word hits, how many gain hits when the reward favors a wrong answer?
At author J-lens rank ≤10, 118/262 clean early-window controls gain a hit in
the same-task/sample-index incentive response; across all saved windows the
count is 130/201. Words in other benign responses are allowed. The two
conditions used different batch seeds, so the report also compares all
within-task incentive samples against other benign samples and keeps lexical
emergence separate from behavioral deception. Rebuild with
`python3 analyze_incentive_transitions.py`; source hashes, per-word counts,
full response pairs and CSVs are in `results/incentive-transitions-20261016/`.

The earlier [target-word specificity report](reports/professor-20261013/target-word-specificity-20261016.html)
applied a stricter population-wide criterion, superseded by the conditional
question above. It tests whether the author-lens words distinguish mismatch examples from
truthful controls. It includes every measured word, ranks 1/5/10, raw-lens
comparisons, and early/reasoning/message windows. No tested word rule meets
the zero-observed-control-hit criterion while flagging a mismatch across the
combined control set; actual scheming sensitivity remains unmeasured because
these mismatches do not establish concealed intent. The task split is explicitly
retrospective. Rebuild with `python3 analyze_target_specificity.py`; detailed
data and source hashes are in `results/target-specificity-20261016/`.

## Explore

- Use **Chat** for conversational instructions, or **Raw text** to continue a passage. Chat uses the native template with an optional system
  message and assistant prefill. The default output budget is **16,819 tokens**;
  generation stops early at EOS or the chat turn-end token. Choose a smaller
  budget or zero to inspect only the prompt. Prompts accept up to 4,096 tokens,
  including template, system, and prefill. Generation is greedy and preserves
  exact token IDs. Opening an older run preserves your chosen output budget.
- Long runs retain their full continuation under **Model continuation**.
  **Inspect positions** pages the token strip, layer grid, and exact traces in
  windows of 512 tokens. All 192 residuals are saved at every token; additional
  readout windows are computed on demand and then cached. A read-only viewer
  can open previously computed windows.
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
  exact traces, and view settings. For long runs, exported readouts and traces
  cover the selected window (with absolute position offsets), while output text
  and token IDs cover the complete sequence. Residual tensors remain on the server.

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

The checkpoint, unembedding, and stored residuals are BF16; Jacobian transport
and probability/rank calculations use FP32. New live runs use SDPA and an
explicit `DynamicCache`, with separate KV
entries for all 192 recurrent sites. Prompt prefill uses 128-token chunks and
readouts use 64-token batches. Prompt interventions are applied as their
positions enter the cache; generated positions are not edited. Historical
runs retain their original uncached-generation provenance. BF16 rounding can
produce differences between cached/chunked and full-sequence computation.

A maximum-length run retains roughly 31 GiB of KV tensors on the GPU in
addition to model weights, and roughly 15 GiB of residuals on the host and
disk. The original server uses a 48 GB L40S on GPU 8; smaller devices may need
a smaller output budget. Runtime device, model revision, lens-family digest,
exact inputs, and
stop flags are recorded. CPU/GPU numerical differences are possible. The
existing research artifacts and their verification manifests are unchanged.
See [WORKSPACE_INVENTORY.md](WORKSPACE_INVENTORY.md) for methodological limits.

## Files and verification

- `ouro_viewer.py`: localhost server, persistent store, validation and job queue.
- `ouro_viewer_engine.py`: generation, readouts, exact traces and interventions.
- `ouro_viewer_inference.py`: chunked prefill, cached decoding and prompt-only edits.
- `ouro_viewer_limits.py`: generation, prompt and readout-window limits.
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
.venv/bin/python -m unittest -v test_ouro_viewer test_ouro_viewer_inference test_ouro_viewer_windows

# Browser testing uses a separate environment; do not change model dependencies.
python3 -m venv /tmp/ouro-browser-env
/tmp/ouro-browser-env/bin/pip install playwright
/tmp/ouro-browser-env/bin/playwright install chromium
/tmp/ouro-browser-env/bin/python test_ouro_viewer_browser.py --live
/tmp/ouro-browser-env/bin/python test_ouro_viewer_long_browser.py
```

The browser acceptance test requires the exported research captures and a
saved live currency example. It exercises both live computation and saved
data: full-vocabulary tracing, sparse pursuit, share/export persistence,
missing sites, capped thought output, native final-site generation alignment,
loop-end identity transport, France→China intervention, cancellation, desktop
and 390px mobile layouts. Omit `--live` for the browsing-only checks.
The long-output test checks the 16,819 default, natural stopping, 512 generated
tokens, late-position readouts/ranks/inventories, cached window navigation and
share/export offsets. [The verification receipt](provenance/long-output-checks.json)
also records a 20,915-position GPU capacity check using synthetic prefill and
one decoding step; it does not claim a full 16,819-step generation benchmark.

## References

The interaction design draws on the [J-Lens Visualizer](https://j-lens-visualizer.vercel.app/),
[Neuronpedia's shared J-lens view](https://www.neuronpedia.org/qwen3.6-27b/jlens?shareId=cmr2kx72r000mpt2x71l23e0z),
and [WeZZard/jlens-qwen36](https://github.com/WeZZard/jlens-qwen36).
The implementation adapts those interaction ideas to Ouro's repeated layers,
local fit and existing validated analysis code. Scientific context is in
[Wang and Reid, arXiv:2609.01924v1](https://arxiv.org/html/2609.01924v1).
