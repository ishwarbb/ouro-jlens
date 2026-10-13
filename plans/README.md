# Run 9 and Run 10 server handoff

The plans are separated by scientific question and can run independently.

| Study | Design | Protocol |
|---|---|---|
| Run 9 | Cross-model capability, example copying, incentives, and richer Section 5.3 transfer | [Plan](../RUN9_PLAN.md), [JSON](run9/protocol.json) |
| Run 10 | Fresh confirmation of Ouro loops 2+3 versus loops 2+3+4 | [Plan](../RUN10_PLAN.md), [JSON](run10/protocol.json) |

These are implementation specifications. There is no `run_deception_run9.py` or `run_workspace_run10.py` in this change. The existing Run 8/audit commands would execute their older protocols, not these plans. Build and validate the new runners, then freeze their records and execution manifests before collecting primary outcomes.

## Get the plans on the other server

For a new checkout:

```bash
git clone https://github.com/ishwarbb/ouro-jlens.git
cd ouro-jlens
git log -1 --oneline
python3 plans/validate_plans.py
```

For an existing checkout, use `git pull --ff-only` and run the same validator. Record the full commit from `git rev-parse HEAD` in the eventual execution manifest. Authenticate with GitHub if the repository or release requires it.

The validator uses only Python 3.10+ and checks plan arithmetic, model pins, site mapping, and repository links. It does not download weights, initialize CUDA, launch experiments, or validate model behavior. Runtime data paths are relative to the checkout unless explicitly overridden in the execution manifest; document/source links are relative to the containing protocol. No paths from the original host or live worker files are required.

## Restore the existing Ouro assets

The existing environment pins are in [requirements.txt](../requirements.txt). Create a dedicated environment on the destination host; do not reuse an incompatible Qwen/Nanbeige environment.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation -e ./jacobian-lens
.venv/bin/python download_model.py
python3 download_artifacts.py --bundle lenses
```

The model downloader uses [model_revision.json](../model_revision.json). The artifact downloader verifies the hashes in [artifacts.json](../artifacts.json). The lens bundle requires approximately 7.47 GB of archive parts and 7.99 GB extracted; allow room for both during restoration, plus weights, environments and new outputs. A local model copy may be reused only after its checkpoint, tokenizer and code hashes are checked against the pinned revision. Git alone does not contain model weights or fitted tensors.

If the pinned PyTorch build is unavailable or unsuitable for the host's CUDA setup, choose a compatible build, record the exact deviation, and re-run numerical preflight before freezing execution. This does not permit changing model weights, intervention precision or scoring to improve observed results.

Recorded tests that help validate the existing components:

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m unittest -v \
  test_ouro_lens test_ouro_family test_workspace_interventions test_workspace_audit
```

These tests do not cover the new Run 9 simulators, cross-model adapters, or Run 10's mixed loop controls and continuation mask; those checks remain part of the implementation work below.

## Additional Run 9 checkpoints

All candidate revisions and suggested relative destinations are in [run9/protocol.json](run9/protocol.json). Download from those exact Hugging Face repository revisions, including tokenizer files, configurations and required model code. Keep a separate environment for any model whose dependencies conflict with Ouro. Use `huggingface_hub.snapshot_download(repo_id=..., revision=..., local_dir=...)`; do not substitute the moving `main` branch.

Download and smoke-test one model at a time. Start with Ouro and Ouro-Thinking, then Nanbeige and Qwen3-4B, and profile Qwen3.8-27B before scheduling it. Its BF16 weights alone occupy about 55.6 GB; model loading, KV state and instrumentation add memory. Use suitable aggregate GPU memory or a separately declared, revalidated quantized configuration. Do not assume a single 48 GB GPU will hold the full BF16 model.

The model panel is open-weight-first. Optional Opus API work is outside these counts and requires a separately bounded design. No API credentials or remote service connections are needed to read, validate or implement the default plans.

## Run 9 implementation work

Reuse the finite-tool and provenance patterns in [run8_tasks.py](../run8_tasks.py), [run7_tasks.py](../run7_tasks.py), and [run8_review.py](../run8_review.py). Use [run7_generation.py](../run7_generation.py) as an Ouro generation reference; it is not a generic cross-model adapter. Preserve the old files and introduce Run 9-specific modules and a new output root.

Required before collection:

1. Implement affine, threshold and lookup simulators, artifact/report scoring, and all five example treatments. Test that A/B/legacy examples are legal but wrong for each selected record, and that the held-out correct implementation is initially correct.
2. Implement model adapters with the pinned native templates, reasoning-history policy, stop tokens, context checks and action parsing. Test token boundaries and actual recurrent invocations where relevant. Save rendered prompts and raw outputs.
3. Materialize disjoint calibration, validation, example-factorial, richer-feasibility and richer-main records. Freeze examples, seeds, eligibility rules, the eight/sixteen lens split, scoring rubrics and paired statistics.
4. Run the eight runtime checks per model, then Stage A calibration and validation. Freeze the resulting decoder/model selection before accessing main behavior. A model-family capability failure excludes that block; it does not trigger prompt tuning within the same run.
5. Run eligible Stage B blocks. Independently run Stage C's richer-interface feasibility gate before its primary comparisons. No gate depends on observing deception or lens hits.
6. Export complete lens-blind review packets. Keep factual outcomes separate from semantic strategy judgments; report human-review coverage explicitly. Fit and validate each new checkpoint's lens only under a separate measured compute budget.

Maximum behavior work is 2,128 episodes across the gated stages, plus 136 factual baselines and 40 runtime checks: at most 16,800 generation calls, excluding separately budgeted lens controls and fitting. The first server milestone is Stage A, not launching that entire ceiling at once.

## Run 10 implementation work

Reuse [workspace_audit.py](../workspace_audit.py), [workspace_interventions.py](../workspace_interventions.py), [ouro_lens.py](../ouro_lens.py), and [ouro_lens_family.py](../ouro_lens_family.py). The old audit fixes its own task population and output layout, so write a separate runner accepting a new record manifest and output directory.

Required before collection:

1. Materialize 64 fresh prompts, eight per semantic template, and three destinations per prompt. Bind expected transformed answers, token IDs, alias rules and rejected candidates before looking at model outcomes.
2. Implement the nine specified conditions, including mixed semantic loop-2+3 edits and random loop-4 edits. Match the norm of **projection transfer**, not coordinate exchange. Test the exact physical/virtual site mapping.
3. Preserve the historical next-token success metric as primary. Implement the separate 32-token completed-answer metric with edits restricted to original prompt positions, cache-free decoding and deterministic scoring.
4. Complete the disjoint engineering preflight, restore-check all tensors, and hash the code/records/runtime in an immutable manifest. Then execute 1,728 logical trial settings and the prespecified paired analysis.

This study uses the existing original Ouro lens family. It does not require fitting a lens for Ouro-Thinking, Qwen or Nanbeige, or waiting for Run 9 reviews.

## Outputs and resumption

Choose unique server-local run IDs, for example `deception-run9-20261014-hostname` and `workspace-run10-20261014-hostname`; the date in the ID should reflect actual execution, not this planning date. Resolve `output_root_template` against the checkout or an explicit new output root. Save the effective paths and GPU IDs in the manifest, not the scientific protocol.

Every case should carry an immutable manifest hash and stable ID. Use atomic result writes and append-only events; resume only missing or infrastructure-failed draws under the same configuration, retaining failed attempts. Never replace an unfavorable valid draw. Keep original Runs 1–8 and their incomplete/excluded dispositions intact. An experiment can proceed on the new server without reading their live status files.

Deliver raw prompts/responses, state transitions, token and model provenance, capability exclusions, review coverage, numerical preflight and all planned statistics. Publish a result even when a hypothesis is unsupported. The two new plans do not change any prior run's conclusions.
