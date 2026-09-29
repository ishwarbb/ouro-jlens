"""Prepare, fit, and read a loop-end Ouro Jacobian lens family."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import time

import torch
import torch.nn.functional as F
import transformers
import jlens
from jlens.hooks import ActivationRecorder

from ouro_lens import OuroLensModel
from ouro_lens_family import OuroLensFamily, TargetModel, source_plan

ROOT = Path(__file__).resolve().parent
WIKITEXT_REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def weight_identity(directory):
    files = sorted(directory.glob('*.safetensors')) or sorted(directory.glob('pytorch_model*.bin'))
    if not files:
        raise ValueError('No local model weight files found')
    return {path.name: digest(path) for path in files}


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def prepare(args):
    """Same first-long-record selection as jlens.examples, with pinned data."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    if args.corpus.exists():
        raise FileExistsError(f'Preserving existing corpus: {args.corpus}')
    prompts, records = [], []
    for shard in range(2):
        filename = f'wikitext-103-raw-v1/train-{shard:05d}-of-00002.parquet'
        path = hf_hub_download('Salesforce/wikitext', filename, repo_type='dataset',
                               revision=WIKITEXT_REVISION)
        row_index = 0
        for batch in pq.ParquetFile(path).iter_batches(columns=['text']):
            for text in batch.column('text').to_pylist():
                if len(text.strip()) >= 600:
                    prompts.append(text)
                    records.append(dict(shard=filename, row=row_index))
                row_index += 1
                if len(prompts) == args.num_prompts:
                    break
            if len(prompts) == args.num_prompts:
                break
        if len(prompts) == args.num_prompts:
            break
    if len(prompts) != args.num_prompts:
        raise ValueError('Not enough qualifying WikiText records')
    save_json(args.corpus, dict(prompts=prompts, metadata=dict(
        dataset='Salesforce/wikitext', configuration='wikitext-103-raw-v1',
        split='train', revision=WIKITEXT_REVISION, min_chars=600, records=records,
        selection='First records with len(text.strip()) >= 600, in dataset order')))
    print(f'Prepared {len(prompts)} prompts: {args.corpus}', flush=True)


@torch.no_grad()
def verify(model, prompt):
    ids = model.encode(prompt, max_length=32)
    with ActivationRecorder(model.layers, at=range(model.n_layers)) as recorder:
        model.forward(ids)
    if len({id(x) for x in recorder.activations.values()}) != model.n_layers:
        raise AssertionError('Virtual activations are aliased')
    targets = list(range(model.physical_layers - 1, model.n_layers, model.physical_layers))
    errors = {}
    for step, target in enumerate(targets):
        stopped = model.forward(ids, stop_at=target)
        if not torch.equal(stopped, recorder.activations[target]):
            raise AssertionError(f'Target {target} changed under prefix execution')
        actual = model.unembed(stopped[:, -1:, :])
        expected = model.hf_model(ids, use_cache=False, exit_at_step=step, logits_to_keep=1).logits
        error = (actual.float() - expected.float()).abs().max().item()
        if error != 0:
            raise AssertionError(f'Loop {step} readout mismatch: {error}')
        errors[target] = error
    with ActivationRecorder(model.layers, at=targets) as repeat:
        model.forward(ids)
    deterministic = all(torch.equal(repeat.activations[t], recorder.activations[t]) for t in targets)
    if not deterministic:
        raise AssertionError('Repeated forwards disagree')
    return dict(distinct_sites=model.n_layers, loop_logits_max_absolute_error=errors,
                prefix_matches_full=True, deterministic=True, use_cache=False)


def fit_family(model, prompts, args):
    plan = source_plan(model.physical_layers, model.recurrent_steps,
                       scope=args.scope, offsets=args.offsets)
    # Validate before upstream fit(), whose generic ValueError handler skips records.
    lengths = [model.encode(p, max_length=args.max_seq_len).shape[1] for p in prompts]
    if any(length <= args.skip_first + 1 for length in lengths):
        raise ValueError('Corpus contains a prompt too short for the position mask')
    metadata = dict(
        paper='https://arxiv.org/html/2609.01924v1',
        model_path=str(args.model.resolve()),
        model_config_sha256=digest(args.model / 'config.json'),
        model_weights_sha256=weight_identity(args.model),
        model_revision=(json.loads((ROOT / 'model_revision.json').read_text())
                        if args.model.resolve() == (ROOT / 'model').resolve() else None),
        model_code_sha256=digest(args.model / 'modeling_ouro.py'),
        implementation_sha256={name: digest(ROOT / name) for name in
                               ('ouro_lens.py', 'ouro_lens_family.py', 'fit_ouro_family.py')},
        estimator_sha256=digest(Path(jlens.__file__).parent / 'fitting.py'),
        corpus_sha256=digest(args.corpus), num_prompts=len(prompts),
        prompts_sha256=hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
        max_seq_len=args.max_seq_len, skip_first=args.skip_first, dim_batch=args.dim_batch,
        source_plan={str(t): s for t, s in plan.items()}, add_bos=True,
        target_location='block output before loop-end normalization',
        torch=str(torch.__version__), transformers=transformers.__version__,
        dtype='bfloat16' if args.device.startswith('cuda') else 'float32', attention='sdpa',
        seed=42, actual_token_lengths=sorted(set(lengths)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / 'fit-manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != metadata:
            raise ValueError('Resume manifest differs; use a new --output-dir')
    elif any(args.output_dir.glob('*.pt')):
        raise ValueError('Refusing to reuse artifacts without their fit manifest')
    save_json(manifest, metadata)
    verification = verify(model, prompts[0])
    save_json(args.output_dir / 'verification.json', verification)
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    lenses = {}
    for target, sources in plan.items():
        print(f'Fitting target {target}: {len(sources)} sources, {len(prompts)} prompts', flush=True)
        lens = jlens.fit(TargetModel(model, target), prompts, source_layers=sources,
                         target_layer=target, dim_batch=args.dim_batch,
                         max_seq_len=args.max_seq_len, skip_first=args.skip_first,
                         checkpoint_path=str(args.output_dir / f'checkpoint-{target}.pt'),
                         checkpoint_every=args.checkpoint_every)
        if lens.n_prompts != len(prompts):
            raise AssertionError('Some prompts were unexpectedly skipped')
        if any(not torch.isfinite(j).all() or j.norm() == 0 for j in lens.jacobians.values()):
            raise AssertionError('Nonfinite or zero fitted Jacobian')
        lenses[target] = lens
    family = OuroLensFamily(lenses, physical_layers=model.physical_layers,
                            recurrent_steps=model.recurrent_steps, metadata=metadata)
    family.save(args.output_dir)
    restored = OuroLensFamily.load(args.output_dir)
    for target, lens in lenses.items():
        for source, matrix in lens.jacobians.items():
            if not torch.equal(matrix, restored.lenses[target].jacobians[source]):
                raise AssertionError('Serialized Jacobian changed')
    save_json(args.output_dir / 'fit-summary.json', dict(
        wall_seconds_this_invocation=time.perf_counter() - start,
        n_prompts=len(prompts), targets=list(plan),
        matrices=sum(len(sources) for sources in plan.values()),
        own_loop_sources=len(family.source_layers), finite_nonzero=True,
        serialization_exact=True, full_paper_reproduction=False,
        peak_allocated_bytes=(torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None)))
    return family


@torch.no_grad()
def readout(model, family, args):
    readouts, actual, ids = family.apply(model, args.prompt)
    raw, _, _ = family.apply(model, args.prompt, use_jacobian=False)

    def top(logits):
        probabilities, indices = logits[-1].softmax(-1).topk(8)
        return [dict(token_id=i, token=model.tokenizer.decode([i]), probability=p)
                for i, p in zip(indices.tolist(), probabilities.tolist())]

    rows = []
    for source, logits in readouts.items():
        target = family.target_for(source)
        reference = actual[target][-1].log_softmax(-1)
        prediction = logits[-1].log_softmax(-1)
        rows.append(dict(source=source, target=target, distance=target-source,
                         top_tokens=top(logits), raw_logit_lens_top_tokens=top(raw[source]),
                         kl_actual_loop_to_lens=F.kl_div(prediction, reference, reduction='sum',
                                                       log_target=True).item()))
    result = dict(prompt=args.prompt, input_token_ids=ids[0].tolist(), rows=rows,
                  loop_outputs={t: top(logits) for t, logits in actual.items()},
                  note='Smoke readouts and bulk KL are not semantic or causal validation.')
    save_json(args.output_dir / 'readout.json', result)
    print(f'Saved {len(rows)} readouts to {args.output_dir / "readout.json"}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['prepare', 'fit', 'apply'], required=True)
    p.add_argument('--model', type=Path, default=ROOT / 'model')
    p.add_argument('--corpus', type=Path, default=ROOT / 'fitting_data/wikitext103-1000.json')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'results/jlens-family')
    p.add_argument('--num-prompts', type=int, default=1000)
    p.add_argument('--dim-batch', type=int, default=4)
    p.add_argument('--max-seq-len', type=int, default=128)
    p.add_argument('--skip-first', type=int, default=16)
    p.add_argument('--checkpoint-every', type=int, default=10)
    p.add_argument('--scope', choices=['all', 'within-loop'], default='all')
    p.add_argument('--offsets', type=lambda s: [int(x) for x in s.split(',')],
                   help='Positive distances from each loop end; overrides --scope')
    p.add_argument('--device', default='cuda')
    p.add_argument('--prompt', default='Fact: The currency used in the country shaped like a boot is')
    args = p.parse_args()
    if min(args.num_prompts, args.dim_batch, args.checkpoint_every) < 1:
        p.error('Counts must be positive')
    if args.skip_first < 0 or args.max_seq_len <= args.skip_first + 1:
        p.error('Need max-seq-len > skip-first + 1 and skip-first >= 0')
    if args.phase == 'prepare':
        prepare(args)
        return
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable; use the GPU-enabled execution environment')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    transformers.set_seed(42)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, attn_implementation='sdpa', device_map=args.device,
        dtype=torch.bfloat16 if args.device.startswith('cuda') else torch.float32).eval()
    model = OuroLensModel(hf, tokenizer)
    try:
        if args.phase == 'fit':
            corpus = json.loads(args.corpus.read_text())
            prompts = corpus['prompts'] if isinstance(corpus, dict) else corpus
            if len(prompts) < args.num_prompts or not all(isinstance(x, str) for x in prompts):
                raise ValueError('Corpus must contain enough text prompts')
            family = fit_family(model, prompts[:args.num_prompts], args)
        else:
            family = OuroLensFamily.load(args.output_dir)
            if family.metadata.get('model_config_sha256') != digest(args.model / 'config.json'):
                raise ValueError('Model configuration differs from fitting')
            if family.metadata.get('model_weights_sha256') != weight_identity(args.model):
                raise ValueError('Model weights differ from fitting')
        readout(model, family, args)
    finally:
        model.close()


if __name__ == '__main__':
    main()
