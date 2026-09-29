"""Exact row-sharded Jacobian fitting with GPU accumulation for Ouro.

The cotangents and position reductions match jlens.fitting. Moving the
per-row CPU transfer out of the backward loop avoids thousands of GPU syncs.
Rows can be partitioned across workers without changing prompt averaging.
"""
from __future__ import annotations
import argparse
import json
import math
import hashlib
import os
from pathlib import Path
import time

import torch
import transformers
from jlens.hooks import ActivationRecorder
from jlens.fitting import valid_position_mask, jacobian_for_prompt
from ouro_lens import OuroLensModel
from fit_ouro_family import ROOT, save_json, digest, weight_identity, verify, WIKITEXT_REVISION
from ouro_lens_family import OuroLensFamily
from jlens import JacobianLens
from jlens.fitting import _atomic_save

DEFAULT_OUTPUT = ROOT/'results/jlens-family-1000'
DEFAULT_CORPUS = ROOT/'fitting_data/wikitext103-1000-128.json'


def prepare_corpus(args):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    if args.corpus.exists():
        raise FileExistsError(args.corpus)
    tokenizer = transformers.AutoTokenizer.from_pretrained(ROOT/'model', trust_remote_code=True)
    prompts, records = [], []
    for shard in range(2):
        name = f'wikitext-103-raw-v1/train-{shard:05d}-of-00002.parquet'
        path = hf_hub_download('Salesforce/wikitext', name, repo_type='dataset',
                              revision=WIKITEXT_REVISION)
        row = 0
        for batch in pq.ParquetFile(path).iter_batches(columns=['text']):
            for text in batch.column('text').to_pylist():
                if len(text.strip()) >= 600 and len(tokenizer.encode(text, add_special_tokens=False)) >= 127:
                    prompts.append(text)
                    records.append(dict(shard=name, row=row))
                row += 1
                if len(prompts) == 1000:
                    break
            if len(prompts) == 1000:
                break
        if len(prompts) == 1000:
            break
    if len(prompts) != 1000:
        raise ValueError('Could not prepare 1000 full-length prompts')
    save_json(args.corpus, dict(prompts=prompts, metadata=dict(dataset='Salesforce/wikitext',
        revision=WIKITEXT_REVISION, configuration='wikitext-103-raw-v1', split='train',
        min_chars=600, min_tokens_without_bos=127, records=records)))
    print(f'Prepared {len(prompts)} prompts of 128 tokens including BOS', flush=True)


def initialize(args):
    prompts = json.loads(args.corpus.read_text())['prompts']
    if len(prompts) != 1000:
        raise ValueError('The full fit requires 1000 prompts')
    manifest = dict(format_version=1, n_prompts=1000, d_model=2048,
        physical_layers=48, recurrent_steps=4, targets=[47,95,143,191],
        source_scope='all upstream layers', workers=args.workers, dim_batch=args.dim_batch,
        max_seq_len=128, skip_first=16, seed=42, dtype='bfloat16', attention='sdpa',
        corpus_path=str(args.corpus.resolve()), corpus_sha256=digest(args.corpus),
        dataset_revision=WIKITEXT_REVISION, prompt_selection='first qualifying records; >=600 characters and >=127 tokens',
        model_config_sha256=digest(ROOT/'model/config.json'),
        model_weights_sha256=weight_identity(ROOT/'model'),
        model_revision=json.loads((ROOT/'model_revision.json').read_text()),
        implementation_sha256={name:digest(ROOT/name) for name in
            ['ouro_full_fit.py','ouro_lens.py','ouro_lens_family.py','fit_ouro_family.py']},
        reference_estimator_sha256=digest(ROOT/'jacobian-lens/jlens/fitting.py'),
        torch=str(torch.__version__), transformers=transformers.__version__)
    path = args.output/'full-manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Existing full-fit manifest differs; preserve it and use another directory')
    save_json(path, manifest)
    print(f'Initialized {path}', flush=True)


def checked_manifest(args):
    path = args.output/'full-manifest.json'
    manifest = json.loads(path.read_text())
    for name, expected in manifest['implementation_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError(f'Implementation changed during the fit: {name}')
    corpus = Path(manifest['corpus_path'])
    if digest(corpus) != manifest['corpus_sha256']:
        raise ValueError('Fitting corpus changed')
    return manifest, digest(path)


def worker(args):
    manifest, manifest_hash = checked_manifest(args)
    rank = args.rank
    workers = manifest['workers']
    if not 0 <= rank < workers or 2048 % workers:
        raise ValueError('Invalid worker partition')
    lo, hi = rank*2048//workers, (rank+1)*2048//workers
    workdir = args.output/'workers'/f'worker-{rank}'
    workdir.mkdir(parents=True, exist_ok=True)
    model = load_model()
    prompts = json.loads(Path(manifest['corpus_path']).read_text())['prompts']
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    try:
        save_json(workdir/'verification.json', verify(model, prompts[0]))
        for target in manifest['targets']:
            shape = (target, hi-lo, model.d_model)
            path = workdir/f'target-{target}-sum.pt'
            header = dict(manifest_sha256=manifest_hash, target=target,
                          row_start=lo, row_end=hi, rank=rank)
            if path.exists():
                state = torch.load(path, weights_only=True, map_location='cpu')
                if state['header'] != header or tuple(state['sum'].shape) != shape:
                    raise ValueError('Worker checkpoint mismatch')
                total, done = state['sum'], state['n_done']
            else:
                total, done = torch.zeros(shape, dtype=torch.float32), 0
            def checkpoint():
                _atomic_save(dict(header=header, sum=total, n_done=done),str(path))
            for index in range(done, len(prompts)):
                tick = time.perf_counter()
                per_prompt, length, valid_count = row_jacobian(model, prompts[index], list(range(target)),
                    target, row_start=lo, row_end=hi, dim_batch=manifest['dim_batch'])
                if length != 128 or valid_count != 111:
                    raise AssertionError('A full-fit prompt does not have the declared geometry')
                if not torch.isfinite(per_prompt).all():
                    raise FloatingPointError(f'Nonfinite Jacobian at target {target}, prompt {index}')
                change = None
                if done:
                    previous = total/done
                    numerator = (per_prompt-previous).flatten(1).norm(dim=1)
                    denominator = (done+1)*previous.flatten(1).norm(dim=1)
                    change = (numerator/denominator.clamp_min(1e-30)).max().item()
                    del previous, numerator, denominator
                total += per_prompt
                done = index+1
                del per_prompt
                elapsed = time.perf_counter()-tick
                progress = dict(rank=rank, target=target, n_done=done, n_prompts=len(prompts),
                    row_start=lo, row_end=hi, prompt_seconds=elapsed, mean_rel_shift=change,
                    elapsed_seconds=time.time()-started, peak_bytes=torch.cuda.max_memory_allocated(),
                    gpu=torch.cuda.get_device_name(), pid=os.getpid(), updated_unix=time.time())
                save_json(workdir/'progress.json', progress)
                with (workdir/'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(progress)+'\n')
                if done % 50 == 0 or done == len(prompts):
                    checkpoint()
                    print(f'worker={rank} target={target} prompts={done}/1000 seconds={elapsed:.3f} shift={change}', flush=True)
                if done == 500:
                    _atomic_save(dict(header=header, sum=total, n_done=done),
                                 str(workdir/f'target-{target}-half.pt'))
            checkpoint()
            del total
        save_json(workdir/'complete.json',dict(manifest_sha256=manifest_hash, rank=rank,
            elapsed_seconds=time.time()-started, peak_bytes=torch.cuda.max_memory_allocated()))
    finally:
        model.close()


def merge(args):
    manifest, manifest_hash = checked_manifest(args)
    workers = manifest['workers']
    lenses, convergence = {}, {}
    for target in manifest['targets']:
        matrix = torch.empty(target,2048,2048,dtype=torch.float32)
        half = torch.empty_like(matrix)
        for rank in range(workers):
            lo, hi = rank*2048//workers,(rank+1)*2048//workers
            base = args.output/'workers'/f'worker-{rank}'
            expected = dict(manifest_sha256=manifest_hash,target=target,row_start=lo,row_end=hi,rank=rank)
            for name, destination, count in [('sum',matrix,1000),('half',half,500)]:
                state = torch.load(base/f'target-{target}-{name}.pt',weights_only=True,map_location='cpu')
                if state['header'] != expected or state['n_done'] != count or state['sum'].shape != (target,hi-lo,2048):
                    raise ValueError(f'Incomplete or incompatible worker {rank}, target {target}')
                destination[:,lo:hi] = state['sum']/count
                del state
        if not torch.isfinite(matrix).all() or not (matrix.flatten(1).norm(dim=1)>0).all():
            raise FloatingPointError('Invalid merged means')
        changes = (matrix-half).flatten(1).norm(dim=1)/matrix.flatten(1).norm(dim=1)
        cosines = torch.nn.functional.cosine_similarity(matrix.flatten(1),half.flatten(1),dim=1)
        convergence[target] = dict(relative_change_500_to_1000=changes.tolist(),
                                    cosine_500_vs_1000=cosines.tolist())
        del half
        lenses[target] = JacobianLens({s:matrix[s] for s in range(target)},n_prompts=1000,d_model=2048)
    family = OuroLensFamily(lenses,physical_layers=48,recurrent_steps=4,metadata=manifest)
    family.save(args.output)
    restored = OuroLensFamily.load(args.output)
    for target,lens in lenses.items():
        for source,value in lens.jacobians.items():
            torch.testing.assert_close(value,restored.lenses[target].jacobians[source],rtol=0,atol=0)
    save_json(args.output/'convergence.json',convergence)
    save_json(args.output/'fit-summary.json',dict(n_prompts=1000,matrices=476,targets=list(lenses),
        own_loop_sources=188,finite_nonzero=True,serialization_exact=True,
        estimator='reference position reduction; disjoint exact output-row partition',
        full_paper_reproduction=False,completed_unix=time.time()))
    print('Merged and verified all 476 full-fit matrices',flush=True)


def row_jacobian(model, prompt, sources, target, *, row_start=0, row_end=None,
                 dim_batch=8, max_seq_len=128, skip_first=16):
    row_end = model.d_model if row_end is None else row_end
    if not 0 <= row_start < row_end <= model.d_model:
        raise ValueError('Invalid output row interval')
    sources = sorted(set(sources))
    if not sources or sources[0] < 0 or sources[-1] >= target or target >= model.n_layers:
        raise ValueError('Invalid source/target geometry')
    if dim_batch < 1:
        raise ValueError('dim_batch must be positive')
    ids = model.encode(prompt, max_length=max_seq_len)
    valid = valid_position_mask(ids.shape[1], skip_first=skip_first)
    n_valid = int(valid.sum())
    device = ids.device
    matrices = torch.empty(len(sources), row_end-row_start, model.d_model,
                           device=device, dtype=torch.float32)
    batch = torch.arange(dim_batch, device=device)
    positions = valid.nonzero(as_tuple=True)[0].to(device)
    with ActivationRecorder(model.layers, at=[*sources, target],
                            start_graph_at=min(sources)) as recorder, torch.enable_grad():
        model.forward(ids.expand(dim_batch, -1), stop_at=target)
        output = recorder.activations[target]
        inputs = [recorder.activations[s] for s in sources]
        cotangent = torch.zeros_like(output)
        for start in range(row_start, row_end, dim_batch):
            count = min(dim_batch, row_end-start)
            cotangent.zero_()
            cotangent[batch[:count, None], positions[None, :], start+batch[:count, None]] = 1
            gradients = torch.autograd.grad(output, inputs, grad_outputs=cotangent,
                                            retain_graph=start+count < row_end)
            for index, grad in enumerate(gradients):
                matrices[index, start-row_start:start-row_start+count].copy_(
                    grad[:count, positions, :].float().mean(dim=1))
            del gradients
    return matrices.cpu(), ids.shape[1], n_valid


def load_model(path=ROOT/'model'):
    tokenizer = transformers.AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, dtype=torch.bfloat16, device_map='cuda',
        attn_implementation='sdpa').eval()
    return OuroLensModel(hf, tokenizer)


def benchmark(args):
    model = load_model()
    prompt = json.loads((ROOT/'fitting_data/wikitext103-1000.json').read_text())['prompts'][0]
    sources = list(range(args.target))
    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        matrices, length, count = row_jacobian(model, prompt, sources, args.target,
            row_start=args.row_start, row_end=args.row_end, dim_batch=args.dim_batch)
        elapsed = time.perf_counter()-start
        result = dict(target=args.target, dim_batch=args.dim_batch,
            row_start=args.row_start, row_end=args.row_end, seconds=elapsed,
            peak_bytes=torch.cuda.max_memory_allocated(), finite=bool(torch.isfinite(matrices).all()),
            nonzero=bool((matrices.flatten(1).norm(dim=1)>0).all()),
            seq_len=length, n_valid=count)
        if args.reference:
            reference, _, _ = jacobian_for_prompt(model, prompt, [sources[0],sources[-1]],
                target_layer=args.target, dim_batch=args.dim_batch, max_seq_len=128, skip_first=16)
            diffs = {}
            for i in (0,len(sources)-1):
                expected = reference[sources[i]][args.row_start:args.row_end]
                torch.testing.assert_close(matrices[i], expected, rtol=0, atol=0)
                diffs[sources[i]] = (matrices[i]-expected).abs().max().item()
            result['reference_max_abs_error'] = diffs
        save_json(args.output, result)
        print(json.dumps(result), flush=True)
    finally:
        model.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare','initialize','worker','merge','benchmark'], default='benchmark')
    parser.add_argument('--target', type=int, default=191)
    parser.add_argument('--row-start', type=int, default=0)
    parser.add_argument('--row-end', type=int, default=256)
    parser.add_argument('--dim-batch', type=int, default=8)
    parser.add_argument('--reference', action='store_true')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--corpus', type=Path, default=DEFAULT_CORPUS)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--rank', type=int, default=0)
    args = parser.parse_args()
    transformers.set_seed(42)
    {'prepare':prepare_corpus,'initialize':initialize,'worker':worker,'merge':merge,
     'benchmark':benchmark}[args.phase](args)


if __name__ == '__main__':
    main()
