"""Verify, benchmark, fit and apply a pass-aware Jacobian lens for Ouro."""
import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
import time
import torch
import transformers
import jlens
from jlens.hooks import ActivationRecorder
from jlens.fitting import valid_position_mask
from ouro_lens import OuroLensModel

ROOT = Path(__file__).resolve().parent


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')


def verify(model, prompt):
    ids = model.encode(prompt, max_length=64)
    with torch.no_grad():
        with ActivationRecorder(model.layers, at=range(model.n_layers)) as recorder:
            model.forward(ids)
        if len({id(x) for x in recorder.activations.values()}) != model.n_layers:
            raise AssertionError('Distinct recurrent sites were aliased')
        logits = model.unembed(recorder.activations[model.n_layers - 1])
        original = model.hf_model(ids, use_cache=False).logits
        forced_final = model.hf_model(ids, use_cache=False, exit_at_step=model.recurrent_steps - 1).logits
        delta = (logits.float() - original.float()).abs().max().item()
        forced_delta = (logits.float() - forced_final.float()).abs().max().item()
        if delta != 0 or forced_delta != 0:
            raise AssertionError(f'Final readout mismatch: default={delta}, forced_final={forced_delta}')
    return dict(sites=model.n_layers, distinct_activations=len(recorder.activations),
                default_logits_max_absolute_error=delta, final_pass_logits_max_absolute_error=forced_delta,
                all_sites_execute_in_order=True, input_tokens=ids.shape[1])


def benchmark(model, prompt, args):
    torch.cuda.reset_peak_memory_stats()
    ids = model.encode(prompt, max_length=args.max_seq_len).expand(args.dim_batch, -1)
    source_layers = args.sources
    target = model.n_layers - 1
    mask = valid_position_mask(ids.shape[1], skip_first=16).to(ids.device)
    positions = mask.nonzero(as_tuple=True)[0]
    timings = []
    with ActivationRecorder(model.layers, at=[*source_layers, target], start_graph_at=min(source_layers)) as recorder, torch.enable_grad():
        torch.cuda.synchronize(); start = time.perf_counter()
        model.forward(ids)
        torch.cuda.synchronize(); forward_seconds = time.perf_counter() - start
        output = recorder.activations[target]
        sources = [recorder.activations[i] for i in source_layers]
        cotangent = torch.zeros_like(output)
        for block in range(args.benchmark_blocks):
            cotangent.zero_()
            dims = (torch.arange(args.dim_batch, device=ids.device) + block * args.dim_batch) % model.d_model
            cotangent[torch.arange(args.dim_batch, device=ids.device)[:,None], positions[None,:], dims[:,None]] = 1
            torch.cuda.synchronize(); start = time.perf_counter()
            grads = torch.autograd.grad(output, sources, grad_outputs=cotangent, retain_graph=block < args.benchmark_blocks-1)
            norms = []
            for grad in grads:
                rows = grad[:, positions, :].float().mean(1).cpu()
                if not torch.isfinite(rows).all():
                    raise AssertionError('Nonfinite activation Jacobian')
                norms.append(rows.norm().item())
            if any(n == 0 for n in norms):
                raise AssertionError('Zero gradient at a selected source site')
            torch.cuda.synchronize(); elapsed = time.perf_counter() - start
            timings.append(elapsed)
            del grads
            print(f'Backward block {block+1}/{args.benchmark_blocks}: {elapsed:.3f}s', flush=True)
    steady = sum(timings[1:])/len(timings[1:]) if len(timings)>1 else timings[0]
    return dict(forward_seconds=forward_seconds, backward_block_seconds=timings,
                estimated_seconds_per_prompt=forward_seconds + math.ceil(model.d_model/args.dim_batch)*steady,
                full_backward_blocks_per_prompt=math.ceil(model.d_model/args.dim_batch),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), finite_nonzero_gradients=True,
                sources=[model.site_label(i) for i in source_layers], token_count=ids.shape[1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['benchmark','fit','apply'], default='benchmark')
    p.add_argument('--num-prompts', type=int, default=1)
    p.add_argument('--dim-batch', type=int, default=4)
    p.add_argument('--max-seq-len', type=int, default=64)
    p.add_argument('--sources', type=lambda s: [int(x) for x in s.split(',')], default=[47,95,143,190])
    p.add_argument('--benchmark-blocks', type=int, default=8)
    p.add_argument('--output-dir', type=Path, default=ROOT/'results/jlens')
    args = p.parse_args()
    if args.num_prompts < 1 or args.dim_batch < 1 or args.benchmark_blocks < 1 or args.max_seq_len <= 17:
        p.error('Positive counts and max-seq-len > 17 are required')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable: run outside the sandbox on GPU 0.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    transformers.set_seed(42)
    prompts = json.loads((ROOT/'fitting_data/neutral_10.json').read_text())
    if args.num_prompts > len(prompts):
        p.error('Requested more prompts than the pinned fitting corpus contains')
    tokenizer = transformers.AutoTokenizer.from_pretrained(ROOT/'model', trust_remote_code=True)
    hf = transformers.AutoModelForCausalLM.from_pretrained(ROOT/'model', trust_remote_code=True, dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa').eval()
    model = OuroLensModel(hf, tokenizer)
    metadata = dict(model=json.loads((ROOT/'model_revision.json').read_text()),
                    adapter_sha256=hashlib.sha256((ROOT/'ouro_lens.py').read_bytes()).hexdigest(),
                    prompts_sha256=hashlib.sha256((ROOT/'fitting_data/neutral_10.json').read_bytes()).hexdigest(),
                    dim_batch=args.dim_batch, max_seq_len=args.max_seq_len, sources=args.sources,
                    target=model.n_layers-1, dtype='bfloat16', attention='sdpa', use_cache=False, add_bos=True,
                    torch=torch.__version__, transformers=transformers.__version__, gpu=torch.cuda.get_device_name())
    try:
        preflight = verify(model, prompts[0])
        save_json(args.output_dir/'verification.json', preflight)
        print('Actual checkpoint readout verification passed.', flush=True)
        if args.phase == 'benchmark':
            result = benchmark(model, prompts[0], args)
            save_json(args.output_dir/f'benchmark-dim{args.dim_batch}.json', dict(metadata=metadata, **result))
            print(json.dumps(result, indent=2))
        elif args.phase == 'fit':
            manifest_path = args.output_dir/'fit-manifest.json'
            if manifest_path.exists() and json.loads(manifest_path.read_text()) != metadata:
                raise ValueError('Existing fit configuration differs; use a new output directory')
            save_json(manifest_path, metadata)
            checkpoint = args.output_dir/'fit-checkpoint.pt'
            previous_prompts = 0
            if checkpoint.exists():
                state = torch.load(checkpoint, map_location='cpu', weights_only=True)
                previous_prompts = state['n_done']
                del state
                if previous_prompts > args.num_prompts:
                    raise ValueError('Checkpoint already contains more prompts than requested')
                if previous_prompts == args.num_prompts and (args.output_dir/'lens.pt').exists():
                    existing = jlens.JacobianLens.load(str(args.output_dir/'lens.pt'))
                    if existing.n_prompts == previous_prompts:
                        print('Requested fit is already complete; preserving its lens and timing summary.')
                        return
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            lens = jlens.fit(model, prompts=prompts[:args.num_prompts], source_layers=args.sources,
                             dim_batch=args.dim_batch, max_seq_len=args.max_seq_len,
                             checkpoint_path=str(checkpoint), checkpoint_every=1, resume=True)
            for site, matrix in lens.jacobians.items():
                if not torch.isfinite(matrix).all():
                    raise AssertionError(f'Nonfinite fitted matrix at site {site}')
            target = args.output_dir/'lens.pt'
            lens.save(str(target))
            restored = jlens.JacobianLens.load(str(target))
            if not all(torch.isfinite(J).all() for J in restored.jacobians.values()):
                raise AssertionError('Serialized fp16 lens overflowed')
            summary = dict(metadata=metadata, n_prompts=lens.n_prompts, requested_prompts=args.num_prompts,
                           previous_prompts=previous_prompts, newly_fitted_prompts=lens.n_prompts-previous_prompts,
                           invocation_seconds=time.perf_counter()-start,
                           peak_allocated_bytes=torch.cuda.max_memory_allocated(), lens_bytes=target.stat().st_size,
                           matrix_shape=[model.d_model,model.d_model],
                           source_labels=[model.site_label(i) for i in lens.source_layers],
                           jacobian_norms={str(i):J.norm().item() for i,J in lens.jacobians.items()})
            save_json(args.output_dir/f'fit-summary-{lens.n_prompts}.json', summary)
            print(json.dumps(summary,indent=2))
        else:
            lens = jlens.JacobianLens.load(str(args.output_dir/'lens.pt'))
            holdout = json.loads((ROOT/'fitting_data/readout_holdout.json').read_text())
            results = []
            for prompt in holdout:
                lens_logits, model_logits, ids = lens.apply(model, prompt, positions=[-1], max_seq_len=128)
                with torch.no_grad():
                    # Match the one-position unembedding GEMM shape in lens.apply.
                    actual_logits = model.hf_model(ids, use_cache=False, logits_to_keep=1).logits[0, -1].float().cpu()
                torch.testing.assert_close(model_logits[0], actual_logits, rtol=0, atol=0)
                if not all(torch.isfinite(logits).all() for logits in lens_logits.values()):
                    raise AssertionError('Nonfinite held-out lens readout')
                def top(logits):
                    values, indices = logits[0].topk(10)
                    return [dict(token_id=i, token=tokenizer.decode([i]), logit=v) for i,v in zip(indices.tolist(),values.tolist())]
                results.append(dict(prompt=prompt, input_ids=ids[0].tolist(), final_readout_matches_checkpoint=True, model_top10=top(model_logits),
                                    lens_top10={str(site):top(logits) for site,logits in lens_logits.items()}))
            save_json(args.output_dir/'heldout-readouts.json',dict(n_fitting_prompts=lens.n_prompts,results=results))
            print('Saved held-out top-token readouts.')
    finally:
        model.close()


if __name__ == '__main__':
    main()
