"""Run a bounded trial or a complete fixed-response replay cohort."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .common import DEFAULT_ASSETS, DEFAULT_INVENTORY, PACKAGE, digest, save, sha, write_jsonl
from .data import load_sample, select
from .inventory import TASKS, WORDS


def configure_device(device, gpu_index, *, brief_gpu8=False):
    if device == 'cpu':
        if gpu_index is not None:
            raise ValueError('A CPU run must not specify a GPU')
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        if gpu_index is None:
            raise ValueError('CUDA requires an explicit physical --gpu-index')
        if gpu_index == 8 and not brief_gpu8:
            raise ValueError('GPU 8 requires the explicitly bounded --brief-gpu8 trial')
        if gpu_index < 0:
            raise ValueError('Invalid GPU index')
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        uuid = subprocess.check_output(['nvidia-smi', '-i', str(gpu_index),
            '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
        if not uuid.startswith('GPU-') or '\n' in uuid:
            raise ValueError('Could not resolve a unique physical GPU UUID')
        os.environ['CUDA_VISIBLE_DEVICES'] = uuid


def enforce_brief_trial(args, sample_count):
    if args.brief_gpu8:
        if args.device != 'cuda' or args.gpu_index != 8:
            raise ValueError('--brief-gpu8 is only for a brief trial on GPU 8')
        if sample_count > 2 or args.window != 'early' or args.loops > 4 or args.max_input_tokens > 512:
            raise ValueError('Brief GPU 8 trial permits at most two early-window responses, four loops and 512 input tokens')


def select_precision(dtype, validate):
    if dtype == 'bfloat16' and not validate:
        raise ValueError('BF16 chunking failed validation in the GPU trial; explicit BF16 runs require --validate before measurement')
    return dtype or 'float32'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument('--assets', type=Path, default=DEFAULT_ASSETS)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--gpu-index', type=int)
    parser.add_argument('--brief-gpu8', action='store_true', help='Explicitly allow a bounded two-response GPU 8 trial with a 60-second deadline')
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'])
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--loops', type=int, default=4)
    parser.add_argument('--tasks', nargs='+', choices=TASKS)
    parser.add_argument('--conditions', nargs='+', choices=['standard', 'direct_conceal', 'monitor_aware'])
    parser.add_argument('--scenarios-per-task', type=int, default=1, help='0 selects all available scenarios')
    parser.add_argument('--rollouts', type=int, default=2, help='0 selects all saved rollout indices')
    parser.add_argument('--window', choices=['early', 'full'], default='early')
    parser.add_argument('--chunk-size', type=int, default=128)
    parser.add_argument('--max-input-tokens', type=int, default=8192, help='Oversized sequences remain explicitly unmeasured')
    parser.add_argument('--validate', action='store_true', help='Compare full/chunked fixed-token replay before measuring')
    args = parser.parse_args()
    configure_device(args.device, args.gpu_index, brief_gpu8=args.brief_gpu8)  # Before importing torch/transformers.
    if args.brief_gpu8:
        def deadline(signum, frame):
            raise TimeoutError('Brief GPU 8 trial exceeded its 60-second deadline')
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(60)
    if args.chunk_size < 1 or args.threads < 1 or args.scenarios_per_task < 0 or args.rollouts < 0:
        raise ValueError('Invalid positive size or selection limit')
    import torch
    import transformers
    from .assay import PHYSICAL_LAYERS, source_positions, token_variants, tokenize
    from .engine import Engine
    from .readout import measure
    from .aggregate import aggregate

    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = select_precision(args.dtype, args.validate)
    samples, pairs = select(args.inventory, loops=args.loops, tasks=args.tasks,
        scenarios_per_task=args.scenarios_per_task, rollouts=args.rollouts, conditions=args.conditions)
    enforce_brief_trial(args, len(samples))
    for sample in samples:
        sample['key'] = digest([sample['id'], sample['prompt']['record_sha256']])[:24]
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def status(stage, **fields):
        result = dict(stage=stage, device=args.device, gpu_index=args.gpu_index,
            cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], pid=os.getpid(),
            planned_responses=len(samples), elapsed_seconds=round(time.monotonic()-started, 2),
            updated_utc=datetime.now(timezone.utc).isoformat(), **fields)
        save(args.output / 'status.json', result)
        print(json.dumps(result), flush=True)

    status('loading_verified_assets')
    engine = Engine(args.assets, device=args.device, dtype=dtype)
    variants = token_variants(engine.tokenizer)
    frozen = dict(model=engine.identity['model'], model_revision=engine.identity['model_revision'],
        lens=engine.identity['lens'], lens_revision=engine.identity['lens_revision'],
        assets_manifest_sha256=sha(args.assets / 'manifest.json'),
        inventory_sha256=sha(args.inventory / 'inventory.json'),
        selected_samples_sha256=digest(samples), pairs_sha256=digest(pairs),
        loops=args.loops, window=args.window, chunk_size=args.chunk_size, max_input_tokens=args.max_input_tokens,
        device=args.device, dtype=dtype, threads=args.threads, torch=torch.__version__, transformers=transformers.__version__,
        gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'] if args.device == 'cuda' else None,
        gpu_name=torch.cuda.get_device_name() if args.device == 'cuda' else None,
        brief_gpu8=args.brief_gpu8, wall_time_limit_seconds=60 if args.brief_gpu8 else None,
        token_variants=variants, unsupported_words=[w for w in WORDS if not variants[w]],
        layers=PHYSICAL_LAYERS, thresholds=[1, 5, 10],
        code_sha256={p.name: sha(p) for p in sorted(PACKAGE.glob('*.py')) if not p.name.startswith('test_')},
        reconstruction='Fixed author checkpoint; original generation checkpoint and token IDs unverified.',
        selection='Hash-ordered source scenarios, then saved response indices; no lens-based selection.',
        exact_historical_trace=False, no_generation=True,
        subset_trial=bool(args.scenarios_per_task or args.rollouts),
        unsupported_scope='1.4B and readouts at loops 5+ require additional matching author matrices.')
    protocol_path = args.output / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != frozen:
        raise ValueError('Existing output has different code, assets, selection or assay; use a new output directory')
    save(protocol_path, frozen)
    protocol_hash = sha(protocol_path)
    write_jsonl(args.output / 'samples.jsonl', samples)
    write_jsonl(args.output / 'pairs.jsonl', pairs)
    if args.validate:
        status('validating_fixed_sequence_replay')
        try:
            messages, response = load_sample(samples[0])
            encoded = tokenize(engine.tokenizer, messages, response)
            validation = engine.validate(encoded['prompt_ids'] + encoded['response_ids'])
            validation['protocol_sha256'] = protocol_hash
            save(args.output / 'replay-validation.json', validation)
        except BaseException as exc:
            save(args.output / 'replay-validation.json', dict(status='failed', protocol_sha256=protocol_hash,
                error=f'{type(exc).__name__}: {exc}'))
            status('failed', completed=0, error=f'{type(exc).__name__}: {exc}')
            raise
    completed = 0
    try:
        for sample in samples:
            path = args.output / 'readouts' / (sample['key'] + '.json')
            if path.exists():
                old = json.loads(path.read_text())
                if old['protocol_sha256'] != protocol_hash or old['id'] != sample['id']:
                    raise ValueError('Stale cached readout')
                completed += 1
                continue
            messages, response = load_sample(sample)
            encoded = tokenize(engine.tokenizer, messages, response)
            ids = encoded['prompt_ids'] + encoded['response_ids']
            positions = source_positions(len(encoded['prompt_ids']), len(encoded['response_ids']), args.window)
            needed = positions[-1] + 1
            record = dict(id=sample['id'], protocol_sha256=protocol_hash,
                source_record_sha256=sample['prompt']['record_sha256'], response_sha256=sample['prompt']['response_sha256'][sample['response_index']],
                reconstructed_token_ids_sha256=digest(ids), prompt_tokens=len(encoded['prompt_ids']),
                saved_response_tokens=len(encoded['response_ids']), required_forward_tokens=needed,
                decoded_response_roundtrip=encoded['roundtrip'], replacement_character=encoded['replacement_character'],
                has_reasoning_close=encoded['reasoning_close'] >= 0,
                supported_words=[w for w in WORDS if variants[w]],
                prompt_mentions=encoded['prompt_mentions'], response_mentions=encoded['response_mentions'],
                source_condition=sample['prompt']['condition'], task=sample['prompt']['task'])
            save(args.output / 'tokens' / (sample['key'] + '.json'), dict(prompt_ids=encoded['prompt_ids'],
                response_ids=encoded['response_ids'], protocol_sha256=protocol_hash,
                status='Reconstructed from saved text; original IDs unavailable'))
            status('replaying', completed=completed, sample=sample['id'], forward_tokens=needed)
            tick = time.monotonic()
            if not encoded['roundtrip']:
                record.update(status='unmeasured', reason='response_text_roundtrip_failed')
            elif needed > min(args.max_input_tokens, engine.model.config.max_position_embeddings):
                record.update(status='unmeasured', reason='input_exceeds_declared_replay_limit')
            else:
                record.update(measure(engine, encoded, variants, loops=args.loops, chunk_size=args.chunk_size, window=args.window), status='measured')
            record['seconds'] = round(time.monotonic() - tick, 3)
            save(path, record)
            completed += 1
            status('replaying', completed=completed, last_status=record['status'], last_seconds=record['seconds'])
        primary = aggregate(args.output)
        status('complete', completed=completed, primary_analysis='primary-analysis.json',
               primary_groups=len(primary['groups']) if primary else 0,
               peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else None)
    except BaseException as exc:
        status('failed', completed=completed, error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
