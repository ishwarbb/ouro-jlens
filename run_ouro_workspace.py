"""Resume logical row shards on a bounded GPU pool, then merge and evaluate."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def read_json(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def process_identity(pid):
    """Start ticks guard against mistaking a reused PID for a live worker."""
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if stat[0] == 'Z' else stat[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def existing_worker(pid, rank, output):
    identity = process_identity(pid)
    if identity is None:
        return None
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
        if str(ROOT / 'ouro_full_fit.py') not in command:
            return None
        if command[command.index('--phase') + 1] != 'worker':
            return None
        if int(command[command.index('--rank') + 1]) != rank:
            return None
        if Path(command[command.index('--output') + 1]).resolve() != output:
            return None
        environment = Path(f'/proc/{pid}/environ').read_bytes().decode().split('\0')
        gpu = next(x.split('=', 1)[1] for x in environment if x.startswith('CUDA_VISIBLE_DEVICES='))
        return dict(rank=rank, pid=pid, gpu=gpu, identity=identity, adopted=True)
    except (FileNotFoundError, ProcessLookupError, ValueError, IndexError, StopIteration):
        return None


def allocate(pending, devices, active):
    """One logical shard per free physical GPU; partition count stays unchanged."""
    occupied = {job['gpu'] for job in active.values()}
    return list(zip(pending, (gpu for gpu in devices if gpu not in occupied)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/jlens-family-1000')
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--adopt', type=Path, help='Optional receipt for a live resource change')
    parser.add_argument('--fit-only', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    devices = [gpu.strip() for gpu in args.gpus.split(',')]
    if not all(devices) or len(set(devices)) != len(devices):
        raise ValueError('Supply distinct nonempty GPU IDs')
    manifest_path = args.output / 'full-manifest.json'
    manifest = read_json(manifest_path)
    fingerprint = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    ranks = list(range(manifest['workers']))
    lock = (args.output / 'supervisor.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise SystemExit('Another supervisor holds the run lock; use its dashboard.')
    (args.output / 'supervisor.pid').write_text(str(os.getpid()) + '\n')
    receipt = read_json(args.adopt) if args.adopt else None
    previous = read_json(args.output / 'status.json') or {}
    start = (receipt or previous).get('started_unix', time.time())
    active, completed, streams = {}, set(), []
    codes = [None] * len(ranks)
    stage, stage_job = 'fitting', None

    def workdir(rank):
        return args.output / 'workers' / f'worker-{rank}'

    def complete(rank):
        record = read_json(workdir(rank) / 'complete.json')
        return bool(record and record.get('manifest_sha256') == fingerprint and record.get('rank') == rank)

    def snapshot(error=None):
        value = dict(stage=stage, pid=os.getpid(), started_unix=start,
            updated_unix=time.time(), devices=devices, logical_shards=len(ranks),
            completed_ranks=sorted(completed), queued_ranks=pending,
            active_workers=[{k:v for k,v in job.items() if k != 'process'} for job in active.values()],
            worker_pids=[active[rank]['pid'] if rank in active else None for rank in ranks],
            exit_codes=codes, progress=[read_json(workdir(rank) / 'progress.json') for rank in ranks])
        if error:
            value['error'] = error
        if stage_job:
            value['stage_pid'] = stage_job.pid
        if stage == 'complete':
            value.update(completed_unix=time.time(), wall_seconds=time.time()-start, fit_only=args.fit_only)
        save_json(args.output / 'status.json', value)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}; checkpoints are preserved')

    previous_signals = {number:signal.signal(number, interrupted) for number in (signal.SIGTERM, signal.SIGINT)}
    pending = []
    try:
        # Adopt survivors after a supervisor restart, avoiding duplicate GPU jobs.
        for rank in ranks:
            progress = read_json(workdir(rank) / 'progress.json') or {}
            adopted = next((x for x in (receipt or {}).get('workers', []) if x['rank'] == rank), {})
            pid = adopted.get('pid', progress.get('pid'))
            job = existing_worker(pid, rank, args.output) if pid else None
            if job:
                if job['gpu'] not in devices or job['gpu'] in {j['gpu'] for j in active.values()}:
                    raise ValueError(f"Existing worker {rank} on GPU {job['gpu']} conflicts with the GPU pool")
                active[rank] = job
                print(f"Adopted shard {rank}: GPU {job['gpu']}, PID {pid}", flush=True)
            elif complete(rank):
                completed.add(rank)
                codes[rank] = 0
            else:
                pending.append(rank)
        (args.output / 'supervisor-error.json').unlink(missing_ok=True)
        while len(completed) < len(ranks):
            for rank, job in list(active.items()):
                process = job.get('process')
                running = process.poll() is None if process else process_identity(job['pid']) == job['identity']
                if running:
                    continue
                code = process.returncode if process else (0 if complete(rank) else -1)
                codes[rank] = code
                if code != 0 or not complete(rank):
                    raise RuntimeError(f'Shard {rank} exited with code {code} without valid completion; checkpoints preserved')
                completed.add(rank)
                del active[rank]
                print(f"Completed shard {rank}; GPU {job['gpu']} available", flush=True)
            for rank, gpu in allocate(pending, devices, active):
                workdir(rank).mkdir(parents=True, exist_ok=True)
                stream = (workdir(rank) / 'worker.log').open('a', buffering=1)
                streams.append(stream)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='2',
                           PYTORCH_ALLOC_CONF='expandable_segments:True')
                job = subprocess.Popen([sys.executable, '-u', str(ROOT / 'ouro_full_fit.py'),
                    '--phase', 'worker', '--rank', str(rank), '--output', str(args.output)],
                    cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                active[rank] = dict(rank=rank, pid=job.pid, gpu=gpu,
                    identity=process_identity(job.pid), adopted=False, process=job)
                pending.remove(rank)
                print(f'Started/resumed shard {rank}: GPU {gpu}, PID {job.pid}', flush=True)
            snapshot()
            if active:
                time.sleep(5)

        def run_stage(command, environment=None):
            nonlocal stage_job
            stage_job = subprocess.Popen(command, cwd=ROOT, env=environment, start_new_session=True)
            while stage_job.poll() is None:
                snapshot()
                time.sleep(5)
            if stage_job.returncode:
                raise RuntimeError(f'{stage} exited with code {stage_job.returncode}')
            stage_job = None

        stage = 'merging'
        run_stage([sys.executable, '-u', str(ROOT / 'ouro_full_fit.py'),
                   '--phase', 'merge', '--output', str(args.output)])
        if not args.fit_only:
            stage = 'workspace_evaluation'
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(devices), OMP_NUM_THREADS='2')
            run_stage([sys.executable, '-u', str(ROOT / 'workspace_reproduction.py'),
                       '--family', str(args.output), '--gpus', ','.join(devices)], env)
        stage = 'complete'
        snapshot()
    except BaseException as exc:
        stage = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        for job in active.values():
            if process_identity(job['pid']) == job['identity']:
                try:
                    os.kill(job['pid'], signal.SIGTERM)
                except ProcessLookupError:
                    pass
        if stage_job and stage_job.poll() is None:
            os.killpg(stage_job.pid, signal.SIGTERM)
        snapshot(repr(exc))
        save_json(args.output / 'supervisor-error.json', dict(error=repr(exc), updated_unix=time.time()))
        raise
    finally:
        for job in active.values():
            if job.get('process'):
                try:
                    job['process'].wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        if stage_job:
            try:
                stage_job.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        for stream in streams:
            stream.close()
        lock.close()
        for number, handler in previous_signals.items():
            signal.signal(number, handler)


if __name__ == '__main__':
    main()
