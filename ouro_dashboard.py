"""Read-only, dependency-free terminal dashboard for the Ouro workspace run."""
import argparse
import collections
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
ANSI = re.compile(r'\x1b\[[0-9;?]*[a-zA-Z]')
COLORS = dict(cyan='36', green='32', yellow='33', red='31', gray='90', bold='1')


def paint(text, color):
    return f'\033[{COLORS[color]}m{text}\033[0m' if sys.stdout.isatty() else str(text)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def tail(path, size=32768):
    try:
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size-size))
            return stream.read().decode(errors='replace').splitlines()
    except FileNotFoundError:
        return []


def duration(seconds):
    if seconds is None:
        return 'warming up'
    hours, remainder = divmod(max(0, int(seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours}h {minutes:02}m' if hours else f'{minutes}m {seconds:02}s'


def bar(fraction, width=24):
    filled = round(max(0, min(1, fraction))*width)
    return paint('━'*filled, 'cyan') + paint('─'*(width-filled), 'gray')


def fit_progress(output, status, manifest):
    active = {job['rank']:job for job in status.get('active_workers', [])}
    complete = set(status.get('completed_ranks', []))
    targets = manifest['targets']
    n = manifest['n_prompts']
    total = n*sum(target+1 for target in targets)
    rows, rates = [], []
    for rank in range(manifest['workers']):
        folder = output / 'workers' / f'worker-{rank}'
        progress = read_json(folder / 'progress.json')
        target, done = progress.get('target', targets[0]), progress.get('n_done', 0)
        checkpoint_target, checkpoint = targets[0], 0
        for line in tail(folder / 'worker.log'):
            match = re.search(r'target=(\d+) prompts=(\d+)/', line)
            if match:
                checkpoint_target, checkpoint = map(int, match.groups())
        state = 'RUN' if rank in active else ('DONE' if rank in complete else 'QUEUED')
        if state == 'QUEUED':
            target, done = checkpoint_target, checkpoint
        units = sum((t+1)*n for t in targets if t < target) + (target+1)*done
        if state == 'DONE':
            units = total
        row = dict(rank=rank, gpu=active.get(rank, {}).get('gpu', '—'), state=state,
            target=target, loop=targets.index(target)+1, done=done, units=units,
            fraction=units/total, remaining=total-units, checkpoint=checkpoint,
            checkpoint_target=checkpoint_target, seconds=progress.get('prompt_seconds'),
            age=time.time()-progress.get('updated_unix', time.time()),
            peak=progress.get('peak_bytes', 0)/2**30)
        rows.append(row)
        if rank in active:
            # Median recent steps avoids checkpoint I/O spikes in the ETA.
            for line in tail(folder / 'metrics.jsonl', 16384)[1:]:
                try:
                    metric = json.loads(line)
                    rates.append(metric['prompt_seconds']/(metric['target']+1))
                except (json.JSONDecodeError, KeyError):
                    pass
    eta = None
    if rates:
        seconds_per_unit = statistics.median(rates)
        loads = {gpu:0. for gpu in status['devices']}
        for row in rows:
            if row['state'] == 'RUN':
                loads[row['gpu']] = row['remaining']*seconds_per_unit
        for row in rows:
            if row['state'] == 'QUEUED':
                gpu = min(loads, key=loads.get)
                loads[gpu] += row['remaining']*seconds_per_unit
        eta = max(loads.values(), default=0)
    return rows, sum(row['units'] for row in rows)/(total*len(rows)), eta


def gpu_lines(devices):
    try:
        result = subprocess.run(['nvidia-smi', '-i', ','.join(devices),
            '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3, check=True)
        lines = []
        for line in result.stdout.splitlines():
            gpu, util, used, total, temperature, watts = [s.strip() for s in line.split(',')]
            percent = int(util)
            lines.append(f' GPU {gpu:>2}  {bar(percent/100, 16)} {percent:3}%  '
                f'{int(used)/1024:4.1f}/{int(total)/1024:4.1f} GiB  {temperature}°C  {float(watts):3.0f} W')
        return lines
    except (OSError, ValueError, subprocess.SubprocessError):
        return [paint(' GPU telemetry unavailable here; the fitting status remains readable.', 'yellow')]


def render(args, width=110):
    status = read_json(args.output / 'status.json')
    manifest = read_json(args.output / 'full-manifest.json')
    stage = status.get('stage', 'starting')
    stale = time.time()-status.get('updated_unix', time.time())
    color = 'red' if stage in ('failed', 'interrupted') or stale > 45 else 'green'
    now = dt.datetime.now().astimezone().strftime('%a %H:%M:%S %Z')
    lines = [paint(' OURO  /  JACOBIAN LENS + WORKSPACE', 'bold') + ' '*5 + paint(now, 'gray'),
             paint('━'*min(width-1, 109), 'cyan'),
             f" {paint(stage.upper().replace('_', ' '), color)}   GPUs {','.join(status.get('devices', []))}"
             f"   supervisor {status.get('pid', '—')}   heartbeat {stale:.0f}s ago"]
    if not manifest:
        return lines + [' Waiting for the fit manifest.']
    rows, fraction, eta = fit_progress(args.output, status, manifest)
    elapsed = time.time()-status.get('started_unix', time.time())
    eta_label = duration(eta) if stage == 'fitting' else (
        'finished' if stage in ('merging', 'workspace_evaluation', 'complete') else 'unavailable')
    lines += [f' Fit   {bar(fraction, min(42, max(12, width-60)))}  {fraction:6.2%}   '
              f'elapsed {duration(elapsed)}',
              f' Fit ETA ~{eta_label}  '
              + paint('(approximate, weighted by target depth; excludes evaluation)', 'gray'), '']
    if stage in ('fitting', 'failed', 'interrupted', 'rescheduling'):
        lines.append(paint(' SHARD  GPU  STATE    LOOP  TARGET  PROMPTS    SHARD FIT              STEP     SAVED', 'gray'))
        for row in rows:
            state = paint(f"{row['state']:6}", 'green' if row['state'] == 'RUN' else 'gray')
            seconds = f"{row['seconds']:.2f}s" if row['state'] == 'RUN' and row['seconds'] else '—'
            saved = f"t{row['checkpoint_target']}:{row['checkpoint']}"
            lines.append(f"  {row['rank']:>2}    {row['gpu']:>2}   {state}    {row['loop']}/4  "
                f" {row['target']:>3}    {row['done']:>4}/{manifest['n_prompts']}  "
                f"{bar(row['fraction'], 13)} {row['fraction']:5.1%}  {seconds:>6}  {saved:>9}")
        lines += ['', paint(' Queued shards show saved work. Eight row partitions share four physical GPUs.', 'gray')]
    if stage in ('workspace_evaluation', 'complete'):
        evaluation = read_json(args.evaluation / 'manifest.json')
        counts = collections.Counter(path.name.split('-')[0] for path in (args.evaluation / 'records').glob('*.json'))
        n_done, n_total = sum(counts.values()), evaluation.get('n_items', 399)
        lines += [f' Workspace trials  {bar(n_done/n_total)}  {n_done}/{n_total} items',
                  ' ' + '  ·  '.join(f'{name}: {count}' for name, count in sorted(counts.items()))]
        for path in sorted(args.evaluation.glob('progress-*.json')):
            progress = read_json(path)
            lines.append(f" Eval worker {progress.get('rank')}: {progress.get('done', 0)}/"
                         f"{progress.get('total', '?')}  {progress.get('last_id', '')}")
        if (args.evaluation / 'REPORT.md').exists():
            lines.append(paint(f' Report: {args.evaluation / "REPORT.md"}', 'green'))
    lines += ['', paint(' GPU TELEMETRY', 'bold')]
    lines += gpu_lines(status.get('devices', ['0', '1', '2', '3'])) if not args.no_gpu else [' (disabled)']
    lines += ['', ' Pipeline   ' + '  →  '.join(paint(label, 'cyan' if value == stage else 'gray') for value, label in
                [('fitting', '1 Fit'), ('merging', '2 Merge + validate'),
                 ('workspace_evaluation', '3 Steering + controls'), ('complete', '4 Report')]),
              paint(' tmux: Ctrl-b then 1 overview / 2 fit logs / 3 eval logs / d detach', 'gray')]
    if status.get('error'):
        lines += [paint(' ERROR: ' + status['error'], 'red')]
    if stale > 45:
        lines += [paint(' Supervisor heartbeat is stale. Check the logs window.', 'red')]
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/jlens-family-1000')
    parser.add_argument('--evaluation', type=Path, default=ROOT / 'results/workspace-full')
    parser.add_argument('--interval', type=float, default=3)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--no-gpu', action='store_true')
    args = parser.parse_args()
    tty = sys.stdout.isatty() and not args.once
    if tty:
        print('\033[?1049h\033[?25l', end='', flush=True)
    try:
        while True:
            size = shutil.get_terminal_size((110, 38))
            lines = render(args, size.columns)
            if tty:
                # Erase each line in place instead of clearing the screen on every refresh.
                sys.stdout.write('\033[H')
                if len(lines) >= size.lines:
                    lines = lines[:size.lines-2] + [paint(' Enlarge this pane (Ctrl-b z) to see the full dashboard.', 'yellow')]
                for line in lines:
                    if len(ANSI.sub('', line)) > size.columns-1:
                        line = ANSI.sub('', line)[:size.columns-2] + '…'
                    sys.stdout.write(line + '\033[K\n')
                sys.stdout.write('\033[J')
                sys.stdout.flush()
            else:
                print('\n'.join(lines), flush=True)
            if args.once:
                break
            time.sleep(max(.5, args.interval))
    except KeyboardInterrupt:
        pass
    finally:
        if tty:
            print('\033[?25h\033[?1049l', end='', flush=True)


if __name__ == '__main__':
    main()
