"""Track unfinished evaluations and execute ready CPU follow-up work.

Run 4 generation remains owned by its existing dispatcher. This queue never
invents annotations, writes scientific phase locks, or repeats failed studies.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
from html import escape
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/evaluation-completion-20261012'
RUN4 = ROOT / 'results/deception-run4-20261011'
RUN8 = ROOT / 'results/deception-run8-20261012'
DISPATCH = ROOT / 'results/evaluation-execution-20261011'
PHASES = ('pilot', 'main_discovery', 'main_held_out', 'secondary_horizons')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def optional(path):
    return read(path) if path.exists() else {}


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'queue.json'
    if path.exists():
        return read(path)
    dispatcher = read(DISPATCH / 'status.json')
    ledger = read(RUN8 / 'plan/deferred-work.json')
    files = [Path(__file__), ROOT / 'test_unfinished_evaluations.py',
             ROOT / 'run8_review.py', ROOT / 'analyze_deception_run4.py',
             ROOT / 'resume_evaluations_run4.py', ROOT / 'continue_evaluations_20261011.py',
             RUN4 / 'execution.json', RUN4 / 'analysis-registration.json',
             RUN8 / 'execution.json', RUN8 / 'plan/deferred-work.json']
    items = [dict(item, disposition_at_enqueue=item['status']) for item in ledger['items']]
    items += [dict(id='run4_' + phase + '_analysis', source='deception-run4-20261011',
                   status='queued_cpu_followup', phase=phase,
                   requires=['All phase jobs generated or excluded', 'All factual reviews resolved'],
                   command=[str(ROOT / '.venv/bin/python'), str(ROOT / 'analyze_deception_run4.py'),
                            '--output', str(RUN4), '--phase', phase]) for phase in PHASES]
    items += [dict(id='run4_discovery_semantic_review', status='queued_review',
                   requires=['main_discovery generation', 'Full-history independent annotations and adjudication'],
                   release_file=str(RUN4 / 'main-analysis-lock.json')),
              dict(id='run4_main_semantic_review', status='queued_review',
                   requires=['main_held_out generation', 'Full-history independent annotations and adjudication'],
                   release_file=str(RUN4 / 'secondary-analysis-lock.json')),
              dict(id='run4_lens_replay', status='conditional_preparation',
                   requires=['Lens-blind semantic review', 'Separate frozen replay manifest and reviewed windows'],
                   command=None, note='No existing replay worker; preparation is queued, GPU inference is not yet runnable.')]
    manifest = dict(schema_version=1, created_utc=utc(), authorized_by='User: queue up other runs that were not properly finished',
                    dispatcher_pid=dispatcher['pid'], dispatcher_gpu=dispatcher['gpu'],
                    run8_pid=read(RUN8 / 'launch-receipt.json')['pid'],
                    items=items, bindings={str(p.relative_to(ROOT)): sha(p) for p in files},
                    policy='Existing dispatcher owns original Run4 generation. Original gates remain mandatory. '
                           'Run5/6 revisions and excluded Run7 families are consolidated into Run8; no duplicate reruns. '
                           'Independent submissions are validated and collected; adjudication and phase locks require actual reviewed evidence.')
    write(path, manifest)
    return manifest


def verify(manifest):
    for relative, expected in manifest['bindings'].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != expected:
            raise ValueError('Queue binding changed: ' + relative)


def phase_state(summary, gate, lock_ready=True):
    if summary.get('generation_complete'):
        return 'awaiting_factual_review' if summary.get('factual_reviews_pending') else 'generation_complete'
    if gate.get('status') == 'failed':
        return 'not_eligible_pilot_gate_failed'
    if gate.get('status') != 'passed':
        return 'queued_awaiting_pilot_gate'
    if not lock_ready:
        return 'queued_awaiting_review_and_analysis_lock'
    return 'queued_with_original_dispatcher'


def dispatcher_alive(pid):
    try:
        raw = Path('/proc') / str(pid) / 'cmdline'
        return b'resume_evaluations_run4.py' in raw.read_bytes()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False


def collect_submissions(output):
    """Ingest complete source-bound files once; a bad submission stops this queue."""
    from run8_review import ingest_reviews
    checkpoint = output / 'collected-submissions.json'
    seen = optional(checkpoint)
    fresh = []
    for path in sorted((RUN8 / 'review-submissions').glob('*/*.json')):
        digest = sha(path)
        if digest not in seen:
            fresh.append(path)
    if fresh:
        result = ingest_reviews(RUN8, fresh)
        for path in fresh:
            seen[sha(path)] = dict(path=str(path.relative_to(ROOT)), collected_utc=utc())
        write(checkpoint, seen)
        write(output / 'review-coverage.json', {study: row['coverage'] for study, row in result['studies'].items()})
    return len(seen)


def analysis_ready(summary):
    return bool(summary.get('generation_complete') and not summary.get('factual_reviews_pending'))


def run_cpu_followups(output, manifest):
    receipt_path = output / 'cpu-followups.json'
    receipts = optional(receipt_path)
    for item in manifest['items']:
        if item.get('status') != 'queued_cpu_followup':
            continue
        phase = item['phase']
        summary = optional(RUN4 / (phase + '-summary.json'))
        if not analysis_ready(summary):
            continue
        # Reviews may change after generation completes. Bind the actual facts.
        inputs = dict(summary_sha256=sha(RUN4 / (phase + '-summary.json')),
                      factual_reviews_sha256=sha(RUN4 / 'fact-reviews.json'))
        previous = receipts.get(phase, {})
        if previous.get('inputs') == inputs and previous.get('status') == 'factual_analysis_complete':
            continue
        with (output / (phase + '-analysis.log')).open('a') as log:
            subprocess.run(item['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'))
        result = read(RUN4 / (phase + '-analysis.json'))
        if result['status'] != 'factual_analysis_complete':
            raise ValueError('Ready phase analysis did not complete: ' + phase)
        receipts[phase] = dict(status=result['status'], completed_utc=utc(), inputs=inputs,
                               result_sha256=sha(RUN4 / (phase + '-analysis.json')))
        write(receipt_path, receipts)
    return receipts


def snapshot(output, manifest, receipts, submission_count):
    gate = optional(RUN4 / 'pilot-gate.json')
    dispatcher = optional(DISPATCH / 'status.json')
    alive = dispatcher_alive(manifest['dispatcher_pid'])
    rows = []
    for phase in PHASES:
        summary = optional(RUN4 / (phase + '-summary.json'))
        lock = ('main-analysis-lock.json' if phase == 'main_held_out'
                else 'secondary-analysis-lock.json' if phase == 'secondary_horizons' else None)
        state = phase_state(summary, gate, not lock or (RUN4 / lock).exists())
        if dispatcher.get('stage') == 'run4_' + phase:
            state = 'running_with_original_dispatcher'
        if not alive and state.startswith('queued'):
            state = 'queued_dispatcher_not_running'
        rows.append(dict(phase=phase, state=state, saved=summary.get('saved', 0), total=summary.get('total'),
                         factual_reviews_pending=summary.get('factual_reviews_pending'),
                         factual_analysis=receipts.get(phase, {}).get('status', 'queued'), required_lock=lock))
    value = dict(utc=utc(), pid=os.getpid(), stage='monitoring_gated_completion_queue',
                 dispatcher_pid=manifest['dispatcher_pid'], dispatcher_alive=alive,
                 dispatcher_gpu=manifest['dispatcher_gpu'], dispatcher_stage=dispatcher.get('stage'),
                 run4_pilot_gate=gate.get('status'), run4_phases=rows,
                 collected_submissions=submission_count,
                 run8=optional(RUN8 / 'status.json'),
                 independent_review_progress={p.stem: read(p) for p in sorted((RUN8 / 'review-workers').glob('reviewer-?.json'))},
                 semantic_locks_written_by_queue=False, duplicate_generation_workers_started=False)
    write(output / 'status.json', value)
    body = ['<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30">',
            '<title>Unfinished evaluation queue</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:32px auto;padding:0 20px}pre{white-space:pre-wrap;overflow-wrap:anywhere}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc;text-align:left}</style>',
            '<h1>Unfinished evaluation queue</h1><p>Run4 generation uses the original dispatcher on GPU ' + escape(str(manifest['dispatcher_gpu'])) + '. Independent review and scientific locks remain required. Run5/6 revisions and excluded Run7 families execute within Run8.</p>',
            '<p><a href="queue.json">Queue manifest</a> · <a href="status.json">Current status</a> · <a href="../deception-run8-20261012/index.html">Run8 progress</a> · <a href="../deception-run4-20261011/index.html">Run4 progress</a></p>',
            '<table><tr><th>Run4 phase</th><th>State</th><th>Saved</th><th>Factual analysis</th></tr>']
    for row in rows:
        body.append('<tr>' + ''.join('<td>' + escape(str(row[k])) + '</td>' for k in ('phase', 'state', 'saved', 'factual_analysis')) + '</tr>')
    body.append('</table><h2>Review progress</h2><pre>' + escape(json.dumps(value['independent_review_progress'], indent=2)) + '</pre>')
    body.append('<h2>Remaining work and prerequisites</h2>')
    for item in manifest['items']:
        body.append('<details><summary>' + escape(item['id']) + '</summary><pre>' + escape(json.dumps(item, indent=2)) + '</pre></details>')
    temporary = output / 'index.html.tmp'
    temporary.write_text(''.join(body))
    temporary.replace(output / 'index.html')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    manifest = prepare(args.output)
    verify(manifest)
    if args.prepare_only:
        print(json.dumps(dict(status='prepared', items=len(manifest['items']), dispatcher_pid=manifest['dispatcher_pid'])))
        return
    with (args.output / '.worker.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                verify(manifest)
                count = collect_submissions(args.output)
                receipts = run_cpu_followups(args.output, manifest)
                result = snapshot(args.output, manifest, receipts, count)
                print(json.dumps({k: result[k] for k in ('utc', 'stage', 'dispatcher_alive', 'run4_pilot_gate', 'collected_submissions')}), flush=True)
            except Exception as error:
                write(args.output / 'status.json', dict(utc=utc(), pid=os.getpid(), stage='stopped_at_error', error=repr(error)))
                raise
            if args.once:
                return
            time.sleep(30)


if __name__ == '__main__':
    main()
