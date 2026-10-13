"""Bound report cadence for Run 8 recovery; scientific execution is unchanged.

The base recovery reuses 303 saved episodes. Its frozen original runner requests
the complete gallery after every reused episode; those requests would reread
roughly 31 GiB of derived HTML each time. This continuation adapter suppresses
bootstrap requests, publishes after each 16 fresh benchmark cases, and always
publishes the completed 288-case benchmark and final status. Original inventory,
summary, generation, scoring and status updates retain their per-episode cadence.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import traceback

from prepare_deception_run4 import read
from run_deception_run4 import dump, now
import run8_recovery as base

CONTINUATION_VERSION = 'run8-recovery-continue-v1'
REPORTING_EVERY = 16


def validate_policy(root, manifest):
    if manifest['adapter_version'] != base.ADAPTER_VERSION:
        raise ValueError('Continuation requires the original recovery adapter')
    if manifest['continuation_adapter_version'] != CONTINUATION_VERSION:
        raise ValueError('Continuation version differs')
    every = manifest['reporting_every']
    initial = manifest['saved_benchmark_episodes']
    if type(every) is not int or every != REPORTING_EVERY:
        raise ValueError('Continuation report interval must be the bound sixteen cases')
    if type(initial) is not int or initial < 0:
        raise ValueError('Initial saved benchmark count must be a nonnegative integer')
    for name in ('run8_recovery_continue.py', 'test_run8_recovery_continue.py'):
        if name not in manifest['code_sha256']:
            raise ValueError('Continuation code is absent from recovery bindings: ' + name)
    inventory = read(base.project_path(manifest['source_inventory']['path']))
    plan = read(root / 'plan/tasks.json')
    benchmark_paths = {str((root / 'cases' / job['id'] / 'episode.json').relative_to(base.ROOT))
                       for job in plan['benchmark']}
    saved = len(benchmark_paths.intersection(inventory['files']))
    total = len(benchmark_paths)
    if initial != saved or total != 288:
        raise ValueError('Report cadence does not match the bound saved inventory and 288-case plan')
    return dict(initial=initial, every=every, total=total)


@contextmanager
def report_cadence(initial, every=REPORTING_EVERY, total=288):
    """Limit only gallery publication; preserve each scientific call and result."""
    import report_deception_run8 as report
    original = report.render

    def paced(root, extra=None, tokenizer=None):
        count = extra.get('benchmark_saved', initial) if isinstance(extra, dict) else initial
        if type(count) is not int or not 0 <= count <= total:
            raise ValueError('Unexpected benchmark count in report publication request')
        fresh = count - initial
        if count == total or fresh > 0 and fresh % every == 0:
            return original(root, extra=extra, tokenizer=tokenizer)
        return None

    report.render = paced
    try:
        yield
    finally:
        report.render = original


def execute(root, manifest_path, mode='run'):
    root = root.resolve()
    policy = validate_policy(root, read(manifest_path))
    if mode != 'run':
        # An explicitly requested gallery refresh always renders immediately.
        return base.execute(root, manifest_path, mode=mode)
    with report_cadence(**policy):
        return base.execute(root, manifest_path, mode=mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    if args.verify_only and args.report_only:
        parser.error('Select one mode')
    try:
        print(execute(args.output, args.manifest,
              mode='verify' if args.verify_only else 'report' if args.report_only else 'run'))
    except BaseException as error:
        dump(args.output / 'recovery-status.json', dict(utc=now(), stage='stopped_at_error',
             adapter_version=base.ADAPTER_VERSION, continuation_adapter_version=CONTINUATION_VERSION,
             error=repr(error)))
        traceback.print_exc()
        raise
