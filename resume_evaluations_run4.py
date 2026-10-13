"""Resume the authorized Run 4 queue after a recorded GPU relocation.

Scientific workers, sampling, phase gates and frozen inputs remain unchanged.
The completed Run 3 replay is verified rather than executed again.
"""
import fcntl
from pathlib import Path
import traceback

import continue_evaluations_20261011 as queue
from run_deception_run4 import verify_frozen


def main():
    worker_lock = (queue.OUT / '.worker.lock').open('a+')
    fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = queue.read(queue.OUT / 'execution-manifest.json')
    for name, expected in manifest['code_sha256'].items():
        assert queue.sha(queue.ROOT / name) == expected, name
    relocation = queue.read(queue.OUT / 'gpu-relocation.json')
    assert relocation['status'] == 'ready'
    assert relocation['resume_entrypoint_sha256'] == queue.sha(Path(__file__))
    assert relocation['target_gpu'] == queue.GPU
    assert relocation['environment_after_sha256'] == queue.sha(queue.RUN4 / 'environment.json')
    for relative, expected in relocation['preserved_files'].items():
        assert queue.sha(queue.RUN4 / relative) == expected, relative
    verify_frozen(queue.RUN4)
    verification = queue.read(queue.RUN3 / 'held-out-verification.json')
    assert verification['status'] == 'passed' and verification['lens_completed'] == 288
    assert queue.read(queue.RUN3 / 'status.json')['stage'] == 'observational_analysis_complete'
    ready = queue.read(queue.RUN4 / 'launch-ready.json')
    assert ready['execution_sha256'] == queue.sha(queue.RUN4 / 'execution.json') and ready['tests_passed']
    if not queue.run4_phase('pilot'):
        return
    gate = queue.read(queue.RUN4 / 'pilot-gate.json')
    if gate['status'] != 'passed':
        queue.status('stopped_pilot_not_feasible', gate=gate)
        return
    if not queue.run4_phase('main_discovery'):
        return
    queue.wait_file(queue.RUN4 / 'main-analysis-lock.json', 'awaiting_run4_discovery_review_and_analysis_lock')
    if not queue.run4_phase('main_held_out'):
        return
    queue.wait_file(queue.RUN4 / 'secondary-analysis-lock.json', 'awaiting_run4_main_review_and_secondary_lock')
    if not queue.run4_phase('secondary_horizons'):
        return
    queue.status('generation_queue_finished_review_pending',
                 note='All authorized generation phases finished; semantic/lens review completion is tracked separately.')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        queue.status('stopped_at_error_or_unmet_gate', error=f'{type(error).__name__}: {error}')
        traceback.print_exc()
        raise
