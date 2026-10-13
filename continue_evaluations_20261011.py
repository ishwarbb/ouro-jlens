"""Durable, gated launch queue for the explicitly authorized evaluation rounds."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from run_deception_run3 import dump, read, sha, now, atomic_text

ROOT=Path(__file__).resolve().parent
RUN3=ROOT/'results/deception-run3-20261009'
RUN4=ROOT/'results/deception-run4-20261011'
OUT=ROOT/'results/evaluation-execution-20261011'
GPU=os.environ.get('CUDA_VISIBLE_DEVICES','8')


def status(stage,**extra):
    value=dict(stage=stage,utc=now(),pid=os.getpid(),gpu=GPU,**extra)
    dump(OUT/'status.json',value)
    from html import escape
    atomic_text(OUT/'index.html','<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="30"><title>Evaluation execution</title><style>body{font:17px/1.6 system-ui;max-width:1000px;margin:40px auto;padding:0 24px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>Evaluation execution</h1><pre>'+escape(str(value))+'</pre><p><a href="../deception-run3-20261009/held-out-analysis.html">Run 3 held-out analysis</a> · <a href="../deception-run4-20261011/index.html">Run 4 progress</a></p></html>')
    print(value,flush=True)


def wait_file(path,stage):
    while not path.exists():
        status(stage,waiting_for=str(path.relative_to(ROOT)))
        time.sleep(30)
    return read(path)


def command(stage,argv,gpu_lock=False):
    status(stage,command=argv)
    env=dict(os.environ,OURO_PROJECT_ROOT=str(ROOT),CUDA_VISIBLE_DEVICES=GPU,
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
             OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    if gpu_lock:
        with (ROOT/f'results/.gpu-evaluation-{GPU}.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            subprocess.run(argv,cwd=ROOT,env=env,check=True)
    else:subprocess.run(argv,cwd=ROOT,env=env,check=True)


def fact_review_digest():
    path=RUN4/'fact-reviews.json'
    return sha(path) if path.exists() else None


def run4_phase(phase):
    """Resume only missing jobs after a hash-matched factual review arrives."""
    while True:
        before=fact_review_digest()
        command('run4_'+phase,[sys.executable,'-u',str(ROOT/'run_deception_run4.py'),
                '--output',str(RUN4),'--phase',phase])
        state=read(RUN4/'status.json')
        if state['stage'] in {'paused_input_token_limit','paused_disk_limit'}:
            status('paused_at_resource_limit',phase=phase,worker_status=state)
            return False
        summary=read(RUN4/f'{phase}-summary.json')
        if summary['generation_complete'] and not summary['factual_reviews_pending']:
            return True
        while fact_review_digest()==before:
            status('awaiting_run4_factual_review',phase=phase,summary=summary,
                   note='No prompts or outputs are changed. Hash-matched reviews may release deferred cases.')
            time.sleep(30)


def main():
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'.worker.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=read(OUT/'execution-manifest.json')
    for name,digest in manifest['code_sha256'].items():assert sha(ROOT/name)==digest,name
    review=wait_file(RUN3/'held-out-semantic-review.json','awaiting_run3_adjudication')
    assert review['adjudicated'] and review['independent_annotations']==2 and len(review['cases'])==288
    python=sys.executable
    command('prepare_run3_held_out_replay',[python,str(ROOT/'replay_deception_run3_heldout.py'),'--output',str(RUN3),'--prepare-only'])
    command('run3_held_out_lens_replay',[python,'-u',str(RUN3/'held-out-lens/runner.py'),'--output',str(RUN3)],gpu_lock=True)
    assert read(RUN3/'held-out-lens/status.json')['stage']=='complete'
    command('verify_run3_held_out',[python,str(ROOT/'analyze_deception_run3_heldout.py'),'--output',str(RUN3),'--verify','--require-complete'])
    command('report_run3_held_out',[python,str(ROOT/'analyze_deception_run3_heldout.py'),'--output',str(RUN3)])
    final=dict(stage='observational_analysis_complete',utc=now(),completed=288,total=288,
               semantic_review_complete=True,lens_analysis_complete=True,
               full_vocabulary_endpoint_available=False,causal_stage_eligible=False,
               note='Independent agent annotations and adjudication; no human annotation. Original discovery files unchanged. See held-out-analysis.html.')
    dump(RUN3/'held_out-generation-summary.json',dict(read(RUN3/'held_out-generation-summary.json'),**final))
    dump(RUN3/'continuation-status.json',final)
    dump(RUN3/'status.json',final)
    ready=wait_file(RUN4/'launch-ready.json','awaiting_run4_preparation_validation')
    assert ready['execution_sha256']==sha(RUN4/'execution.json') and ready['tests_passed']
    if not run4_phase('pilot'):return
    gate=read(RUN4/'pilot-gate.json')
    if gate['status']!='passed':
        status('stopped_pilot_not_feasible',gate=gate,
               note='Main and secondary phases remain unlaunched; the registered feasibility gate did not pass.')
        return
    # Subsequent phases enforce their own immutable gates and stop safely for
    # missing reviews. A queued phase is never described as already executing.
    if not run4_phase('main_discovery'):return
    wait_file(RUN4/'main-analysis-lock.json','awaiting_run4_discovery_review_and_analysis_lock')
    if not run4_phase('main_held_out'):return
    wait_file(RUN4/'secondary-analysis-lock.json','awaiting_run4_main_review_and_secondary_lock')
    if not run4_phase('secondary_horizons'):return
    status('generation_queue_finished_review_pending',note='All authorized generation phases finished; semantic/lens review completion is tracked separately.')


if __name__=='__main__':
    try:main()
    except Exception as error:
        status('stopped_at_error_or_unmet_gate',error=f'{type(error).__name__}: {error}')
        traceback.print_exc()
        raise
