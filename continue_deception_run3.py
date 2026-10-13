"""Persistent sequential discovery replay -> freeze -> held-out generation."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import traceback

from run_deception_run3 import dump, now, read, sha

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/deception-run3-20261009'


def main():
    lock=(OUT/'.continuation.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,OURO_PROJECT_ROOT=str(ROOT),CUDA_VISIBLE_DEVICES='0',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
             OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    for archived in (OUT/'analysis-code').glob('*.py'):
        assert sha(archived)==sha(ROOT/archived.name),f'Analysis code changed: {archived.name}'
    def status(stage,**extra):
        value=dict(stage=stage,utc=now(),pid=os.getpid(),**extra)
        dump(OUT/'continuation-status.json',value);print(value,flush=True)
    commands=[
        ('discovery_lens_replay',[sys.executable,'-u',str(OUT/'lens/runner.py'),'--output',str(OUT)]),
        ('verify_and_freeze_discovery',[sys.executable,'-u',str(ROOT/'advance_deception_run3.py'),'--output',str(OUT),'--freeze']),
        ('held_out_generation',[sys.executable,'-u',str(OUT/'code/run_deception_run3.py'),'--output',str(OUT),'--phase','held_out']),
        ('verify_saved_outputs',[sys.executable,'-u',str(ROOT/'verify_deception_run3.py'),'--output',str(OUT),'--require-lens'])]
    try:
        for stage,command in commands:
            status(stage)
            subprocess.run(command,cwd=ROOT,env=env,check=True)
        summary=read(OUT/'held_out-generation-summary.json')
        assert summary['generation_complete'] and summary['completed']==288
        status('awaiting_held_out_semantic_review',held_out_saved=288,
               note='Read full held-out outputs and annotate before any held-out lens replay. No causal interventions authorized by the failed discovery/instrument gates.')
    except Exception as error:
        status('failed',error=f'{type(error).__name__}: {error}')
        traceback.print_exc();raise


if __name__=='__main__':main()
