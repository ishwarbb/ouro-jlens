"""Dynamic scheduling of the unchanged, frozen workspace audit tasks.

Only scheduling differs from workspace_audit.run; scientific code and manifest
remain frozen. Per-item locks allow four workers to share the expensive tasks.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
from workspace_audit import Audit, check, FAMILY, OUT, ROOT
from fit_ouro_family import digest, save_json, verify
from ouro_full_fit import load_model
from ouro_lens_family import OuroLensFamily


def weight(item):
    n={'report':10,'flexible':3}.get(item['family'],1)
    return n*(29 if item['profile']=='primary' and item['family'] in ('flexible','probe','report') else 1)


def worker(args):
    m=check(args);fingerprint=digest(args.output/'manifest.json')
    plan=json.loads((args.output/'execution-plan.json').read_text())
    if plan['driver_sha256']!=digest(Path(__file__)):raise ValueError('Scheduling driver changed')
    torch.set_num_threads(2);model=load_model();family=OuroLensFamily.load(FAMILY);exp=Audit(model,family)
    save_json(args.output/f'verification-{args.rank}.json',verify(model,'The capital of France is'))
    selected=sorted(m['tasks'],key=lambda item:(-weight(item),item['id']))
    started=time.time();done=0
    try:
        for item in selected:
            path=args.output/'records'/f'{item["id"]}.json'
            with (args.output/'claims'/f'{item["id"]}.lock').open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                if path.exists():
                    if json.loads(path.read_text())['manifest_sha256']!=fingerprint:raise ValueError('Mixed run')
                    continue
                tick=time.time();exp.profile=item['profile'];model.add_bos=exp.profile!='no_bos'
                with torch.no_grad():
                    if item['family'] in ('flexible','probe','report'):result=exp.causal(item)
                    elif item['family']=='language':result=exp.language_audit(item)
                    else:result=exp.linecount(item)
                save_json(path,dict(id=item['id'],spec=item,result=result,seconds=time.time()-tick,
                    manifest_sha256=fingerprint,execution_plan_sha256=digest(args.output/'execution-plan.json')))
                done+=1
                save_json(args.output/f'progress-{args.rank}.json',dict(rank=args.rank,done=done,total='shared queue',
                    last_id=item['id'],elapsed_seconds=time.time()-started,updated_unix=time.time()))
                print(item['id'],f'{time.time()-tick:.2f}s',flush=True)
        save_json(args.output/f'worker-{args.rank}-complete.json',dict(items_this_invocation=done,seconds=time.time()-started))
    finally:model.close()


def run(args):
    with (args.output/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        m=check(args);started=time.time();jobs=[];streams=[]
        (args.output/'claims').mkdir(exist_ok=True)
        save_json(args.output/'execution-plan.json',dict(created_unix=started,
            reason='Original round-robin assignment concentrated primary items on two workers; dynamic shared queue balances remaining work',
            driver_sha256=digest(Path(__file__)),manifest_sha256=digest(args.output/'manifest.json'),
            preserved_records=len(list((args.output/'records').glob('*.json'))),gpus=m['gpus'],
            scientific_inputs_changed=False))
        try:
            for rank,gpu in enumerate(m['gpus']):
                stream=(args.output/f'worker-{rank}.log').open('a');streams.append(stream)
                jobs.append(subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--rank',str(rank),'--output',str(args.output)],
                    cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')))
            while any(p.poll() is None for p in jobs):
                codes=[p.poll() for p in jobs]
                save_json(args.output/'status.json',dict(stage='running',scheduler='shared queue',pid=os.getpid(),
                    started_unix=started,updated_unix=time.time(),gpus=m['gpus'],pids=[p.pid for p in jobs],
                    exit_codes=codes,records=len(list((args.output/'records').glob('*.json'))),total=m['n_items']))
                if any(c not in (None,0) for c in codes):raise RuntimeError(str(codes))
                time.sleep(5)
            if any(p.returncode for p in jobs):raise RuntimeError('Worker failure')
            subprocess.run([sys.executable,str(ROOT/'workspace_audit_analysis.py'),'--output',str(args.output)],check=True,cwd=ROOT)
            save_json(args.output/'status.json',dict(stage='complete',scheduler='shared queue',exit_codes=[p.returncode for p in jobs],
                completed_unix=time.time(),wall_seconds_this_invocation=time.time()-started,
                records=len(list((args.output/'records').glob('*.json'))),total=m['n_items'],gpus=m['gpus']))
        except BaseException as e:
            save_json(args.output/'queue-error.json',dict(error=repr(e),time=time.time()))
            for p in jobs:
                if p.poll() is None:p.terminate()
            for p in jobs:p.wait()
            raise
        finally:
            for stream in streams:stream.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['run','worker']);p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--rank',type=int,default=0);a=p.parse_args();{'run':run,'worker':worker}[a.phase](a)
