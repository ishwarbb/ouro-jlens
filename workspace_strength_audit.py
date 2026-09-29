"""Explicit post-results strength semantics diagnostic on all 192 flexible swaps."""
from contextlib import contextmanager
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
from workspace_audit import Audit,FAMILY,ROOT,load_data,tasks
from workspace_interventions import transfer_delta
from fit_ouro_family import save_json,digest
from ouro_full_fit import load_model
from ouro_lens_family import OuroLensFamily

OUT=ROOT/'results/workspace-strength-audit'


def change(h,a,b,mode):
    projection=(h@a).unsqueeze(-1)
    if mode=='update_x2':return 2*projection*(b-a)
    if mode=='destination_x2':return projection*(2*b-a)
    if mode=='directions_x2':return 4*projection*(b-a)
    if mode=='positive_transfer_x2':return 2*projection.clamp_min(0)*(b-a)
    raise ValueError(mode)


@contextmanager
def edit(exp,a,b,mode):
    handles=[]
    def hook_for(site):
        source=exp.bank.get(site,a);dest=exp.bank.get(site,b)
        def hook(module,args,output):
            h=output.float();return (h+change(h,source,dest,mode)).to(output.dtype)
        return hook
    try:
        for site in exp.sites:handles.append(exp.model.layers[site].register_forward_hook(hook_for(site)))
        yield
    finally:
        for handle in handles:handle.remove()


def worker(rank):
    torch.set_num_threads(2);model=load_model();exp=Audit(model,OuroLensFamily.load(FAMILY))
    try:
        for i,item in enumerate([t for t in tasks() if t['family']=='flexible' and t['profile']=='primary']):
            if i%4!=rank:continue
            fn=item['function'];source=item['argument'];text=fn['template'].format(arg=source);a=exp.direction(source);trials=[]
            with torch.no_grad():
                for dest in item['arguments']:
                    if dest==source:continue
                    b=exp.direction(dest);row=dict(destination=dest,conditions={})
                    for mode in ['update_x2','destination_x2','directions_x2','positive_transfer_x2']:
                        with edit(exp,a,b,mode):_,_,logits=exp.forward(text)
                        row['conditions'][mode]=exp.output_record(logits,{'answer':fn['answers'][source],'swap_answer':fn['answers'][dest]})
                    trials.append(row)
            save_json(OUT/'records'/f'{i:02d}.json',dict(id=item['id'],spec=item,prompt=text,trials=trials,
                manifest_sha256=digest(OUT/'manifest.json')))
            print(item['id'],flush=True)
    finally:model.close()


def analyze():
    from collections import defaultdict
    from analyze_workspace import rates
    from workspace_audit_analysis import score,cell
    files=sorted((OUT/'records').glob('*.json'));assert len(files)==64
    rows=[];checked=0
    for p in files:
        r=json.loads(p.read_text());main=json.loads((ROOT/'results/workspace-audit/records'/(r['id']+'.json')).read_text())['result']
        original={t['destination']:t for t in main['trials']}
        for trial in r['trials']:
            # Same operation must reproduce the fixed grid exactly.
            assert trial['conditions']['update_x2']==original[trial['destination']]['conditions']['strength_2']
            checked+=1
            rows.append((r['spec']['category']+':'+r['spec']['argument'],{k:score(v) for k,v in trial['conditions'].items()}))
    s=dict(rates=rates(rows),exact_primary_strength_regressions=checked)
    save_json(OUT/'summary.json',s)
    definitions={'update_x2':'Double the whole residual update (frozen main definition)',
        'destination_x2':'Remove source once; add twice its magnitude in destination direction',
        'directions_x2':'Scale both unnormalized direction vectors by two (four times the update)',
        'positive_transfer_x2':'Transfer only positive source projections; double the update'}
    lines=['# Strength semantics diagnostic','',
        'Exploratory follow-up to a discrepancy with the paper’s doubled-strength row. The primary definition remains unchanged; '
        'the authors’ exact strength implementation is unknown. Every interpretation is tested on all 192 swaps.', '',
        '| Interpretation | Flexible top-1 |','|---|---:|']
    for k,v in s['rates'].items():lines.append(f'| {definitions[k]} | {cell(v)} |')
    lines+=['','Paper reference: 14/192 under its doubled-strength setting ([Table 7](https://arxiv.org/html/2609.01924v1#A2)). '
        'A close number under an alternative is not evidence that the author implementation has been recovered.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def run():
    import fcntl
    OUT.mkdir(exist_ok=True)
    with (OUT/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'manifest.json').exists():raise FileExistsError('Preserving prior run')
        save_json(OUT/'manifest.json',dict(created_unix=time.time(),code_sha256=digest(Path(__file__)),
            primary_manifest_sha256=digest(ROOT/'results/workspace-audit/manifest.json'),
            rationale='Discrepant doubled-strength result; fixed four algebraic interpretations, all released flexible trials',
            exploratory=True,gpus=[0,1,2,3]))
        jobs=[];streams=[];start=time.time()
        try:
            for rank in range(4):
                f=(OUT/f'worker-{rank}.log').open('w');streams.append(f)
                jobs.append(subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--rank',str(rank)],
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false'),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT))
            while any(p.poll() is None for p in jobs):
                codes=[p.poll() for p in jobs];save_json(OUT/'status.json',dict(stage='running',exit_codes=codes,records=len(list((OUT/'records').glob('*.json')))))
                if any(c not in (None,0) for c in codes):raise RuntimeError(str(codes))
                time.sleep(5)
            assert all(p.returncode==0 for p in jobs)
            analyze();save_json(OUT/'status.json',dict(stage='complete',exit_codes=[0]*4,records=64,seconds=time.time()-start))
        finally:
            for p in jobs:
                if p.poll() is None:p.terminate()
            for p in jobs:p.wait()
            for f in streams:f.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['run','worker']);p.add_argument('--rank',type=int,default=0);a=p.parse_args()
    run() if a.phase=='run' else worker(a.rank)
