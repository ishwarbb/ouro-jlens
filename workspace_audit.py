"""Recorded replication audit: all released affected tasks and a fixed sensitivity grid.

Preserves workspace-full. Run init, then run (four GPUs), then analyze.
Every result retains its condition, exact prompt, token IDs, and loop-end ranks.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import torch
from jlens.hooks import ActivationRecorder
from fit_ouro_family import ROOT, digest, save_json, verify
from ouro_full_fit import load_model
from ouro_lens_family import OuroLensFamily
from workspace_interventions import Directions, Vocabulary, score_tokens, swap_delta, transfer_delta
from workspace_reproduction import Experiment, load_data, items as original_items

OUT = ROOT/'results/workspace-audit'
FAMILY = ROOT/'results/jlens-family-1000'
CODE = ['workspace_audit.py','workspace_audit_analysis.py','workspace_reproduction.py',
        'workspace_interventions.py','ouro_lens.py','ouro_lens_family.py','ouro_full_fit.py',
        'fit_ouro_family.py','jacobian-lens/jlens/hooks.py']


def conditions():
    base = dict(mode='transfer', loops=[2,3,4], band=[24,40], strength=1.0, kind='own')
    result = {'transfer':base}
    for n in range(1,4):
        for loops in itertools.combinations([2,3,4],n):
            if n<3: result['loops_'+''.join(map(str,loops))] = dict(base,loops=list(loops))
    result['loops_1234'] = dict(base,loops=[1,2,3,4])
    for strength in [.25,.5,.75,1.25,1.5,2.0]:
        result[f'strength_{strength:g}'] = dict(base,strength=strength)
    for lo,hi in [(20,40),(23,39),(24,36),(28,40),(32,40),(0,16)]:
        result[f'band_{lo}_{hi}'] = dict(base,band=[lo,hi])
    for mode in ['swap','replace_target','dual_clamp']:
        result[mode] = dict(base,mode=mode)
    for kind in ['raw','final']:
        result[kind] = dict(base,kind=kind)
    for seed in [42,43,44]:
        result[f'random_{seed}'] = dict(base,mode='random_transfer',seed=seed)
    result['last_position'] = dict(base,last_position=True)
    return result


def tasks():
    result=[]
    for item in original_items():
        if item['family'] not in ('flexible','probe','report','language'):continue
        result.append(dict(item,profile='primary'))
        if item['family'] in ('flexible','probe','report'):
            for profile in ['no_bos','no_space_direction','lower_direction','alternate_format','rstrip']:
                result.append(dict(item,id=item['id']+'-'+profile,profile=profile))
            if item['family']=='report':
                for profile in ['colon_prefill','answer_prefill','no_prefill']:
                    result.append(dict(item,id=item['id']+'-'+profile,profile=profile))
        if item['family']=='language':
            result.append(dict(item,id=item['id']+'-no_bos',profile='no_bos'))
    for i,passage in enumerate(load_data('selectivity-linecount')['passages']):
        result.append(dict(id=f'linecount-{i:02d}',family='linecount',passage=passage,profile='primary'))
    return result


def delta_for(h,a,b,mode):
    if mode in ('transfer','random_transfer'):return transfer_delta(h,a,b)
    if mode=='swap':return swap_delta(h,a,b)
    if mode=='replace_target':
        # Sequentially remove source and then set destination projection to old source.
        x=(h@a).unsqueeze(-1); after=h-x*a
        return -x*a+(x-(after@b).unsqueeze(-1))*b
    if mode=='dual_clamp':
        # Minimum-norm change that sets source projection to 0, destination to old source.
        v=torch.stack([a,b],dim=1)
        current=h@v
        desired=torch.stack([torch.zeros_like(current[...,0]),current[...,0]],dim=-1)
        return (desired-current)@torch.linalg.pinv(v)
    raise ValueError(mode)


@contextmanager
def edited(model,bank,source,destination,condition,key):
    sites=[48*(loop-1)+p for loop in condition['loops']
           for p in range(condition['band'][0],condition['band'][1]+1)]
    prepared={}
    for site in sites:
        a=bank.get(site,source,condition['kind']);b=bank.get(site,destination,condition['kind'])
        r=bank.random(site,f'{key}:{condition.get("seed",42)}') if condition['mode']=='random_transfer' else None
        prepared[site]=(a,b,r)
    handles=[]
    def hook_for(site):
        a,b,r=prepared[site]
        def hook(module,args,output):
            selector=[output.shape[1]-1] if condition.get('last_position') else slice(None)
            h=output[:,selector,:].float()
            delta=delta_for(h,a,b,condition['mode'])
            if r is not None:delta=delta.norm(dim=-1,keepdim=True)*r
            updated=output.clone()
            updated[:,selector,:]=(h+condition['strength']*delta).to(output.dtype)
            return updated
        return hook
    try:
        for site in sites:handles.append(model.layers[site].register_forward_hook(hook_for(site)))
        yield
    finally:
        for h in handles:h.remove()


class Audit(Experiment):
    def __init__(self,model,family):
        super().__init__(model,family)
        self.profile='primary'

    def direction(self,word):
        if self.profile=='lower_direction':word=word.lower()
        return self.vocab.direction_token(word,leading_space=self.profile!='no_space_direction')

    def prefix_ids(self,word):
        # Explicit diagnostic only. Prefix hits never masquerade as complete-word answers.
        return sorted({self.model.tokenizer.encode(v,add_special_tokens=False)[0]
                       for v in [' '+word,' '+word.lower(),word,word.lower()] if v})

    def scores(self,logits,words):
        return {str(t):{w:dict(whole=score_tokens(value,self.vocab.candidates(word)),
                             prefix=score_tokens(value,self.prefix_ids(word)))
                       for w,word in words.items()} for t,value in logits.items()}

    def output_record(self,logits,words):
        return dict(scores=self.scores(logits,words),top5=self.top(logits[191]))

    @torch.no_grad()
    def causal(self,item):
        family=item['family']
        if family=='flexible':
            fn=item['function'];source=item['argument'];answer=fn['answers'][source]
            text=fn['template'].format(arg=source)
            swaps=[(target,fn['answers'][target]) for target in item['arguments'] if target!=source]
        elif family=='probe':
            d=item['item'];text=d['prompt'];source=d['intermediate'];answer=d['answer']
            swaps=[(d['swap_to'],d['swap_answer'])]
        else:
            user=f'Think of a {item["category"]}. Answer in one word.'
            prefill={'colon_prefill':':','answer_prefill':'Answer:','no_prefill':''}.get(self.profile,'One word:')
            text=self.chat(user,prefill)
            if self.profile=='alternate_format':text=user+'\nOne word:'
        if self.profile=='alternate_format' and family!='report':text=self.chat(text)
        if self.profile=='rstrip':text=text.rstrip()
        _,_,baseline=self.forward(text)
        source_token=None
        if family=='report':
            source_token=int(baseline[191].argmax());source=self.model.tokenizer.decode([source_token]).strip();answer=source
            swaps=[(w,w) for w in item['candidates'][:10] if w.casefold()!=source.casefold()]
        if source_token is None:source_token=self.direction(source)
        grids=conditions() if self.profile=='primary' else {'transfer':conditions()['transfer']}
        result=dict(prompt=text,input_ids=self.model.encode(text,max_length=2048)[0].tolist(),
                    baseline=self.output_record(baseline,{'answer':answer}),source=source,answer=answer,
                    source_token_id=source_token,source_is_category_member=(source.casefold() in
                    {w.casefold() for w in item.get('candidates',[])}) if family=='report' else None,trials=[])
        for dest,expected in swaps:
            destination=self.direction(dest)
            words={'answer':answer,'swap_answer':expected}
            trial=dict(destination=dest,expected=expected,destination_token_id=destination,
                       whole_answer_eligible=bool(self.vocab.candidates(expected)),
                       conditions={'baseline':self.output_record(baseline,words)})
            trial['intervention_eligible']=source_token is not None and destination is not None and source_token!=destination
            if trial['intervention_eligible']:
                for name,condition in grids.items():
                    with edited(self.model,self.bank,source_token,destination,condition,item['id']+':'+dest):
                        _,_,output=self.forward(text)
                    trial['conditions'][name]=self.output_record(output,words)
            else:trial['exclusion']='No distinct whole-token direction; retained in attempted-trial denominator'
            result['trials'].append(trial)
        if family=='probe' and self.profile=='primary':
            record=sorted({48*l+p for l in range(4) for p in range(47)})
            ids,acts,_=self.forward(text,record=record)
            foils=[w for w in ['violin','copper','velvet','planet'] if w.casefold()!=source.casefold() and w.casefold() not in text.casefold()]
            queries={'intermediate':[source],**{f'foil_{i}':[w] for i,w in enumerate(foils)}}
            result['latent']=dict(concept_present_in_prompt=any(t in ids[0].tolist() for t in self.vocab.candidates(source)),
                                  foils=foils,readouts=self.detailed_readout(acts,list(range(1,ids.shape[1])),queries))
        return result

    @torch.no_grad()
    def detailed_readout(self,acts,positions,queries,full=True):
        tokens={q:sorted({t for word in words for t in self.vocab.candidates(word)}) for q,words in queries.items()}
        result={}
        for site in sorted(acts):
            if site%48==47:continue
            # Full physical profile for own-loop lens; controls at matched band sites.
            modes=['own','raw','final'] if 24<=site%48<=40 else ['own']
            h=acts[site][0,positions].float()
            for mode in modes:
                mapped=h
                if mode!='raw':
                    target=self.family.target_for(site) if mode=='own' else 191
                    key=(site,target)
                    if key not in self.bank.matrices:self.bank.matrices[key]=self.family.lenses[target].jacobians[site].to(h.device)
                    mapped=h@self.bank.matrices[key].T
                logits=self.model.unembed(mapped).float()
                top=logits.topk(8,dim=-1).indices
                row={}
                for q,ts in tokens.items():
                    if not ts:row[q]=dict(eligible=False);continue
                    ts_tensor=torch.tensor(ts,device=top.device)
                    ranks=1+(logits>logits[:,ts].max(-1).values[:,None]).sum(-1)
                    row[q]=dict(eligible=True,min_rank=int(ranks.min()),last_rank=int(ranks[-1]),
                        hit1=bool(torch.isin(top[:,:1],ts_tensor).any()),
                        hit5=bool(torch.isin(top[:,:5],ts_tensor).any()),
                        hit8=bool(torch.isin(top,ts_tensor).any()),
                        hit5_positions=int(torch.isin(top[:,:5],ts_tensor).any(-1).sum()),positions=len(positions))
                result[f'{site}:{mode}']=row
        return result

    @torch.no_grad()
    def language_audit(self,item):
        d=load_data('selectivity-language');passage=item['passage'];results={}
        for name,key in [('explicit','explicit_q'),('automatic','automatic_q')]:
            text=self.chat(d['task'][key].format(text=passage['text']))
            start=text.index(passage['text'])+len(passage['text']);positions=self.span(text,start,len(text))
            _,acts,output=self.forward(text,record=[48*l+p for l in range(4) for p in range(47)])
            results[name]=dict(prompt=text,positions=positions,top5=self.top(output[191]),
                readouts=self.detailed_readout(acts,positions,{'language':d['intermediates'][passage['category']]}))
        return dict(passage=passage,conditions=results)

    @torch.no_grad()
    def linecount(self,item):
        d=load_data('selectivity-linecount');p=item['passage'];passage=textwrap.fill(p['text'],width=p['width'])
        number_words=['twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety']
        candidates=[str(i) for i in range(10,100)]+number_words
        results={}
        for name in ['none','direct','letter','continue']:
            spec=d['conditions'].get(name,dict(question=d['explicit_q'],prefill=''))
            user=spec['question']+'\n\n'+passage
            text=self.chat(user,spec['prefill'])
            ids,acts,output=self.forward(text,record=self.sites)
            results[name]=dict(prompt=text,top5=self.top(output[191]),
                readouts=self.detailed_readout(acts,list(range(1,ids.shape[1])),{'number':candidates,'exact':[str(len(passage.splitlines()[0]))]}))
        return dict(passage=passage,answer=len(passage.splitlines()[0]),conditions=results)


def initialize(args):
    if (args.output/'manifest.json').exists():raise FileExistsError('Audit is already frozen')
    protocol=dict(version=1,paper='https://arxiv.org/html/2609.01924v1',scope='All released inconclusive families; line-count selectivity added',
        conditions=conditions(),profiles=sorted({t['profile'] for t in tasks()}),
        primary_operator='h + (h dot source) * (destination - source); inference from clamp-transfer description, exact author implementation unavailable',
        scoring='Whole-token exact case-insensitive top1 and top5, plus separately labelled first-token prefix diagnostic. All attempted trials retained; no answer-eligibility gate on intervention.',
        metrics='Verbal top5; flexible/probe top1; recurrence paired transfer-versus-single-loop; language delta hit5; latent per-loop own versus final and foil, matched physical sites excluding loop4 equality.',
        sensitivity_rule='Fixed complete grid; no best-setting replacement of main result; already-seen original projection-transfer outcomes disclosed',
        interpretation='Post-results method audit, not an independent preregistered replication. Repeated prompts/conditions do not add independent samples.',
        uncertainty='Exact fitting examples, causal-band endpoints/indexing, prompt wrappers, token-selection rules, clamp algebra and author code unavailable.',
        paper_metrics=dict(flexible=dict(successes=73,n=192,loop2=40,loop3_percent=29,loop4=48,double_strength=14,baseline_successes=42,baseline_n=64),
            probe=dict(successes=26,n=90,baseline_successes=43),report=dict(top5_successes=106,n=133,top1_percent=67),language_delta=.25,
            linecount=dict(direct=1.,none=.09)))
    save_json(args.output/'protocol.json',protocol)
    manifest=dict(created_unix=time.time(),tasks=tasks(),n_items=len(tasks()),
        protocol_sha256=digest(args.output/'protocol.json'),code_sha256={f:digest(ROOT/f) for f in CODE},
        data_sha256={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'jacobian-lens/data/experiments').glob('*.json')},
        family_manifest_sha256=digest(FAMILY/'family.json'),fit_manifest_sha256=digest(FAMILY/'full-manifest.json'),
        model_code_sha256=digest(ROOT/'model/modeling_ouro.py'),model_revision=json.loads((ROOT/'model_revision.json').read_text()),
        gpus=['0','1','2','3'])
    save_json(args.output/'manifest.json',manifest)
    for name in CODE:
        dest=args.output/'code'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    print('Frozen',len(tasks()),'items;',len(conditions()),'intervention conditions on primary causal items',flush=True)


def check(args):
    m=json.loads((args.output/'manifest.json').read_text())
    for name,expected in {**m['code_sha256'],**m['data_sha256']}.items():
        if digest(ROOT/name)!=expected:raise ValueError('Input/code changed: '+name)
    if digest(args.output/'protocol.json')!=m['protocol_sha256']:raise ValueError('Protocol changed')
    if digest(FAMILY/'family.json')!=m['family_manifest_sha256']:raise ValueError('Fit changed')
    return m


def worker(args):
    m=check(args);fingerprint=digest(args.output/'manifest.json')
    torch.set_num_threads(2)
    family=OuroLensFamily.load(FAMILY);model=load_model();exp=Audit(model,family)
    selected=m['tasks'][args.rank::4]
    if args.limit:selected=selected[:args.limit]
    started=time.time()
    save_json(args.output/f'verification-{args.rank}.json',verify(model,'The capital of France is'))
    try:
        for index,item in enumerate(selected):
            path=args.output/'records'/f'{item["id"]}.json'
            if path.exists():
                if json.loads(path.read_text())['manifest_sha256']!=fingerprint:raise ValueError('Mixed run')
                continue
            tick=time.time();exp.profile=item['profile'];model.add_bos=exp.profile!='no_bos'
            with torch.no_grad():
                if item['family'] in ('flexible','probe','report'):result=exp.causal(item)
                elif item['family']=='language':result=exp.language_audit(item)
                else:result=exp.linecount(item)
            save_json(path,dict(id=item['id'],spec=item,result=result,seconds=time.time()-tick,manifest_sha256=fingerprint))
            save_json(args.output/f'progress-{args.rank}.json',dict(rank=args.rank,done=index+1,total=len(selected),
                last_id=item['id'],elapsed_seconds=time.time()-started,updated_unix=time.time()))
            print(item['id'],f'{time.time()-tick:.2f}s',flush=True)
        save_json(args.output/f'worker-{args.rank}-complete.json',dict(items=len(selected),seconds=time.time()-started))
    finally:model.close()


def run(args):
    import fcntl
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'supervisor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        m=check(args);jobs=[];streams=[];started=time.time()
        try:
            for rank,gpu in enumerate(m['gpus']):
                stream=(args.output/f'worker-{rank}.log').open('a');streams.append(stream)
                command=[sys.executable,'-u',str(ROOT/'workspace_audit.py'),'worker','--output',str(args.output),'--rank',str(rank)]
                if args.limit:command+=['--limit',str(args.limit)]
                jobs.append(subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')))
            while any(p.poll() is None for p in jobs):
                codes=[p.poll() for p in jobs]
                save_json(args.output/'status.json',dict(stage='running',pid=os.getpid(),started_unix=started,
                    updated_unix=time.time(),gpus=m['gpus'],pids=[p.pid for p in jobs],exit_codes=codes,
                    records=len(list((args.output/'records').glob('*.json'))),total=m['n_items']))
                if any(c not in (None,0) for c in codes):raise RuntimeError(f'Worker failure: {codes}')
                time.sleep(5)
            codes=[p.returncode for p in jobs]
            if any(codes):raise RuntimeError(str(codes))
            if not args.limit:
                subprocess.run([sys.executable,str(ROOT/'workspace_audit_analysis.py'),'--output',str(args.output)],check=True,cwd=ROOT)
            save_json(args.output/'status.json',dict(stage='smoke_complete' if args.limit else 'complete',exit_codes=codes,
                completed_unix=time.time(),wall_seconds=time.time()-started,records=len(list((args.output/'records').glob('*.json'))),gpus=m['gpus']))
        except BaseException as e:
            save_json(args.output/'error.json',dict(error=repr(e),time=time.time()))
            for p in jobs:
                if p.poll() is None:p.terminate()
            for p in jobs:p.wait()
            raise
        finally:
            for stream in streams:stream.close()


def status(args):
    def read(name):
        path=args.output/name
        return json.loads(path.read_text()) if path.exists() else {}
    state=read('status.json');manifest=read('manifest.json')
    count=len(list((args.output/'records').glob('*.json')));total=manifest.get('n_items',0)
    print('OURO WORKSPACE — PAPER AUDIT')
    print('Stage:',state.get('stage','starting'),' | GPUs: 0,1,2,3')
    print(f'Items: {count}/{total} ({100*count/max(1,total):.1f}%)')
    if state.get('started_unix'):
        print(f'Elapsed: {(time.time()-state["started_unix"])/60:.1f} min')
    print('')
    for rank in range(4):
        v=read(f'progress-{rank}.json')
        print(f'GPU {rank}: {v.get("done",0):4}/{v.get("total","?")}  {v.get("last_id","loading")}')
    print('\nReport:',args.output/'REPORT.md')
    subprocess.run(['nvidia-smi','-i','0,1,2,3','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],check=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['init','run','worker','status'])
    p.add_argument('--output',type=Path,default=OUT);p.add_argument('--rank',type=int,default=0);p.add_argument('--limit',type=int)
    a=p.parse_args();{'init':initialize,'run':run,'worker':worker,'status':status}[a.phase](a)
