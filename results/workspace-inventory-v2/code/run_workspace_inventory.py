"""Frozen four-GPU pipeline: validated capture, layer-wise pursuit, then ablation."""
import argparse
from contextlib import nullcontext
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import transformers
from fit_ouro_family import ROOT, digest, save_json, verify
from ouro_full_fit import load_model
from ouro_lens_family import OuroLensFamily
from workspace_reproduction import items, load_data
from workspace_interventions import Directions
from workspace_validated import ValidatedIO, word_atom, prefix_transfer
from workspace_inventory import (dictionary, pursuit, orthonormal_basis,
                                 random_atom_indices, ablation)

OUT = ROOT / 'results/workspace-inventory-v2'
FAMILY = ROOT / 'results/jlens-family-1000'
SITES = [48*l+p for l in (1, 2, 3) for p in range(24, 41)]
READ_SITES = [48*l+p for l in range(4) for p in range(47)]
CODE = ['run_workspace_inventory.py', 'workspace_inventory.py', 'workspace_validated.py',
        'analyze_workspace_inventory.py', 'test_workspace_inventory.py',
        'workspace_audit_analysis.py', 'analyze_workspace.py',
        'workspace_interventions.py', 'workspace_reproduction.py', 'ouro_lens.py',
        'ouro_lens_family.py', 'ouro_full_fit.py', 'fit_ouro_family.py',
        'jacobian-lens/jlens/hooks.py']
DEMOS = [dict(id='demo-animal', family='demo', user='I’m thinking of the largest land animal. What do they eat? Answer briefly.'),
         dict(id='demo-sport', family='demo', user='Think of a specific sport, but don’t say its name. Describe the atmosphere at a big match of that sport without naming it.')]


def tasks():
    return [i for i in items() if i['family'] in ('flexible', 'report')] + DEMOS


def save_tensor(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def initialize(out):
    if (out / 'manifest.json').exists():
        raise FileExistsError('Existing frozen run is preserved; choose another output directory')
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(paper='https://arxiv.org/html/2609.01924v1', version=1,
        interpretation='Post-audit development study; no independent confirmation claim',
        gpus=[0, 1, 2, 3], model_dtype='bfloat16', arithmetic='float32; TF32 disabled',
        report_wrapper='Native chat template + Answer:; selected from already-observed audit, fixed across all 14 categories',
        report_scoring='Whole-token top1/top5 primary. Exact-ID greedy completed-word and teacher-forced sequence probabilities separately. All attempts retained; invalid category sources flagged.',
        report_continuation='Original prefill positions only edited during greedy/teacher-forced continuation; max 8 generated tokens; no retokenization',
        dictionary='Normalized J.T @ raw LM-head rows. Special tokens excluded; decoded stripped word must consist of Unicode alphabetic parts separated only by apostrophe/hyphen. Exact author filter unavailable.',
        pursuit='Greedy maximum positive residual inner product; unique tokens; positivity floor 1e-7 at unit state norm; accelerated projected-gradient NNLS on Gram support; normalized KKT tolerance 2e-6, fail if unconverged after 4000 iterations',
        inventory=dict(demos=[d['id'] for d in DEMOS], max_generated_tokens=128, generation_stop='EOS 0 or chat turn-end 2 only; thought markers retained and do not terminate generation',
            layers=READ_SITES, k=[8,16,25,50], theta=[0,.01,.05], primary=[25,.05],
            zero_threshold='Counts selected atoms including coefficients refitted to zero, matching >= threshold definition',
            scope='Two exact prompts quoted in paper, all prefill/completion positions at 188 fitted within-loop source sites; not the paper eight-prompt aggregate'),
        ablation=dict(prompts='All 64 released flexible-generalization prompts, including baseline failures',
            denominator='Primary retained-answer rate restricted to baseline whole-token-top1 correct, fixed identically across conditions; all-attempt counts also reported',
            sites=SITES, positions='all prefill positions including BOS', k=10,
            basis='Frozen clean-state pursuit atoms; rank-aware QR with reorthogonalization, dependence tolerance 1e-6',
            controls='Ten seeded random-vocabulary atoms, fixed per layer and shared over positions/trials; three seeds 42,43,44. Single argument, answer and unrelated directions.',
            paper_counts=dict(n=42,pursuit_all=0,pursuit_loop2=6,pursuit_loop4=0,random_42=37,argument_all=30,argument_loop2=42,argument_loop4=32,answer_all=8,unrelated_all=42)),
        uncertainties=['Exact author wrapper/filter/solver settings and demo transcripts unavailable',
                       'Independent confirmation, capacity under pursuit, and six remaining demo prompts not included'])
    save_json(out / 'protocol.json', protocol)
    data = [ROOT/'jacobian-lens/data/experiments'/f'{name}.json' for name in ['flexible-generalization','verbal-report']]
    manifest = dict(created_unix=time.time(), tasks=tasks(), protocol_sha256=digest(out/'protocol.json'),
        code_sha256={p:digest(ROOT/p) for p in CODE}, data_sha256={str(p.relative_to(ROOT)):digest(p) for p in data},
        family_manifest_sha256=digest(FAMILY/'family.json'), fit_manifest_sha256=digest(FAMILY/'full-manifest.json'),
        model_revision=json.loads((ROOT/'model_revision.json').read_text()),
        model_files_sha256={p.name:digest(p) for p in (ROOT/'model').iterdir() if p.is_file() and p.suffix in ('.json','.py','.jinja')},
        torch=str(torch.__version__), transformers=transformers.__version__, gpus=[0,1,2,3])
    save_json(out/'manifest.json', manifest)
    for name in CODE:
        path=out/'code'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/name).read_bytes())
    for name in ['capture','inventory','ablation','claims']:
        (out/name).mkdir(exist_ok=True)
    print('Frozen', len(tasks()), 'capture tasks,', len(READ_SITES), 'inventory layers, 64 ablation trials', flush=True)


def check(out):
    m=json.loads((out/'manifest.json').read_text())
    for name, expected in {**m['code_sha256'], **m['data_sha256']}.items():
        if digest(ROOT/name)!=expected:raise ValueError('Changed code/input: '+name)
    for path, expected in [(out/'protocol.json', m['protocol_sha256']), (FAMILY/'family.json',m['family_manifest_sha256']),
                           (FAMILY/'full-manifest.json',m['fit_manifest_sha256'])]:
        if digest(path)!=expected:raise ValueError('Changed protocol/family: '+str(path))
    for name, expected in m['model_files_sha256'].items():
        if digest(ROOT/'model'/name)!=expected:raise ValueError('Model/tokenizer changed: '+name)
    return m


def top(logits, tokenizer):
    values, ids=logits.softmax(-1).topk(5)
    return [dict(id=i,text=tokenizer.decode([i]),probability=p) for i,p in zip(ids.tolist(),values.tolist())]


@torch.no_grad()
def report(io, bank, item):
    prompt=io.chat(f'Think of a {item["category"]}. Answer in one word.', 'Answer:')
    logits,_=io.forward(prompt.ids)
    source=int(logits.argmax());source_text=io.tokenizer.decode([source]).strip()
    greedy=io.generate(prompt.ids,limit=8,stop_word=True,initial_logits=logits)
    source_valid=source_text.casefold() in {w.casefold() for w in item['candidates']}
    trials=[]
    for word in item['candidates'][:10]:
        if word.casefold()==source_text.casefold():continue
        spec=io.target(word); destination=spec['direction_token_id']
        baseline=dict(whole=io.score(logits,word),greedy=greedy,
            completed_word_hit=greedy['complete_first_word']==word.casefold(),
            sequence=io.sequence_score(prompt.ids,word,initial_logits=logits),top5=top(logits,io.tokenizer))
        trial=dict(target=spec, eligible=destination is not None and destination!=source,
                   conditions=dict(baseline=baseline))
        if trial['eligible']:
            for name,key in [('transfer',None),('random','validated-report:'+item['category']+':'+word)]:
                with prefix_transfer(io.model,bank,source,destination,SITES,len(prompt.ids),random_key=key):
                    changed,_=io.forward(prompt.ids)
                    emitted=io.generate(prompt.ids,limit=8,stop_word=True,initial_logits=changed)
                    trial['conditions'][name]=dict(whole=io.score(changed,word),greedy=emitted,
                        completed_word_hit=emitted['complete_first_word']==word.casefold(),
                        sequence=io.sequence_score(prompt.ids,word,initial_logits=changed),top5=top(changed,io.tokenizer))
        trials.append(trial)
    return dict(prompt=prompt.record(),source_token_id=source,source_text=source_text,
                source_in_category=source_valid,greedy_source=greedy,trials=trials)


@torch.no_grad()
def capture(io, bank, item, out):
    if item['family']=='report':return report(io,bank,item)
    if item['family']=='flexible':
        fn=item['function']; prompt=io.prompt(fn['template'].format(arg=item['argument']))
        ids=prompt.ids; sites=SITES; generated=None; answer=fn['answers'][item['argument']]
    else:
        prompt=io.chat(item['user'],wrapper='chat-demo-v1')
        generated=io.generate(prompt.ids,limit=128)
        ids=prompt.ids+tuple(generated['token_ids']); sites=READ_SITES;answer=None
    logits,acts=io.forward(ids,sites=sites)
    # Real-model replay of exact IDs checks determinism and state capture.
    repeated,_=io.forward(ids)
    if not torch.equal(logits,repeated):raise AssertionError('Exact-ID replay changed output')
    path=out/'capture'/f'{item["id"]}.pt'
    save_tensor(path,dict(seq_ids=list(ids),states={s:acts[s][0].cpu() for s in sites}))
    return dict(prompt=prompt.record(),seq_ids=list(ids),generated=generated,
        answer=io.target(answer) if answer else None, baseline=io.score(logits,answer) if answer else None,
        top5=top(logits,io.tokenizer),replay_exact=True,states_sha256=digest(path))


@torch.no_grad()
def inventory(io, family, captured, token_ids, site, out):
    start=time.time();device=io.model.hf_model.lm_head.weight.device
    j=family.lenses[family.target_for(site)].jacobians[site].to(device)
    atoms=dictionary(io.model.hf_model.lm_head.weight,j,token_ids)
    records={}; bases={}; summaries=[]
    for item,states in captured:
        if site not in states['states']:continue
        h=states['states'][site].to(device).float()
        k=50 if item['family']=='demo' else 10
        result=pursuit(h,atoms,token_ids,k=k)
        if item['family']=='demo':
            mapped=h@j.T
            lens=io.model.unembed(mapped).float();raw=io.model.unembed(h).float()
            probs,ids=lens.softmax(-1).topk(8,-1)
            raw_probs,raw_ids=raw.softmax(-1).topk(8,-1)
            result.update(top8_ids=ids.cpu(),top8_probabilities=probs.cpu(),raw_top8_ids=raw_ids.cpu(),raw_top8_probabilities=raw_probs.cpu())
        else:
            selected=atoms[result['selected_indices'].to(device)]*result['valid'].to(device).unsqueeze(-1)
            q=orthonormal_basis(selected)
            # Verify actual projector, including rank-deficient cells.
            gram=q.transpose(1,2)@q; diagonal=torch.diag_embed((q.norm(dim=1)>.5).float())
            orth_error=float((gram-diagonal).abs().max())
            if orth_error>1e-4:raise ArithmeticError('Nonorthogonal ablation basis')
            bases[item['id']]=q.cpu()
            result.update(basis_rank=(q.norm(dim=1)>.5).sum(-1).cpu(),basis_orthogonality_error=orth_error)
        summaries.append(dict(id=item['id'],cells=len(h),kkt=result['max_kkt'],refit_steps=result['max_refit_steps'],
            explained=float(1-result['snapshots'][k]['residual_fraction'].mean())))
        records[item['id']]=result
    random_bases={}
    if site in SITES:
        for seed in [42,43,44]:
            indices=random_atom_indices(len(token_ids),site,seed=seed).to(device)
            random_bases[seed]=dict(token_ids=token_ids[indices].cpu(),q=orthonormal_basis(atoms[indices].unsqueeze(0)).cpu())
    path=out/'inventory'/f'{site:03d}.pt'
    save_tensor(path,dict(site=site,records=records,bases=bases,random_bases=random_bases))
    return dict(site=site,seconds=time.time()-start,dictionary_size=len(token_ids),records=summaries,artifact_sha256=digest(path))


@torch.no_grad()
def ablate(io, bank, layer_records, item, out):
    captured=json.loads((out/'capture'/f'{item["id"]}.json').read_text())['result']
    ids=captured['seq_ids'];answer=captured['answer']['word']
    device=io.model.hf_model.lm_head.weight.device
    own={s:layer_records[s]['bases'][item['id']].to(device) for s in SITES}
    random={seed:{s:layer_records[s]['random_bases'][seed]['q'].to(device).expand(len(ids),-1,-1) for s in SITES} for seed in [42,43,44]}
    argument=io.vocab.direction_token(item['argument']); answer_id=io.vocab.direction_token(answer)
    unrelated=next(w for w in ['violin','copper','velvet','planet'] if w.casefold() not in (item['argument']+' '+answer+' '+captured['prompt']['text']).casefold())
    unrelated_id=io.vocab.direction_token(unrelated)
    def single(token):
        if token is None:return None
        return {s:bank.get(s,token).reshape(1,-1,1).expand(len(ids),-1,-1) for s in SITES}
    arg=single(argument);ans=single(answer_id);foil=single(unrelated_id)
    specs={'baseline':(None,SITES),'sham':({s:torch.zeros_like(q) for s,q in own.items()},SITES),
        'pursuit_all':(own,SITES),'pursuit_loop2':(own,[s for s in SITES if s//48==1]),
        'pursuit_loop4':(own,[s for s in SITES if s//48==3]),
        **{f'random_{seed}':(q,SITES) for seed,q in random.items()},
        'argument_all':(arg,SITES),'argument_loop2':(arg,[s for s in SITES if s//48==1]),
        'argument_loop4':(arg,[s for s in SITES if s//48==3]),
        'answer_all':(ans,SITES),'unrelated_all':(foil,SITES)}
    results={}; clean_logits=None
    for name,(basis,sites) in specs.items():
        if basis is None and name!='baseline':
            results[name]=dict(eligible=False,reason='No whole-token direction');continue
        with ablation(io.model,basis,sites=sites) if basis is not None else nullcontext():
            logits,_=io.forward(ids)
        if name=='baseline':clean_logits=logits
        if name=='sham' and not torch.equal(logits,clean_logits):raise AssertionError('Sham changed logits')
        results[name]=dict(eligible=True,whole=io.score(logits,answer),top5=top(logits,io.tokenizer))
    if results['baseline']['whole']!=captured['baseline']:raise AssertionError('Baseline score drift')
    return dict(prompt=captured['prompt'],answer=answer,source=item['argument'],unrelated=unrelated,
                baseline_correct=captured['baseline']['top1'],conditions=results,sham_exact=True)


def worker(args):
    manifest=check(args.output);fingerprint=digest(args.output/'manifest.json')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    model=load_model();io=ValidatedIO(model);family=OuroLensFamily.load(FAMILY);bank=Directions(model,family)
    save_json(args.output/f'verification-{args.stage}-{args.rank}.json',verify(model,'The capital of France is'))
    captured=[];layer_records={};token_ids=None
    if args.stage=='capture':queue=manifest['tasks']
    elif args.stage=='inventory':
        for item in manifest['tasks']:
            if item['family']=='report':continue
            path=args.output/'capture'/f'{item["id"]}.pt'
            meta=json.loads(path.with_suffix('.json').read_text())
            if digest(path)!=meta['result']['states_sha256']:raise ValueError('Captured states changed')
            captured.append((item,torch.load(path,map_location='cpu',weights_only=True)))
        allowed=[i for i in range(len(io.tokenizer)) if i not in io.tokenizer.all_special_ids and word_atom(io.tokenizer.decode([i]))]
        token_ids=torch.tensor(allowed,device=model.hf_model.lm_head.weight.device)
        save_json(args.output/f'dictionary-{args.rank}.json',dict(token_ids=allowed,decoded=[io.tokenizer.decode([i]) for i in allowed]))
        queue=[dict(id=f'{s:03d}',site=s) for s in READ_SITES]
    else:
        for site in SITES:
            path=args.output/'inventory'/f'{site:03d}.pt'
            meta=json.loads(path.with_suffix('.json').read_text())
            if digest(path)!=meta['result']['artifact_sha256']:raise ValueError('Inventory changed')
            layer_records[site]=torch.load(path,map_location='cpu',weights_only=True)
        queue=[i for i in manifest['tasks'] if i['family']=='flexible']
    done=0
    try:
        for item in queue:
            path=args.output/args.stage/f'{item["id"]}.json'
            with (args.output/'claims'/f'{args.stage}-{item["id"]}.lock').open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                if path.exists():
                    if json.loads(path.read_text())['manifest_sha256']!=fingerprint:raise ValueError('Mixed run')
                    continue
                started=time.time()
                if args.stage=='capture':result=capture(io,bank,item,args.output)
                elif args.stage=='inventory':result=inventory(io,family,captured,token_ids,item['site'],args.output)
                else:result=ablate(io,bank,layer_records,item,args.output)
                save_json(path,dict(spec=item,result=result,seconds=time.time()-started,manifest_sha256=fingerprint))
                done+=1
                save_json(args.output/f'progress-{args.rank}.json',dict(stage=args.stage,done=done,last_id=item['id'],seconds=time.time()-started,updated_unix=time.time()))
                print(args.stage,item['id'],f'{time.time()-started:.1f}s',flush=True)
    finally:model.close()


def run(args):
    manifest=check(args.output)
    with (args.output/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started=time.time()
        for stage,total in [('capture',len(manifest['tasks'])),('inventory',len(READ_SITES)),('ablation',64)]:
            jobs=[];streams=[]
            try:
                for rank in range(4):
                    stream=(args.output/f'{stage}-worker-{rank}.log').open('a');streams.append(stream)
                    jobs.append(subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--stage',stage,'--rank',str(rank),'--output',str(args.output)],cwd=ROOT,
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false'),stdout=stream,stderr=subprocess.STDOUT))
                while any(p.poll() is None for p in jobs):
                    codes=[p.poll() for p in jobs]
                    save_json(args.output/'status.json',dict(stage=stage,records=len(list((args.output/stage).glob('*.json'))),total=total,
                        exit_codes=codes,pids=[p.pid for p in jobs],gpus=[0,1,2,3],started_unix=started,updated_unix=time.time()))
                    if any(c not in (None,0) for c in codes):raise RuntimeError('Worker failed: '+str(codes))
                    time.sleep(5)
                if any(p.returncode for p in jobs):raise RuntimeError('Worker failed')
                if len(list((args.output/stage).glob('*.json')))!=total:raise RuntimeError('Incomplete stage')
            except BaseException as error:
                save_json(args.output/'error.json',dict(stage=stage,error=repr(error),unix=time.time()))
                raise
            finally:
                for p in jobs:
                    if p.poll() is None:p.terminate()
                for p in jobs:p.wait()
                for stream in streams:stream.close()
        subprocess.run([sys.executable,str(ROOT/'analyze_workspace_inventory.py'),'--output',str(args.output)],cwd=ROOT,check=True)
        save_json(args.output/'status.json',dict(stage='complete',gpus=[0,1,2,3],capture=80,inventory=188,ablation=64,seconds=time.time()-started,exit_codes=[0]*4))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['init','run','worker']);p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--stage',choices=['capture','inventory','ablation']);p.add_argument('--rank',type=int,default=0);a=p.parse_args()
    if a.phase=='init':initialize(a.output)
    elif a.phase=='run':run(a)
    else:worker(a)
