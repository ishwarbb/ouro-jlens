"""Post-audit diagnostic of punctuation and split-token report sources.

This is separate from the frozen main grid: no diagnostic replaces its primary.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import torch
from workspace_audit import Audit, FAMILY, ROOT, conditions, edited, load_data
from fit_ouro_family import digest,save_json
from ouro_full_fit import load_model
from ouro_lens_family import OuroLensFamily

OUT=ROOT/'results/workspace-report-tokens'


def record(model,exp,category,candidates):
    original=exp.chat(f'Think of a {category}. Answer in one word.','One word:')
    prefix=original;formatting=[]
    with torch.no_grad():
        # Consume only punctuation/whitespace. Never select a source to improve success.
        for _ in range(4):
            _,_,output=exp.forward(prefix);token=int(output[191].argmax());piece=model.tokenizer.decode([token])
            if any(c.isalnum() for c in piece):break
            if token in model.tokenizer.all_special_ids:break
            prefix+=piece;formatting.append(dict(token_id=token,text=piece))
        _,_,baseline=exp.forward(prefix);first=int(baseline[191].argmax());first_text=model.tokenizer.decode([first]).strip()
        continuation='';text=prefix;emitted=[];concept=None
        # Identify the greedy word, limited to released candidates; retain unresolved cases.
        for _ in range(8):
            _,_,output=exp.forward(text);token=int(output[191].argmax());piece=model.tokenizer.decode([token]);emitted.append(dict(token_id=token,text=piece))
            if token in model.tokenizer.all_special_ids:break
            continuation+=piece;text+=piece
            normalized=continuation.strip().strip('"\'.,:;!?)(').casefold()
            matches=[w for w in candidates if w.casefold()==normalized]
            if matches:concept=matches[0];break
            if len(continuation.strip().split())>1:break
        result=dict(category=category,original_prompt=original,scored_prompt=prefix,formatting_tokens=formatting,
            first_token=first,first_text=first_text,greedy_tokens=emitted,decoded_concept=concept,variants={})
        grid={k:v for k,v in conditions().items() if k in ['transfer','loops_2','loops_3','loops_4','swap','raw','random_42']}
        for name,source,source_id in [('skip_formatting',first_text,first),
                                    ('decode_concept',concept,exp.direction(concept) if concept else None)]:
            trials=[]
            for dest in candidates[:10]:
                if source and dest.casefold()==source.casefold():continue
                destination=exp.direction(dest);words={'swap_answer':dest,'answer':source or first_text}
                trial=dict(destination=dest,destination_id=destination,conditions={'baseline':exp.output_record(baseline,words)})
                trial['eligible']=source_id is not None and destination is not None and source_id!=destination
                if trial['eligible']:
                    for key,condition in grid.items():
                        with edited(model,exp.bank,source_id,destination,condition,'report-token:'+category+':'+dest):
                            _,_,output=exp.forward(prefix)
                        trial['conditions'][key]=exp.output_record(output,words)
                trials.append(trial)
            result['variants'][name]=dict(source=source,source_token_id=source_id,
                source_in_category=bool(source and source.casefold() in {w.casefold() for w in candidates}),trials=trials)
        return result


def worker(rank):
    torch.set_num_threads(2);model=load_model();exp=Audit(model,OuroLensFamily.load(FAMILY))
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert manifest['code_sha256']==digest(Path(__file__))
    try:
        for i,(category,candidates) in enumerate(load_data('verbal-report')['candidates'].items()):
            if i%4!=rank:continue
            start=time.time();r=record(model,exp,category,candidates)
            save_json(OUT/'records'/f'{i:02d}.json',dict(result=r,seconds=time.time()-start,manifest_sha256=digest(OUT/'manifest.json')))
            print(category,r['first_text'],r['decoded_concept'],time.time()-start,flush=True)
    finally:model.close()


def analyze():
    from collections import defaultdict
    from analyze_workspace import rates,cluster_contrast
    from workspace_audit_analysis import score,cell
    rows=defaultdict(list);sources=defaultdict(list);details=[]
    files=sorted((OUT/'records').glob('*.json'));assert len(files)==14
    for p in files:
        r=json.loads(p.read_text())['result'];details.append({k:r[k] for k in ['category','first_text','decoded_concept','formatting_tokens']})
        for variant,v in r['variants'].items():
            sources[variant].append(v['source_in_category'])
            for trial in v['trials']:
                for metric in ['top1','top5']:
                    values={c:score(trial['conditions'].get(c,{}),metric=metric) for c in ['baseline','transfer','loops_2','loops_3','loops_4','swap','raw','random_42']}
                    rows[variant+'/'+metric].append((r['category'],values))
    s=dict(rates={k:rates(v) for k,v in rows.items()},valid_source_categories={k:sum(v) for k,v in sources.items()},
           contrasts={k:cluster_contrast(v,'transfer','baseline') for k,v in rows.items()},sources=details)
    save_json(OUT/'summary.json',s)
    lines=['# Verbal-report token diagnostic','',
        'Exploratory follow-up after the fixed audit exposed punctuation as the source in many categories. '
        'Only formatting tokens are consumed before scoring. A second variant identifies the greedy reported concept, '
        'then uses a complete-token concept direction if one exists. Unresolved concepts remain failures. '
        'This changes the prompt/scored position and cannot be claimed identical to the paper. It does not replace the frozen main result.','',
        '| Variant / score | Transfer | Unedited | Coordinate exchange | Random |','|---|---:|---:|---:|---:|']
    for k,v in s['rates'].items():lines.append('| '+k+' | '+' | '.join(cell(v[c]) for c in ['transfer','baseline','swap','random_42'])+' |')
    lines+=['','Valid category-source counts: '+str(s['valid_source_categories']), '',
        'Complete prefixes, emitted token sequences, unresolved cases, and all seven conditions are in the per-category records.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def run():
    import fcntl
    OUT.mkdir(exist_ok=True)
    with (OUT/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'manifest.json').exists():raise FileExistsError('Preserving existing diagnostic')
        save_json(OUT/'manifest.json',dict(created_unix=time.time(),code_sha256=digest(Path(__file__)),
            primary_manifest_sha256=digest(ROOT/'results/workspace-audit/manifest.json'),
            trigger='Primary prompt produced many punctuation/split-token source answers',
            status='Exploratory diagnostic; not an independent replication',formatting_token_limit=4,greedy_word_token_limit=8,gpus=[0,1,2,3]))
        jobs=[];streams=[];start=time.time()
        try:
            for rank in range(4):
                stream=(OUT/f'worker-{rank}.log').open('w');streams.append(stream)
                jobs.append(subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--rank',str(rank)],
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false'),
                    cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT))
            while any(p.poll() is None for p in jobs):
                codes=[p.poll() for p in jobs]
                save_json(OUT/'status.json',dict(stage='running',exit_codes=codes,records=len(list((OUT/'records').glob('*.json')))))
                if any(c not in (None,0) for c in codes):raise RuntimeError(str(codes))
                time.sleep(5)
            assert all(p.returncode==0 for p in jobs)
            analyze();save_json(OUT/'status.json',dict(stage='complete',exit_codes=[0]*4,records=14,seconds=time.time()-start))
        finally:
            for p in jobs:
                if p.poll() is None:p.terminate()
            for p in jobs:p.wait()
            for f in streams:f.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['run','worker']);p.add_argument('--rank',type=int,default=0);a=p.parse_args()
    run() if a.phase=='run' else worker(a.rank)
