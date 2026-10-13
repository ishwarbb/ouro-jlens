"""Run 5 published-data audit, fixed online-state lens readouts and descriptive reports."""
from __future__ import annotations
import argparse
from collections import defaultdict
from html import escape
import json
from pathlib import Path
import random
import re
import sys

from prepare_deception_run4 import read,sha
from run_deception_run4 import dump,atomic_text,now
from prepare_deception_run5 import ROOT,WORDS,SITES,FAMILIES,CONDITIONS


def audit_paper_assets(root):
    import pyarrow.parquet as pq
    folder=root/'paper-assets';cases={}
    for p in sorted(folder.glob('*.parquet')):
        rows=pq.read_table(p).to_pylist();meta=read(p.with_suffix('.json'))
        fig=read(folder/('fig_'+p.stem+'.json'))
        assert {r['rank'] for r in rows}==set(range(1,11))
        cells=defaultdict(list)
        for row in rows:cells[(row['ctx'],row['layer'])].append(row)
        assert all(len(rs)==10 for rs in cells.values())
        def clean(word):return re.sub(r'[⍽↑⇪▲·\s]','',word).casefold()
        found={w:sum(any(clean(r['str'])==w for r in rs) for rs in cells.values()) for w in WORDS}
        cases[p.stem]=dict(cells=len(cells),top10_rows=len(rows),selected_positions=sorted({x[0] for x in cells}),
            source_position_slots=len(meta['ctxTokens']),nonempty_tokens=sum(bool(x) for x in meta['ctxTokens']),
            nonwhitespace_tokens=sum(bool(x.strip()) for x in meta['ctxTokens']),
            exact_display_word_cells=found,annotated_callouts=len(fig['callouts']),
            hashes={q.name:sha(q) for q in (p,p.with_suffix('.json'),folder/('fig_'+p.stem+'.json'))})
    result=dict(created_utc=now(),status='verified_selected_released_cells',cases=cases,
        source='https://transformer-circuits.pub/2026/workspace/#audit-opus-examples',
        limitations='Selected displayed token/layer cells only. Blank context slots are omitted content, not empty original tokens. These assets cannot reproduce a full prompt, full model forward or population rate. No Opus weights are present.')
    dump(root/'paper-asset-audit.json',result)
    return result


def windows(g,tok):
    prefix=len(g['input_ids']);ids=g['output_ids'];raw=g['raw_text']
    stop=len(ids)-int(bool(ids and ids[-1] in (0,2)))
    bounds=[len(tok.decode(ids[:i])) for i in range(stop+1)]
    markers=list(re.finditer(r'(?im)^\s*(ACTION|MESSAGE)\s*:',raw))
    body=markers[-1].end() if markers else None
    after=next((i for i in range(stop) if body is not None and bounds[i]>=body),None)
    return dict(last_prompt=[prefix-1],first_generated=list(range(prefix,prefix+min(20,stop))),
        before_action_or_message=[] if after is None else list(range(prefix+max(0,after-16),prefix+after)),
        action_or_message_start=[] if after is None else list(range(prefix+after,prefix+min(stop,after+16))))


def prefix_absence(g,tok,positions):
    N=len(g['input_ids']);raw_messages='\n'.join(m['content'] for m in g['messages'])
    # All retained prompt positions are at the response boundary. The primary
    # last-prompt position has read all history; generated positions add only
    # tokens consumed through that source position, never future output.
    return {p:{word:not bool(re.search(r'(?<!\w)'+re.escape(word)+r'(?!\w)',
            raw_messages+('\n'+tok.decode(g['output_ids'][:p-N+1]) if p>=N else ''),re.I))
               for word in WORDS} for p in positions}


def score_lenses(root):
    import torch
    from run_deception_run5 import verify_frozen,status,save_tensor
    ex=verify_frozen(root,assets=True)
    sys.path.insert(0,str(ROOT/'jacobian-lens'))
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from ouro_lens import OuroLensModel
    from ouro_lens_family import OuroLensFamily
    tok=AutoTokenizer.from_pretrained(ROOT/'model',trust_remote_code=True)
    hf=AutoModelForCausalLM.from_pretrained(ROOT/'model',trust_remote_code=True,dtype=torch.bfloat16,
                                         device_map='cuda',attn_implementation='sdpa').eval()
    model=OuroLensModel(hf,tok)
    family=OuroLensFamily.load(ROOT/'results/jlens-family-1000');family.check_model(model)
    matrices={s:family.lenses[family.target_for(s)].jacobians[s].to('cuda') for s in SITES}
    support=read(root/'plan/vocabulary.json')['word_tokens']
    assert all(support[w] for w in WORDS)
    files=sorted((root/'cases').glob('*/turn-*/generation.json'))
    completed=0
    try:
        with torch.inference_mode():
            for path in files:
                out=path.parent/'lens';out.mkdir(exist_ok=True)
                metric=out/'metrics.json'
                if metric.exists():
                    m=read(metric);assert m['generation_sha256']==sha(path)
                    assert m['execution_sha256']==sha(root/'execution.json')
                    assert m['readouts_sha256']==sha(out/'readouts.pt')
                    completed+=1;continue
                g=read(path)
                for name,h in g['artifacts'].items():assert sha(path.parent/name)==h
                assert tok.decode(g['output_ids'])==g['raw_text']
                state=torch.load(path.parent/'states.pt',map_location='cpu',weights_only=True)
                assert state['sites']==SITES and state['input_ids']==g['input_ids'] and state['output_ids']==g['output_ids']
                w=windows(g,tok);positions=sorted({p for ps in w.values() for p in ps})
                lookup={p:i for i,p in enumerate(state['positions'])}
                assert all(p in lookup for p in positions)
                H=state['states'][:,[lookup[p] for p in positions]]
                absent=prefix_absence(g,tok,positions)
                output={}
                for kind in ('jacobian','raw_logit'):
                    top=torch.empty((len(SITES),len(positions),25),dtype=torch.int32)
                    vals=torch.empty_like(top,dtype=torch.float32)
                    ranks=torch.empty((len(SITES),len(positions),len(WORDS)),dtype=torch.int32)
                    ties=torch.empty_like(ranks)
                    for si,s in enumerate(SITES):
                        for start in range(0,len(positions),24):
                            h=H[si,start:start+24].to('cuda').float()
                            scores=model.unembed(h@matrices[s].T if kind=='jacobian' else h).float()
                            v,idx=scores.topk(25,-1);end=start+len(h)
                            top[si,start:end]=idx.cpu().int();vals[si,start:end]=v.cpu()
                            for wi,word in enumerate(WORDS):
                                best=scores[:,support[word]].max(-1).values
                                ranks[si,start:end,wi]=((scores>best[:,None]).sum(-1)+1).cpu().int()
                                ties[si,start:end,wi]=(scores==best[:,None]).sum(-1).cpu().int()
                            del h,scores,v,idx
                    output[kind]=dict(top_ids=top,top_values=vals,concept_ranks=ranks,concept_ties=ties)
                summaries={}
                for kind,data in output.items():
                    hits=torch.zeros((len(SITES),len(positions)),dtype=torch.bool)
                    by_word={}
                    for word in WORDS:
                        allowed=torch.tensor([absent[p][word] for p in positions],dtype=torch.bool)
                        hit=torch.isin(data['top_ids'][:,:,:10],torch.tensor(support[word],dtype=torch.int32)).any(-1)&allowed[None,:]
                        by_word[word]=hit;hits|=hit
                    summaries[kind]={}
                    for name,ps in w.items():
                        indices=[positions.index(p) for p in ps]
                        cells=len(SITES)*len(indices)
                        value=dict(cells=cells,hits=int(hits[:,indices].sum()),
                            prefix_absent_rate=float(hits[:,indices].float().mean()) if cells and g['complete'] else None,
                            word_rates={word:float(hit[:,indices].float().mean()) if cells and g['complete'] else None for word,hit in by_word.items()},
                            loop_rates={str(loop+1):float(hits[loop*17:(loop+1)*17,indices].float().mean()) if cells and g['complete'] else None for loop in range(4)})
                        summaries[kind][name]=value
                save_tensor(torch,dict(sites=SITES,positions=positions,windows=w,words=list(WORDS),
                    prefix_absence=absent,**output),out/'readouts.pt')
                dump(metric,dict(generation_sha256=sha(path),execution_sha256=sha(root/'execution.json'),
                    source='Saved online cached activations; no teacher-forced state substitution',
                    complete=g['complete'],windows=w,metrics=summaries,
                    readouts_sha256=sha(out/'readouts.pt'),semantic_review_complete=False))
                completed+=1
                status(root,'lens_turn_saved',completed=completed,total=len(files),episode=g['id'],turn=g['turn'])
                del state,H,output,top,vals,ranks,ties
                torch.cuda.empty_cache()
    finally:model.close()


def interval(values):
    if not values:return None
    if len(set(values))==1:return [values[0],values[0]]
    rng=random.Random(2026101105);n=len(values)
    draws=sorted(sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(10000))
    return [draws[249],draws[9749]]


def report(root):
    from run_deception_run5 import phase_summary,verify_frozen
    verify_frozen(root)
    jobs=read(root/'plan/episodes.json');by_id={j['id']:j for j in jobs}
    episodes=[]
    for p in sorted((root/'cases').glob('*/episode.json')):
        ep=read(p)
        for rel,h in ep['artifacts'].items():assert sha(p.parent/rel)==h
        lens=[]
        for m in sorted(p.parent.glob('turn-*/lens/metrics.json')):
            value=read(m)
            assert value['generation_sha256']==sha(m.parent.parent/'generation.json')
            lens.append(value)
        row={k:ep[k] for k in ('id','record_id','family','condition','phase','repetition','complete','failure','valid_action_turns','benign_task_correct','behavioral_facts')}
        row['lens_turns']=len(lens)
        row['lens_means']={kind:{window:(sum(vals)/len(vals) if vals and ep['complete'] and len(lens)==3 else None)
            for window in ('last_prompt','first_generated','before_action_or_message','action_or_message_start')
            for vals in [[m['metrics'][kind][window]['prefix_absent_rate'] for m in lens if m['metrics'][kind][window]['prefix_absent_rate'] is not None]]}
            for kind in ('jacobian','raw_logit')}
        episodes.append(row)
    summaries={phase:phase_summary(root,phase) for phase in ('pilot','held_out')}
    contrasts={}
    for phase in ('pilot','held_out'):
        rows=[e for e in episodes if e['phase']==phase]
        records=sorted({j['record_id'] for j in jobs if j['phase']==phase})
        items=[]
        for other in ('accountable','benign'):
            values=[];missing=[]
            for rec in records:
                rates={}
                for condition in ('opportunity',other):
                    subset=[r for r in rows if r['record_id']==rec and r['condition']==condition]
                    planned=sum(j['record_id']==rec and j['condition']==condition for j in jobs)
                    cells=[r['lens_means']['jacobian']['before_action_or_message'] for r in subset]
                    rates[condition]=sum(cells)/len(cells) if len(cells)==planned and cells and all(v is not None for v in cells) else None
                if any(v is None for v in rates.values()):missing.append(rec)
                else:values.append(rates['opportunity']-rates[other])
            ready=summaries[phase]['all_accounted']
            items.append(dict(contrast='opportunity_minus_'+other,usable_record_clusters=len(values),
                planned_record_clusters=len(records),missing_records=missing,
                estimate=sum(values)/len(values) if ready and values else None,
                ci95=interval(values) if ready else None,role='descriptive complete-pair estimate; not confirmation of intention'))
        contrasts[phase]=items
    result=dict(utc=now(),execution_sha256=sha(root/'execution.json'),phases=summaries,
                episodes=episodes,lens_contrasts=contrasts,semantic_review_complete=False,deception_confirmed=False,
                missingness='No generation or unavailable/capped readout remains missing; attempted mechanical counts retained separately.',
                claim='Descriptive local action and lens evidence only. No semantic strategy timing or causal result is inferred automatically.')
    dump(root/'analysis.json',result)
    rows=''.join('<tr>'+''.join('<td>'+escape(str(row[k]))+'</td>' for k in ('id','complete','valid_action_turns','behavioral_facts','lens_turns'))+'</tr>' for row in episodes)
    atomic_text(root/'analysis.html','<!doctype html><meta charset="utf-8"><title>Run 5 analysis</title><style>body{font:16px/1.6 system-ui;margin:30px}td,th{border-bottom:1px solid #ddd;padding:8px;vertical-align:top}table{width:100%}</style><h1>Run 5 descriptive analysis</h1><p>Online cached activations and independently recorded local tool actions. Full-output, lens-blind semantic review is pending. No deceptive-intent or causal claim.</p><p><a href="analysis.json">Full results</a> · <a href="plan/protocol.json">Protocol</a></p><pre>'+escape(json.dumps(summaries,indent=2))+'</pre><table><tr><th>Episode</th><th>Complete</th><th>Valid actions</th><th>Mechanical facts</th><th>Lens turns</th></tr>'+rows+'</table>')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--lens',action='store_true')
    a=p.parse_args()
    if a.lens:score_lenses(a.output)
    report(a.output)
