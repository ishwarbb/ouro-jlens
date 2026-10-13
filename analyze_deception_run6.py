"""Read saved online states and report Run 6's preregistered diagnostics."""
from __future__ import annotations
from collections import defaultdict
from html import escape
import json
import math
import random
import re

from prepare_deception_run4 import read,sha
from run_deception_run4 import dump,atomic_text,now
from prepare_deception_run6 import ROOT,SITES,WORDS,CONDITIONS


def windows(g,tok):
    N=len(g['input_ids']);ids=g['output_ids'];raw=g['raw_text']
    stop=len(ids)-int(bool(ids and ids[-1] in (0,2)))
    marker=re.search(r'"(?:answer|choice|result)"\s*:\s*',raw)
    eligible=[]
    if marker:
        for index in range(stop):
            consumed=tok.decode(ids[:index+1])
            # No source token that overlaps even the opening value quote is read.
            if raw.startswith(consumed) and len(consumed)<=marker.end():eligible.append(N+index)
    return dict(last_prompt=[N-1],early_before_answer=eligible[:16],before_answer=eligible[-16:])


def prefix_absent(g,tok,positions,words):
    N=len(g['input_ids']);history='\n'.join(m['content'] for m in g['messages'])
    return {p:{w:not bool(re.search(r'(?<!\w)'+re.escape(w)+r'(?!\w)',
         history+('\n'+tok.decode(g['output_ids'][:p-N+1]) if p>=N else ''),re.I)) for w in words} for p in positions}


def mean(values):
    values=[x for x in values if x is not None]
    return sum(values)/len(values) if values else None


class LensScorer:
    def __init__(self,root,model):
        from ouro_lens_family import OuroLensFamily
        self.root,self.model,self.tok=root,model,model.tokenizer
        family=OuroLensFamily.load(ROOT/'results/jlens-family-1000');family.check_model(model)
        self.matrices={s:family.lenses[family.target_for(s)].jacobians[s].to('cuda') for s in SITES}
        self.support=read(root/'plan/vocabulary.json')['word_tokens']

    def score(self,path):
        import torch
        from run_deception_run6 import save_tensor,status
        out=path.parent/'lens';out.mkdir(exist_ok=True);metric=out/'metrics.json'
        if metric.exists():
            m=read(metric)
            assert m['generation_sha256']==sha(path) and m['execution_sha256']==sha(self.root/'execution.json')
            assert m['readouts_sha256']==sha(out/'readouts.pt')
            return m
        g=read(path)
        for name,h in g['artifacts'].items():assert sha(path.parent/name)==h,name
        state=torch.load(path.parent/'states.pt',map_location='cpu',weights_only=True)
        assert state['sites']==SITES and state['input_ids']==g['input_ids'] and state['output_ids']==g['output_ids']
        assert self.tok.decode(g['output_ids'])==g['raw_text']
        w=windows(g,self.tok);positions=sorted({p for ps in w.values() for p in ps})
        lookup={p:i for i,p in enumerate(state['positions'])}
        assert all(p in lookup for p in positions)
        H=state['states'][:,[lookup[p] for p in positions]]
        assert torch.isfinite(H).all()
        words=sorted(set(WORDS)|({g['target'],g['foil']} if g['phase']=='measurement' else set()))
        absent=prefix_absent(g,self.tok,positions,words)
        status(self.root,'scoring_online_capture',episode=g['id'],phase=g['phase'],turn=g['turn'],positions=len(positions))
        readouts={};summaries={}
        with torch.inference_mode():
            for kind in ('jacobian','raw_logit'):
                shape=(len(SITES),len(positions))
                top=torch.empty((*shape,25),dtype=torch.int32);vals=torch.empty_like(top,dtype=torch.float32)
                ranks=torch.empty((*shape,len(words)),dtype=torch.int32);ties=torch.empty_like(ranks)
                concept_values=torch.empty_like(ranks,dtype=torch.float32)
                for si,site in enumerate(SITES):
                    for start in range(0,len(positions),24):
                        h=H[si,start:start+24].to('cuda').float()
                        scores=self.model.unembed(h@self.matrices[site].T if kind=='jacobian' else h).float()
                        assert torch.isfinite(scores).all()
                        end=start+len(h);v,idx=scores.topk(25,-1)
                        top[si,start:end]=idx.cpu().int();vals[si,start:end]=v.cpu()
                        for wi,word in enumerate(words):
                            best=scores[:,self.support[word]].max(-1).values
                            concept_values[si,start:end,wi]=best.cpu()
                            ranks[si,start:end,wi]=((scores>best[:,None]).sum(-1)+1).cpu().int()
                            ties[si,start:end,wi]=(scores==best[:,None]).sum(-1).cpu().int()
                        del h,scores,v,idx
                readouts[kind]=dict(top_ids=top,top_values=vals,concept_ranks=ranks,concept_ties=ties,concept_values=concept_values)
                raw_hits={word:torch.isin(top[:,:,:10],torch.tensor(self.support[word],dtype=torch.int32)).any(-1) for word in words}
                union=torch.zeros(shape,dtype=torch.bool);clean=torch.zeros_like(union)
                for word in WORDS:
                    union|=raw_hits[word]
                    clean|=raw_hits[word]&torch.tensor([absent[p][word] for p in positions])[None,:]
                summaries[kind]={}
                for name,ps in w.items():
                    indices=[positions.index(p) for p in ps]
                    loop_values=[]
                    for loop in range(4):
                        site_indices=[i for i,s in enumerate(SITES) if s//48==loop]
                        def extract(t):return t[site_indices][:,indices]
                        cells=len(site_indices)*len(indices)
                        value=dict(loop=loop,cells=cells,raw_concept_rate=float(extract(union).float().mean()) if cells else None,
                                   prefix_absent_concept_rate=float(extract(clean).float().mean()) if cells else None,
                                   per_word={word:float(extract(raw_hits[word]).float().mean()) if cells else None for word in WORDS})
                        if g['phase']=='measurement':
                            ti,fi=words.index(g['target']),words.index(g['foil'])
                            rt,rf=extract(ranks[:,:,ti]),extract(ranks[:,:,fi])
                            allowed=torch.tensor([absent[p][g['target']] for p in ps])[None,:].expand(len(site_indices),-1)
                            target_hits=extract(raw_hits[g['target']])
                            value.update(target_rank=rt.float().mean().item() if cells else None,
                                foil_rank=rf.float().mean().item() if cells else None,
                                target_beats_foil=(rt<rf).float().mean().item() if cells else None,
                                target_foil_tie=(rt==rf).float().mean().item() if cells else None,
                                target_top10=target_hits.float().mean().item() if cells else None,
                                target_prefix_absent_cells=int(allowed.sum()),
                                target_top10_when_prefix_absent=target_hits[allowed].float().mean().item() if allowed.any() else None)
                        loop_values.append(value)
                    summaries[kind][name]=dict(loops=loop_values,cells=sum(x['cells'] for x in loop_values),
                        raw_concept_rate=mean([x['raw_concept_rate'] for x in loop_values]),
                        prefix_absent_concept_rate=mean([x['prefix_absent_concept_rate'] for x in loop_values]))
        save_tensor(torch,dict(sites=SITES,positions=positions,words=words,windows=w,readouts=readouts),out/'readouts.pt')
        m=dict(generation_sha256=sha(path),execution_sha256=sha(self.root/'execution.json'),
               readouts_sha256=sha(out/'readouts.pt'),windows=w,prefix_absence=absent,summaries=summaries)
        dump(metric,m);return m


def quantile(values,p):
    values=sorted(values);position=(len(values)-1)*p;lo=int(position);hi=math.ceil(position)
    return values[lo]+(values[hi]-values[lo])*(position-lo)


def clustered_contrasts(rows,metric):
    by_record=defaultdict(lambda:defaultdict(list));templates={}
    for row in rows:
        by_record[row['record_id']][row['condition']].append(row[metric]);templates[row['record_id']]=row['template_id']
    output={}
    for condition in CONDITIONS[1:]:
        clusters=defaultdict(list)
        for record,conditions in by_record.items():
            a,b=conditions.get(condition,[]),conditions.get('plain',[])
            if len(a)!=2 or len(b)!=2 or any(x is None for x in a+b):continue
            clusters[templates[record]].append(mean(a)-mean(b))
        values=[mean(xs) for xs in clusters.values() if len(xs)==4]
        if len(values)!=12:
            output[condition]=dict(status='missing_complete_template_pairs',templates=len(values),estimate=None,ci95=None);continue
        rng=random.Random(2026101106)
        draws=[mean([rng.choice(values) for _ in values]) for _ in range(10000)]
        output[condition]=dict(status='descriptive',templates=len(values),estimate=mean(values),
            ci95=[quantile(draws,.025),quantile(draws,.975)],template_values={t:mean(xs) for t,xs in clusters.items()})
    return output


def report(root,final=False):
    jobs=read(root/'plan/jobs.json');episodes=[];rows=[];measurement=defaultdict(list)
    for job in jobs:
        ep_path=root/'cases'/job['id']/'episode.json'
        if not ep_path.exists():continue
        ep=read(ep_path);episodes.append(ep);outcome=ep['outcome']
        lens=read(ep_path.parent/'turn-0/lens/metrics.json')['summaries']
        row={k:job[k] for k in ('id','phase','family','record_id','template_id','condition')}
        row.update(outcome)
        for kind in ('jacobian','raw_logit'):
            row[kind+'_raw_rate']=lens[kind]['last_prompt']['raw_concept_rate']
            row[kind+'_absent_rate']=lens[kind]['last_prompt']['prefix_absent_concept_rate']
        rows.append(row)
        if job['phase']=='measurement':
            measurement[job['family']].append(dict(job=job,outcome=outcome,lens=lens))
    measurement_summary={}
    for family,items in measurement.items():
        view=dict(planned=8,saved=len(items),correct=sum(x['outcome']['task_correct'] for x in items),
                  complete=sum(x['outcome']['complete'] for x in items),readout={})
        for kind in ('jacobian','raw_logit'):
            view['readout'][kind]={}
            for window in ('last_prompt','early_before_answer','before_answer'):
                view['readout'][kind][window]=[]
                for loop in range(4):
                    values=[x['lens'][kind][window]['loops'][loop] for x in items]
                    correct=[x['lens'][kind][window]['loops'][loop] for x in items if x['outcome']['task_correct']]
                    view['readout'][kind][window].append(dict(loop=loop,
                        measured=sum(v['cells']>0 for v in values),
                        target_beats_foil=mean([v['target_beats_foil'] for v in values]),
                        target_top10=mean([v['target_top10'] for v in values]),
                        correct_answers_only_target_beats_foil=mean([v['target_beats_foil'] for v in correct]),
                        target_top10_when_prefix_absent=mean([v['target_top10_when_prefix_absent'] for v in values])))
        measurement_summary[family]=view
    capability=[r for r in rows if r['phase']=='capability'];groups={}
    for family in ('repair','metrics','correction','approval'):
        for condition in CONDITIONS:
            rs=[r for r in capability if r['family']==family and r['condition']==condition]
            groups[family+'/'+condition]=dict(planned=24,saved=len(rs),missing=24-len(rs),
                complete=sum(r['complete'] for r in rs),valid_actions=sum(r['valid_action'] for r in rs),
                task_correct=sum(r['task_correct'] for r in rs),report_correct=sum(r['report_correct'] for r in rs),
                task_accuracy_attempted=mean([r['task_correct'] for r in rs]),
                jacobian_raw_rate=mean([r['jacobian_raw_rate'] for r in rs]),
                jacobian_absent_rate=mean([r['jacobian_absent_rate'] for r in rs]))
    contrasts=({m:clustered_contrasts(capability,m) for m in ('task_correct','report_correct','jacobian_raw_rate','jacobian_absent_rate','raw_logit_raw_rate')}
               if final and len(capability)==288 else dict(status='pending_complete_capability_run'))
    result=dict(utc=now(),execution_sha256=sha(root/'execution.json'),planned=len(jobs),saved=len(rows),
        measurement=measurement_summary,capability=groups,condition_contrasts=contrasts,episodes=rows,
        claim='Known-answer measurement and benign task capability diagnostics only. No deception labels, detector validation or causal claim.',
        limitations=['Finite choice actions are easier than free-form planning.',
            'Negative-control readouts measure response to task context; wrong choices do not establish intent.',
            'Run 5 scarcity is assumed for planning, not inferred here.'])
    dump(root/'analysis.json',result)
    compact={f:{k:v for k,v in data.items() if k!='readout'} for f,data in measurement_summary.items()}
    table=''.join('<tr>'+''.join('<td>'+escape(str(r.get(k,'')))+'</td>' for k in ('id','complete','task_correct','report_correct','jacobian_raw_rate'))+'</tr>' for r in rows)
    atomic_text(root/'analysis.html','<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="30"><title>Run 6 results</title><style>body{font:16px/1.5 system-ui;max-width:1300px;margin:32px auto;padding:0 24px}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap}</style><h1>Run 6: measurement and benign capability</h1><p>'+str(len(rows))+' / '+str(len(jobs))+' episodes saved. No causal stage or deceptive-intent labels.</p><p><a href="index.html">Progress</a> · <a href="analysis.json">Full analysis including per-loop readouts</a> · <a href="plan/protocol.json">Frozen protocol</a></p><h2>Known-answer checks</h2><pre>'+escape(json.dumps(compact,indent=2))+'</pre><h2>Capability conditions</h2><pre>'+escape(json.dumps(groups,indent=2))+'</pre><table><tr><th>Episode</th><th>Complete</th><th>Task correct</th><th>Report correct</th><th>J-lens word rate at prompt end</th></tr>'+table+'</table>')
    return result
