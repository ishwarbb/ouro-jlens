"""Shared, versioned evaluation interface and exact pre-value capture boundaries."""
from __future__ import annotations
import json
from html import escape
import re
import time
from pathlib import Path

from prepare_deception_run4 import read,sha,digest
from run_deception_run4 import dump,atomic_text,now
from prepare_deception_run6 import ROOT,SITES,WORDS

CAPTURE_SITES=sorted(set(SITES)|{47,95,143,191})
SETTINGS=('greedy','sampled_0_9')


def visible(raw):
    text=raw.replace('<|im_end|>','').replace('<|endoftext|>','')
    return re.sub(r'<think>.*?</think>',lambda m:' '*len(m.group()),text,flags=re.S)


def json_objects(text):
    """Top-level balanced objects; never salvage a nested object from invalid JSON."""
    objects=[];errors=[];start=None;depth=0;quoted=False;escaped=False
    for i,char in enumerate(text):
        if start is None:
            if char=='{':start=i;depth=1;quoted=False;escaped=False
            continue
        if quoted:
            if escaped:escaped=False
            elif char=='\\':escaped=True
            elif char=='"':quoted=False
        elif char=='"':quoted=True
        elif char=='{':depth+=1
        elif char=='}':
            depth-=1
            if depth==0:
                def unique(pairs):
                    result={}
                    for k,v in pairs:
                        if k in result:raise ValueError('duplicate_key')
                        result[k]=v
                    return result
                try:
                    obj=json.loads(text[start:i+1],object_pairs_hook=unique)
                    objects.append((obj,(start,i+1)))
                except ValueError as err:errors.append(str(err))
                start=None
    if start is not None:errors.append('unclosed_json_object')
    return objects,errors


def parse_action(raw,complete=True):
    if not complete:return dict(value=None,error='incomplete_generation',span=None,format_compliant=False)
    text=visible(raw);objects,errors=json_objects(text)
    candidates=[(o,s) for o,s in objects if isinstance(o,dict) and 'tool' in o]
    if errors or len(candidates)!=1:
        return dict(value=None,error='invalid_json' if errors else 'ambiguous_or_missing_action',span=None,format_compliant=False)
    value,span=candidates[0]
    if type(value.get('tool')) is not str:
        return dict(value=None,error='invalid_tool_name',span=span,format_compliant=False)
    clean=text.strip()
    compliant=clean==text[span[0]:span[1]]
    return dict(value=value,error=None,span=list(span),format_compliant=compliant)


def scalar(raw,field,complete=True,choices=None):
    """Extract a unique assertion independently of the expected answer."""
    if not complete:return dict(value=None,error='incomplete_generation',span=None,format_compliant=False)
    text=visible(raw);objects,errors=json_objects(text)
    candidates=[(o,s) for o,s in objects if isinstance(o,dict) and field in o]
    if errors or len(candidates)>1:
        return dict(value=None,error='ambiguous_or_invalid_json',span=None,format_compliant=False)
    atom=r'([A-Za-z_]+|[+-]?\d+(?:\.\d+)?)'
    labels=list(re.finditer(r'(?i)\b'+re.escape(field)+r'\s*:\s*["\']?'+atom+r'["\']?(?=\s|[.!]|$)',text))
    if candidates:
        obj,span=candidates[0];value=obj[field]
        if any(not span[0]<=m.start()<span[1] for m in labels):
            return dict(value=None,error='ambiguous_scalar_assertions',span=None,format_compliant=False)
        if set(obj)!={field} or type(value) not in (str,int,float):
            return dict(value=None,error='invalid_scalar_schema',span=list(span),format_compliant=False)
        if type(value) is float:
            if not __import__('math').isfinite(value):return dict(value=None,error='nonfinite_value',span=list(span),format_compliant=False)
            value=str(int(value)) if value.is_integer() else str(value)
        else:value=str(value).strip()
        # The schema diagnostic remains strict, even when numerical correctness passes.
        compliant=type(obj[field]) is str and text.strip()==text[span[0]:span[1]]
        if choices is not None and value.upper() not in choices:
            return dict(value=None,error='unknown_choice',span=list(span),format_compliant=compliant)
        return dict(value=value.upper() if choices else value,error=None,span=list(span),format_compliant=compliant)
    if objects:return dict(value=None,error='missing_scalar_field',span=None,format_compliant=False)
    if len(labels)>1:return dict(value=None,error='ambiguous_scalar_assertions',span=None,format_compliant=False)
    if len(labels)==1:
        value=labels[0].group(1)
        if choices is None or value.upper() in choices:
            return dict(value=value.upper() if choices else value,error=None,span=list(labels[0].span(1)),format_compliant=False)
    if choices is not None:
        patterns=[r'\s*([ABC])\s*[.!]?\s*',r'\s*(?:choice|action)\s*:\s*["\']?([ABC])["\']?\s*[.!]?\s*',
                  r'\s*I (?:will )?(?:choose|select) (?:action |option )?["\']?([ABC])["\']?\s*[.!]?\s*']
        matches=[re.fullmatch(p,text,re.I) for p in patterns]
        match=next((m for m in matches if m),None)
        if match:return dict(value=match.group(1).upper(),error=None,span=list(match.span(1)),format_compliant=False)
    else:
        try:
            quoted=json.loads(text.strip())
        except ValueError:quoted=None
        if type(quoted) is str:
            start=text.index('"')+1
            return dict(value=quoted.strip(),error=None,span=[start,start+len(quoted)],format_compliant=False)
        match=re.fullmatch(r'\s*(?:(?:ANSWER|RESULT|FACT)\s*:\s*)?'+atom+r'\s*[.!]?\s*',text,re.I)
        if match:return dict(value=match.group(1),error=None,span=list(match.span(1)),format_compliant=False)
    return dict(value=None,error='ambiguous_or_missing_scalar',span=None,format_compliant=False)


def report_text(raw,complete=True):
    if not complete:return None
    text=visible(raw).strip()
    matches=list(re.finditer(r'(?im)^\s*MESSAGE\s*:',text))
    if len(matches)>1:return None
    result=text[matches[0].end():].strip() if matches else text
    return result if result else None


def value_boundary(g,tok,fields=('answer','choice','result','fact','tool')):
    """Last consumed source token containing no part of the asserted value."""
    raw=g['raw_text'];text=visible(raw);objects,errors=json_objects(text)
    starts=[]
    if not errors:
        for obj,span in objects:
            for field in fields:
                if field not in obj:continue
                m=re.search(r'"'+re.escape(field)+r'"\s*:\s*',text[span[0]:span[1]])
                if m:
                    begin=span[0]+m.end()
                    if begin<len(text) and text[begin]=='"':begin+=1
                    starts.append(begin)
    if len(starts)!=1:
        # Bare, unambiguous diagnostic values use their exact parsed assertion span.
        parsed=None
        if 'choice' in fields:parsed=scalar(raw,'choice',g['complete'],choices='ABC')
        elif 'answer' in fields:parsed=scalar(raw,'answer',g['complete'])
        elif 'fact' in fields:parsed=scalar(raw,'fact',g['complete'])
        elif 'result' in fields:parsed=scalar(raw,'result',g['complete'])
        if parsed and parsed['value'] is not None and parsed['span'] is not None and not objects:
            starts=[parsed['span'][0]]
        else:return dict(source_position=None,value_char_start=None,reason='ambiguous_or_missing_value')
    begin=starts[0];N=len(g['input_ids']);ids=g['output_ids']
    stop=len(ids)-int(bool(ids and ids[-1] in (0,2)))
    # If separator and answer share one token, withhold that entire token.
    for index in range(stop):
        consumed=tok.decode(ids[:index+1])
        if not raw.startswith(consumed):continue
        if len(consumed)>begin:
            return dict(source_position=N+index-1,value_char_start=begin,reason=None,
                        first_value_output_index=index)
    return dict(source_position=None,value_char_start=begin,reason='value_not_generated')


def corrected_windows(g,tok,fields=('answer','choice','result','fact','tool')):
    N=len(g['input_ids']);anchor=value_boundary(g,tok,fields);p=anchor['source_position']
    return dict(last_prompt=[N-1],immediate_pre_value=[] if p is None else [p],
        before_value=[] if p is None else list(range(max(0,N-32,p-15),p+1)),
        early_before_value=[] if p is None else list(range(N,min(N+16,p+1)))),anchor


def update_status(root,stage,**extra):
    value=dict(utc=now(),stage=stage,**extra);dump(root/'status.json',value)
    title=read(root/'metadata.json')['title'] if (root/'metadata.json').exists() else root.name
    rows=''.join(f'<tr><th>{escape(str(k))}</th><td>{escape(str(v))}</td></tr>' for k,v in value.items())
    atomic_text(root/'index.html','<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="30"><title>'+escape(title)+'</title><style>body{font:16px/1.6 system-ui;max-width:1200px;margin:32px auto;padding:0 20px}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}</style><h1>'+escape(title)+'</h1><p><a href="analysis.html">Results</a> · <a href="plan/protocol.json">Protocol</a> · <a href="status.json">Status data</a></p><table>'+rows+'</table>')
    with (root/'events.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
    print(json.dumps(value),flush=True)


def immutable(path,value):
    if path.exists():assert read(path)==value,str(path)
    else:dump(path,value)


def save_tensor(torch,value,path):
    temp=path.with_suffix(path.suffix+'.tmp');torch.save(value,temp);temp.replace(path)


def generate(root,job,turn,messages,model,execution_hash,setting,resources):
    import torch
    from transformers import set_seed
    from ouro_viewer_inference import CachedDecoder
    from run_deception_run6 import thought_closed
    directory=root/'cases'/job['id']/f'turn-{turn}';directory.mkdir(parents=True,exist_ok=True)
    path=directory/'generation.json';tok=model.tokenizer
    if path.exists():
        g=read(path);assert g['job_sha256']==digest(job) and g['messages']==messages
        assert g['revision_execution_sha256']==execution_hash and g['setting']==setting
        for name,h in g['artifacts'].items():assert sha(directory/name)==h
        return g,path
    prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    ids=tok.encode(prompt,add_special_tokens=False);N=len(ids)
    if N>4096:raise RuntimeError('Prompt exceeds limit; no truncation')
    set_seed(job['seed']+turn);outputs=[];positions=[];captures=[];start=time.time()
    rng=dict(seed=job['seed']+turn,cpu_before=torch.get_rng_state(),cuda_before=torch.cuda.get_rng_state())
    update_status(root,'generating',episode=job['id'],turn=turn,setting=setting,input_tokens=N)
    with torch.inference_mode(),CachedDecoder(model) as decoder:
        for begin in range(0,N,256):
            chunk=ids[begin:begin+256];logits,states=decoder.forward(chunk)
            keep=[i for i in range(len(chunk)) if begin+i>=N-32]
            if keep:captures.append(states[CAPTURE_SITES][:,keep].clone());positions.extend(begin+i for i in keep)
            del states
        first=logits.cpu()
        for step in range(resources['max_new_tokens']):
            if setting=='greedy':token=int(logits.argmax())
            elif setting=='sampled_0_9':
                probabilities=torch.softmax(logits/.9,dim=-1);values,order=probabilities.sort(descending=True)
                values[values.cumsum(-1)-values>=.95]=0
                token=int(order[int(torch.multinomial(values,1))])
            else:raise ValueError(setting)
            outputs.append(token)
            if token in (0,2):break
            logits,states=decoder.forward([token]);captures.append(states[CAPTURE_SITES].clone())
            positions.append(N+len(outputs)-1);del states
            if (step+1)%128==0:update_status(root,'generating',episode=job['id'],turn=turn,generated_tokens=step+1)
        natural=bool(outputs and outputs[-1] in (0,2));raw=tok.decode(outputs)
        rng.update(cpu_after=torch.get_rng_state(),cuda_after=torch.cuda.get_rng_state())
        save_tensor(torch,rng,directory/'rng.pt')
        save_tensor(torch,dict(states=torch.cat(captures,dim=1),sites=CAPTURE_SITES,positions=positions,
              input_ids=ids,output_ids=outputs,first_next_token_logits=first),directory/'states.pt')
        atomic_text(directory/'response.txt',raw)
        g=dict(job,job_sha256=digest(job),turn=turn,messages=messages,prompt_text=prompt,input_ids=ids,output_ids=outputs,
            raw_text=raw,assistant_history_text=tok.decode(outputs[:-1] if natural else outputs).strip(),
            complete=natural and thought_closed(tok,outputs),natural_stop=natural,capped=not natural,setting=setting,
            revision_execution_sha256=execution_hash,generated_utc=now(),seconds=time.time()-start,
            artifacts={name:sha(directory/name) for name in ('states.pt','rng.pt','response.txt')})
        dump(path,g)
    del captures,rng;torch.cuda.empty_cache()
    return g,path


class RevisionLens:
    def __init__(self,model,vocabulary):
        from ouro_lens_family import OuroLensFamily
        self.model=model;self.tok=model.tokenizer;self.support=vocabulary
        self.family=OuroLensFamily.load(ROOT/'results/jlens-family-1000');self.family.check_model(model)
        self.matrices={s:self.family.lenses[self.family.target_for(s)].jacobians[s].to('cuda') for s in SITES}

    def score(self,path,out,execution_hash,fields=None):
        import torch
        from analyze_deception_run6 import prefix_absent
        out.mkdir(parents=True,exist_ok=True);metric=out/'metrics.json'
        if metric.exists():
            m=read(metric);assert m['generation_sha256']==sha(path) and m['revision_execution_sha256']==execution_hash
            assert m['readouts_sha256']==sha(out/'readouts.pt');return m
        g=read(path)
        for name,h in g['artifacts'].items():assert sha(path.parent/name)==h
        assert self.tok.decode(g['output_ids'])==g['raw_text']
        state=torch.load(path.parent/'states.pt',map_location='cpu',weights_only=True)
        assert state['input_ids']==g['input_ids'] and state['output_ids']==g['output_ids']
        if fields is None:fields=('answer',) if g.get('phase')=='measurement' else ('choice','result')
        windows,anchor=corrected_windows(g,self.tok,fields)
        positions=sorted({p for ps in windows.values() for p in ps});lookup={p:i for i,p in enumerate(state['positions'])}
        assert all(p in lookup for p in positions)
        site_lookup={s:i for i,s in enumerate(state['sites'])};assert all(s in site_lookup for s in SITES)
        h=state['states'][[site_lookup[s] for s in SITES]][:,[lookup[p] for p in positions]]
        words=sorted(set(WORDS)|({g['target'],g['foil']} if 'target' in g else set()))
        absence=prefix_absent(g,self.tok,positions,words);readouts={};summaries={}
        with torch.inference_mode():
            for kind in ('jacobian','raw_logit'):
                ranks=torch.empty((68,len(positions),len(words)),dtype=torch.int32)
                values=torch.empty_like(ranks,dtype=torch.float32);ties=torch.empty_like(ranks)
                top=torch.empty((68,len(positions),25),dtype=torch.int32)
                for si,s in enumerate(SITES):
                    H=h[si].to('cuda').float();scores=self.model.unembed(H@self.matrices[s].T if kind=='jacobian' else H).float()
                    assert torch.isfinite(scores).all()
                    top[si]=scores.topk(25,-1).indices.cpu().int()
                    for wi,word in enumerate(words):
                        best=scores[:,self.support[word]].max(-1).values
                        values[si,:,wi]=best.cpu();ranks[si,:,wi]=((scores>best[:,None]).sum(-1)+1).cpu().int()
                        ties[si,:,wi]=(scores==best[:,None]).sum(-1).cpu().int()
                readouts[kind]=dict(top_ids=top,concept_ranks=ranks,concept_values=values,concept_ties=ties)
                summaries[kind]={}
                for name,ps in windows.items():
                    pi=[positions.index(p) for p in ps];loops=[]
                    for loop in range(4):
                        rows=[i for i,s in enumerate(SITES) if s//48==loop];cells=len(rows)*len(pi)
                        raw=torch.zeros((len(rows),len(pi)),dtype=torch.bool);absent=torch.zeros_like(raw)
                        for word in WORDS:
                            hits=torch.isin(top[rows][:,pi,:10],torch.tensor(self.support[word],dtype=torch.int32)).any(-1)
                            raw|=hits;absent|=hits&torch.tensor([absence[p][word] for p in ps],dtype=torch.bool)[None,:]
                        item=dict(loop=loop,cells=cells,raw_concept_rate=float(raw.float().mean()) if cells else None,
                                  prefix_absent_concept_rate=float(absent.float().mean()) if cells else None)
                        if 'target' in g:
                            ti,fi=words.index(g['target']),words.index(g['foil'])
                            rt,rf=ranks[rows][:,pi,ti],ranks[rows][:,pi,fi]
                            margins=values[rows][:,pi,ti]-values[rows][:,pi,fi]
                            hits=torch.isin(top[rows][:,pi,:10],torch.tensor(self.support[g['target']],dtype=torch.int32)).any(-1)
                            item.update(target_rank=float(rt.float().mean()) if cells else None,
                                foil_rank=float(rf.float().mean()) if cells else None,
                                target_beats_foil=float((rt<rf).float().mean()) if cells else None,
                                target_foil_margin=float(margins.mean()) if cells else None,
                                target_top10=float(hits.float().mean()) if cells else None)
                        loops.append(item)
                    summaries[kind][name]=dict(loops=loops,cells=sum(i['cells'] for i in loops))
        save_tensor(torch,dict(sites=SITES,positions=positions,words=words,windows=windows,readouts=readouts),out/'readouts.pt')
        m=dict(generation_sha256=sha(path),revision_execution_sha256=execution_hash,
               readouts_sha256=sha(out/'readouts.pt'),boundary=anchor,windows=windows,summaries=summaries)
        dump(metric,m);return m
