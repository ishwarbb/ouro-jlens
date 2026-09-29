"""Inference, exact concept traces and single-cell sparse inventories for the UI."""
from contextlib import contextmanager,nullcontext
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import shutil
import time
import uuid

import torch
from fit_ouro_family import ROOT,digest
from ouro_full_fit import load_model
from ouro_lens_family import OuroLensFamily
from workspace_validated import ValidatedIO,terminal_ids,answer_text,word_atom
from workspace_interventions import Directions
from workspace_inventory import dictionary,pursuit

from ouro_viewer_inference import CachedDecoder,prefix_edit
from ouro_viewer_limits import DEFAULT_NEW_TOKENS,MAX_PROMPT_TOKENS,READOUT_WINDOW,PREFILL_CHUNK,READOUT_BATCH,window_bounds

FAMILY=ROOT/'results/jlens-family-1000'
MODEL=dict(name='Ouro-2.6B',revision='1ed04250da1a9936042725d302e81c8fa2ab5abd',loops=4,layers_per_loop=48,
           fit_prompts=1000,routing='Each source uses its own loop-end Jacobian. Loop ends use identity transport.',
           workspace_band=[24,40],layer_indexing='zero based')


def rounded(tensor):
    return tensor.detach().float().cpu().numpy().astype('float64').round(7).tolist()


def cuda_has_room():
    """Avoid competing with resident GPU jobs; inspect without creating a context."""
    try:
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]
        result=subprocess.run(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],
                              capture_output=True,text=True,timeout=5,check=True)
        return int(result.stdout.strip())>=10240
    except (OSError,ValueError,subprocess.SubprocessError):return False


def inventory_record(result):
    norms=result['state_norm'].clamp_min(1e-30)
    return dict(ids=result['token_ids'].tolist(),snapshots={str(k):dict(
        coefficients=rounded(v['coefficients']/norms[:,None]),residual_fraction=rounded(v['residual_fraction']))
        for k,v in result['snapshots'].items()},coefficient_units='coefficient / state norm')


class Engine:
    def __init__(self,store):
        self.preference=os.environ.get('OURO_VIEWER_DEVICE','auto')
        use_cuda=self.preference!='cpu' and cuda_has_room() and torch.cuda.is_available()
        if self.preference=='cuda' and not use_cuda:raise RuntimeError('The selected GPU needs at least 10 GiB free. Use automatic mode for CPU fallback.')
        torch.set_num_threads(2 if use_cuda else 8);torch.backends.cuda.matmul.allow_tf32=False
        self.store=store
        if use_cuda:self.model=load_model()
        else:
            from transformers import AutoTokenizer,AutoModelForCausalLM
            from ouro_lens import OuroLensModel
            tokenizer=AutoTokenizer.from_pretrained(ROOT/'model',trust_remote_code=True)
            hf=AutoModelForCausalLM.from_pretrained(ROOT/'model',trust_remote_code=True,dtype=torch.bfloat16,
                device_map='cpu',attn_implementation='sdpa').eval()
            self.model=OuroLensModel(hf,tokenizer)
        self.family=OuroLensFamily.load(FAMILY)
        self.io=ValidatedIO(self.model);self.bank=Directions(self.model,self.family);self.allowed=None

    @property
    def device(self):return self.model.hf_model.lm_head.weight.device

    def execute(self,payload,update):
        kind=payload['kind']
        if self.device.type=='cpu' and self.preference=='auto' and cuda_has_room() and torch.cuda.is_available():
            update('A GPU is free; moving Ouro to it',.03)
            self.model.hf_model.to('cuda');self.bank=Directions(self.model,self.family);self.allowed=None
            torch.set_num_threads(2)
        with torch.inference_mode():
            if kind=='analyze':return self.analyze(payload,update)
            if kind=='trace':return self.trace(payload,update)
            if kind=='inventory':return self.inventory(payload,update)
            if kind=='window':return self.window(payload,update)
            base=self.store.get(payload['run_id'])
            return self.analyze(base['input'],update,edit=payload,baseline=base)

    def prompt(self,payload):
        if payload.get('mode','raw')=='raw':return self.io.prompt(payload['prompt'],limit=MAX_PROMPT_TOKENS)
        messages=[]
        if payload.get('system'):messages.append(dict(role='system',content=payload['system']))
        messages.append(dict(role='user',content=payload['prompt']))
        text=self.io.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)+payload.get('prefill','')
        return self.io.prompt(text,wrapper='viewer-chat-v1',limit=MAX_PROMPT_TOKENS)

    def matrix(self,site):
        target=self.family.target_for(site)
        return None if site==target else self.family.lenses[target].jacobians[site].to(self.device)

    @contextmanager
    def edit(self,spec,prefix_length,position):
        source=self.io.vocab.direction_token(spec['source'])
        if source is None:raise ValueError(f'“{spec["source"]}” has no complete-token direction. Choose another word.')
        sites=[(loop-1)*48+p for loop in sorted(set(spec['loops'])) for p in range(spec['band'][0],spec['band'][1]+1)]
        if spec['operation']=='transfer':
            destination=self.io.vocab.direction_token(spec['destination'])
            if destination is None:raise ValueError(f'“{spec["destination"]}” has no complete-token direction. Choose another word.')
            if destination==source:raise ValueError('Source and destination resolve to the same token')
        else:destination=None
        with prefix_edit(self.model,self.bank,source,destination,sites,prefix_length,position):yield

    def analyze(self,payload,update,*,edit=None,baseline=None):
        started=time.time();prompt=self.prompt(payload);limit=payload.get('max_new_tokens',DEFAULT_NEW_TOKENS)
        capacity=len(prompt.ids)+limit
        if capacity>self.model.hf_model.config.max_position_embeddings:raise ValueError('Prompt plus output exceeds the model context limit')
        ids=list(prompt.ids);generated=[];stopped=False;stops=terminal_ids(self.io.tokenizer)
        # Host storage grows with the sequence; GPU captures grow only with the chunk.
        states=torch.empty((self.model.n_layers,capacity,self.model.d_model),dtype=self.model.hf_model.lm_head.weight.dtype)
        decoder=CachedDecoder(self.model)
        with self.edit(edit,len(ids),lambda:decoder.position) if edit else nullcontext():
            with decoder:
                for start in range(0,len(ids),PREFILL_CHUNK):
                    chunk=ids[start:start+PREFILL_CHUNK]
                    update('Reading prompt',.04+.04*start/len(ids))
                    logits,captured=decoder.forward(chunk)
                    states[:,start:start+len(chunk)].copy_(captured)
                for step in range(limit):
                    token=int(logits.argmax());ids.append(token);generated.append(token)
                    extra=dict(partial_text=self.io.tokenizer.decode(generated)) if step%8==0 or token in stops else {}
                    update(f'Generating · {step+1:,} / {limit:,} tokens',.08+.60*(step+1)/max(1,limit),generated_tokens=step+1,**extra)
                    logits,captured=decoder.forward([token])
                    states[:,len(ids)-1:len(ids)].copy_(captured)
                    if token in stops:stopped=True;break
        del logits,captured,decoder
        if self.device.type=='cuda':torch.cuda.empty_cache()
        run_id=uuid.uuid4().hex[:12];directory=self.store.path(run_id)
        try:
            for site in range(self.model.n_layers):
                update(f'Saving residuals · {site+1} / {self.model.n_layers}',.69+.08*(site+1)/self.model.n_layers,
                       partial_text=self.io.tokenizer.decode(generated) if site==0 else None)
                path=directory/'states'/f'site-{site:03d}.pt';path.parent.mkdir(parents=True,exist_ok=True)
                # Clone avoids serializing unused capacity or the other sites' storage.
                torch.save(states[site,:len(ids)].clone(),path)
            from ouro_viewer import write_json
            write_json(directory/'states/index.json',dict(format='per-site-v1',seq_ids=ids,sites=list(range(self.model.n_layers)),
                       dtype=str(states.dtype),shape=[self.model.n_layers,len(ids),self.model.d_model]))
            del states
            start=(len(prompt.ids)-1)//READOUT_WINDOW*READOUT_WINDOW
            page=self.readouts(run_id,start,len(ids),update)
            visible,inside,had_thought=answer_text(self.io.tokenizer,generated)
            clean_input={k:payload.get(k,default) for k,default in [('prompt',''),('mode','raw'),('max_new_tokens',DEFAULT_NEW_TOKENS),('system',''),('prefill','')]}
            title=payload['prompt'].strip().splitlines()[0][:68]
            if edit:title=f'{edit["source"]} '+('→ '+edit['destination'] if edit['operation']=='transfer' else 'removed')+' · '+title[:40]
            run=dict(schema_version=2,id=run_id,title=title,kind='intervention' if edit else 'live',
                created_at=datetime.now(timezone.utc).isoformat(),model=MODEL,input=clean_input,
                prompt_text=prompt.text,prompt_length=len(prompt.ids),layers=page['layers'],readout_window=page['window'],
                tokens=[dict(id=t,text=self.io.tokenizer.decode([t]),position=p,segment='prompt' if p<len(prompt.ids) else 'completion',special=t in self.io.tokenizer.all_special_ids) for p,t in enumerate(ids)],
                vocabulary=page['vocabulary']|{str(t):self.io.tokenizer.decode([t]) for t in set(ids)},
                generation=dict(raw_text=self.io.tokenizer.decode(generated),answer_text=visible,output_tokens=len(generated),
                    limit_reached=len(generated)==limit and not stopped and limit>0,unclosed_thought=inside,had_thought=had_thought,stopped=stopped),
                elapsed_seconds=round(time.time()-started,2),notes=[],provenance=dict(family_sha256=digest(FAMILY/'family.json'),
                    native_precision='bfloat16',transport_precision='float32',state_format='per-site-v1',top_k=8,attention='sdpa',
                    generation='greedy; explicit DynamicCache, separate KV entries for all recurrent sites',prefill_chunk=PREFILL_CHUNK,
                    readout_window_size=READOUT_WINDOW,generated_ids_preserved=True,fit_prompts=1000,device=self.device.type))
            if edit:run.update(intervention=edit,baseline_id=baseline['id'],baseline_generation=baseline['generation']);run['notes'].append('Intervention acts at original prompt positions in the selected loops and layers.')
            if inside:run['notes'].append('The continuation ends inside a thought block; no completed final answer was produced.')
            if run['generation']['limit_reached']:run['notes'].append(f'Generation reached its {limit:,}-token budget.')
            if len(ids)>READOUT_WINDOW:run['notes'].append('All tokens and residuals are saved. Layer readouts and exact concept traces are computed in 512-token windows.')
            self.store.save(run)
        except BaseException:
            shutil.rmtree(directory,ignore_errors=True)
            raise
        update('Saved',1)
        return dict(kind='run',run_id=run_id)

    def states(self,run_id,site=None):
        directory=self.store.path(run_id)
        index=directory/'states/index.json'
        if index.exists():
            sites=json.loads(index.read_text())['sites']
            if site is not None:sites=[site] if site in sites else []
            return {s:torch.load(directory/'states'/f'site-{s:03d}.pt',map_location='cpu',weights_only=True,mmap=True) for s in sites}
        path=directory/'states.pt'
        if not path.exists():raise ValueError('Residual states are not available for this imported run')
        values=torch.load(path,map_location='cpu',weights_only=True,mmap=True)['states']
        return values if site is None else {site:values[site]} if site in values else {}

    def readouts(self,run_id,start,length,update):
        start,end=window_bounds(start,length);layers=[];vocab_ids=set();states=self.states(run_id)
        for index,site in enumerate(sorted(states)):
            update(f'Reading positions {start}–{end-1} · loop {site//48+1} · layer {site%48}',.78+.21*(index+1)/len(states))
            j=self.matrix(site)
            row=dict(site=site,loop=site//48+1,physical=site%48,target=self.family.target_for(site),position_start=start,
                     jlens=dict(ids=[],probabilities=[]),logit=dict(ids=[],probabilities=[]))
            for offset in range(start,end,READOUT_BATCH):
                h=states[site][offset:min(offset+READOUT_BATCH,end)].to(self.device).float()
                for name,residual in [('jlens',h if j is None else h@j.T),('logit',h)]:
                    logits=self.model.unembed(residual).float()
                    probs,token_ids=logits.softmax(-1).topk(8,dim=-1)
                    row[name]['ids'].extend(token_ids.cpu().tolist());row[name]['probabilities'].extend(rounded(probs))
                    vocab_ids.update(token_ids.flatten().tolist())
            layers.append(row)
        return dict(kind='window',run_id=run_id,window=dict(start=start,end=end,size=READOUT_WINDOW,total=length),layers=layers,
                    vocabulary={str(t):self.io.tokenizer.decode([t]) for t in sorted(vocab_ids)})

    def window(self,payload,update):
        run=self.store.get(payload['run_id']);start,_=window_bounds(payload['start'],len(run['tokens']))
        try:return self.store.window(run['id'],start)
        except FileNotFoundError:pass
        result=self.readouts(run['id'],start,len(run['tokens']),update)
        from ouro_viewer import write_json
        write_json(self.store.path(run['id'])/f'window-{start:06d}.json',result)
        return result

    def trace(self,payload,update):
        run=self.store.get(payload['run_id']);start,end=window_bounds(payload.get('start',0),len(run['tokens']))
        word=payload['word'].strip();key=uuid.uuid5(uuid.NAMESPACE_OID,word.casefold()).hex
        cache_path=self.store.path(run['id'])/f'trace-{key}-{start:06d}.json'
        if cache_path.exists():return json.loads(cache_path.read_text())
        legacy=self.store.path(run['id'])/f'trace-{key}.json'
        if start==0 and legacy.exists():return json.loads(legacy.read_text())
        states=self.states(run['id']);candidates=self.io.vocab.candidates(word)
        if not candidates:raise ValueError(f'“{word}” has no complete-token spelling. Exact single-token ranks are unavailable.')
        rows={};selected=torch.tensor(candidates,device=self.device)
        for index,site in enumerate(sorted(states)):
            update(f'Tracing {word} · positions {start}–{end-1} · loop {site//48+1}',.05+.94*(index+1)/len(states))
            j=self.matrix(site);row={name:dict(ranks=[],probabilities=[]) for name in ['jlens','logit']}
            for offset in range(start,end,READOUT_BATCH):
                h=states[site][offset:min(offset+READOUT_BATCH,end)].to(self.device).float()
                for name,residual in [('jlens',h if j is None else h@j.T),('logit',h)]:
                    logits=self.model.unembed(residual).float();scores=logits[:,selected]
                    row[name]['ranks'].extend((1+(logits>scores.max(-1).values[:,None]).sum(-1)).cpu().tolist())
                    row[name]['probabilities'].extend(rounded((scores.logsumexp(-1)-logits.logsumexp(-1)).exp()))
            rows[str(site)]=row
        result=dict(kind='trace',run_id=run['id'],word=word,token_ids=candidates,layers=rows,position_start=start,position_end=end,
                    scoring='Exact full-vocabulary rank; all case-insensitive whole-token surface forms; ties use strict greater-than')
        from ouro_viewer import write_json
        write_json(cache_path,result)
        return result

    def inventory(self,payload,update):
        cache_path=self.store.path(payload['run_id'])/f'cell-{payload["site"]:03d}-{payload["position"]:03d}.json'
        if cache_path.exists():return json.loads(cache_path.read_text())
        site=payload['site'];position=payload['position'];states=self.states(payload['run_id'],site)
        if site not in states:raise ValueError('This layer was not recorded in the selected run')
        if position>=len(states[site]):raise ValueError('This token position was not recorded')
        update('Building the concept dictionary',.1)
        if self.allowed is None:
            self.allowed=torch.tensor([i for i in range(len(self.io.tokenizer)) if i not in self.io.tokenizer.all_special_ids and word_atom(self.io.tokenizer.decode([i]))],device=self.device)
        j=self.matrix(site)
        if j is None:j=torch.eye(self.model.d_model,device=self.device)
        atoms=dictionary(self.model.hf_model.lm_head.weight,j,self.allowed)
        update('Fitting the sparse inventory',.45)
        result=pursuit(states[site][position:position+1].to(self.device).float(),atoms,self.allowed,k=50)
        record=inventory_record(result)
        result=dict(kind='inventory',run_id=payload['run_id'],site=site,position=position,inventory=record,
                    vocabulary={str(t):self.io.tokenizer.decode([t]) for t in result['token_ids'][0].tolist() if t>=0})
        from ouro_viewer import write_json
        write_json(cache_path,result)
        return result
