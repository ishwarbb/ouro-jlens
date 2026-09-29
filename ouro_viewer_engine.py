"""Inference, exact concept traces and single-cell sparse inventories for the UI."""
from contextlib import contextmanager,nullcontext
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

import torch
from fit_ouro_family import ROOT,digest
from ouro_full_fit import load_model
from ouro_lens_family import OuroLensFamily
from workspace_validated import ValidatedIO,terminal_ids,answer_text,word_atom,prefix_transfer
from workspace_interventions import Directions
from workspace_inventory import dictionary,pursuit,ablation

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
            base=self.store.get(payload['run_id'])
            return self.analyze(base['input'],update,edit=payload,baseline=base)

    def prompt(self,payload):
        if payload.get('mode','raw')=='raw':return self.io.prompt(payload['prompt'],limit=512)
        messages=[]
        if payload.get('system'):messages.append(dict(role='system',content=payload['system']))
        messages.append(dict(role='user',content=payload['prompt']))
        text=self.io.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)+payload.get('prefill','')
        return self.io.prompt(text,wrapper='viewer-chat-v1',limit=512)

    def matrix(self,site):
        target=self.family.target_for(site)
        return None if site==target else self.family.lenses[target].jacobians[site].to(self.device)

    @contextmanager
    def edit(self,spec,prefix_length):
        source=self.io.vocab.direction_token(spec['source'])
        if source is None:raise ValueError(f'“{spec["source"]}” has no complete-token direction. Choose another word.')
        sites=[(loop-1)*48+p for loop in sorted(set(spec['loops'])) for p in range(spec['band'][0],spec['band'][1]+1)]
        if spec['operation']=='transfer':
            destination=self.io.vocab.direction_token(spec['destination'])
            if destination is None:raise ValueError(f'“{spec["destination"]}” has no complete-token direction. Choose another word.')
            if destination==source:raise ValueError('Source and destination resolve to the same token')
            with prefix_transfer(self.model,self.bank,source,destination,sites,prefix_length):yield
        else:
            bases={s:self.bank.get(s,source).reshape(1,-1,1).expand(prefix_length,-1,-1) for s in sites}
            with ablation(self.model,bases):yield

    def analyze(self,payload,update,*,edit=None,baseline=None):
        started=time.time();prompt=self.prompt(payload);limit=payload.get('max_new_tokens',24)
        if len(prompt.ids)+limit>512:raise ValueError(f'Prompt has {len(prompt.ids)} tokens. Prompt plus output must fit within 512; shorten the prompt or output.')
        ids=list(prompt.ids);generated=[];stopped=False;stops=terminal_ids(self.io.tokenizer)
        with self.edit(edit,len(ids)) if edit else nullcontext():
            for step in range(limit):
                update('Generating continuation',.05+.28*step/max(1,limit),generated_tokens=step,
                       partial_text=self.io.tokenizer.decode(generated))
                logits,_=self.io.forward(ids)
                token=int(logits.argmax());ids.append(token);generated.append(token)
                if token in stops:stopped=True;break
            update('Recording all four loops',.35,partial_text=self.io.tokenizer.decode(generated))
            _,acts=self.io.forward(ids,sites=range(192))
        states={s:acts[s][0].cpu() for s in range(192)};del acts
        layers=[];vocab_ids=set(ids)
        for site in range(192):
            update(f'Reading loop {site//48+1} · layer {site%48}',.37+.60*(site+1)/192)
            h=states[site].to(self.device).float();j=self.matrix(site)
            row=dict(site=site,loop=site//48+1,physical=site%48,target=self.family.target_for(site))
            for name,residual in [('jlens',h if j is None else h@j.T),('logit',h)]:
                logits=self.model.unembed(residual).float()
                probs,token_ids=logits.softmax(-1).topk(8,dim=-1)
                row[name]=dict(ids=token_ids.cpu().tolist(),probabilities=rounded(probs))
                vocab_ids.update(token_ids.flatten().tolist())
            layers.append(row)
        run_id=uuid.uuid4().hex[:12]
        visible,inside,had_thought=answer_text(self.io.tokenizer,generated)
        clean_input={k:payload.get(k,default) for k,default in [('prompt',''),('mode','raw'),('max_new_tokens',24),('system',''),('prefill','')]}
        title=payload['prompt'].strip().splitlines()[0][:68]
        if edit:title=f'{edit["source"]} '+('→ '+edit['destination'] if edit['operation']=='transfer' else 'removed')+' · '+title[:40]
        run=dict(schema_version=1,id=run_id,title=title,kind='intervention' if edit else 'live',
            created_at=datetime.now(timezone.utc).isoformat(),model=MODEL,input=clean_input,
            prompt_text=prompt.text,prompt_length=len(prompt.ids),layers=layers,
            tokens=[dict(id=t,text=self.io.tokenizer.decode([t]),position=p,segment='prompt' if p<len(prompt.ids) else 'completion',special=t in self.io.tokenizer.all_special_ids) for p,t in enumerate(ids)],
            vocabulary={str(t):self.io.tokenizer.decode([t]) for t in sorted(vocab_ids)},
            generation=dict(raw_text=self.io.tokenizer.decode(generated),answer_text=visible,output_tokens=len(generated),
                limit_reached=len(generated)==limit and not stopped and limit>0,unclosed_thought=inside,had_thought=had_thought,stopped=stopped),
            elapsed_seconds=round(time.time()-started,2),notes=[],provenance=dict(family_sha256=digest(FAMILY/'family.json'),
                native_precision='bfloat16',transport_precision='float32',top_k=8,attention='sdpa',generation='greedy; no KV cache',
                generated_ids_preserved=True,fit_prompts=1000,device=self.device.type))
        if edit:run.update(intervention=edit,baseline_id=baseline['id'],baseline_generation=baseline['generation']);run['notes'].append('Intervention acts at original prompt positions in the selected loops and layers.')
        if inside:run['notes'].append('The continuation ends inside a thought block; no completed final answer was produced.')
        if run['generation']['limit_reached']:run['notes'].append(f'Generation reached its {limit}-token budget.')
        directory=self.store.path(run_id);directory.mkdir(parents=True,exist_ok=True)
        torch.save(dict(states=states,seq_ids=ids),directory/'states.pt')
        self.store.save(run);update('Saved',1)
        return dict(kind='run',run_id=run_id)

    def states(self,run_id):
        path=self.store.path(run_id)/'states.pt'
        if not path.exists():raise ValueError('Residual states are not available for this imported run')
        return torch.load(path,map_location='cpu',weights_only=True)['states']

    def trace(self,payload,update):
        cache_path=self.store.path(payload['run_id'])/('trace-'+uuid.uuid5(uuid.NAMESPACE_OID,payload['word'].strip().casefold()).hex+'.json')
        if cache_path.exists():return json.loads(cache_path.read_text())
        run=self.store.get(payload['run_id']);states=self.states(payload['run_id']);word=payload['word'].strip()
        candidates=self.io.vocab.candidates(word)
        if not candidates:raise ValueError(f'“{word}” has no complete-token spelling. Exact single-token ranks are unavailable.')
        rows={};selected=torch.tensor(candidates,device=self.device)
        for index,site in enumerate(sorted(states)):
            update(f'Tracing {word} · loop {site//48+1}',.05+.94*(index+1)/len(states))
            h=states[site].to(self.device).float();j=self.matrix(site);row={}
            for name,residual in [('jlens',h if j is None else h@j.T),('logit',h)]:
                logits=self.model.unembed(residual).float();scores=logits[:,selected]
                row[name]=dict(ranks=(1+(logits>scores.max(-1).values[:,None]).sum(-1)).cpu().tolist(),
                    probabilities=rounded((scores.logsumexp(-1)-logits.logsumexp(-1)).exp()))
            rows[str(site)]=row
        result=dict(kind='trace',run_id=run['id'],word=word,token_ids=candidates,layers=rows,
                    scoring='Exact full-vocabulary rank; all case-insensitive whole-token surface forms; ties use strict greater-than')
        write_path=self.store.path(run['id'])/('trace-'+uuid.uuid5(uuid.NAMESPACE_OID,word.casefold()).hex+'.json')
        from ouro_viewer import write_json
        write_json(write_path,result)
        return result

    def inventory(self,payload,update):
        cache_path=self.store.path(payload['run_id'])/f'cell-{payload["site"]:03d}-{payload["position"]:03d}.json'
        if cache_path.exists():return json.loads(cache_path.read_text())
        states=self.states(payload['run_id']);site=payload['site'];position=payload['position']
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
