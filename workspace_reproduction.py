"""Run the predeclared causal and readout tests on a fitted Ouro lens family."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import transformers
from jlens.hooks import ActivationRecorder
from ouro_full_fit import load_model
from ouro_lens_family import OuroLensFamily
from fit_ouro_family import ROOT,save_json,digest,weight_identity
from workspace_interventions import Directions,Vocabulary,intervention,score_tokens

DATA=ROOT/'jacobian-lens/data/experiments'
DEFAULT_OUTPUT=ROOT/'results/workspace-full'
PROTOCOL=ROOT/'workspace_protocol.json'


def load_data(name):return json.loads((DATA/f'{name}.json').read_text())


def items():
    result=[]
    for category in load_data('flexible-generalization')['categories']:
        for function in category['funcs']:
            for argument in category['args']:
                result.append(dict(family='flexible',category=category['name'],function=function,
                    argument=argument,arguments=category['args']))
    result.extend(dict(family='probe',item=item) for item in load_data('probe-swap')['items'])
    result.extend(dict(family='report',category=name,candidates=words)
                  for name,words in load_data('verbal-report')['candidates'].items())
    result.extend(dict(family='language',passage=item) for item in load_data('selectivity-language')['passages'])
    modulation=load_data('directed-modulation')
    for index,topic in enumerate(modulation['topic_categories']):
        result.append(dict(family='modulation',topic=topic,
                           carrier=modulation['carrier_sentences'][index%len(modulation['carrier_sentences'])]))
    result.extend(dict(family='introspection',concept=item)
                  for item in load_data('verbal-introspection')['concepts'])
    for index,item in enumerate(result):item['id']=f'{index:04d}-{item["family"]}'
    return result


class Experiment:
    def __init__(self,model,family):
        self.model,self.family=model,family
        self.vocab=Vocabulary(model.tokenizer)
        self.bank=Directions(model,family)
        self.targets=list(family.lenses)
        self.band=list(range(24,41))
        self.sites=[loop*48+physical for loop in (1,2,3) for physical in self.band]

    def chat(self,user,prefill=''):
        return self.model.tokenizer.apply_chat_template([dict(role='user',content=user)],
            tokenize=False,add_generation_prompt=True)+prefill

    def span(self,text,start,end):
        encoded=self.model.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
        actual=self.model.encode(text,max_length=2048)[0].tolist()
        raw=encoded['input_ids']
        shift=1 if len(actual)==len(raw)+1 else 0
        if actual[shift:]!=raw:raise AssertionError('Token span mapping differs from model input')
        return [index+shift for index,(a,b) in enumerate(encoded['offset_mapping'])
                if b>a and b>start and a<end]

    @torch.no_grad()
    def forward(self,text,*,record=(),edit=None):
        ids=self.model.encode(text,max_length=2048)
        context=intervention(self.model,self.bank,**edit) if edit else nullcontext()
        with context,ActivationRecorder(self.model.layers,at=sorted(set(record)|set(self.targets))) as recorder:
            self.model.forward(ids)
        logits={target:self.model.unembed(recorder.activations[target][:,-1:,:])[0,0].float()
                for target in self.targets}
        if any(not torch.isfinite(value).all() for value in logits.values()):
            raise FloatingPointError('Nonfinite intervention output')
        return ids,recorder.activations,logits

    def outputs(self,logits,words):
        return {str(target):{name:score_tokens(value,self.vocab.candidates(word))
                            for name,word in words.items()} for target,value in logits.items()}

    def top(self,logits):
        values,ids=logits.softmax(-1).topk(5)
        return [dict(token_id=i,token=self.model.tokenizer.decode([i]),probability=p)
                for i,p in zip(ids.tolist(),values.tolist())]

    def conditions(self,source,destination,trial_id):
        core=dict(source=source,destination=destination,sites=self.sites,mode='swap',strength=1.0)
        result={'sustained':core,
            'loop2':dict(core,sites=[48+p for p in self.band]),
            'loop3':dict(core,sites=[96+p for p in self.band]),
            'loop4':dict(core,sites=[144+p for p in self.band]),
            'random_matched':dict(core,mode='random_matched',random_key=trial_id),
            'raw_lens':dict(core,kind='raw'),
            'early_band':dict(core,sites=[48*loop+p for loop in (1,2,3) for p in range(17)]),
            'projection_transfer':dict(core,mode='transfer'),
            'band24_36':dict(core,sites=[48*loop+p for loop in (1,2,3) for p in range(24,37)]),
            'double_strength':dict(core,strength=2.0)}
        if all(site in self.family.lenses[191].jacobians for site in self.sites):
            result['final_target']=dict(core,kind='final')
        return result

    @torch.no_grad()
    def readouts(self,activations,positions,queries,*,loops=(0,1,2,3),modes=('own','raw','final')):
        candidates={key:sorted({token for word in words for token in self.vocab.candidates(word)})
                    for key,words in queries.items()}
        result={}
        if not positions:raise ValueError('Readout span is empty')
        for loop in loops:
            for mode in modes:
                summary={key:dict(min_rank=None,hit5=False,max_probability=0.0)
                         for key,tokens in candidates.items() if tokens}
                for physical in self.band:
                    site=48*loop+physical
                    if mode=='final' and site not in self.family.lenses[191].jacobians:continue
                    h=activations[site][0,positions].float()
                    if mode!='raw':
                        target=self.family.target_for(site) if mode=='own' else 191
                        key=(site,target)
                        if key not in self.bank.matrices:
                            self.bank.matrices[key]=self.family.lenses[target].jacobians[site].to(h.device)
                        h=h@self.bank.matrices[key].T
                    logits=self.model.unembed(h).float()
                    top=logits.topk(5,dim=-1).indices
                    normalizer=logits.logsumexp(-1)
                    for key,tokens in candidates.items():
                        if not tokens:continue
                        values=logits[:,tokens]
                        ranks=1+(logits>values.max(-1).values[:,None]).sum(-1)
                        score=summary[key]
                        rank=int(ranks.min())
                        score['min_rank']=rank if score['min_rank'] is None else min(rank,score['min_rank'])
                        score['hit5']|=bool(torch.isin(top,torch.tensor(tokens,device=top.device)).any())
                        probability=(values.logsumexp(-1)-normalizer).exp().max().item()
                        score['max_probability']=max(score['max_probability'],probability)
                result[f'loop{loop+1}_{mode}']=summary
        return result

    def swap_trial(self,text,source_word,destination_word,answer,swap_answer,trial_id,*,baseline=None,source_token=None):
        source=self.vocab.direction_token(source_word) if source_token is None else source_token
        destination=self.vocab.direction_token(destination_word)
        result=dict(source=source_word,destination=destination_word,answer=answer,swap_answer=swap_answer,
                    source_token_id=source,destination_token_id=destination)
        if source is None or destination is None or not self.vocab.candidates(answer) or not self.vocab.candidates(swap_answer):
            return dict(result,eligible=False,reason='At least one concept/answer lacks a whole-token form')
        if source==destination:return dict(result,eligible=False,reason='Identical intervention tokens')
        clean=baseline if baseline is not None else self.forward(text)[2]
        words=dict(answer=answer,swap_answer=swap_answer)
        conditions={'baseline':dict(scores=self.outputs(clean,words),top5=self.top(clean[191]))}
        for name,edit in self.conditions(source,destination,trial_id).items():
            _,_,output=self.forward(text,edit=edit)
            conditions[name]=dict(scores=self.outputs(output,words),top5=self.top(output[191]))
        return dict(result,eligible=True,conditions=conditions)

    def flexible(self,item):
        function=item['function'];source=item['argument']
        text=function['template'].format(arg=source)
        baseline=self.forward(text)[2]
        trials=[]
        for target in item['arguments']:
            if target==source:continue
            trials.append(self.swap_trial(text,source,target,function['answers'][source],
                function['answers'][target],item['id']+':'+target,baseline=baseline))
        return dict(prompt=text,category=item['category'],argument=source,function=function['name'],
                    baseline_answer=score_tokens(baseline[191],self.vocab.candidates(function['answers'][source])),trials=trials)

    def probe(self,item):
        data=item['item'];text=data['prompt'];concept=data['intermediate']
        read_sites=[48*loop+p for loop in range(4) for p in self.band]
        ids,activations,baseline=self.forward(text,record=read_sites)
        result=self.swap_trial(text,concept,data['swap_to'],data['answer'],data['swap_answer'],item['id'],baseline=baseline)
        result.update(prompt=text,name=data['name'],category=data['category'])
        result['baseline_answer']=score_tokens(baseline[191],self.vocab.candidates(data['answer']))
        result['concept_present_in_prompt']=any(token in ids[0].tolist() for token in self.vocab.candidates(concept))
        foils=['violin','copper','velvet','planet']
        foils=[word for word in foils if word.casefold() not in {concept.casefold(),data['swap_to'].casefold()}
               and word.casefold() not in text.casefold()]
        result['foils']=foils
        queries={'intermediate':[concept],**{f'foil_{i}':[word] for i,word in enumerate(foils)}}
        result['readouts']=self.readouts(activations,list(range(1,ids.shape[1])),queries)
        source=self.vocab.direction_token(concept)
        answer=self.vocab.direction_token(data['answer'])
        unrelated=self.vocab.direction_token(foils[0])
        ablations={}
        specs={'argument':(source,self.sites),'answer':(answer,self.sites),
               'unrelated':(unrelated,self.sites),
               'argument_loop2':(source,[48+p for p in self.band]),
               'argument_loop4':(source,[144+p for p in self.band])}
        for name,(token,sites) in specs.items():
            if token is None:continue
            _,_,output=self.forward(text,edit=dict(sites=sites,source=token,mode='ablate'))
            ablations[name]=dict(scores=self.outputs(output,dict(answer=data['answer'],swap_answer=data['swap_answer'])),
                                  top5=self.top(output[191]))
        result['ablations']=ablations
        return result

    def report(self,item):
        text=self.chat(f'Think of a {item["category"]}. Answer in one word.','One word:')
        _,_,baseline=self.forward(text)
        answer_id=int(baseline[191].argmax())
        answer=self.model.tokenizer.decode([answer_id]).strip()
        trials=[]
        for target in item['candidates'][:10]:
            if target.casefold()==answer.casefold():continue
            trial=self.swap_trial(text,answer,target,answer,target,item['id']+':'+target,
                                  baseline=baseline,source_token=answer_id)
            trial['baseline_target_outside_top10']=(score_tokens(baseline[191],self.vocab.candidates(target))['rank'] or 0)>10
            trials.append(trial)
        return dict(prompt=text,category=item['category'],baseline_answer=answer,
                    baseline_in_category=answer.casefold() in {w.casefold() for w in item['candidates']},trials=trials)

    def language(self,item):
        data=load_data('selectivity-language');passage=item['passage']
        results={}
        for condition,question in [('explicit','explicit_q'),('automatic','automatic_q')]:
            user=data['task'][question].format(text=passage['text'])
            text=self.chat(user)
            start=text.index(passage['text'])+len(passage['text'])
            positions=self.span(text,start,len(text))
            _,acts,output=self.forward(text,record=self.sites)
            results[condition]=dict(prompt=text,positions=positions,
                readouts=self.readouts(acts,positions,{'language':data['intermediates'][passage['category']]},
                                       loops=(1,2,3),modes=('own','raw')),
                top5=self.top(output[191]),
                author=score_tokens(output[191],sorted({token for word in data['authors'][passage['category']]
                                                      for token in self.vocab.candidates(word)})))
        return dict(passage=passage,conditions=results)

    def modulation(self,item):
        data=load_data('directed-modulation');topic=item['topic'];carrier=item['carrier']
        trials=[]
        for phrasing in data['phrasings']:
            user=phrasing['text'].format(x=topic['name'])+' Copy this sentence exactly: '+carrier
            prefix=self.chat(user);text=prefix+carrier
            positions=self.span(text,len(prefix),len(text))
            _,acts,_=self.forward(text,record=self.sites)
            trials.append(dict(phrasing=phrasing['name'],group=data['group_kind'][phrasing['group']],
                prompt=text,positions=positions,
                readouts=self.readouts(acts,positions,{'topic':topic['members']},loops=(1,2,3),modes=('own','raw'))))
        return dict(topic=topic,carrier=carrier,trials=trials)

    def introspection(self,item):
        data=load_data('verbal-introspection');word=item['concept']['surface']
        token=self.vocab.direction_token(word,leading_space=False)
        if token is None:return dict(concept=word,eligible=False,reason='No whole-token concept form')
        messages=data['intro_prompt'][:-1]
        prefix=self.model.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        text=prefix+data['prefills']['default']
        user=messages[-1]['content'];start=text.rindex(user)
        positions=self.span(text,start,start+len(user))
        ids,acts,baseline=self.forward(text,record=self.sites)
        scales={site:acts[site][:,1:,:].float().norm(dim=-1).mean().item() for site in self.sites}
        conditions={'0':dict(scores=self.outputs(baseline,{'concept':word}),top5=self.top(baseline[191]))}
        for strength in (0.5,1.,2.,4.,8.):
            _,_,output=self.forward(text,edit=dict(sites=self.sites,destination=token,mode='inject',
                strength=strength,positions=positions,scales=scales))
            conditions[str(strength)]=dict(scores=self.outputs(output,{'concept':word}),top5=self.top(output[191]))
        _,_,output=self.forward(text,edit=dict(sites=self.sites,destination=token,mode='random_inject',
            strength=8.,positions=positions,scales=scales,random_key=item['id']))
        conditions['random8']=dict(scores=self.outputs(output,{'concept':word}),top5=self.top(output[191]))
        return dict(concept=word,eligible=True,token_id=token,prompt=text,positions=positions,
                    concept_present_in_prompt=any(t in ids[0].tolist() for t in self.vocab.candidates(word)),
                    scales=scales,conditions=conditions)


def worker(args):
    manifest=json.loads((args.output/'manifest.json').read_text())
    fingerprint=digest(args.output/'manifest.json')
    for name,expected in manifest['code_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError(f'Evaluation code changed: {name}')
    for name,expected in manifest['data_sha256'].items():
        if digest(DATA/f'{name}.json')!=expected:raise ValueError(f'Evaluation data changed: {name}')
    if digest(PROTOCOL)!=manifest['protocol_sha256']:raise ValueError('Protocol changed')
    family=OuroLensFamily.load(args.family)
    model=load_model()
    experiment=Experiment(model,family)
    started=time.time()
    selected=[item for index,item in enumerate(items()) if index%args.workers==args.rank]
    if args.limit:
        kept=[];counts={}
        for item in selected:
            count=counts.get(item['family'],0)
            if count<args.limit:kept.append(item)
            counts[item['family']]=count+1
        selected=kept
    try:
        with torch.no_grad():
            for index,item in enumerate(selected):
                path=args.output/'records'/f'{item["id"]}.json'
                if path.exists():
                    if json.loads(path.read_text())['manifest_sha256']!=fingerprint:
                        raise ValueError('Existing trial uses another manifest')
                    continue
                tick=time.time()
                result=getattr(experiment,item['family'])(item)
                save_json(path,dict(id=item['id'],family=item['family'],specification=item,
                    manifest_sha256=fingerprint,result=result,seconds=time.time()-tick))
                save_json(args.output/f'progress-{args.rank}.json',dict(rank=args.rank,done=index+1,
                    total=len(selected),last_id=item['id'],updated_unix=time.time(),elapsed_seconds=time.time()-started))
                print(f'{item["id"]}: {time.time()-tick:.2f}s',flush=True)
    finally:model.close()


def run(args):
    devices=args.gpus.split(',')
    family_manifest=json.loads((args.family/'family.json').read_text())
    n_prompts=family_manifest['metadata'].get('n_prompts',family_manifest['metadata'].get('num_prompts'))
    if n_prompts!=1000 and not args.debug:
        raise ValueError('Primary evaluation requires the completed 1000-prompt family')
    if family_manifest['metadata']['model_weights_sha256']!=weight_identity(ROOT/'model'):
        raise ValueError('Model weights do not match fitted lens')
    frozen=ROOT/'results/workspace-full/protocol-frozen.json'
    if not frozen.exists() or frozen.read_bytes()!=PROTOCOL.read_bytes():
        raise ValueError('Protocol differs from the frozen specification')
    filenames=['flexible-generalization','probe-swap','verbal-report','selectivity-language','directed-modulation','verbal-introspection']
    manifest=dict(protocol=json.loads(PROTOCOL.read_text()),protocol_sha256=digest(PROTOCOL),
        family_path=str(args.family.resolve()),family_manifest_sha256=digest(args.family/'family.json'),
        family_weights_sha256={p.name:digest(p) for p in sorted(args.family.glob('target-*.pt'))},
        data_sha256={name:digest(DATA/f'{name}.json') for name in filenames},
        code_sha256={name:digest(ROOT/name) for name in ['workspace_reproduction.py','workspace_interventions.py','analyze_workspace.py']},
        tokenizer_sha256={p.name:digest(p) for p in sorted((ROOT/'model').glob('tokenizer*.json'))},
        debug=args.debug,limit_per_family_per_worker=args.limit,workers=len(devices),
        n_items=len(items()),torch=str(torch.__version__),transformers=transformers.__version__)
    path=args.output/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:
        raise ValueError('Evaluation manifest changed; use a new output directory')
    save_json(path,manifest)
    (args.output/'code').mkdir(exist_ok=True)
    for name in manifest['code_sha256']:
        (args.output/'code'/name).write_bytes((ROOT/name).read_bytes())
    jobs=[];streams=[]
    try:
        for rank,device in enumerate(devices):
            log=(args.output/f'worker-{rank}.log').open('a',buffering=1);streams.append(log)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=device,OMP_NUM_THREADS='2',
                     PYTORCH_ALLOC_CONF='expandable_segments:True')
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',
                '--rank',str(rank),'--workers',str(len(devices)),'--family',str(args.family),
                '--output',str(args.output),'--limit',str(args.limit)]
            jobs.append(subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT))
        codes=[job.wait() for job in jobs]
        if any(codes):raise RuntimeError(f'Evaluation workers failed: {codes}')
        if not args.debug:
            subprocess.run([sys.executable,str(ROOT/'analyze_workspace.py'),'--output',str(args.output)],check=True,cwd=ROOT)
        save_json(args.output/'complete.json',dict(completed_unix=time.time(),exit_codes=codes,debug=args.debug))
    finally:
        for stream in streams:stream.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',type=Path,default=ROOT/'results/jlens-family-1000')
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--gpus',default='0,1,2,3')
    p.add_argument('--worker',action='store_true')
    p.add_argument('--rank',type=int,default=0)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--debug',action='store_true')
    p.add_argument('--limit',type=int,default=0)
    args=p.parse_args();transformers.set_seed(42)
    if args.worker:worker(args)
    else:run(args)


if __name__=='__main__':main()
