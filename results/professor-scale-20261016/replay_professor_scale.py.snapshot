"""Author-lens and raw-lens fixed-window screening of every scaled response."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time

import torch

from analyze_professor_scale import OUT, classify, sections
from replay_professor_followups import ROOT, save, sha


def positions_for(g,tok):
    prefix=len(g['input_ids']);ids=g['input_ids']+g['output_ids'];blocks=sections(g['raw_text'])
    ends=[len(tok.decode(g['output_ids'][:i])) for i in range(len(g['output_ids'])+1)]
    region={prefix-1:'prompt'}
    for i in range(len(g['output_ids'])-1):
        region[prefix+i]=next((b['name'] for b in blocks if b['start']<ends[i+1]<=b['end']),'header_or_other')
    windows=dict(primary_early=[p for p in range(prefix-1,min(prefix+20,len(ids)-1))],
        reasoning_tail=[p for p,r in region.items() if r=='reasoning'][-8:],
        message_start=[p for p,r in region.items() if r=='message'][:12],final=[len(ids)-2])
    positions=sorted({p for ps in windows.values() for p in ps})
    assert positions and min(positions)>=0 and max(positions)<len(ids)-1
    return positions,region,windows


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',type=int,required=True);parser.add_argument('--limit',type=int);args=parser.parse_args()
    from ouro_full_fit import load_model
    from ouro_lens_family import OuroLensFamily
    from workspace_interventions import Vocabulary
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    protocol=json.loads((OUT/'protocol.json').read_text());ph=sha(OUT/'protocol.json');config=protocol['readout']
    family_path=ROOT/protocol['lens_path'];assert sha(family_path/'family.json')==protocol['lens_manifest_sha256']
    family=OuroLensFamily.load(family_path);assert family.metadata['revision']==protocol['lens_revision']
    model=load_model();family.check_model(model);tok=model.tokenizer;sites=config['sites'];methods=config['methods']
    vocab=Vocabulary(tok);tokens={w:vocab.candidates(w) for w in config['words']};token_word={t:w for w,ts in tokens.items() for t in ts}
    matrices=torch.stack([family.lenses[family.target_for(s)].jacobians[s] for s in sites]).cuda().float()
    env=dict(protocol_sha256=ph,code_sha256=sha(Path(__file__)),lens_manifest_sha256=protocol['lens_manifest_sha256'],
        lens_repo=protocol['lens_repo'],lens_revision=protocol['lens_revision'],model_revision=protocol['model_revision'],
        gpu=torch.cuda.get_device_name(),cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),torch=torch.__version__,
        vocabulary=tokens,sites=sites,methods=methods,
        precision='BF16 full-causal replay residuals; FP32 author transport; original BF16 norm/head; TF32 disabled.',
        scope='All completed generations, including capped and error outputs. Fixed windows defined before generation; no lens-based selection.')
    save(OUT/f'lens-worker-{args.worker}-environment.json',env)
    files=[p for p in sorted((OUT/'cases').glob('*/generation.json')) if int(p.parent.name.split('-')[1])%protocol['workers']==args.worker]
    if args.limit:files=files[:args.limit]
    start=time.monotonic();done=0

    def status(stage,**kwargs):
        record=dict(worker=args.worker,stage=stage,completed=done,planned=len(files),elapsed_seconds=round(time.monotonic()-start,2),
            updated_utc=datetime.now(timezone.utc).isoformat(),**kwargs)
        save(OUT/f'lens-worker-{args.worker}-status.json',record);print(json.dumps(record),flush=True)

    status('screening')
    try:
        for path in files:
            destination=path.parent/'screen';destination.mkdir(exist_ok=True)
            generation_hash=sha(path)
            if (destination/'summary.json').exists():
                existing=json.loads((destination/'summary.json').read_text())
                assert existing['source_sha256']==generation_hash and existing['protocol_sha256']==ph
                done+=1;continue
            g=json.loads(path.read_text());assert g['protocol_sha256']==ph
            behavior=classify(g)
            # Record behavior-derived triage before examining any activations.
            save(destination/'pre-lens-triage.json',dict(source_sha256=generation_hash,**behavior))
            positions,region,windows=positions_for(g,tok);ids=g['input_ids']+g['output_ids'];prefix=len(g['input_ids'])
            assert tok.decode(g['output_ids'])==g['raw_text']
            index=torch.tensor(positions,device='cuda');recorded={};handles=[]
            for site in sites:
                def hook(module,args,output,site=site):
                    recorded[site]=output[0].index_select(0,index).detach()
                handles.append(model.layers[site].register_forward_hook(hook))
            try:model.forward(torch.tensor([ids],device='cuda'))
            finally:
                for handle in handles:handle.remove()
            states=torch.stack([recorded[s] for s in sites]);del recorded
            torch.save(dict(ids=ids,sites=sites,positions=positions,states=states.cpu()),destination/'states.pt')
            transported=torch.bmm(states.float(),matrices.transpose(1,2))
            top_ids=torch.empty((2,len(sites),len(positions),25),dtype=torch.int32)
            top_values=torch.empty_like(top_ids,dtype=torch.float32)
            hits=[];metrics={m:{w:dict(cells=len(ps)*len(sites),any_word_top10_cells=0,false_top10_cells=0) for w,ps in windows.items()} for m in methods}
            for mi,method in enumerate(methods):
                h=(transported if method=='jlens' else states.float()).flatten(0,1)
                for offset in range(0,len(h),128):
                    logits=model.unembed(h[offset:offset+128]).float();assert torch.isfinite(logits).all()
                    values,order=logits.topk(25,-1);logz=logits.logsumexp(-1)
                    top_ids[mi].view(-1,25)[offset:offset+len(order)]=order.cpu().int()
                    top_values[mi].view(-1,25)[offset:offset+len(order)]=values.cpu()
                    probs=(values-logz[:,None]).exp().cpu().tolist()
                    for local,(row_ids,row_values,row_probs) in enumerate(zip(order.tolist(),values.tolist(),probs)):
                        si,pi=divmod(offset+local,len(positions));site=sites[si];position=positions[pi]
                        present=set()
                        for ordinal,token in enumerate(row_ids):
                            if token not in token_word:continue
                            rank=1+sum(v>row_values[ordinal] for v in row_values);word=token_word[token]
                            if rank<=10:present.add(word)
                            hits.append(dict(method=method,site=site,loop=site//48+1,layer=site%48,position=position,
                                generated_position=position-prefix,region=region[position],word=word,token_id=token,rank=rank,
                                ordinal=ordinal+1,probability=row_probs[ordinal],source_token=tok.decode([ids[position]]),
                                next_token=tok.decode([ids[position+1]]),absent_from_full_output=not bool(re.search(r'\b'+re.escape(word)+r'\b',g['raw_text'],re.I)),
                                top10=[tok.decode([t]) for t in row_ids[:10]]))
                        for window,ps in windows.items():
                            if position in ps:
                                metrics[method][window]['any_word_top10_cells']+=int(bool(present))
                                metrics[method][window]['false_top10_cells']+=int('false' in present)
            torch.save(dict(methods=methods,sites=sites,positions=positions,ids=top_ids,values=top_values),destination/'top25.pt')
            with (destination/'hits.jsonl').open('w') as stream:
                for hit in hits:stream.write(json.dumps(hit,ensure_ascii=False)+'\n')
            best={}
            for hit in sorted(hits,key=lambda h:(h['rank'],h['position'],h['site'])):
                best.setdefault(f'{hit["method"]}/{hit["region"]}/{hit["word"]}',hit)
            save(destination/'summary.json',dict(id=g['id'],task=g['task'],condition=g['condition'],source_sha256=generation_hash,
                protocol_sha256=ph,environment_sha256=sha(OUT/f'lens-worker-{args.worker}-environment.json'),
                windows=windows,positions=positions,region_by_position=region,metrics=metrics,best_hits=best,
                top25_sha256=sha(destination/'top25.pt'),state_sha256=sha(destination/'states.pt'),
                triage=behavior['category'],scope='Fixed-window screening; omitted positions are unmeasured, not negative.'))
            del states,transported,top_ids,top_values,h,logits
            done+=1
            if done%8==0:status('screening',last_case=g['id'])
        status('complete')
    finally:model.close()


if __name__=='__main__':main()
