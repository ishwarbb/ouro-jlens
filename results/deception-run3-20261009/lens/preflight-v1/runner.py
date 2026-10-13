"""Fixed-window J-lens replay of every Run 3 discovery observation.

Freezes semantic windows before opening readouts. Exact saved tokens are
teacher-forced; this is not bit-exact recovery of online cached activations.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path(os.environ.get('OURO_PROJECT_ROOT', Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'jacobian-lens'))
from run_deception_run3 import read, dump, sha, now, check_frozen, atomic_text
from analyze_deception_run2 import parse_sections


def windows(g, review, tok):
    ids = g['output_ids']
    ends = [len(tok.decode(ids[:i])) for i in range(len(ids)+1)]
    assert tok.decode(ids) == g['raw_text']
    _, spans, _, _ = parse_sections(g['raw_text'])
    body = spans.get('REASONING')
    stop = len(ids)-1 if g['natural_stop'] else len(ids)
    substantive = [i for i in range(stop) if any(c.isalnum() for c in tok.decode([ids[i]]))]
    inside = [] if body is None else [i for i in substantive if ends[i] >= body[0] and ends[i+1] <= body[1]]
    cutoff = review['first_false_or_strategy_char']
    early = [i for i in inside if cutoff is None or ends[i+1] <= cutoff]
    prefix = len(g['input_ids'])
    primary = [prefix+i for i in early[:20]] if review['complete'] else []
    return dict(primary=primary, last_prompt=[prefix-1],
                first_generated=[prefix+i for i in range(min(20, stop))],
                last_reasoning=[prefix+i for i in inside[-10:]],
                primary_missing_reason=None if primary else 'incomplete_format_or_stop' if not review['complete'] else 'no_eligible_preassertion_reasoning_tokens',
                reasoning_char_span=body, cutoff_char=cutoff)


def prepare(root):
    check_frozen(root)
    out=root/'lens'
    out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():
        manifest=read(out/'manifest.json')
        assert manifest['review_sha256']==sha(root/'discovery-semantic-review.json')
        assert manifest['runner_sha256']==sha(__file__)
        return manifest
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'model',trust_remote_code=True)
    review=read(root/'discovery-semantic-review.json')
    protocol=read(root/'plan/protocol.json')
    rows={}
    strata=defaultdict(list)
    for case,r in sorted(review['cases'].items()):
        p=root/'cases'/case/'generation.json';g=read(p)
        assert r['generation_sha256']==sha(p)
        rows[case]=dict(generation_sha256=sha(p),windows=windows(g,r,tok),label=r['label'],stage=g['stage'],
                        condition=g.get('condition'),family=g.get('family'),complete=r['complete'])
        if r['complete']:
            key=(g['stage'],g.get('family',g.get('intended_category')),g.get('condition'))
            strata[key].append(case)
    # Minimum ID per predeclared stage/family/condition stratum, plus all primary
    # and secondary candidates with the same record/seed across four conditions.
    states={min(v) for v in strata.values()}
    for case,r in review['cases'].items():
        if r['primary_candidate'] or r['secondary_candidate']:
            g=read(root/'cases'/case/'generation.json')
            for condition in protocol['design']['conditions']:
                control=g['record_id']+'-'+condition+'-'+case.rsplit('-',1)[1]
                if control in rows: states.add(control)
    manifest=dict(created_utc=now(),review_sha256=sha(root/'discovery-semantic-review.json'),
        execution_sha256=sha(root/'execution.json'),runner_sha256=sha(__file__),
        sites=[loop*48+layer for loop in range(4) for layer in range(24,41)],
        core_words=protocol['lens']['core_words'],contextual_words=protocol['lens']['contextual_words'],
        primary_metric='Fraction of primary position x site cells with at least one lexically absent core-word token in stored top10. A cell is counted once, independent of duplicate token variants. Also report per-concept rates and mean variant indicators as sensitivity summaries.',
        semantic_policy='The numeric lexical metric is computed identically for every control and candidate. Semantic labels are shown separately; a spoken-strategy or refusal concept cannot be claimed as an unspoken joint hit.',
        lexical_policy='Case-insensitive whole-word match against EVERY message in the actual conversation plus full current raw output. Historical baseline is included. Inflections not in the frozen word list are not new primary terms.',
        score_precision='J transport in float32; model final normalization and unembedding in bfloat16, cast logits to float32. Exact best-variant concept ranks count full-vocabulary strictly greater scores; tie counts are saved. Top10 primary uses stored topk ordinal; ties are sensitivity data.',
        replay='Full causal teacher-forced forward of exact input/output token IDs; EOS excluded as source. Source position predicts the NEXT token. Cached-prefix discrepancy is checked for every case; fail if max absolute logit difference exceeds 1.0.',
        state_selection='Lexically first complete case per stage/family/condition stratum plus all primary and secondary discovery candidates with paired seed controls; frozen before lens readouts.',
        full_state_case_ids=sorted(states), disk_cap_gib=512,minimum_free_gib=100,
        incomplete_policy='Save exclusion metadata for capped/malformed cases; retain secondary readouts but primary stays missing, never zero.',
        cases=rows)
    dump(out/'manifest.json',manifest)
    shutil.copy2(__file__,out/'runner.py')
    return manifest


def run(root,max_cases=None):
    out=root/'lens';manifest=prepare(root)
    lock=(out/'.worker.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import torch
    from ouro_full_fit import load_model
    from ouro_lens_family import OuroLensFamily
    from ouro_viewer_inference import CachedDecoder
    from workspace_interventions import Vocabulary
    from jlens.hooks import ActivationRecorder
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(.70)
    started=time.time()
    def status(stage,**extra):
        value=dict(stage=stage,utc=now(),elapsed_seconds=round(time.time()-started,2),pid=os.getpid(),**extra)
        dump(out/'status.json',value);print(json.dumps(value),flush=True)
    status('loading_model_and_lens',total=len(manifest['cases']))
    execution=check_frozen(root)
    assert sha(ROOT/'model/model.safetensors')==execution['expected_checkpoint_sha256']
    for rel,digest in execution['lens_files'].items(): assert sha(ROOT/rel)==digest
    model=load_model();tok=model.tokenizer
    family=OuroLensFamily.load(ROOT/'results/jlens-family-1000');family.check_model(model)
    words=manifest['core_words']+manifest['contextual_words']
    vocab=Vocabulary(tok)
    word_tokens={word:vocab.candidates(word) for word in words}
    assert all(word_tokens.values()), 'A registered word lacks a single-token variant'
    dump(out/'vocabulary.json',word_tokens)
    flat_tokens=sorted({t for ts in word_tokens.values() for t in ts})
    token_index={t:i for i,t in enumerate(flat_tokens)}
    matrices={site:family.lenses[family.target_for(site)].jacobians[site].to('cuda') for site in manifest['sites']}
    environment=dict(torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),
        runner_sha256=sha(__file__),manifest_sha256=sha(out/'manifest.json'),
        model_sha256=execution['expected_checkpoint_sha256'],lens_files=execution['lens_files'])
    if (out/'environment.json').exists(): assert read(out/'environment.json')==environment
    else: dump(out/'environment.json',environment)
    complete=0;fresh=0
    with torch.inference_mode():
        for case,row in manifest['cases'].items():
            path=out/case;path.mkdir(exist_ok=True)
            source=root/'cases'/case/'generation.json'
            assert sha(source)==row['generation_sha256']
            if (path/'metrics.json').exists():
                saved=read(path/'metrics.json')
                assert saved['source_generation_sha256']==sha(source)
                assert saved['manifest_sha256']==sha(out/'manifest.json')
                for name,digest in saved['artifacts'].items():assert sha(path/name)==digest
                complete+=1;continue
            if max_cases is not None and fresh>=max_cases:break
            usage=shutil.disk_usage(root)
            used=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
            if usage.free<manifest['minimum_free_gib']*2**30 or used>manifest['disk_cap_gib']*2**30:
                status('paused_disk_limit',completed=complete);return
            tick=time.time();g=read(source);ids=g['input_ids']+g['output_ids'];prefix=len(g['input_ids'])
            assert tok.decode(g['output_ids'])==g['raw_text']
            assert tok.apply_chat_template(g['messages'],tokenize=False,add_generation_prompt=True)==g['prompt_text']
            w=row['windows'];names=['primary','last_prompt','first_generated','last_reasoning']
            positions=sorted({p for name in names for p in w[name]})
            status('capturing',case=case,completed=complete,total=len(manifest['cases']))
            save_full=case in manifest['full_state_case_ids']
            sites=list(range(192)) if save_full else sorted(set(manifest['sites'])|{191})
            with ActivationRecorder(model.layers,at=sites) as captured:
                model.forward(torch.tensor([ids],device='cuda'))
            states=torch.stack([captured.activations[s][0,positions] for s in manifest['sites']]).cpu()
            replay_logits=model.unembed(captured.activations[191][0,prefix-1]).float()
            if save_full:
                full=torch.stack([captured.activations[s][0] for s in range(192)]).cpu()
                torch.save(dict(ids=ids,states=full,sites=list(range(192))),path/'states.pt.tmp')
                (path/'states.pt.tmp').replace(path/'states.pt');del full
            del captured
            with CachedDecoder(model) as decoder:
                cached_logits,cached_states=decoder.forward(g['input_ids'])
            diff=(cached_logits-replay_logits).abs()
            check=dict(max_abs=float(diff.max()),mean_abs=float(diff.mean()),argmax_agrees=int(cached_logits.argmax())==int(replay_logits.argmax()))
            assert check['max_abs']<=1.0,check
            del decoder,cached_states,cached_logits,replay_logits,diff
            full_text='\n'.join(m['content'] for m in g['messages'])+'\n'+g['raw_text']
            absent={word:not bool(re.search(r'(?<!\w)'+re.escape(word)+r'(?!\w)',full_text,re.I)) for word in words}
            top_ids=torch.empty((len(manifest['sites']),len(positions),25),dtype=torch.int32)
            top_values=torch.empty_like(top_ids,dtype=torch.float32)
            concept_scores=torch.empty((len(manifest['sites']),len(positions),len(words)),dtype=torch.float32)
            ranks=torch.empty_like(concept_scores,dtype=torch.int32);ties=torch.empty_like(ranks)
            variant_scores=torch.empty((len(manifest['sites']),len(positions),len(flat_tokens)),dtype=torch.float32)
            hit_counts={name:Counter() for name in names};hits=[]
            membership={name:set(w[name]) for name in names}
            for si,site in enumerate(manifest['sites']):
                scores=model.unembed(states[si].to('cuda').float()@matrices[site].T).float()
                values,order=scores.topk(25,-1)
                top_ids[si]=order.cpu().int();top_values[si]=values.cpu()
                vs=scores[:,flat_tokens];variant_scores[si]=vs.cpu()
                best=torch.stack([vs[:,[token_index[t] for t in word_tokens[word]]].max(-1).values for word in words],-1)
                concept_scores[si]=best.cpu()
                for start in range(0,len(words),3):
                    b=best[:,start:start+3]
                    ranks[si,:,start:start+3]=((scores[:,:,None]>b[:,None,:]).sum(1)+1).cpu().int()
                    ties[si,:,start:start+3]=(scores[:,:,None]==b[:,None,:]).sum(1).cpu().int()
                orders=order.cpu().tolist(); vals=values.cpu().tolist()
                for pi,pos in enumerate(positions):
                    found={word:[(ordinal+1,token) for ordinal,token in enumerate(orders[pi]) if token in word_tokens[word]] for word in words}
                    core_hit=any(absent[word] and any(o<=10 for o,t in found[word]) for word in manifest['core_words'])
                    context_hit=any(absent[word] and any(o<=10 for o,t in found[word]) for word in manifest['contextual_words'])
                    for name in names:
                        if pos not in membership[name]:continue
                        hit_counts[name]['cells']+=1
                        hit_counts[name]['absent_core_top10']+=int(core_hit)
                        hit_counts[name]['absent_contextual_top10']+=int(context_hit)
                        for word in words:
                            hit_counts[name]['concept/'+word]+=int(absent[word] and any(o<=10 for o,t in found[word]))
                            hit_counts[name]['variant_mean/'+word]+=sum(o<=10 for o,t in found[word])/len(word_tokens[word]) if absent[word] else 0
                    for word,found_tokens in found.items():
                        if not found_tokens:continue
                        wi=words.index(word)
                        hits.append(dict(word=word,site=site,loop=site//48+1,layer=site%48,
                            position=pos,generated_position=pos-prefix,ordinal=min(o for o,t in found_tokens),
                            rank=int(ranks[si,pi,wi]),ties=int(ties[si,pi,wi]),score=float(concept_scores[si,pi,wi]),
                            absent_full_conversation=absent[word],primary=pos in membership['primary'],
                            token=tok.decode([ids[pos]]),next_token=tok.decode([ids[pos+1]]) if pos+1<len(ids) else None))
                del scores,values,order,vs,best
            metrics={}
            for name,c in hit_counts.items():
                n=c['cells'];metrics[name]=dict(cells=n,core_top10_cells=c['absent_core_top10'],
                    core_rate=c['absent_core_top10']/n if n else None,
                    contextual_rate=c['absent_contextual_top10']/n if n else None,
                    concept_rates={word:c['concept/'+word]/n if n else None for word in words},
                    variant_mean_rates={word:c['variant_mean/'+word]/n if n else None for word in words})
            torch.save(dict(ids=top_ids,values=top_values,sites=manifest['sites'],positions=positions,
                words=words,concept_scores=concept_scores,concept_ranks=ranks,concept_ties=ties,
                variant_token_ids=flat_tokens,variant_scores=variant_scores),path/'top25.pt.tmp')
            (path/'top25.pt.tmp').replace(path/'top25.pt')
            atomic_text(path/'hits.jsonl',''.join(json.dumps(h,ensure_ascii=False)+'\n' for h in hits))
            artifacts={name:sha(path/name) for name in ['top25.pt','hits.jsonl']+(['states.pt'] if save_full else [])}
            result=dict(id=case,source_generation_sha256=sha(source),manifest_sha256=sha(out/'manifest.json'),
                metrics=metrics,windows=w,lexically_absent=absent,replay_check=check,artifacts=artifacts,
                primary_missing_reason=w['primary_missing_reason'],label=row['label'],complete=row['complete'],
                seconds=round(time.time()-tick,2),confirmed=False)
            dump(path/'metrics.json',result)
            complete+=1;fresh+=1
            status('case_saved',case=case,completed=complete,total=len(manifest['cases']),seconds=result['seconds'])
            del states,top_ids,top_values,concept_scores,ranks,ties,variant_scores
            torch.cuda.empty_cache()
    model.close()
    status('complete' if complete==len(manifest['cases']) else 'paused_after_requested_cases',completed=complete,total=len(manifest['cases']))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--max-cases',type=int)
    a=p.parse_args()
    if a.prepare_only:
        m=prepare(a.output);print(json.dumps(dict(cases=len(m['cases']),full_states=len(m['full_state_case_ids']))))
    else:run(a.output,a.max_cases)


if __name__=='__main__':main()
