"""Read-only provenance and fixed-window validation for saved Run 3 data."""
import argparse
from collections import Counter
from pathlib import Path
import json

from run_deception_run3 import read, sha, now, check_frozen, validate_resume, resolve_job, extract, dump


def verify(root, require_lens=False):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(Path(__file__).resolve().parent/'model',trust_remote_code=True)
    check_frozen(root)
    jobs=read(root/'generation-jobs.json')
    reviews=read(root/'discovery-semantic-review.json')['cases']
    counts=Counter();hashes={};held_out=0
    for job in jobs:
        p=root/'cases'/job['id']/'generation.json'
        if not p.exists():continue
        messages,parent,gate=resolve_job(root,job)
        assert messages is not None,job['id']
        g=validate_resume(p,job,parent)
        assert g['messages']==messages
        assert tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)==g['prompt_text']
        assert tok.encode(g['prompt_text'],add_special_tokens=False)==g['input_ids']
        assert tok.decode(g['output_ids'])==g['raw_text']==(p.parent/'response.txt').read_text()
        sections,proper=extract(g['raw_text'])
        assert sections==g['sections'] and bool(proper)==g['format_complete']
        assert g['natural_stop']==(g['output_ids'][-1] in [0,2])
        assert g['capped']==(not g['natural_stop'] and len(g['output_ids'])>=1024)
        hashes[g['id']]=sha(p);counts[g['stage']]+=1
        if g['stage']=='held_out':held_out+=1
        else:assert reviews[g['id']]['generation_sha256']==sha(p)
    assert counts['baseline']==24 and counts['calibration']==48 and counts['discovery']==240
    excluded=read(root/'discovery-excluded.json')
    assert len(excluded)==48
    assert {k.rsplit('-',2)[0] for k in excluded}=={'allocation-00','allocation-01'}
    assert not read(root/'discovery-deferred.json')
    lens_count=0;missing=0;checks=[]
    lens=root/'lens'
    if (lens/'manifest.json').exists():
        import torch
        from replay_deception_run3 import windows
        m=read(lens/'manifest.json')
        assert m['review_sha256']==sha(root/'discovery-semantic-review.json')
        assert m['runner_sha256']==sha(lens/'runner.py')
        for case,row in m['cases'].items():
            assert row['generation_sha256']==hashes[case]
            g=read(root/'cases'/case/'generation.json')
            assert windows(g,reviews[case],tok)==row['windows']
            p=lens/case/'metrics.json'
            if not p.exists():
                assert not require_lens,f'Missing lens capture: {case}'
                continue
            result=read(p)
            assert result['source_generation_sha256']==hashes[case]
            assert result['manifest_sha256']==sha(lens/'manifest.json')
            assert result['windows']==row['windows']
            for name,digest in result['artifacts'].items():assert sha(p.parent/name)==digest
            top=torch.load(p.parent/'top25.pt',map_location='cpu',weights_only=True)
            assert top['sites']==m['sites']
            assert tuple(top['ids'].shape)==(68,len(top['positions']),25)
            assert bool((top['values'][:,:,1:]<=top['values'][:,:,:-1]).all())
            full_text='\n'.join(message['content'] for message in g['messages'])+'\n'+g['raw_text']
            import re
            for word,absent in result['lexically_absent'].items():
                assert absent==(not bool(re.search(r'(?<!\w)'+re.escape(word)+r'(?!\w)',full_text,re.I)))
            core_tokens=[token for word in m['core_words'] if result['lexically_absent'][word] for token in m['word_tokens'][word]]
            present=torch.isin(top['ids'][:,:,:10],torch.tensor(core_tokens,dtype=torch.int32)).any(-1)
            for window in ['primary','last_prompt','first_generated','last_reasoning']:
                selected=[i for i,pos in enumerate(top['positions']) if pos in row['windows'][window]]
                measured=result['metrics'][window]
                assert measured['core_top10_cells']==int(present[:,selected].sum())
                assert measured['cells']==68*len(selected)
                for word in m['unmeasurable_core_words']:
                    wi=top['words'].index(word)
                    assert bool(torch.isnan(top['concept_scores'][:,:,wi]).all())
                    assert bool((top['concept_ranks'][:,:,wi]==-1).all())
                    assert measured['concept_rates'][word] is None
                assert measured['registered_full_vocabulary_core_rate'] is None
            metric=result['metrics']['primary'];cells=len(row['windows']['primary'])*68
            assert metric['cells']==cells
            assert (metric['core_rate'] is None)==(cells==0)
            if cells:assert metric['core_rate']==metric['core_top10_cells']/cells
            missing+=int(cells==0)
            assert result['replay_check']['max_abs']<=1.0
            checks.append(result['replay_check']['max_abs']);lens_count+=1
    if require_lens:assert lens_count==312
    return dict(status='passed',utc=now(),generation_counts=dict(counts),generation_total=sum(counts.values()),
        excluded_discovery=48,deferred_discovery=0,lens_completed=lens_count,lens_expected=312,
        lens_primary_missing=missing,max_replay_logit_difference=max(checks,default=None),
        held_out_generated=held_out,frozen_inputs_match=True,raw_token_roundtrips=True,
        prompt_histories_match=True,rng_hashes_match=True,artifact_hashes_match=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--require-lens',action='store_true')
    a=p.parse_args();result=verify(a.output,a.require_lens)
    dump(a.output/'continuation-verification.json',result);print(json.dumps(result,indent=2))
