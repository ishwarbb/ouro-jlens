"""Independent artifact reconstruction and reporting-quality review.

Run after the frozen pipeline completes, on one of the authorized GPUs.
This validator does not call the pursuit or projector construction routines.
"""
import argparse
import json
from pathlib import Path
import time

import torch
from fit_ouro_family import save_json,digest
from ouro_full_fit import load_model
from ouro_lens_family import OuroLensFamily
from run_workspace_inventory import check,FAMILY
from workspace_validated import ValidatedIO


@torch.no_grad()
def verify(out):
    start=time.time();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    manifest=check(out);model=load_model();family=OuroLensFamily.load(FAMILY);io=ValidatedIO(model)
    states={};capture={};demo_quality={};score_replays=0;report_replays=0;report_exclusions=0
    peak=dict(residual_fraction_error=0.,normalized_kkt=0.,orthogonality_error=0.,span_error=0.,native_logits_error=0.)
    for item in manifest['tasks']:
        name=item['id'];r=json.loads((out/'capture'/f'{name}.json').read_text())['result'];capture[name]=r
        prompt=io.prompt(r['prompt']['text'],wrapper=r['prompt']['wrapper'])
        assert prompt.record()==r['prompt']
        if item['family']=='report':
            oldpath=out.parent/'workspace-audit/records'/f'{name}-answer_prefill.json'
            if oldpath.exists():
                old=json.loads(oldpath.read_text())['result']
                old_trials={t['destination']:t for t in old['trials']}
                for trial in r['trials']:
                    word=trial['target']['word'];before=old_trials[word]
                    for cond in ['baseline','transfer']:
                        if cond not in before['conditions']:
                            assert cond not in trial['conditions']
                            assert not trial['eligible'] and not before['intervention_eligible']
                            report_exclusions+=1
                            continue
                        expected=before['conditions'][cond]['scores']['191']['swap_answer']['whole']
                        actual=trial['conditions'][cond]['whole']
                        assert expected==actual,(name,word,cond,expected,actual)
                        report_replays+=1
            continue
        path=out/'capture'/f'{name}.pt'
        assert digest(path)==r['states_sha256']
        stored=torch.load(path,map_location='cpu',weights_only=True);states[name]=stored['states']
        assert stored['seq_ids']==r['seq_ids']
        if item['family']=='demo':
            g=r['generated'];assert r['seq_ids']==list(prompt.ids)+g['token_ids']
            demo_quality[name]=dict(generated_tokens=len(g['token_ids']),limit_reached=g['limit_reached'],
                unclosed_thought=g['unclosed_thought'],final_text=g['text'],
                has_final_text=bool(g['text'].strip()),
                comparable_completed_answer=bool(g['text'].strip()) and not g['unclosed_thought'] and not g['limit_reached'])
        else:
            logits,_=io.forward(r['seq_ids'])
            native=model.hf_model(io.tensor(r['seq_ids']),use_cache=False,exit_at_step=3,logits_to_keep=1).logits[0,0].float()
            peak['native_logits_error']=max(peak['native_logits_error'],float((logits-native).abs().max()))
            assert torch.equal(logits,native)
            assert io.score(logits,r['answer']['word'])==r['baseline']
            score_replays+=1
    dictionary=json.loads((out/'dictionary-0.json').read_text());allowed=set(dictionary['token_ids'])
    counts=dict(snapshots=0,pursuit_cells=0,projector_cells=0,layer_artifacts=0)
    weight=model.hf_model.lm_head.weight;device=weight.device
    for path in sorted((out/'inventory').glob('*.pt')):
        meta=json.loads(path.with_suffix('.json').read_text())['result'];assert digest(path)==meta['artifact_sha256']
        data=torch.load(path,map_location='cpu',weights_only=True);site=data['site']
        j=family.lenses[family.target_for(site)].jacobians[site].to(device)
        for name,r in data['records'].items():
            ids=r['token_ids'];valid=r['valid'];n,k=ids.shape
            for row,mask in zip(ids.tolist(),valid.tolist()):
                selected=[t for t,v in zip(row,mask) if v]
                assert len(selected)==len(set(selected)) and set(selected)<=allowed
            token_ids=ids.clamp_min(0).to(device)
            # Independently gather only selected atoms, avoiding the original full dictionary.
            unique,inverse=token_ids.flatten().unique(return_inverse=True)
            transported=(j.T@weight[unique].float().T).T
            transported=transported/transported.norm(dim=-1,keepdim=True)
            atoms=transported[inverse].reshape(n,k,-1)*valid.to(device).unsqueeze(-1)
            h=states[name][site].to(device).float();norm=h.norm(dim=-1)
            previous=torch.ones(n,device=device)
            for size,snap in sorted(r['snapshots'].items()):
                coeff=snap['coefficients'].to(device)
                assert bool((coeff>=0).all())
                a=atoms[:,:size]
                reconstruction=torch.einsum('nk,nkd->nd',coeff,a)
                squared=(h-reconstruction).square().sum(-1)/norm.square()
                error=float((squared-snap['residual_fraction'].to(device)).abs().max())
                peak['residual_fraction_error']=max(peak['residual_fraction_error'],error)
                assert error<2e-5,(site,name,size,error)
                assert bool((squared<=previous+3e-5).all())
                previous=squared
                x=coeff/norm[:,None]
                gradient=torch.einsum('nkd,nd->nk',a,(reconstruction-h)/norm[:,None])
                kkt=float((x-(x-gradient).clamp_min(0)).abs().max())
                peak['normalized_kkt']=max(peak['normalized_kkt'],kkt)
                assert kkt<1e-5,(site,name,size,kkt)
                counts['snapshots']+=n
            if name in data['bases']:
                q=data['bases'][name].to(device);rank=r['basis_rank'].to(device)
                gram=q.transpose(1,2)@q
                diag=torch.diag_embed((q.norm(dim=1)>.5).float())
                error=float((gram-diag).abs().max());peak['orthogonality_error']=max(peak['orthogonality_error'],error)
                assert error<1e-4
                assert torch.equal((q.norm(dim=1)>.5).sum(-1),rank)
                a=atoms.transpose(1,2)
                span_error=float((a-q@(q.transpose(1,2)@a)).norm(dim=1).max())
                peak['span_error']=max(peak['span_error'],span_error);assert span_error<1e-4
                counts['projector_cells']+=n
            counts['pursuit_cells']+=n
        counts['layer_artifacts']+=1
        if counts['layer_artifacts']%24==0:print('verified layers',counts['layer_artifacts'],flush=True)
    assert counts['layer_artifacts']==188 and score_replays==64 and report_replays==257 and report_exclusions==5
    result=dict(passed=True,seconds=time.time()-start,validator_sha256=digest(Path(__file__)),
        manifest_sha256=digest(out/'manifest.json'),counts=counts,maximum_errors=peak,
        native_baseline_replays=score_replays,previous_report_score_replays=report_replays,
        previous_report_eligibility_replays=report_exclusions,
        demo_quality=demo_quality,limitations=['Only two demo prompts; four loop-end targets excluded from inventory sources',
          'Sport continuation reaches 128-token cap inside thought; cannot replicate hidden-sport final-answer claim',
          'Report wrapper calibrated on earlier results; 5/14 first-token sources outside released category candidates',
          'Author word-filter/solver/wrapper details and independent confirmation still unresolved'])
    save_json(out/'VALIDATION.json',result)
    summary=json.loads((out/'summary.json').read_text())
    lines=['# Independent reconstruction and quality review','',
        f"Passed: {counts['layer_artifacts']} layer artifacts, {counts['pursuit_cells']:,} pursuit cells, {counts['snapshots']:,} coefficient snapshots, and {counts['projector_cells']:,} frozen projector cells.",'',
        'This separate verifier reconstructs atoms from the LM head and transposed saved Jacobians. It recomputes residual fractions and nonnegative-fit KKT conditions from stored coefficients, and verifies each frozen basis is orthogonal and spans its selected atoms. It does not invoke the pursuit solver or basis constructor.','',
        f"All {score_replays} baseline forwards match the native model's logits exactly. All {report_replays} baseline/transfer report score records and {report_exclusions} whole-token direction exclusions match the earlier frozen `Answer:` audit exactly. Excluded interventions remain failures in the all-attempt denominator. This is measurement hardening, not a newly discovered accuracy improvement.",'',
        '## Numerical errors','',*['- '+key+': '+f'{value:.3g}' for key,value in peak.items()],'',
        '## Interpretation limits','',
        'The animal completion differs from the paper transcript and explicitly says **trees**. Tree therefore cannot be labeled latent merely by importing the paper’s classification. Class labels must be derived from our own saved prompt and completion.','',
        'The hidden-sport prompt emits `<think>` and uses all 128 tokens in that thought block. Its inventory is a valid measurement of that trace, but it is not evidence for a concept hidden from the final answer: no final answer was produced. The complete raw text and cap flag are retained.','',
        'Report has 9/14 category-valid first-token sources. The 131-attempt rates retain the five invalid-source categories; the valid-source subset is descriptive. The fixed `Answer:` wrapper was chosen using the preceding audit, so improved report scores are not an independent confirmation.','',
        'The fitted-source grid omits the four loop-end targets and uses two of the paper’s eight demonstration prompts. The paper’s 21.5% explained-variance aggregate has a different population. Capacity under pursuit and fresh confirmation examples remain outstanding.','',
        'The corrected pipeline supersedes the preserved v1 development run, whose overly broad special-token stopping rule was caught during preflight. EOS and turn-end stop generation; thought markers do not.','']
    (out/'VALIDATION.md').write_text('\n'.join(lines))
    model.close()
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);verify(p.parse_args().output)
