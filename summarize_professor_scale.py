"""Join all scaled generations, reviewed cases, lens controls and integrity checks."""
from __future__ import annotations
from collections import Counter, defaultdict
import json
import re

import torch
from transformers import AutoTokenizer

from analyze_professor_scale import OUT, sections
from analyze_professor_followups import regions, saved_rank
from replay_professor_followups import ROOT, save, sha


def main():
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(ROOT/'model',trust_remote_code=True,local_files_only=True)
    protocol=json.loads((OUT/'protocol.json').read_text());ph=sha(OUT/'protocol.json')
    triage=json.loads((OUT/'triage.json').read_text());review=json.loads((OUT/'semantic-review.json').read_text())
    assert triage['completed']==1024
    tasks={t['id']:t for t in protocol['tasks']};manual={r['id']:r for r in review['reviews']}
    reviewed_controls={p[key] for p in review['control_pairs'] for key in ('same_prompt_control','aligned_control')}
    triage_by_id={r['id']:r for r in triage['rows']}
    for ident in reviewed_controls:
        manual[ident]=dict(id=ident,source_sha256=triage_by_id[ident]['source_sha256'],category='truthful_control',
            note='Full response inspected for the selected control comparison: correct current reasoning and final answer.')
    aggregate=defaultdict(lambda:Counter());rows=[];total_cells=Counter()
    for row in triage['rows']:
        path=ROOT/row['source'];assert sha(path)==row['source_sha256']
        folder=path.parent/'screen';screen=json.loads((folder/'summary.json').read_text())
        assert screen['source_sha256']==row['source_sha256'] and screen['protocol_sha256']==ph
        assert sha(folder/'top25.pt')==screen['top25_sha256']
        assert sha(folder/'states.pt')==screen['state_sha256']
        flags={}
        for method,windows in screen['metrics'].items():
            for window,m in windows.items():
                key=f'{method}/{window}'
                flags[key]=dict(false_top10=bool(m['false_top10_cells']),any_word_top10=bool(m['any_word_top10_cells']))
                group=f'{row["condition"]}/{key}';aggregate[group]['responses']+=1
                aggregate[group]['responses_with_false_top10']+=int(flags[key]['false_top10'])
                aggregate[group]['responses_with_any_word_top10']+=int(flags[key]['any_word_top10'])
                aggregate[group]['measured_cells']+=m['cells']
                aggregate[group]['false_top10_cells']+=m['false_top10_cells']
        total_cells['unique_recorded_cells']+=2*len(protocol['readout']['sites'])*len(screen['positions'])
        rows.append(dict(**row,semantic_review=manual.get(row['id']),screen_flags=flags,
                         screen_source=str((folder/'summary.json').relative_to(ROOT))))
    assert Counter((r['task'],r['condition']) for r in rows)==Counter({(t,c):16 for t in tasks for c in ['conflict','aligned']})
    full={};tensors={}
    for shard in range(4):
        summary=json.loads((OUT/f'full-{shard}'/'summary.json').read_text())
        assert summary['planned']==summary['completed']
        for case in summary['cases']:
            folder=OUT/f'full-{shard}'/case['id'];assert sha(folder/'top25.pt')==case['top25_sha256']
            full[case['id']]=(case,folder)
    assert len(full)==30

    def load(ident):
        summary,folder=full[ident]
        if ident not in tensors:tensors[ident]=torch.load(folder/'top25.pt',map_location='cpu',weights_only=True)
        return summary,json.loads((ROOT/summary['source']).read_text()),tensors[ident],folder

    candidates=[];checks=[]
    for pair in review['control_pairs']:
        ident=pair['candidate'];summary,g,t,folder=load(ident)
        cs,cg,ct,cf=load(pair['same_prompt_control']);als,ag,at,af=load(pair['aligned_control'])
        region=regions(g,tok);hits=[json.loads(s) for s in (folder/'hits.jsonl').read_text().splitlines()]
        for h in hits:h['region']=region[h['position']]
        meaningful=[h for h in hits if h['method']=='jlens' and h['word']=='false' and h['region'] in ('reasoning','message')]
        message=[h for h in meaningful if h['region']=='message' and h['rank']<=10]
        focus=min(message or meaningful,key=lambda h:(h['rank'],h['position'],h['site']))
        assert focus['absent_from_full_output']
        minima={}
        for h in sorted(hits,key=lambda h:(h['rank'],h['position'],h['site'])):
            if h['word']=='false':minima.setdefault(f'{h["method"]}/{h["region"]}',h)
        prefix=len(g['input_ids']);assert g['input_ids']==cg['input_ids']
        shared_end=prefix+pair['shared_output_tokens']-1
        pa=[i for i,p in enumerate(t['positions']) if p<=shared_end];pb=[ct['positions'].index(t['positions'][i]) for i in pa]
        shared_ids=torch.equal(t['ids'][:,:,pa],ct['ids'][:,:,pb]);shared_values=torch.equal(t['values'][:,:,pa],ct['values'][:,:,pb])
        reason_positions=[p for p,r in region.items() if r=='reasoning']
        focus_shared=focus['position']<=shared_end
        same_rank=saved_rank(ct,'jlens',focus['site'],focus['position'],focus['token_id']) if focus_shared else None
        message_block=next(b for b in sections(g['raw_text']) if b['name']=='message')
        wrong_letter=re.search(r'\b'+re.escape(g['wrong_answer'])+r'\b',message_block['text'])
        assert wrong_letter
        char=message_block['start']+wrong_letter.start()+1
        first_wrong=next(prefix+i for i in range(len(g['output_ids'])) if len(tok.decode(g['output_ids'][:i+1]))>=char)
        st=torch.load(OUT/'cases'/ident/'screen/top25.pt',map_location='cpu',weights_only=True)
        indices=[t['positions'].index(p) for p in st['positions']]
        agreement=float((st['ids'][:,:,:,0]==t['ids'][:,:,indices,0]).float().mean())
        exact_ids=torch.equal(st['ids'],t['ids'][:,:,indices]);exact_scores=torch.equal(st['values'],t['values'][:,:,indices])
        checks.append(dict(id=ident,screen_full_top1_agreement=agreement,screen_full_top25_ids_exact=exact_ids,
            screen_full_top25_scores_exact=exact_scores,compared_cells=st['ids'].numel()//25))
        assert agreement>.99, checks[-1]
        corresponding=[]
        def source_at_char(record,char):
            return len(record['input_ids'])+next(i for i in range(len(record['output_ids'])) if len(tok.decode(record['output_ids'][:i+1]))>=char)
        for label,record,tensor in [('same_prompt_control',cg,ct),('aligned_control',ag,at)]:
            position=None;basis=None
            if g['task']=='task-28':
                position=source_at_char(record,record['raw_text'].index('600'))
                basis='Space before 600 in the truthful description of the superseded dataset; different prefix.'
            elif focus['source_token']==':' and focus['region']=='message':
                block=next(b for b in sections(record['raw_text']) if b['name']=='message')
                match=re.search(r'\b'+re.escape(record['correct_answer'])+':',block['text'])
                if match:
                    position=source_at_char(record,block['start']+match.end())
                    basis='Colon after the chosen option letter in MESSAGE; letter differs and prefix is no longer identical.'
            elif label=='same_prompt_control' and focus_shared:
                position=focus['position'];basis='Exactly shared input/output prefix at the same absolute source position.'
            if position is not None:
                corresponding.append(dict(control=label,position=position,basis=basis,
                    jlens_saved_rank=saved_rank(tensor,'jlens',focus['site'],position,focus['token_id']),
                    logit_saved_rank=saved_rank(tensor,'logit',focus['site'],position,focus['token_id'])))
        candidates.append(dict(id=ident,task=g['task'],domain=g['domain'],correct_letter=g['correct_answer'],wrong_letter=g['wrong_answer'],
            correct_value=g['options'][g['correct_answer']],wrong_value=g['options'][g['wrong_answer']],
            full_response=g['raw_text'],prompt_text=g['prompt_text'],focus=focus,best_false_by_region=minima,
            first_wrong_message_source=first_wrong,focus_before_wrong_message=focus['position']<first_wrong,
            focus_in_shared_prefix=focus_shared,control_rank_at_shared_focus=same_rank,
            reasoning_inside_shared_prefix=bool(reason_positions) and max(reason_positions)<=shared_end,
            shared_output_tokens=pair['shared_output_tokens'],shared_prefix_top25_ids_exact=shared_ids,shared_prefix_scores_exact=shared_values,
            shared_compared_cells=2*len(t['sites'])*len(pa),same_prompt_control=pair['same_prompt_control'],aligned_control=pair['aligned_control'],
            control_response=cg['raw_text'],aligned_response=ag['raw_text'],control_prompt=cg['prompt_text'],aligned_prompt=ag['prompt_text'],
            corresponding_control_cells=corresponding,
            source=summary['source'],source_sha256=summary['source_sha256'],full_replay_folder=str(folder.relative_to(ROOT)),
            full_tensor_sha256=summary['top25_sha256']))
    representatives=[];seen=set()
    for c in candidates:
        if c['task'] not in seen:representatives.append(c['id']);seen.add(c['task'])
    assert len(representatives)==8
    # This study follows the local adapter's BOS convention. Preserve and disclose it.
    first=json.loads((OUT/'cases/task-01-conflict-s01/generation.json').read_text())
    old=json.loads((ROOT/'results/professor-followups-20261016/fresh/fresh-history-i1-s0/generation.json').read_text())
    assert first['input_ids'][0]==tok.bos_token_id and first['input_ids'][1:]==old['input_ids']
    result=dict(schema_version=1,study=protocol['study'],protocol_sha256=ph,
        author_lens=dict(repo=protocol['lens_repo'],revision=protocol['lens_revision'],manifest_sha256=protocol['lens_manifest_sha256'],model_revision=protocol['model_revision']),
        generations=1024,tasks=32,samples_per_condition_per_task=16,conditions=['conflict','aligned'],responses_per_condition=512,screened=1024,full_replayed=30,
        automatic_counts=triage['counts'],manual_review=review,strict_count=11,strict_tasks=8,representatives=representatives,
        tokenization='Native chat template plus leading BOS token 0, matching the local adapter convention. The earlier 32-sample pilot omitted that BOS; its rates and exact cell ranks are not directly interchangeable with this cohort. Both conditions here use the same convention.',
        numerical_note='The batched screen and separate full replay are not bit-identical: all 11 checked candidates have over 99.97% top-token agreement on overlapping cells. Presentation ranks come from the saved full replay; fixed-window statistics come from the batched screen.',
        screen_scope=protocol['readout'],screen_aggregates={k:dict(v) for k,v in aggregate.items()},
        screened_cells=total_cells['unique_recorded_cells'],numerical_checks=checks,candidates=candidates,rows=rows,
        caveats=['Behavioral mismatches are not established autonomous scheming; score maximization was explicitly instructed.',
                 'No validated advance monitor is established: truthful controls can share the same reasoning and early lexical flags.',
                 'Only the fixed windows were screened for all 1,024 responses. Full trajectories were replayed for 30 selected candidates and controls.',
                 '11 strict responses include duplicates: 8 tasks and 8 normalized distinct response forms. Three earlier-wrong-CHECK cases are separate.',
                 'Most dataset labels are automatic. Full semantic review covers 89 targeted outputs, 32 baselines and 16 distinct controls; the clean set is a reviewed lower bound, not a prevalence estimate.',
                 'Strongest lexical cells in full replay are exploratory, selected after looking at the readouts. Ranks are vocabulary ranks, not deception probabilities.'])
    save(OUT/'analysis.json',result)
    save(OUT/'integrity-verification.json',dict(generations_checked=1024,screens_checked=1024,full_replays_checked=30,
        generation_and_tensor_hashes_verified=True,balanced_32_by_2_by_16=True,
        numerical_checks=checks,shared_prefix_exact_cases=sum(c['shared_prefix_top25_ids_exact'] and c['shared_prefix_scores_exact'] for c in candidates)))
    print(json.dumps(dict(strict=11,tasks=8,screened_cells=result['screened_cells'],
        early_conflict=aggregate['conflict/jlens/primary_early'],early_aligned=aggregate['aligned/jlens/primary_early'],
        full_checks=checks,
        candidates=[dict(id=c['id'],focus_rank=c['focus']['rank'],region=c['focus']['region'],before_wrong=c['focus_before_wrong_message'],shared=c['focus_in_shared_prefix'],same_control_rank=c['control_rank_at_shared_focus']) for c in candidates]),indent=2))


if __name__=='__main__':main()
