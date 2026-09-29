"""Paper comparisons and paired semantic-cluster uncertainty for the fixed audit."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

from analyze_workspace import cluster_contrast, rates
from fit_ouro_family import save_json, digest


def wilson(k,n):
    if not n:return None
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-half,center+half]


def hit(readouts,query,mode='own',loops=(2,3,4),band=(24,40),metric='hit5'):
    return any(row.get(query,{}).get(metric,False) for key,row in readouts.items()
               for s,m in [key.split(':')] if m==mode and int(s)//48+1 in loops and band[0]<=int(s)%48<=band[1])


def score(condition,word='swap_answer',metric='top1',scoring='whole',target=191):
    return bool(condition.get('scores',{}).get(str(target),{}).get(word,{}).get(scoring,{}).get(metric,False))


def summarize(records):
    groups=defaultdict(list);eligibility=defaultdict(lambda:dict(attempted=0,direction_eligible=0,whole_answer_eligible=0))
    profiles=defaultdict(list);categories=defaultdict(list);trajectories=defaultdict(list);baseline=defaultdict(list)
    for record in records:
        spec=record['spec'];r=record['result'];family=spec['family'];profile=spec['profile']
        if family in ('flexible','probe','report'):
            cluster=(spec['category']+':'+spec['argument']) if family=='flexible' else (
                spec['item']['intermediate'].casefold() if family=='probe' else spec['category'])
            for scoring in ['whole','prefix']:
                baseline[f'{family}/{profile}/{scoring}'].append((cluster,{'correct':score(r['baseline'],'answer',scoring=scoring)}))
            for trial in r['trials']:
                ek=f'{family}/{profile}';eligibility[ek]['attempted']+=1
                eligibility[ek]['direction_eligible']+=trial['intervention_eligible']
                eligibility[ek]['whole_answer_eligible']+=trial['whole_answer_eligible']
                # Every attempted intervention stays in denominator; missing direction is a failure.
                names=set(trial['conditions'])
                if profile=='primary':
                    from workspace_audit import conditions
                    names|=set(conditions())
                else:names.add('transfer')
                for scoring in ['whole','prefix']:
                    for metric in ['top1','top5']:
                        values={c:score(trial['conditions'].get(c,{}),scoring=scoring,metric=metric) for c in sorted(names)}
                        key=f'{family}/{scoring}/{metric}'
                        profiles[f'{key}/{profile}'].append((cluster,values))
                        if profile=='primary':
                            groups[key].append((cluster,values))
                            if family=='flexible':categories[f'{spec["category"]}/{scoring}/{metric}'].append((cluster,values))
                            if trial['intervention_eligible'] and trial['whole_answer_eligible']:
                                groups[key+'/eligible'].append((cluster,values))
                if profile=='primary':
                    for condition in ['baseline','transfer','loops_2','loops_3','loops_4']:
                        if condition not in trial['conditions']:continue
                        entry=trial['conditions'][condition]
                        outcome='success' if score(entry) else 'failure'
                        for target in [47,95,143,191]:
                            for word in ['answer','swap_answer']:
                                rank=entry['scores'][str(target)][word]['whole']['rank']
                                if rank is not None:trajectories[f'{family}/{condition}/{outcome}/{word}/loop{target//48+1}'].append(rank)
            if family=='probe' and profile=='primary':
                lat=r['latent'];reads=lat['readouts']
                if not lat['concept_present_in_prompt']:
                    for metric in ['hit1','hit5','hit8']:
                        for loops,label in [((1,),'loop1'),((2,),'loop2'),((3,),'loop3'),((4,),'loop4'),((2,3),'loops23'),((2,3,4),'loops234')]:
                            vals={mode:hit(reads,'intermediate',mode,loops,metric=metric) for mode in ['own','raw','final']}
                            vals['foil']=statistics.mean([hit(reads,f'foil_{i}','own',loops,metric=metric) for i in range(len(lat['foils']))])
                            groups[f'latent/{label}/{metric}'].append((cluster,vals))
        elif family=='language':
            for metric in ['hit1','hit5','hit8']:
                for loops,label in [((1,),'loop1'),((2,),'loop2'),((3,),'loop3'),((4,),'loop4'),((2,3),'loops23'),((2,3,4),'loops234')]:
                    vals={c:hit(v['readouts'],'language',loops=loops,metric=metric) for c,v in r['conditions'].items()}
                    groups[f'language/{profile}/{label}/{metric}'].append((spec['passage']['key'],vals))
        elif family=='linecount':
            for query in ['number','exact']:
                for metric in ['hit1','hit5','hit8']:
                    vals={c:hit(v['readouts'],query,metric=metric) for c,v in r['conditions'].items()}
                    groups[f'linecount/{query}/{metric}'].append((spec['passage']['tag'],vals))
    results={k:rates(v) for k,v in groups.items()}
    for key,vs in results.items():
        if not key.startswith('latent'):
            for value in vs.values():value['wilson95']=wilson(value['sum'],value['n'])
    contrasts={}
    for family,metric in [('flexible','top1'),('probe','top1'),('report','top5')]:
        key=f'{family}/whole/{metric}'
        for control in ['baseline','random_42','random_43','random_44','loops_2','loops_3','loops_4','swap','raw','final']:
            contrasts[f'{family}: transfer - {control}']=cluster_contrast(groups[key],'transfer',control)
    for loops in ['loop1','loop2','loop3','loop4','loops23','loops234']:
        for control in ['foil','raw','final']:
            contrasts[f'latent {loops}: own - {control}']=cluster_contrast(groups[f'latent/{loops}/hit5'],'own',control)
    contrasts['language: explicit - automatic']=cluster_contrast(groups['language/primary/loops234/hit5'],'explicit','automatic')
    for metric in ['hit1','hit5']:
        contrasts[f'linecount {metric}: direct - none']=cluster_contrast(groups[f'linecount/number/{metric}'],'direct','none')
    return dict(records=len(records),eligibility=dict(eligibility),rates=results,contrasts=contrasts,
        baseline={k:rates(v) for k,v in baseline.items()},profiles={k:rates(v) for k,v in profiles.items()},
        categories={k:rates(v) for k,v in categories.items()},
        trajectories={k:dict(n=len(v),median_rank=statistics.median(v)) for k,v in trajectories.items()})


def pct(value):return f'{100*value:.1f}%'
def cell(value):return f'{int(value["sum"])}/{value["n"]} ({pct(value["rate"])})'


def render(summary,protocol):
    s=summary;r=s['rates'];p=protocol['paper_metrics']
    lines=['# Ouro workspace replication audit','',
      'Completed the entire fixed grid over all released prompts for the previously inconclusive families. '
      'This is a method audit informed by earlier results, not an independent preregistered replication. '
      'The original run remains in `../workspace-full/`.', '',
      '## Comparison with the paper','',
      'Reference: [Wang and Reid, v1, Tables 6–7](https://arxiv.org/html/2609.01924v1#A2). '
      'The transfer formula is an explicit interpretation of its clamp-transfer description; author code was not located. '
      'Denominators and tokenization differences are retained, so these are not guaranteed identical experiments.','',
      '| Measurement | Paper | Audit primary, whole-token metric |','|---|---:|---:|',
      f'| Flexible steering | 73/192 (38%) | {cell(r["flexible/whole/top1"]["transfer"])} |',
      f'| Loop 2 only | 40/192 (21%) | {cell(r["flexible/whole/top1"]["loops_2"])} |',
      f'| Loop 3 only | 29% | {cell(r["flexible/whole/top1"]["loops_3"])} |',
      f'| Loop 4 only | 48/192 (25%) | {cell(r["flexible/whole/top1"]["loops_4"])} |',
      f'| Double strength | 14/192 (7%) | {cell(r["flexible/whole/top1"]["strength_2"])} |',
      f'| Two-hop steering | 26/90 (29%) | {cell(r["probe/whole/top1"]["transfer"])} |',
      f'| Verbal report top-5 | 106/133 (80%) | {cell(r["report/whole/top5"]["transfer"])} |',
      f'| Verbal report top-1 | 67% | {cell(r["report/whole/top1"]["transfer"])} |']
    lang=r['language/primary/loops234/hit5'];line=r['linecount/number/hit1']
    lines += [f'| Language selectivity difference | +25 pp | {100*(lang["explicit"]["rate"]-lang["automatic"]["rate"]):+.1f} pp |',
      f'| Line-count label readout, direct / unasked | 100% / 9% | {pct(line["direct"]["rate"])} / {pct(line["none"]["rate"])} |','',
      '## Method corrections and interpretation','',
      '- The original primary used a coordinate-exchange reflection. This audit uses source-projection transfer and retains reflection plus two alternative clamp interpretations as sensitivities.',
      '- Verbal report uses top-5 as the paper headline does, and also retains top-1. Every original-answer token is recorded; punctuation sources and invalid-category answers are visible.',
      '- Interventions no longer depend on the original answer having a complete single-token form. All attempts remain in denominators, with eligibility counts and separately labelled prefix scores.',
      '- The previous language effect already equalled the reported +25 points. A confidence interval including zero is uncertainty, not numerical disagreement.',
      '- Latent readout is examined per loop at matched physical positions. The former all-loops criterion included loop 4, where own-loop and final-target lenses are identical, and demanded superiority not required by the paper.',
      '- Line-count selectivity was absent from the first run and is included here. Exact author prompt wrapping remains unknown.',
      '- The adapter and existing fit use all four loops, distinct virtual sites, native loop-end logits, disabled cache/checkpointing, and the reference Jacobian estimator. Real-checkpoint verification is repeated on each worker.',
      '- The fit shares corpus size, token length, and skip geometry with the stated recipe. Exact author corpus rows and code are unavailable. Software and GPU differ. The existing fitted matrices are reused, not re-estimated.', '',
      '## Paired evidence','',
      'Semantic-cluster bootstrap intervals (10,000 resamples). The wider intervals reuse the earlier seven-family tail adjustment as a sensitivity check; they do not control all comparisons in this expanded grid. '
      'Repeated conditions do not create new independent observations.','',
      '| Contrast | Difference | 95% cluster interval | Conservative interval | Clusters |','|---|---:|---|---|---:|']
    for key,v in s['contrasts'].items():
        def interval(x):return 'unavailable' if x is None else f'[{100*x[0]:.1f}, {100*x[1]:.1f}] pp'
        lines.append(f'| {key} | {100*v["difference"]:+.1f} pp | {interval(v["ci95"])} | {interval(v["ci_familywise"])} | {v["clusters"]} |')
    lines+=['','## Complete intervention grid','',
      'All settings were fixed before this rerun. These rows describe sensitivity; no winning setting replaces the primary.','',
      '| Condition | Flexible top-1 | Two-hop top-1 | Report top-5 |','|---|---:|---:|---:|']
    for condition in r['flexible/whole/top1']:
        lines.append('| '+condition+' | '+' | '.join(cell(r[f'{family}/whole/{metric}'][condition]) for family,metric in [('flexible','top1'),('probe','top1'),('report','top5')])+' |')
    lines+=['','## Prompt and direction-token sensitivities','',
      '| Profile | Flexible top-1 | Two-hop top-1 | Report top-5 |','|---|---:|---:|---:|']
    for profile in protocol['profiles']:
        cells=[]
        for family,metric in [('flexible','top1'),('probe','top1'),('report','top5')]:
            value=s['profiles'].get(f'{family}/whole/{metric}/{profile}',{}).get('transfer')
            cells.append(cell(value) if value else 'not applicable')
        lines.append('| '+profile+' | '+' | '.join(cells)+' |')
    lines+=['','## Eligibility and denominator audit','',
      '| Primary family | Attempts | Valid directions | Whole-token target answer |','|---|---:|---:|---:|']
    for family in ['flexible','probe','report']:
        v=s['eligibility'][family+'/primary'];lines.append(f'| {family} | {v["attempted"]} | {v["direction_eligible"]} | {v["whole_answer_eligible"]} |')
    lines+=['','Prefix-only outcomes, per-category effects, matched-loop latent readout, baseline accuracy, and loop-end rank trajectories are in `summary.json`. '
      'Prefix hits are not complete-word success. Every prompt, token ID, condition and score is retained in `records/`.', '',
      '## Remaining limits','',
      'This audit covers the previously inconclusive families and both released selectivity tasks. It does not execute all eleven suites, re-fit on the unreleased author sample, '
      'test other checkpoints, or independently establish the proposed boundary mechanism. Capacity, ignition, dual-task interference, summoning, and the sparse inventory remain outside this run. '
      'An exhaustive finite grid is not an exhaustive search over every possible implementation.']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    m=json.loads((a.output/'manifest.json').read_text());fingerprint=digest(a.output/'manifest.json')
    records=[json.loads(f.read_text()) for f in sorted((a.output/'records').glob('*.json'))]
    if {r['id'] for r in records}!={t['id'] for t in m['tasks']}:raise ValueError('Incomplete or extra records')
    if any(r['manifest_sha256']!=fingerprint for r in records):raise ValueError('Mixed manifests')
    s=summarize(records);save_json(a.output/'summary.json',s)
    protocol=json.loads((a.output/'protocol.json').read_text())
    (a.output/'REPORT.md').write_text(render(s,protocol))
    with (a.output/'rates.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['measurement','condition','n','successes','rate'])
        for key,values in s['rates'].items():
            for c,v in values.items():writer.writerow([key,c,v['n'],v['sum'],v['rate']])
    print('Analyzed',len(records),'records:',a.output/'REPORT.md',flush=True)


if __name__=='__main__':main()
