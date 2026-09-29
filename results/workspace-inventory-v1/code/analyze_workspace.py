"""Recompute workspace claims from per-trial records, with paired cluster intervals."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from fit_ouro_family import save_json,digest


def cluster_contrast(rows,left,right,*,draws=10000):
    """Rows are (cluster, {condition: outcome}); bootstrap complete clusters."""
    groups=defaultdict(list)
    for group,values in rows:
        if left in values and right in values:
            groups[str(group)].append(float(values[left])-float(values[right]))
    if not groups:return dict(n=0,clusters=0,difference=None,ci95=None,ci_familywise=None)
    sums=np.array([sum(values) for values in groups.values()])
    counts=np.array([len(values) for values in groups.values()])
    observed=float(sums.sum()/counts.sum())
    if len(groups)<2:return dict(n=int(counts.sum()),clusters=len(groups),difference=observed,ci95=None,ci_familywise=None)
    rng=np.random.default_rng(42)
    sampled=rng.integers(0,len(groups),size=(draws,len(groups)))
    estimates=sums[sampled].sum(1)/counts[sampled].sum(1)
    # Seven predeclared primary hypothesis families. Also report ordinary 95% CIs.
    tail=0.05/(2*7)
    return dict(n=int(counts.sum()),clusters=len(groups),difference=observed,
                ci95=np.quantile(estimates,[.025,.975]).tolist(),
                ci_familywise=np.quantile(estimates,[tail,1-tail]).tolist())


def rates(rows):
    totals=defaultdict(list)
    for _,values in rows:
        for name,value in values.items():totals[name].append(float(value))
    return {name:dict(n=len(values),sum=float(sum(values)),rate=float(np.mean(values)))
            for name,values in totals.items()}


def causal_values(trial,word='swap_answer',metric='top1'):
    return {name:record['scores']['191'][word][metric]
            for name,record in trial['conditions'].items()
            if record['scores']['191'][word]['eligible']}


def any_hit(readouts,query,mode='own',loops=(2,3,4)):
    return any(readouts.get(f'loop{loop}_{mode}',{}).get(query,{}).get('hit5',False)
               for loop in loops)


def analyze(records):
    collections=defaultdict(list)
    eligibility=defaultdict(lambda:dict(total=0,eligible=0))
    examples=[]
    for record in records:
        family=record['family'];result=record['result'];spec=record['specification']
        if family in ('flexible','report'):
            group=(result['category']+':'+result['argument']) if family=='flexible' else result['category']
            if family=='flexible' and result.get('baseline_answer',{}).get('eligible'):
                collections['flexible_baseline_accuracy'].append((group,{'correct':result['baseline_answer']['top1']}))
            for trial in result['trials']:
                eligibility[family]['total']+=1
                if not trial['eligible']:continue
                eligibility[family]['eligible']+=1
                values=causal_values(trial)
                collections[family].append((group,values))
                collections[family+'_top5'].append((group,causal_values(trial,metric='top5')))
                baseline=trial['conditions']['baseline']['scores']['191']['answer']['top1']
                if baseline:collections[family+'_baseline_correct'].append((group,values))
                if family=='report' and trial.get('baseline_target_outside_top10'):
                    collections['report_outside_top10'].append((group,values))
                if family=='report' and result.get('baseline_in_category'):
                    collections['report_valid_category'].append((group,values))
                if values.get('sustained') and not values.get('baseline') and len(examples)<12:
                    examples.append(dict(family=family,prompt=result['prompt'],source=trial['source'],
                        destination=trial['destination'],expected=trial['swap_answer'],
                        before=trial['conditions']['baseline']['top5'],after=trial['conditions']['sustained']['top5']))
        elif family=='probe':
            eligibility[family]['total']+=1
            group=spec['item']['intermediate'].casefold()
            if result.get('baseline_answer',{}).get('eligible'):
                collections['probe_baseline_accuracy'].append((group,{'correct':result['baseline_answer']['top1']}))
            if result['eligible']:
                eligibility[family]['eligible']+=1
                values=causal_values(result);collections['probe'].append((group,values))
                baseline=result['conditions']['baseline']['scores']['191']['answer']['top1']
                if baseline:
                    collections['probe_baseline_correct'].append((group,values))
                    retained={name:record['scores']['191']['answer']['top1'] for name,record in result['ablations'].items()}
                    collections['ablation_retention'].append((group,retained))
                if values.get('sustained') and not values.get('baseline') and len(examples)<12:
                    examples.append(dict(family=family,prompt=result['prompt'],source=result['source'],
                        destination=result['destination'],expected=result['swap_answer'],
                        before=result['conditions']['baseline']['top5'],after=result['conditions']['sustained']['top5']))
            if not result['concept_present_in_prompt']:
                reads=result['readouts']
                for loop in range(1,5):
                    own=reads[f'loop{loop}_own'].get('intermediate',{})
                    if own.get('min_rank') is None:continue
                    foil_hits=[reads[f'loop{loop}_own'].get(f'foil_{i}',{}).get('hit5',False)
                               for i in range(len(result['foils']))]
                    collection={'own':float(own['hit5']),
                        'raw':float(reads[f'loop{loop}_raw']['intermediate']['hit5']),
                        'final':float(reads[f'loop{loop}_final']['intermediate']['hit5']),
                        'foil':float(np.mean(foil_hits)) if foil_hits else 0.0}
                    collections[f'latent_loop{loop}'].append((group,collection))
                collections['latent_primary'].append((group,dict(
                    own=any_hit(reads,'intermediate'),raw=any_hit(reads,'intermediate','raw'),
                    final=any_hit(reads,'intermediate','final'),
                    foil=np.mean([any_hit(reads,f'foil_{i}') for i in range(len(result['foils']))]))))
        elif family=='language':
            values={name:any_hit(condition['readouts'],'language') for name,condition in result['conditions'].items()}
            collections['language'].append((result['passage']['key'],values))
        elif family=='modulation':
            groups=defaultdict(list)
            for trial in result['trials']:groups[trial['group']].append(any_hit(trial['readouts'],'topic'))
            values={name:float(np.mean(values)) for name,values in groups.items()}
            collections['modulation'].append((result['topic']['name'],values))
        elif family=='introspection':
            eligibility[family]['total']+=1
            if not result['eligible']:continue
            eligibility[family]['eligible']+=1
            values={name:entry['scores']['191']['concept']['top1'] for name,entry in result['conditions'].items()}
            collections['introspection'].append((result['concept'],values))
            rr={name:1/entry['scores']['191']['concept']['rank'] for name,entry in result['conditions'].items()}
            collections['introspection_reciprocal_rank'].append((result['concept'],rr))
            if not result.get('concept_present_in_prompt',False):
                collections['introspection_absent'].append((result['concept'],values))
    summaries={name:rates(rows) for name,rows in collections.items()}
    contrasts={}
    for name in ('flexible','probe','report'):
        for control in ('baseline','random_matched','loop2','loop3','loop4','raw_lens','final_target','early_band'):
            contrasts[f'{name}: sustained - {control}']=cluster_contrast(collections[name],'sustained',control)
    contrasts['necessity: unrelated - argument retention']=cluster_contrast(collections['ablation_retention'],'unrelated','argument')
    for control in ('foil','raw','final'):
        contrasts[f'latent: own - {control}']=cluster_contrast(collections['latent_primary'],'own',control)
    contrasts['language: explicit - automatic']=cluster_contrast(collections['language'],'explicit','automatic')
    contrasts['modulation: focus - control']=cluster_contrast(collections['modulation'],'focus','control')
    contrasts['modulation: control - suppress']=cluster_contrast(collections['modulation'],'control','suppress')
    for control in ('0','random8'):
        contrasts[f'introspection: 8 - {control}']=cluster_contrast(collections['introspection_absent'],'8.0',control)
    criteria={
        'Flexible steering':['flexible: sustained - baseline','flexible: sustained - random_matched'],
        'Internal reasoning':['probe: sustained - baseline','probe: sustained - random_matched'],
        'Verbal report':['report: sustained - baseline','report: sustained - random_matched'],
        'Sustained recurrence interface':['flexible: sustained - loop2'],
        'Causal necessity':['necessity: unrelated - argument retention'],
        'Latent readout':['latent: own - foil','latent: own - raw','latent: own - final'],
        'Task selectivity':['language: explicit - automatic'],
        'Directed modulation':['modulation: focus - control','modulation: control - suppress'],
        'Injected-content report':['introspection: 8 - 0','introspection: 8 - random8']}
    verdicts={}
    for name,keys in criteria.items():
        ordinary=all(contrasts[key]['ci95'] is not None and contrasts[key]['ci95'][0]>0 for key in keys)
        conservative=all(contrasts[key]['ci_familywise'] is not None and contrasts[key]['ci_familywise'][0]>0 for key in keys)
        verdicts[name]=dict(positive_95ci=ordinary,positive_conservative_ci=conservative,contrasts=keys)
    return dict(eligibility=dict(eligibility),rates=summaries,contrasts=contrasts,verdicts=verdicts,
                examples=examples,records=len(records),
                interpretation='Fixed qualitative tests, not exact numerical replication. Nonpositive intervals mean not established by this test, not absence of representations.')


def render(summary):
    lines=['# Ouro workspace reproduction','',
        'Full 1,000-prompt fitting and predeclared tests of the base Ouro-2.6B checkpoint. '
        'Results below are operational tests of workspace properties; the authors’ exact code and fitting sample were unavailable.','',
        '## Primary evidence','',
        '| Property | Paired 95% intervals positive | Conservative intervals positive |',
        '|---|---|---|']
    for name,value in summary['verdicts'].items():
        lines.append(f'| {name} | {"Yes" if value["positive_95ci"] else "Not established"} | {"Yes" if value["positive_conservative_ci"] else "Not established"} |')
    lines+=['','Conservative intervals use percentile tails 0.05/(2×7), covering seven primary '
        'hypothesis families. This is a conservative descriptive check, not a simultaneous guarantee '
        'for every individual contrast or a substitute for independent replication.','',
        '## Steering counts','',
        '| Experiment | Condition | Successes / trials | Rate |','|---|---|---:|---:|']
    for family in ('flexible','probe','report','report_top5','ablation_retention','introspection'):
        for condition,value in summary['rates'].get(family,{}).items():
            lines.append(f'| {family} | {condition} | {value["sum"]:.0f} / {value["n"]} | {100*value["rate"]:.1f}% |')
    lines+=['','## Paired differences','',
        '| Contrast | Difference (percentage points) | 95% cluster interval | Clusters |',
        '|---|---:|---|---:|']
    for name,value in summary['contrasts'].items():
        if value['difference'] is None:continue
        interval=value['ci95']
        formatted='Unavailable' if interval is None else f'[{100*interval[0]:.1f}, {100*interval[1]:.1f}]'
        lines.append(f'| {name} | {100*value["difference"]:.1f} | {formatted} | {value["clusters"]} |')
    lines+=['','## Interpretation limits','',
        'The protocol and sensitivities were fixed before full-fit outcomes. Bootstrap units '
        'are semantic clusters, not individual repeated swaps. Language selectivity has only '
        'eight passages. Tokenization exclusions and baseline-correct subsets are retained '
        'in summary.json. Readout hits alone do not establish causal use. Sensitivity results '
        'are shown but never replace failed primary conditions.','',
        'Not assessed: all eleven suites, capacity, ignition, dual-task interference, '
        'or the sparse concept inventory. Differences in the undisclosed causal-band and '
        'prompt-formatting details limit exact comparison with the paper’s percentages.','',
        '## Example successful interventions','']
    for example in summary['examples']:
        before=example['before'][0]['token'];after=example['after'][0]['token']
        lines += [f'- {example["prompt"]!r}: swap {example["source"]!r} → {example["destination"]!r}; '
                  f'top output {before!r} → {after!r}; expected {example["expected"]!r}.']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-incomplete',action='store_true')
    args=p.parse_args();manifest=json.loads((args.output/'manifest.json').read_text())
    fingerprint=digest(args.output/'manifest.json')
    records=[json.loads(path.read_text()) for path in sorted((args.output/'records').glob('*.json'))]
    if any(record['manifest_sha256']!=fingerprint for record in records):raise ValueError('Mixed trial manifests')
    if len(records)!=manifest['n_items'] and not args.allow_incomplete:raise ValueError('Evaluation is incomplete')
    summary=analyze(records)
    save_json(args.output/'summary.json',summary)
    (args.output/'REPORT.md').write_text(render(summary))
    print(f'Analyzed {len(records)} items: {args.output / "REPORT.md"}',flush=True)


if __name__=='__main__':main()
