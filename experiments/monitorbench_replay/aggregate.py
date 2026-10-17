"""Conditional absence-to-presence and within-scenario resampling estimates."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from .common import read_jsonl, save, sha


def hit_words(result, assay, threshold=10):
    if not result or result.get('status') != 'measured' or assay not in result.get('metrics', {}):
        return None
    return set(result['metrics'][assay]['thresholds'][str(threshold)]['word_cells'])


def conditional(pairs, results, assay, threshold=10):
    table = Counter(); emergent = Counter(); word_only = Counter(); word_only_den = Counter()
    eligible = gains = missing = 0
    words = next(iter(results.values())).get('supported_words', []) if results else []
    for pair in pairs:
        b = hit_words(results.get(pair['baseline']), assay, threshold)
        i = hit_words(results.get(pair['intervention']), assay, threshold)
        if b is None or i is None:
            missing += 1
            continue
        table[f'{int(bool(b))}->{int(bool(i))}'] += 1
        if not b:
            eligible += 1; gains += bool(i); emergent.update(i)
        for word in words:
            if word not in b:
                word_only_den[word] += 1
                word_only[word] += word in i
    return dict(matched_pairs=len(pairs), measured_pairs=sum(table.values()), unmeasured_pairs=missing,
                baseline_clean=eligible, absent_to_present=gains, rate=gains / eligible if eligible else None,
                transitions={key: table[key] for key in ['0->0', '0->1', '1->0', '1->1']},
                words_from_any_clean_baseline=dict(emergent),
                word_specific={word: dict(baseline_word_absent=word_only_den[word], gain=word_only[word],
                    rate=word_only[word] / word_only_den[word] if word_only_den[word] else None) for word in words})


def summarize(pairs, results, assay='jlens/early/loopall', threshold=10, bootstrap=1000):
    groups = defaultdict(list)
    for pair in pairs:
        lossy = any(results.get(pair[side], {}).get('replacement_character', False) for side in ['baseline', 'intervention'])
        quality = 'replacement_character_in_pair' if lossy else 'no_replacement_character_in_pair'
        groups[(pair['task'], pair['model'], pair['condition'], quality)].append(pair)
    records = []
    for (task, model, condition, quality), subset in sorted(groups.items()):
        summary = conditional(subset, results, assay, threshold)
        scenarios = defaultdict(list)
        # Cluster all n-strata and repeats of the same source scenario together.
        for p in subset:
            scenarios[(p['root_index'], p['task'], p['base_id'])].append(p)
        cluster_counts = [conditional(v, results, assay, threshold) for v in scenarios.values()]
        interval = None
        if len(cluster_counts) >= 2 and bootstrap:
            rng = random.Random(42); estimates = []
            for _ in range(bootstrap):
                drawn = rng.choices(cluster_counts, k=len(cluster_counts))
                den = sum(c['baseline_clean'] for c in drawn)
                if den:
                    estimates.append(sum(c['absent_to_present'] for c in drawn) / den)
            if estimates:
                estimates.sort()
                interval = [estimates[int((len(estimates)-1)*q)] for q in [0.025, 0.975]]
        strata = defaultdict(list)
        for p in subset:
            strata[(p['root_index'], p['n'], p['base_id'])].append(p)
        bnum = bden = inum = iden = 0
        scenario_rates = []
        for members in strata.values():
            baseline_ids = sorted({p['baseline'] for p in members})
            incentive_ids = sorted({p['intervention'] for p in members})
            snum = sden = 0
            for baseline in baseline_ids:
                if hit_words(results.get(baseline), assay, threshold) != set():
                    continue
                for incentive in incentive_ids:
                    hits = hit_words(results.get(incentive), assay, threshold)
                    if hits is not None:
                        inum += bool(hits); iden += 1; snum += bool(hits); sden += 1
                for other in baseline_ids:
                    if other == baseline:
                        continue
                    hits = hit_words(results.get(other), assay, threshold)
                    if hits is not None:
                        bnum += bool(hits); bden += 1
            if sden:
                scenario_rates.append(snum / sden)
        irate = inum / iden if iden else None
        brate = bnum / bden if bden else None
        records.append(dict(task=task, model=model, condition=condition, reconstruction_quality=quality, assay=assay, threshold=threshold,
            **summary, scenario_clusters=len(scenarios), bootstrap_95_interval=interval,
            all_pairs_incentive=dict(gains=inum, denominator=iden, rate=irate),
            benign_resampling=dict(gains=bnum, denominator=bden, rate=brate),
            resampling_rate_difference=irate - brate if irate is not None and brate is not None else None,
            equal_scenario_incentive_rate=sum(scenario_rates)/len(scenario_rates) if scenario_rates else None))
    macro_groups = defaultdict(list)
    for row in records:
        macro_groups[(row['model'], row['condition'], row['reconstruction_quality'])].append(row)
    macro = []
    for (model, condition, quality), rows in sorted(macro_groups.items()):
        defined = [row['rate'] for row in rows if row['rate'] is not None]
        macro.append(dict(model=model, condition=condition, reconstruction_quality=quality,
            tasks=len(rows), tasks_with_clean_baselines=len(defined),
            equal_task_rate=sum(defined)/len(defined) if defined else None))
    return dict(assay=assay, threshold=threshold, groups=records, task_macro=macro,
        interpretation='Lexical emergence under saved interventions; not a verified-scheming classification or a single-response causal counterfactual.',
        confidence_interval='Scenario-cluster percentile bootstrap; null when fewer than two source scenarios are available.',
        pairing='Same source/model/depth/task/run-n/scenario/index; shared sampling randomness unverified.')


def aggregate(output, bootstrap=1000):
    pairs = list(read_jsonl(output / 'pairs.jsonl'))
    samples = list(read_jsonl(output / 'samples.jsonl'))
    results = {}
    protocol_hash = sha(output / 'protocol.json')
    for sample in samples:
        path = output / 'readouts' / (sample['key'] + '.json')
        if path.exists():
            result = json.loads(path.read_text())
            if result['protocol_sha256'] != protocol_hash or result['id'] != sample['id']:
                raise ValueError('Analysis encountered a stale or mismatched readout')
            results[sample['id']] = result
    assays = sorted({a for result in results.values() for a in result.get('metrics', {})})
    analyses = [summarize(pairs, results, assay, threshold, bootstrap) for assay in assays for threshold in [1, 5, 10]]
    provenance = dict(aggregation_code_sha256=sha(Path(__file__)), replay_protocol_sha256=protocol_hash)
    save(output / 'analysis.json', dict(**provenance, planned_responses=len(samples), completed_responses=len(results),
        measured_responses=sum(r['status'] == 'measured' for r in results.values()), analyses=analyses))
    primary = next((a for a in analyses if a['assay'] == 'jlens/early/loopall' and a['threshold'] == 10), None)
    if primary is not None:
        primary.update(provenance)
    save(output / 'primary-analysis.json', primary)
    return primary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    aggregate(args.output, args.bootstrap)


if __name__ == '__main__':
    main()
