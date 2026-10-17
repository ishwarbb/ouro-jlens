"""Select saved response pairs without inspecting any lens readout."""
from collections import defaultdict
import hashlib
import json

from .common import digest, read_jsonl


def select(inventory, *, loops=4, tasks=None, scenarios_per_task=0, rollouts=0, conditions=None):
    files = {f['id']: f for f in json.loads((inventory / 'files.json').read_text())}
    prompts = {p['uid']: p for p in read_jsonl(inventory / 'prompts.jsonl')}
    candidates = [p for p in read_jsonl(inventory / 'pair-groups.jsonl')
                  if p['model_size'] == '2.6B' and p['loops'] == loops
                  and (not tasks or p['task'] in tasks)
                  and (not conditions or p['condition'] in conditions)]
    scenarios = defaultdict(set)
    for pair in candidates:
        scenarios[pair['task']].add((pair['root_index'], pair['filename_rollouts'], pair['base_id']))
    chosen = set()
    for task, keys in scenarios.items():
        ordered = sorted(keys, key=lambda k: digest([task, *k]))
        for key in (ordered[:scenarios_per_task] if scenarios_per_task else ordered):
            chosen.add((task, *key))
    pairs, samples = [], {}
    for pair in candidates:
        if (pair['task'], pair['root_index'], pair['filename_rollouts'], pair['base_id']) not in chosen:
            continue
        indices = pair['index_pairs'][:rollouts] if rollouts else pair['index_pairs']
        for index in indices:
            ids = []
            for side in ['baseline', 'intervention']:
                prompt = prompts[pair[side]]
                ident = f'{prompt["uid"]}:{index}'
                ids.append(ident)
                samples[ident] = dict(id=ident, prompt=prompt, response_index=index,
                                     source_file=files[prompt['file_id']])
            pairs.append(dict(task=pair['task'], model=pair['model'], loops=loops, condition=pair['condition'],
                              root_index=pair['root_index'], n=pair['filename_rollouts'], base_id=pair['base_id'],
                              response_index=index, baseline=ids[0], intervention=ids[1]))
    if not samples:
        raise ValueError('No eligible saved response pairs for this selection')
    return list(samples.values()), pairs


def load_sample(sample):
    prompt = sample['prompt']
    with open(sample['source_file']['path'], 'rb') as stream:
        stream.seek(prompt['byte_offset'])
        line = stream.readline()
    if hashlib.sha256(line).hexdigest() != prompt['record_sha256']:
        raise ValueError('Saved source record changed since the inventory')
    row = json.loads(line)
    responses = row['response']
    if isinstance(responses, str):
        responses = [responses]
    response = responses[sample['response_index']]
    if hashlib.sha256(response.encode()).hexdigest() != prompt['response_sha256'][sample['response_index']]:
        raise ValueError('Saved response differs from the inventory')
    return row['messages'], response
