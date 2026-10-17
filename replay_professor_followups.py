"""Author-lens readouts of behavior-selected archived transcripts; no resampling."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'jacobian-lens'))
import torch
from analyze_deception_run2 import parse_sections
from ouro_lens_config import DEFAULT_FAMILY


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


@torch.inference_mode()
def replay(out):
    from ouro_full_fit import load_model
    from ouro_lens_family import OuroLensFamily
    from workspace_interventions import Vocabulary
    from jlens.hooks import ActivationRecorder
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    protocol = json.loads((out / 'protocol.json').read_text())
    protocol_hash = sha(out / 'protocol.json')
    family = OuroLensFamily.load(DEFAULT_FAMILY)
    assert family.metadata['revision'] == protocol['lens_revision']
    model = load_model()
    family.check_model(model)
    tok, sites, words = model.tokenizer, protocol['sites'], protocol['words']
    vocab = Vocabulary(tok)
    tokens = {word: vocab.candidates(word) for word in words}
    token_word = {t: word for word, ids in tokens.items() for t in ids}
    environment = dict(protocol_sha256=protocol_hash, code_sha256=sha(Path(__file__)),
        lens_path=str(DEFAULT_FAMILY.relative_to(ROOT)), lens_manifest_sha256=sha(DEFAULT_FAMILY/'family.json'),
        lens_repo=family.metadata['repo'], lens_revision=family.metadata['revision'],
        model_revision=family.metadata['model_revision'], vocabulary=tokens,
        torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        precision='Saved or re-captured BF16 residuals; FP32 transport; original BF16 RMSNorm/head',
        scope='Exact saved token IDs, no generation. Top25 readouts across 68 workspace sites and all generated source positions; EOS source excluded.')
    save(out / 'environment.json', environment)
    results = []
    try:
        for entry in protocol['rows']:
            destination = out / entry['id']
            destination.mkdir(exist_ok=True)
            source = ROOT / entry['source']
            assert sha(source) == entry['source_sha256']
            generation = json.loads(source.read_text())
            ids = generation['input_ids'] + generation['output_ids']
            prefix = len(generation['input_ids'])
            raw = generation['raw_text']
            assert tok.decode(generation['output_ids']) == raw
            original = ROOT / entry['original_state']
            if original.exists():
                state_path = original
                state = torch.load(original, weights_only=True, map_location='cpu', mmap=True)
                assert state['ids'] == ids
                states = state['states']
                state_mode = 'existing full-causal replay states; not exact online generation states'
            else:
                state_path = destination / 'states.pt'
                with ActivationRecorder(model.layers, at=range(192)) as recorder:
                    model.forward(torch.tensor([ids], device='cuda'))
                states = torch.stack([recorder.activations[s][0].cpu() for s in range(192)])
                del recorder
                torch.save(dict(ids=ids, states=states), state_path)
                state_mode = 'new full-causal replay of exact saved IDs; not resampled'
            assert states.shape == (192, len(ids), 2048)
            sections, spans, _, _ = parse_sections(raw)
            ends = [len(tok.decode(generation['output_ids'][:i])) for i in range(len(generation['output_ids'])+1)]
            positions = list(range(prefix - 1, len(ids) - 1))
            phases = {}
            for position in positions:
                index = position-prefix
                phase = 'prompt' if index < 0 else 'other'
                if index >= 0:
                    for name, (start, end) in spans.items():
                        if start < ends[index+1] <= end and re.search(r'\w', tok.decode([ids[position]])):
                            phase = name.lower()
                            break
                phases[position] = phase
            top_ids = torch.empty((2,len(sites),len(positions),25),dtype=torch.int32)
            top_values = torch.empty_like(top_ids,dtype=torch.float32)
            hits = []
            for si, site in enumerate(sites):
                matrix = family.lenses[family.target_for(site)].jacobians[site].cuda()
                for start in range(0,len(positions),32):
                    ps = positions[start:start+32]
                    h = states[site,ps].cuda().float()
                    for mi, method in enumerate(('jlens','logit')):
                        logits = model.unembed(h@matrix.T if mi==0 else h).float()
                        assert torch.isfinite(logits).all()
                        values, order = logits.topk(25,-1)
                        logz = logits.logsumexp(-1)
                        top_ids[mi,si,start:start+len(ps)] = order.cpu().int()
                        top_values[mi,si,start:start+len(ps)] = values.cpu()
                        for pi, (row_ids, row_values) in enumerate(zip(order.tolist(),values.tolist())):
                            for ordinal, token in enumerate(row_ids):
                                if token not in token_word:
                                    continue
                                word = token_word[token]
                                position = ps[pi]
                                hits.append(dict(method=method,site=site,loop=site//48+1,layer=site%48,
                                    position=position,generated_position=position-prefix,phase=phases[position],
                                    word=word,token_id=token,rank=1+sum(v>row_values[ordinal] for v in row_values),
                                    ordinal=ordinal+1,probability=float((values[pi,ordinal]-logz[pi]).exp()),
                                    source_token=tok.decode([ids[position]]),next_token=tok.decode([ids[position+1]]),
                                    absent_from_full_output=not bool(re.search(r'\b'+re.escape(word)+r'\b',raw,re.I)),
                                    top10=[tok.decode([t]) for t in row_ids[:10]]))
            torch.save(dict(methods=['jlens','logit'],sites=sites,positions=positions,ids=top_ids,values=top_values),destination/'top25.pt')
            with (destination/'hits.jsonl').open('w') as stream:
                for hit in hits:
                    stream.write(json.dumps(hit,ensure_ascii=False,allow_nan=False)+'\n')
            best = {}
            for hit in sorted(hits,key=lambda h:(h['rank'],h['position'],h['site'])):
                key=f'{hit["method"]}/{hit["phase"]}/{hit["word"]}'
                best.setdefault(key,hit)
            result = dict(**entry,protocol_sha256=protocol_hash,full_response=raw,prompt_text=generation['prompt_text'],
                sections=sections,token_count=len(ids),source_positions=len(positions),phase_positions=dict(Counter(phases.values())),
                state_path=str(state_path.relative_to(ROOT)),state_sha256=sha(state_path),state_mode=state_mode,
                top25_sha256=sha(destination/'top25.pt'),best_hits=best,
                ranking='Exact competition rank for saved top25 hits; absence means outside the saved set, not zero concept probability.',
                total_hits=len(hits))
            save(destination/'summary.json',result)
            results.append(result)
            brief = {k:v['rank'] for k,v in best.items() if k.startswith('jlens/') and k.endswith('/false')}
            print(entry['id'],json.dumps(brief),flush=True)
            save(out/'summary.json',dict(environment=environment,planned=len(protocol['rows']),completed=len(results),cases=results))
            del states, top_ids, top_values
    finally:
        model.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/professor-followups-20261016')
    args=parser.parse_args()
    replay(args.output.resolve())
