"""Run 7: complete captured-token readouts for the Section 5.3 case gallery.

Every saved capture is scored at every registered site. Fixed-window summaries
are diagnostics; they never determine which transcripts enter the gallery.
"""
from __future__ import annotations

from pathlib import Path
import re

from evaluation_revisions import RevisionLens, save_tensor
from prepare_deception_run4 import read, sha
from prepare_deception_run6 import SITES, WORDS
from run_deception_run4 import dump


def lexical_absence(g, tok, positions, words):
    """Whether a literal tracked word is absent from the consumed token prefix."""
    count = len(g['input_ids'])
    patterns = {w: re.compile(r'(?<!\w)' + re.escape(w) + r'(?!\w)', re.I)
                for w in words}
    result = {}
    for position in positions:
        ids = (g['input_ids'][:position + 1] if position < count else
               g['input_ids'] + g['output_ids'][:position - count + 1])
        prefix = tok.decode(ids)
        result[position] = {word: not bool(pattern.search(prefix))
                            for word, pattern in patterns.items()}
    return result


def capture_limit(g, state=None):
    """Resolve bounded prompt coverage and check declared metadata consistency."""
    cap = g.get('prompt_capture_tokens', 32)
    assert type(cap) is int and 1 <= cap <= 512, 'Invalid prompt capture limit'
    if state is not None and 'prompt_capture_tokens' in state:
        saved = state['prompt_capture_tokens']
        assert type(saved) is int and 1 <= saved <= 512, 'Invalid saved prompt capture limit'
        assert saved == cap, 'Prompt capture metadata differs'
    return cap


def validate_capture(g, state, tok, d_model=None):
    """Reject partial or shifted capture inventories before any readout is made."""
    import torch
    assert state['input_ids'] == g['input_ids'], 'Capture input IDs differ'
    assert state['output_ids'] == g['output_ids'], 'Capture output IDs differ'
    assert tok.decode(g['output_ids']) == g['raw_text'], 'Output decoding differs'
    count = len(g['input_ids'])
    assert count > 0, 'Empty prompt'
    outputs = g['output_ids']
    stop = len(outputs) - int(bool(outputs and outputs[-1] in (0, 2)))
    assert all(t not in (0, 2) for t in outputs[:stop]), 'EOS inside output'
    cap = capture_limit(g, state)
    expected = list(range(max(0, count - cap), count + stop))
    assert state['positions'] == expected, 'Incomplete or shifted capture positions'
    assert len(state['sites']) == len(set(state['sites'])), 'Duplicate capture site'
    assert all(s in state['sites'] for s in SITES), 'Missing registered lens site'
    h = state['states']
    assert h.ndim == 3 and tuple(h.shape[:2]) == (len(state['sites']), len(expected)), 'Capture shape differs'
    if d_model is not None:
        assert h.shape[-1] == d_model, 'Capture width differs'
    assert torch.isfinite(h).all(), 'Nonfinite captured state'
    return expected


def summarize(readout, sites, positions, words, support, absence, windows, g):
    """Equal-site summaries across the four frozen loops and fixed windows."""
    import torch
    top = readout['top_ids']
    ranks = readout['concept_ranks']
    values = readout['concept_values']
    hits = {word: torch.isin(top[:, :, :10], torch.tensor(support[word], dtype=torch.int32)).any(-1)
            for word in words}
    union = torch.zeros(top.shape[:2], dtype=torch.bool)
    absent_hits = torch.zeros_like(union)
    for word in WORDS:
        union |= hits[word]
        absent_hits |= hits[word] & torch.tensor([absence[p][word] for p in positions])[None, :]
    p_lookup = {p: i for i, p in enumerate(positions)}
    result = {}
    for name, window in windows.items():
        pi = [p_lookup[p] for p in window]
        loops = []
        for loop in range(4):
            rows = [i for i, site in enumerate(sites) if site // 48 == loop]
            cells = len(rows) * len(pi)
            def extract(t):
                return t[rows][:, pi]
            item = dict(loop=loop, cells=cells,
                raw_concept_rate=float(extract(union).float().mean()) if cells else None,
                prefix_absent_concept_rate=float(extract(absent_hits).float().mean()) if cells else None,
                per_word={w: float(extract(hits[w]).float().mean()) if cells else None for w in WORDS})
            if 'target' in g and 'foil' in g:
                ti, fi = words.index(g['target']), words.index(g['foil'])
                rt, rf = extract(ranks[:, :, ti]), extract(ranks[:, :, fi])
                margin = extract(values[:, :, ti]) - extract(values[:, :, fi])
                allowed = torch.tensor([absence[p][g['target']] for p in window])[None, :].expand(len(rows), -1)
                target_hits = extract(hits[g['target']])
                item.update(target_rank=float(rt.float().mean()) if cells else None,
                    foil_rank=float(rf.float().mean()) if cells else None,
                    target_beats_foil=float((rt < rf).float().mean()) if cells else None,
                    target_foil_tie=float((rt == rf).float().mean()) if cells else None,
                    target_foil_margin=float(margin.mean()) if cells else None,
                    target_top10=float(target_hits.float().mean()) if cells else None,
                    target_prefix_absent_cells=int(allowed.sum()),
                    target_top10_when_prefix_absent=float(target_hits[allowed].float().mean()) if allowed.any() else None)
            loops.append(item)
        result[name] = dict(loops=loops, cells=sum(item['cells'] for item in loops))
    return result


def _same_saved(actual, expected):
    """Check an orphaned atomic tensor artifact without overwriting it."""
    import torch
    if isinstance(expected, torch.Tensor):
        return isinstance(actual, torch.Tensor) and actual.dtype == expected.dtype and torch.equal(actual, expected)
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(_same_saved(actual[k], v) for k,v in expected.items())
    if isinstance(expected, (list, tuple)):
        return type(actual) is type(expected) and len(actual) == len(expected) and all(_same_saved(a, b) for a,b in zip(actual, expected))
    return actual == expected


class Run7Lens(RevisionLens):
    def __init__(self, model, vocabulary):
        super().__init__(model, vocabulary.get('word_tokens', vocabulary))

    def score(self, path, out, execution_hash, fields=None):
        import torch
        from run7_tasks import boundary
        path, out = Path(path), Path(out)
        out.mkdir(parents=True, exist_ok=True)
        g = read(path)
        assert g['revision_execution_sha256'] == execution_hash, 'Generation execution hash differs'
        cap = capture_limit(g)
        fields = tuple(fields or (('answer',) if g.get('phase') == 'measurement' else ('choice', 'result')))
        metric = out / 'metrics.json'
        if metric.exists():
            m = read(metric)
            assert m['generation_sha256'] == sha(path), 'Generation hash changed'
            assert m['revision_execution_sha256'] == execution_hash, 'Execution hash changed'
            assert m['fields'] == list(fields), 'Boundary field selection changed'
            assert m.get('prompt_capture_tokens', 32) == cap, 'Readout prompt coverage differs'
            assert m['readouts_sha256'] == sha(out / 'readouts.pt'), 'Readout hash changed'
            for name, expected in g['artifacts'].items():
                assert sha(path.parent / name) == expected, name
            return m
        for name, expected in g['artifacts'].items():
            assert sha(path.parent / name) == expected, name
        state = torch.load(path.parent / 'states.pt', map_location='cpu', weights_only=True)
        positions = validate_capture(g, state, self.tok, self.model.d_model)
        windows, anchor = boundary(g, self.tok, fields)
        assert all(p in positions for window in windows.values() for p in window), 'Uncaptured fixed-window position'
        site_lookup = {site: i for i, site in enumerate(state['sites'])}
        words = sorted(set(WORDS) | ({g['target'], g['foil']} if 'target' in g and 'foil' in g else set()))
        for word in words:
            assert word in self.support and self.support[word], f'Unsupported word: {word}'
        absence = lexical_absence(g, self.tok, positions, words)
        readouts, summaries = {}, {}
        with torch.inference_mode():
            for kind in ('jacobian', 'raw_logit'):
                shape = (len(SITES), len(positions))
                top_ids = torch.empty((*shape, 25), dtype=torch.int32)
                top_values = torch.empty_like(top_ids, dtype=torch.float32)
                ranks = torch.empty((*shape, len(words)), dtype=torch.int32)
                ties = torch.empty_like(ranks)
                concept_values = torch.empty_like(ranks, dtype=torch.float32)
                for si, site in enumerate(SITES):
                    source = state['states'][site_lookup[site]]
                    matrix = self.matrices[site]
                    for start in range(0, len(positions), 32):
                        h = source[start:start + 32].to(matrix.device).float()
                        # Captures are raw post-block residuals. unembed performs
                        # the model's final normalization exactly once.
                        transported = h @ matrix.T if kind == 'jacobian' else h
                        scores = self.model.unembed(transported).float()
                        assert torch.isfinite(scores).all(), 'Nonfinite readout logits'
                        end = start + len(h)
                        values, ids = scores.topk(25, -1)
                        top_ids[si, start:end] = ids.cpu().int()
                        top_values[si, start:end] = values.cpu()
                        for wi, word in enumerate(words):
                            best = scores[:, self.support[word]].max(-1).values
                            concept_values[si, start:end, wi] = best.cpu()
                            ranks[si, start:end, wi] = ((scores > best[:, None]).sum(-1) + 1).cpu().int()
                            ties[si, start:end, wi] = (scores == best[:, None]).sum(-1).cpu().int()
                        del h, transported, scores, values, ids
                readout = dict(top_ids=top_ids, top_values=top_values,
                    concept_ranks=ranks, concept_ties=ties, concept_values=concept_values)
                readouts[kind] = readout
                summaries[kind] = summarize(readout, SITES, positions, words,
                    self.support, absence, windows, g)
        count = len(g['input_ids'])
        data = dict(format_version=1, sites=list(SITES), positions=positions, words=words,
            windows=windows, boundary=anchor, fields=list(fields), readouts=readouts,
            prefix_absence=torch.tensor([[absence[p][word] for word in words] for p in positions], dtype=torch.bool),
            input_ids=g['input_ids'], output_ids=g['output_ids'],
            generation_sha256=sha(path), states_sha256=g['artifacts']['states.pt'],
            revision_execution_sha256=execution_hash,
            site_mapping=[dict(site=s, loop=s // 48, physical_layer=s % 48, target=(s // 48 + 1) * 48 - 1) for s in SITES],
            position_mapping=[dict(position=p, section='prompt' if p < count else 'generated',
                token_id=(g['input_ids'][p] if p < count else g['output_ids'][p - count])) for p in positions],
            prompt_capture_tokens=capture_limit(g, state),
            uncaptured_prompt=dict(begin=0, end_exclusive=max(0, count - capture_limit(g, state))),
            normalization='Raw captured residuals; own-loop h @ J.T; model.unembed normalizes once.')
        readout_path = out / 'readouts.pt'
        if readout_path.exists():
            existing = torch.load(readout_path, map_location='cpu', weights_only=True)
            assert _same_saved(existing, data), 'Existing readouts differ; immutable artifact was retained'
        else:
            save_tensor(torch, data, readout_path)
        m = dict(format_version=1, generation_sha256=sha(path), states_sha256=g['artifacts']['states.pt'],
            revision_execution_sha256=execution_hash, readouts_sha256=sha(out / 'readouts.pt'),
            fields=list(fields), boundary=anchor, windows=windows, prefix_absence={str(p): row for p,row in absence.items()},
            captured_positions=len(positions), registered_sites=len(SITES), uncaptured_prompt=data['uncaptured_prompt'],
            prompt_capture_tokens=data['prompt_capture_tokens'],
            tracked_words=words, summaries=summaries,
            claim_limit='Lexical readouts and mechanical facts are diagnostics, not evidence of intent by themselves.')
        dump(metric, m)
        return m
