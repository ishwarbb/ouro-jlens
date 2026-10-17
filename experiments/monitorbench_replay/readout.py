"""Stream author-lens and raw-lens ranks into compact, explicit assays."""
from collections import Counter

import torch

from .assay import THRESHOLDS, source_positions


def windows_for(position, encoded, window):
    prompt_len = len(encoded['prompt_ids'])
    result = []
    if position in source_positions(prompt_len, len(encoded['response_ids']), 'early'):
        result.append('early')
    if window == 'full':
        result.append('full')
        if position >= prompt_len:
            end = encoded['offsets'][position - prompt_len][1]
            close = encoded['reasoning_close']
            if close >= 0:
                if end <= close:
                    result.append('reasoning')
                elif end > close + len('</think>'):
                    result.append('answer')
            else:
                result.append('reasoning_unclosed')
    return result


@torch.inference_mode()
def measure(engine, encoded, variants, *, loops, chunk_size, window):
    ids = encoded['prompt_ids'] + encoded['response_ids']
    prompt_len = len(encoded['prompt_ids'])
    positions = source_positions(prompt_len, len(encoded['response_ids']), window)
    token_to_word = {token: word for word, tokens in variants.items() for token in tokens}
    token_ids = torch.tensor(sorted(token_to_word), device=engine.device)
    metrics, evidence, seen_evidence = {}, [], Counter()
    measured_positions = 0
    for selected, states, _ in engine.chunks(ids, positions, loops=loops, chunk_size=chunk_size):
        measured_positions += len(selected)
        for site, h in sorted(states.items()):
            loop = site // 48 + 1
            for method in ['jlens', 'raw']:
                transported = h.float() @ engine.matrices[site].T if method == 'jlens' else h
                # Bound transient logits memory independently of token chunk size.
                for start in range(0, len(selected), 32):
                    logits = engine.unembed(transported[start:start + 32])
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError('Nonfinite lens readout')
                    top_values, top_ids = logits.topk(10, dim=-1)
                    target_logits = logits.index_select(1, token_ids)
                    candidate_rows = (target_logits >= top_values[:, -1:]).nonzero().tolist()
                    hits_by_row = {}
                    for row, target_index in candidate_rows:
                        token = int(token_ids[target_index])
                        rank = int((logits[row] > target_logits[row, target_index]).sum()) + 1
                        word = token_to_word[token]
                        old = hits_by_row.setdefault(row, {}).get(word)
                        if old is None or rank < old[0]:
                            hits_by_row[row][word] = (rank, token)
                    for row, position in enumerate(selected[start:start + 32]):
                        windows = windows_for(position, encoded, window)
                        hits = hits_by_row.get(row, {})
                        for assay_window in windows:
                            for loop_scope in ['all', str(loop)]:
                                key = f'{method}/{assay_window}/loop{loop_scope}'
                                metric = metrics.setdefault(key, dict(cells=0, thresholds={str(k): dict(any_word_cells=0, word_cells={}) for k in THRESHOLDS}))
                                metric['cells'] += 1
                                for threshold in THRESHOLDS:
                                    present = [word for word, (rank, _) in hits.items() if rank <= threshold]
                                    level = metric['thresholds'][str(threshold)]
                                    level['any_word_cells'] += bool(present)
                                    for word in present:
                                        level['word_cells'][word] = level['word_cells'].get(word, 0) + 1
                        for word, (rank, token) in hits.items():
                            evidence_key = (method, loop, word)
                            if seen_evidence[evidence_key] >= 3:
                                continue
                            seen_evidence[evidence_key] += 1
                            response_index = position - prompt_len
                            visible_end = encoded['offsets'][response_index][1] if response_index >= 0 else 0
                            earlier_mentions = [p for p in encoded['response_mentions'][word] if p < visible_end]
                            evidence.append(dict(method=method, word=word, rank=rank, token_id=token,
                                virtual_site=site, loop=loop, layer=site % 48, source_position=position,
                                response_source_index=response_index, windows=windows,
                                source_token=engine.tokenizer.decode([ids[position]]),
                                next_token=engine.tokenizer.decode([ids[position + 1]]),
                                top10=[engine.tokenizer.decode([t]) for t in top_ids[row].tolist()],
                                absent_from_prompt=not bool(encoded['prompt_mentions'][word]),
                                absent_from_full_response=not bool(encoded['response_mentions'][word]),
                                precedes_literal_response_mention=not bool(earlier_mentions)))
    expected_cells = len(positions) * 17 * min(loops, 4)
    primary_key = f'jlens/{"early" if window == "early" else "full"}/loopall'
    if metrics[primary_key]['cells'] != expected_cells or measured_positions != len(positions):
        raise AssertionError('Incomplete readout coverage')
    return dict(metrics=metrics, evidence=evidence, measured_source_positions=len(positions),
                executed_loops=loops, measured_loops=list(range(1, min(loops, 4) + 1)),
                unmeasured_loops=list(range(5, loops + 1)), scope=window,
                evidence_policy='First three hits per method/loop/word; aggregate counts include every measured cell')
