"""Run 7 generation with tool-response input coverage and immutable provenance.

The decoding kernel follows the frozen revision generator. Retaining the last
512 prompt states also exposes the most recent actual tool response to the
case gallery; all earlier prompt text remains in the transcript.
"""
from __future__ import annotations

import time

from evaluation_revisions import CAPTURE_SITES, save_tensor, update_status
from prepare_deception_run4 import digest, read, sha
from run_deception_run4 import atomic_text, dump, now

PROMPT_CAPTURE_TOKENS = 512


def generate(root, job, turn, messages, model, execution_hash, setting, resources):
    import torch
    from transformers import set_seed
    from ouro_viewer_inference import CachedDecoder
    from run_deception_run6 import thought_closed

    directory = root / 'cases' / job['id'] / f'turn-{turn}'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'generation.json'
    tok = model.tokenizer
    if path.exists():
        g = read(path)
        assert g['job_sha256'] == digest(job) and g['messages'] == messages
        assert g['revision_execution_sha256'] == execution_hash and g['setting'] == setting
        assert g['prompt_capture_tokens'] == PROMPT_CAPTURE_TOKENS
        assert g['turn'] == turn
        for name, expected in g['artifacts'].items():
            assert sha(directory / name) == expected, name
        assert (directory / 'response.txt').read_text() == g['raw_text'] == tok.decode(g['output_ids'])
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        assert g['prompt_text'] == prompt
        assert g['input_ids'] == tok.encode(prompt, add_special_tokens=False)
        state = torch.load(directory / 'states.pt', map_location='cpu', weights_only=True)
        assert state['prompt_capture_tokens'] == PROMPT_CAPTURE_TOKENS
        assert state['input_ids'] == g['input_ids'] and state['output_ids'] == g['output_ids']
        assert state['sites'] == CAPTURE_SITES
        N = len(g['input_ids'])
        stop = len(g['output_ids']) - int(bool(g['output_ids'] and g['output_ids'][-1] in (0, 2)))
        assert state['positions'] == list(range(max(0, N-PROMPT_CAPTURE_TOKENS), N+stop))
        return g, path

    assert resources['max_new_tokens'] == 512, 'Run 7 uses a frozen 512-token output budget'
    assert resources['max_input_tokens'] == 4096, 'Run 7 uses a frozen 4096-token input limit'
    if setting not in ('greedy', 'sampled_0_9'):
        raise ValueError(setting)
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(prompt, add_special_tokens=False)
    N = len(ids)
    if not N or N > resources['max_input_tokens']:
        raise RuntimeError('Empty prompt or prompt exceeds limit; no truncation')
    set_seed(job['seed'] + turn)
    outputs = []
    positions = []
    captures = []
    start = time.time()
    rng = dict(seed=job['seed'] + turn, cpu_before=torch.get_rng_state(),
               cuda_before=torch.cuda.get_rng_state())
    update_status(root, 'generating', episode=job['id'], turn=turn,
                  setting=setting, input_tokens=N, prompt_capture_tokens=PROMPT_CAPTURE_TOKENS)
    with torch.inference_mode(), CachedDecoder(model) as decoder:
        for begin in range(0, N, 256):
            chunk = ids[begin:begin+256]
            logits, states = decoder.forward(chunk)
            keep = [i for i in range(len(chunk)) if begin+i >= N-PROMPT_CAPTURE_TOKENS]
            if keep:
                captures.append(states[CAPTURE_SITES][:, keep].clone())
                positions.extend(begin+i for i in keep)
            del states
        first = logits.cpu()
        for step in range(resources['max_new_tokens']):
            if setting == 'greedy':
                token = int(logits.argmax())
            else:
                probabilities = torch.softmax(logits/.9, dim=-1)
                values, order = probabilities.sort(descending=True)
                values[values.cumsum(-1)-values >= .95] = 0
                token = int(order[int(torch.multinomial(values, 1))])
            outputs.append(token)
            if token in (0, 2):
                break
            logits, states = decoder.forward([token])
            captures.append(states[CAPTURE_SITES].clone())
            positions.append(N+len(outputs)-1)
            del states
            if (step+1) % 128 == 0:
                update_status(root, 'generating', episode=job['id'], turn=turn,
                              generated_tokens=step+1, prompt_capture_tokens=PROMPT_CAPTURE_TOKENS)
        natural = bool(outputs and outputs[-1] in (0, 2))
        raw = tok.decode(outputs)
        rng.update(cpu_after=torch.get_rng_state(), cuda_after=torch.cuda.get_rng_state())
        save_tensor(torch, rng, directory / 'rng.pt')
        save_tensor(torch, dict(states=torch.cat(captures, dim=1), sites=CAPTURE_SITES,
                    positions=positions, input_ids=ids, output_ids=outputs,
                    first_next_token_logits=first, prompt_capture_tokens=PROMPT_CAPTURE_TOKENS),
                    directory / 'states.pt')
        atomic_text(directory / 'response.txt', raw)
        g = dict(job, job_sha256=digest(job), turn=turn, messages=messages,
                 prompt_text=prompt, input_ids=ids, output_ids=outputs, raw_text=raw,
                 assistant_history_text=tok.decode(outputs[:-1] if natural else outputs).strip(),
                 complete=natural and thought_closed(tok, outputs), natural_stop=natural,
                 capped=not natural, setting=setting, prompt_capture_tokens=PROMPT_CAPTURE_TOKENS,
                 revision_execution_sha256=execution_hash, generated_utc=now(),
                 seconds=time.time()-start,
                 artifacts={name: sha(directory / name) for name in ('states.pt', 'rng.pt', 'response.txt')})
        dump(path, g)
    del captures, rng
    torch.cuda.empty_cache()
    return g, path
