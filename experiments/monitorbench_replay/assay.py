"""Frozen words and windows. No tensor or model imports."""
import re

from .inventory import WORDS

PHYSICAL_LAYERS = list(range(24, 41))
THRESHOLDS = [1, 5, 10]


def source_positions(prompt_length, response_length, window='early'):
    if prompt_length < 1 or response_length < 1:
        raise ValueError('Replay requires nonempty prompt and completion token sequences')
    end = prompt_length + response_length - 1
    if window == 'early':
        end = min(end, prompt_length + 20)
    elif window != 'full':
        raise ValueError('Unknown source window')
    return list(range(prompt_length - 1, end))


def token_variants(tokenizer):
    variants = {word: [] for word in WORDS}
    specials = set(tokenizer.all_special_ids)
    for index in range(len(tokenizer)):
        if index not in specials:
            word = tokenizer.decode([index], clean_up_tokenization_spaces=False).strip().casefold()
            if word in variants:
                variants[word].append(index)
    return variants


def mentions(text):
    return {word: [m.start() for m in re.finditer(r'\b' + re.escape(word) + r'\b', text, re.I)] for word in WORDS}


def tokenize(tokenizer, messages, response):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    # vLLM receives a prompt string and encodes it with the tokenizer defaults.
    prompt_ids = tokenizer(prompt, add_special_tokens=True)['input_ids']
    encoded = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
    response_ids = encoded['input_ids']
    roundtrip = tokenizer.decode(response_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == response
    return dict(prompt=prompt, prompt_ids=prompt_ids, response_ids=response_ids,
                offsets=encoded['offset_mapping'], roundtrip=roundtrip,
                replacement_character='\ufffd' in response,
                reasoning_close=response.find('</think>'), prompt_mentions=mentions(prompt), response_mentions=mentions(response))
