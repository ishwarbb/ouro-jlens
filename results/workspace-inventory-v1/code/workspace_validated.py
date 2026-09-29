"""Explicit prompt boundaries and separate token/sequence scores.

No text is silently truncated, and generated IDs are never decoded and
re-tokenized as model inputs. Complete-word and first-token scores stay separate.
"""
from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import json
import re

import torch
from jlens.hooks import ActivationRecorder
from workspace_interventions import Vocabulary, score_tokens


@dataclass(frozen=True)
class Prompt:
    text: str
    ids: tuple[int, ...]
    wrapper: str

    def record(self):
        encoded = json.dumps(list(self.ids), separators=(',', ':')).encode()
        return dict(text=self.text, seq_ids=list(self.ids), wrapper=self.wrapper,
                    text_sha256=hashlib.sha256(self.text.encode()).hexdigest(),
                    ids_sha256=hashlib.sha256(encoded).hexdigest())


def encode_prompt(tokenizer, text, *, wrapper='raw-v1', add_bos=True, limit=4096):
    ids = tokenizer.encode(text, add_special_tokens=False)
    bos = tokenizer.bos_token_id
    if add_bos and bos is not None and (not ids or ids[0] != bos):
        ids = [bos, *ids]
    if not ids or len(ids) > limit:
        raise ValueError(f'Invalid prompt length {len(ids)}; limit={limit}; no truncation')
    return Prompt(text, tuple(ids), wrapper)


def word_atom(text):
    """Declared Unicode policy: a whole alphabetic word, optional internal '-/’.

    Whitespace, numbers, punctuation, replacement characters and mixed strings
    are excluded. Author code's exact non-word filter is unavailable.
    """
    text = text.strip()
    parts = re.split("[-'’]", text)
    return bool(parts) and all(part.isalpha() for part in parts)


def first_word(text, *, ended=False):
    match = re.search(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", text, re.UNICODE)
    if not match:
        return None
    # A trailing partial token sequence is not a completed word.
    if match.end() == len(text) and not ended:
        return None
    return match.group().casefold()


class ValidatedIO:
    def __init__(self, model):
        self.model = model
        self.tokenizer = model.tokenizer
        self.vocab = Vocabulary(self.tokenizer)

    def prompt(self, text, **kwargs):
        return encode_prompt(self.tokenizer, text, add_bos=self.model.add_bos, **kwargs)

    def chat(self, user, prefill='', *, wrapper='chat-answer-v1'):
        text = self.tokenizer.apply_chat_template([dict(role='user', content=user)],
                                                 tokenize=False, add_generation_prompt=True) + prefill
        return self.prompt(text, wrapper=wrapper)

    def tensor(self, ids):
        return torch.tensor([list(ids)], dtype=torch.long,
                            device=self.model.decoder.embed_tokens.weight.device)

    @torch.no_grad()
    def forward(self, ids, *, sites=()):
        if not 0 < len(ids) <= 4096:
            raise ValueError('Invalid forward length; no truncation')
        final = self.model.n_layers - 1
        with ActivationRecorder(self.model.layers, at=sorted(set(sites) | {final})) as recorder:
            self.model.forward(self.tensor(ids))
        logits = self.model.unembed(recorder.activations[final][0, -1]).float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite output')
        return logits, recorder.activations

    def target(self, word):
        return dict(word=word, whole_token_ids=self.vocab.candidates(word),
                    direction_token_id=self.vocab.direction_token(word),
                    continuation_ids=self.tokenizer.encode(' ' + word, add_special_tokens=False))

    def score(self, logits, word):
        return score_tokens(logits, self.vocab.candidates(word))

    @torch.no_grad()
    def generate(self, ids, *, limit=128, stop_word=False, initial_logits=None):
        prefix = tuple(ids)
        emitted = []
        ended = False
        logits = initial_logits
        for _ in range(limit):
            if logits is None:
                logits, _ = self.forward(prefix + tuple(emitted))
            token = int(logits.argmax())
            emitted.append(token)
            ended = token in self.tokenizer.all_special_ids
            decoded = self.tokenizer.decode(emitted, skip_special_tokens=True)
            if ended or (stop_word and first_word(decoded, ended=ended) is not None):
                break
            logits = None
        decoded = self.tokenizer.decode(emitted, skip_special_tokens=True)
        return dict(token_ids=emitted, text=decoded, stopped_on_special=ended,
                    complete_first_word=first_word(decoded, ended=ended),
                    limit_reached=len(emitted) == limit and not ended)

    @torch.no_grad()
    def sequence_score(self, ids, word, *, initial_logits=None):
        """Log P(encode(' '+word) | exact prefix), not an answer success metric."""
        continuation = self.target(word)['continuation_ids']
        logits = initial_logits
        values = []
        for i, token in enumerate(continuation):
            if logits is None:
                logits, _ = self.forward(tuple(ids) + tuple(continuation[:i]))
            values.append(float(logits.log_softmax(-1)[token]))
            logits = None
        return dict(token_ids=continuation, token_logprobs=values, log_probability=sum(values))


@contextmanager
def prefix_transfer(model, bank, source, destination, sites, length, *, random_key=None):
    """Transfer (or norm-matched random delta) at frozen original prompt positions."""
    prepared = {s:(bank.get(s, source), bank.get(s, destination),
                   bank.random(s, random_key) if random_key is not None else None) for s in sites}
    handles = []
    def make_hook(values):
        a, b, random = values
        def hook(module, args, output):
            if output.shape[1] < length:
                raise ValueError('Transfer prefix is longer than model input')
            h = output[:, :length].float()
            delta = (h @ a).unsqueeze(-1) * (b - a)
            if random is not None:
                delta = delta.norm(dim=-1, keepdim=True) * random
            result = output.clone()
            result[:, :length] = (h + delta).to(output.dtype)
            return result
        return hook
    try:
        for site, values in prepared.items():
            handles.append(model.layers[site].register_forward_hook(make_hook(values)))
        yield
    finally:
        for handle in handles:
            handle.remove()
