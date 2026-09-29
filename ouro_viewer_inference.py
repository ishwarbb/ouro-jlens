"""Cache-aware inference for the viewer, without changing the frozen fitter.

Explicit DynamicCache avoids the checkpoint's legacy cache mask bug. The
adapter's identity taps still expose every recurrent invocation. Captures are
copied to host memory once per forward; GPU activation retention is bounded by
the prefill chunk size, independently of the requested output length.
"""
from contextlib import contextmanager

import torch
from transformers import DynamicCache


class CachedDecoder:
    def __init__(self, model):
        self.model = model
        self.cache = DynamicCache()
        self.position = 0
        self.captured = [None] * model.n_layers
        self.handles = []

    def __enter__(self):
        for site, layer in enumerate(self.model.layers):
            def record(module, args, output, site=site):
                self.captured[site] = output.detach()
            self.handles.append(layer.register_forward_hook(record))
        return self

    @torch.inference_mode()
    def forward(self, ids):
        model = self.model
        if model._active or model._closed or not ids:
            raise ValueError('Invalid cached forward')
        model._counts = [0] * model.physical_layers
        model.last_site_order = []
        model._stop_at = None
        model._active = True
        self.captured = [None] * model.n_layers
        device = model.decoder.embed_tokens.weight.device
        try:
            result = model.hf_model(
                input_ids=torch.tensor([ids], dtype=torch.long, device=device),
                past_key_values=self.cache, use_cache=True, logits_to_keep=1,
                exit_at_step=model.recurrent_steps - 1,
            )
        finally:
            model._active = False
        if model.last_site_order != list(range(model.n_layers)):
            raise RuntimeError('Cached forward missed recurrent sites')
        if any(value is None for value in self.captured):
            raise RuntimeError('Cached activation capture is incomplete')
        logits = result.logits[0, -1].float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite cached output')
        states = torch.stack([value[0] for value in self.captured]).cpu()
        self.captured = [None] * model.n_layers
        self.position += len(ids)
        if len(self.cache.layers) != model.n_layers or self.cache.get_seq_length() != self.position:
            raise RuntimeError('Recurrent cache geometry is incorrect')
        return logits, states

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.captured = []
        self.cache = None


@contextmanager
def prefix_edit(model, bank, source, destination, sites, prefix_length, position):
    """Edit original prompt positions once as each chunk enters the KV cache."""
    directions = {s: (bank.get(s, source), bank.get(s, destination) if destination is not None else None)
                  for s in sites}
    handles = []
    for site, (a, b) in directions.items():
        def edit(module, args, output, a=a, b=b):
            count = min(output.shape[1], max(0, prefix_length - position()))
            if count == 0:
                return None
            h = output[:, :count].float()
            projection = (h @ a).unsqueeze(-1)
            edited = h - projection * a if b is None else h + projection * (b - a)
            result = output.clone()
            result[:, :count] = edited.to(output.dtype)
            return result
        handles.append(model.layers[site].register_forward_hook(edit))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
