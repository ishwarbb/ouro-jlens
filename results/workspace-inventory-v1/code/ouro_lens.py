"""J-lens adapter with one hookable residual tap per Ouro layer invocation.

Site r * L + l is the output of physical block l on pass r, BEFORE the
pass-ending norm. The target is the last block on the last pass. Unembedding
applies the final norm once. The original blocks, recurrent computation and
weights are unchanged; identity taps expose their individual invocations.
"""
from __future__ import annotations
import torch
from torch import nn


class _TargetReached(Exception):
    """Internal early return after the target's hooks have fired."""
    def __init__(self, residual):
        self.residual = residual


class OuroLensModel:
    def __init__(self, hf_model, tokenizer, *, add_bos=True):
        self.hf_model = hf_model.eval()
        self.tokenizer = tokenizer
        self.add_bos = add_bos
        self.decoder = hf_model.model
        self.physical_layers = len(self.decoder.layers)
        self.recurrent_steps = self.decoder.total_ut_steps
        if self.recurrent_steps != hf_model.config.total_ut_steps:
            raise ValueError('Decoder and configuration disagree about recurrent passes')
        self.n_layers = self.physical_layers * self.recurrent_steps
        self.d_model = hf_model.config.hidden_size
        self.layers = nn.ModuleList([nn.Identity() for _ in range(self.n_layers)])
        for parameter in hf_model.parameters():
            parameter.requires_grad_(False)
        hf_model.gradient_checkpointing_disable()
        self._active = False
        self._closed = False
        self._handles = []
        try:
            for index, block in enumerate(self.decoder.layers):
                self._handles.append(block.register_forward_hook(self._hook(index)))
        except Exception:
            self.close()
            raise

    def _hook(self, physical_index):
        def hook(module, args, output):
            if not self._active:
                return None
            recurrent_pass = self._counts[physical_index]
            if recurrent_pass >= self.recurrent_steps:
                raise RuntimeError('Unexpected extra invocation of an Ouro block')
            site = recurrent_pass * self.physical_layers + physical_index
            self._counts[physical_index] += 1
            self.last_site_order.append(site)
            if torch.is_tensor(output):
                result = self.layers[site](output)
                residual = result
            elif isinstance(output, tuple) and torch.is_tensor(output[0]):
                residual = self.layers[site](output[0])
                result = (residual, *output[1:])
            else:
                raise TypeError('Unsupported Ouro residual-block output')
            if site == self._stop_at:
                raise _TargetReached(residual)
            return result
        return hook

    def site_label(self, index):
        if not 0 <= index < self.n_layers:
            raise IndexError(index)
        return dict(site=index, recurrent_pass=index // self.physical_layers + 1,
                    physical_layer=index % self.physical_layers + 1,
                    location='block output before pass-ending normalization')

    def encode(self, text, *, max_length=512):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        bos = self.tokenizer.bos_token_id
        if self.add_bos and bos is not None and (not ids or ids[0] != bos):
            ids = [bos, *ids]
        if not ids or max_length < 1:
            raise ValueError('A nonempty token sequence is required')
        return torch.tensor([ids[:max_length]], dtype=torch.long,
                            device=self.decoder.embed_tokens.weight.device)

    def forward(self, input_ids, *, stop_at=None):
        """Run all loops, or return the raw residual at an explicit target.

        Stopping saves the downstream graph during intermediate-target fits.
        The prefix still includes every preceding loop and its normalization.
        """
        if self._closed or self._active:
            raise RuntimeError('Adapter is closed or already running a forward')
        if self.hf_model.training or self.hf_model.is_gradient_checkpointing:
            raise RuntimeError('Lens forwards require eval mode and checkpointing disabled')
        if stop_at is not None and not 0 <= stop_at < self.n_layers:
            raise ValueError('stop_at must be a valid virtual site')
        self._counts = [0] * self.physical_layers
        self.last_site_order = []
        self._stop_at = stop_at
        self._active = True
        try:
            output = self.decoder(input_ids=input_ids, use_cache=False)
        except _TargetReached as reached:
            output = reached.residual
        finally:
            self._active = False
            self._stop_at = None
        expected = self.n_layers if stop_at is None else stop_at + 1
        if self.last_site_order != list(range(expected)):
            raise RuntimeError('Unexpected recurrent layer execution order')
        return output

    def unembed(self, residual):
        weight = self.hf_model.lm_head.weight
        residual = residual.to(device=weight.device, dtype=weight.dtype)
        return self.hf_model.lm_head(self.decoder.norm(residual))

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._closed = True
