"""Causal fixed-token replay. This module never samples a completion."""
import inspect
import json
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .assay import PHYSICAL_LAYERS
from .common import LENS_ID, LENS_REVISION, MODEL_ID, MODEL_REVISION, sha


class Engine:
    def __init__(self, assets, *, device='cpu', dtype='float32'):
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.identity = json.loads((assets / 'manifest.json').read_text())
        if (self.identity['model'], self.identity['model_revision'], self.identity['lens'], self.identity['lens_revision']) != (
                MODEL_ID, MODEL_REVISION, LENS_ID, LENS_REVISION):
            raise ValueError('Only the pinned author Thinking model/lens is accepted')
        for entry in self.identity['files']:
            path = Path(entry['path'])
            if path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
                raise ValueError(f'Asset changed: {path}')
        self.tokenizer = AutoTokenizer.from_pretrained(assets / 'model', trust_remote_code=True, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(assets / 'model', trust_remote_code=True,
            local_files_only=True, dtype=self.dtype, attn_implementation='sdpa').eval().to(self.device)
        self.model.requires_grad_(False)
        # Accommodate the upstream keyword rename without editing pinned code.
        module = sys.modules[type(self.model).__module__]
        from transformers.masking_utils import create_causal_mask
        params = inspect.signature(create_causal_mask).parameters

        def compatible_mask(**kwargs):
            if 'input_embeds' in kwargs and 'input_embeds' not in params:
                kwargs['inputs_embeds'] = kwargs.pop('input_embeds')
            return create_causal_mask(**{k: v for k, v in kwargs.items() if k in params})

        module.create_causal_mask = compatible_mask
        self.matrices = {}
        for entry in self.identity['files']:
            if entry['repo'] != LENS_ID:
                continue
            name = Path(entry['path']).stem
            target = int(name.rsplit('_t', 1)[1]) if '_t' in name else 191
            bank = torch.load(entry['path'], map_location='cpu', weights_only=True, mmap=True)
            if bank['d_model'] != self.model.config.hidden_size or bank['n_prompts'] < 1:
                raise ValueError('Author bank has incompatible geometry')
            for layer in PHYSICAL_LAYERS:
                site = (target // 48) * 48 + layer
                matrix = bank['J'][site].float().to(self.device)
                if matrix.shape != (2048, 2048) or not torch.isfinite(matrix).all():
                    raise ValueError('Invalid author transport matrix')
                self.matrices[site] = matrix
        if len(self.matrices) != 68 or len(self.model.model.layers) != 48:
            raise ValueError('Unexpected author/model geometry')

    def set_depth(self, loops):
        if loops < 1:
            raise ValueError('Depth must be positive')
        self.model.config.total_ut_steps = loops
        self.model.model.config.total_ut_steps = loops
        self.model.model.total_ut_steps = loops
        self.loops = loops

    @torch.inference_mode()
    def unembed(self, states):
        return self.model.lm_head(self.model.model.norm(states.to(self.dtype))).float()

    @torch.inference_mode()
    def chunks(self, ids, positions, *, loops, chunk_size, include_targets=False):
        """Yield selected states and final logits; preserve separate KV per loop."""
        self.set_depth(loops)
        sites = [48 * loop + layer for loop in range(min(loops, 4)) for layer in PHYSICAL_LAYERS]
        if include_targets:
            sites += [48 * loop + 47 for loop in range(loops)]
        wanted = set(sites)
        positions = sorted(set(positions))
        if not positions or positions[0] < 0 or positions[-1] >= len(ids):
            raise ValueError('Invalid selected source positions')
        cache = None
        try:
            for offset in range(0, positions[-1] + 1, chunk_size):
                end = min(offset + chunk_size, positions[-1] + 1)
                selected = [p for p in positions if offset <= p < end]
                local = torch.tensor([p - offset for p in selected], dtype=torch.long, device=self.device)
                captured, handles = {}, []
                for physical in sorted({site % 48 for site in wanted}):
                    def hook(module, args, kwargs, output, physical=physical):
                        site = int(kwargs['current_ut']) * 48 + physical
                        if site in wanted and selected:
                            h = output[0] if isinstance(output, tuple) else output
                            captured[site] = h[0].index_select(0, local).detach()
                    handles.append(self.model.model.layers[physical].register_forward_hook(hook, with_kwargs=True))
                try:
                    output = self.model(input_ids=torch.tensor([ids[offset:end]], device=self.device),
                        past_key_values=cache, use_cache=True, exit_at_step=loops - 1,
                        logits_to_keep=local if selected else 1)
                finally:
                    for handle in handles:
                        handle.remove()
                cache = output.past_key_values
                if cache.get_seq_length() != end:
                    raise AssertionError('KV cache sequence length differs from the replay prefix')
                if selected:
                    if set(captured) != wanted:
                        raise AssertionError(f'Missing virtual hooks: {wanted - captured.keys()}')
                    if not torch.isfinite(output.logits).all():
                        raise FloatingPointError('Nonfinite replay logits')
                    yield selected, captured, output.logits[0].float()
                del output, captured
        finally:
            del cache

    @torch.inference_mode()
    def validate(self, ids):
        """Compare fixed sequences across chunk layouts, loops and final readout."""
        ids = ids[:16]
        positions = list(range(len(ids)))
        report = []
        prefixes = {}
        for loops in [1, 4, 8]:
            runs = []
            for chunk_size in [len(ids), max(1, len(ids) // 2)]:
                parts, logits = {}, []
                for _, states, scores in self.chunks(ids, positions, loops=loops, chunk_size=chunk_size, include_targets=True):
                    for site, h in states.items():
                        parts.setdefault(site, []).append(h.clone())
                    logits.append(scores.clone())
                runs.append(({s: torch.cat(v) for s, v in parts.items()}, torch.cat(logits)))
            full, chunked = runs
            tolerance = 2e-4 if self.dtype == torch.float32 else 0.35
            torch.testing.assert_close(full[1], chunked[1], rtol=0.01, atol=tolerance)
            state_error = max((full[0][s].float() - chunked[0][s].float()).abs().max().item() for s in full[0])
            for site in full[0]:
                torch.testing.assert_close(full[0][site], chunked[0][site], rtol=0.01, atol=tolerance)
            final_readout = self.unembed(full[0][loops * 48 - 1])
            torch.testing.assert_close(final_readout, full[1], rtol=0, atol=0)
            for site in [24, 40, 47]:
                if site in prefixes:
                    torch.testing.assert_close(full[0][site], prefixes[site], rtol=0, atol=0)
                else:
                    prefixes[site] = full[0][site].clone()
            site = 24
            transport = full[0][site].float() @ self.matrices[site].T
            reference = torch.einsum('ij,tj->ti', self.matrices[site], full[0][site].float())
            torch.testing.assert_close(transport, reference, rtol=1e-5, atol=1e-5)
            report.append(dict(loops=loops, tokens=len(ids), hooks=len(full[0]),
                chunk_logit_max_abs_error=(full[1] - chunked[1]).abs().max().item(),
                chunk_state_max_abs_error=state_error, final_readout_exact=True, loop1_prefix_exact=True,
                author_transport_orientation_passed=True))
        return dict(status='passed', checks=report, backend='Hugging Face only; vLLM equivalence not tested',
                    historical_replay_exact=False, dtype=str(self.dtype))
