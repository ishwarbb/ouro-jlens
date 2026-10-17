"""Loop-end Jacobian lenses for Ouro (arXiv:2609.01924, sections 2–3).

Uses the unchanged upstream Jacobian estimator. Targets and model geometry
are persisted here because the upstream portable lens format omits them.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
from pathlib import Path

import torch
from jlens import JacobianLens
from jlens.hooks import ActivationRecorder
from ouro_lens_config import DEFAULT_FAMILY


class TargetModel:
    """LensModel protocol view that stops computation at a chosen target."""
    def __init__(self, model, target):
        if not 0 <= target < model.n_layers:
            raise ValueError('Invalid target')
        self.model, self.target = model, target
        self.n_layers, self.d_model = model.n_layers, model.d_model
        self.layers = model.layers
        self.encode, self.unembed = model.encode, model.unembed

    def forward(self, input_ids):
        return self.model.forward(input_ids, stop_at=self.target)


def source_plan(physical_layers, recurrent_steps, *, scope='all', offsets=None):
    """Map each loop-end target to sources. Offsets are distances to target.

    ``all`` includes all upstream sites for cross-loop transport analysis;
    ``within-loop`` fits only sources read by the default own-loop routing.
    Explicit positive offsets select matched distances inside every loop.
    """
    if physical_layers < 2 or recurrent_steps < 1:
        raise ValueError('Need at least two physical layers and one loop')
    if scope not in ('all', 'within-loop'):
        raise ValueError('Unknown source scope')
    if offsets is not None:
        offsets = sorted(set(offsets))
        if not offsets or min(offsets) < 1 or max(offsets) >= physical_layers:
            raise ValueError('Offsets must be in [1, physical_layers - 1]')
    plan = {}
    for step in range(recurrent_steps):
        target = (step + 1) * physical_layers - 1
        start = 0 if scope == 'all' else step * physical_layers
        plan[target] = (sorted(target - offset for offset in offsets)
                        if offsets is not None else list(range(start, target)))
    return plan


class OuroLensFamily:
    def __init__(self, lenses, *, physical_layers, recurrent_steps, metadata=None):
        self.physical_layers = physical_layers
        self.recurrent_steps = recurrent_steps
        self.n_layers = physical_layers * recurrent_steps
        self.lenses = dict(sorted(lenses.items()))
        self.metadata = metadata or {}
        expected = {(i + 1) * physical_layers - 1 for i in range(recurrent_steps)}
        if set(self.lenses) != expected:
            raise ValueError(f'Need one lens at every loop end: {sorted(expected)}')
        self.d_model = next(iter(self.lenses.values())).d_model
        for target, lens in self.lenses.items():
            if lens.d_model != self.d_model or lens.n_prompts < 1 or not lens.source_layers:
                raise ValueError('Inconsistent or empty lens')
            for source, matrix in lens.jacobians.items():
                if not 0 <= source < target:
                    raise ValueError('Every fitted source must precede its target')
                if matrix.shape != (self.d_model, self.d_model) or not torch.isfinite(matrix).all():
                    raise ValueError('Invalid Jacobian matrix')

    def target_for(self, source):
        if not 0 <= source < self.n_layers:
            raise ValueError(f'Invalid source {source}')
        return (source // self.physical_layers + 1) * self.physical_layers - 1

    @property
    def source_layers(self):
        return [s for s in range(self.n_layers)
                if s in self.lenses[self.target_for(s)].jacobians]

    def _lens(self, source, target):
        own_target = self.target_for(source)
        target = own_target if target is None else target
        if target not in self.lenses or source > target:
            raise ValueError('Target must be a fitted downstream loop end')
        lens = self.lenses[target]
        if source != target and source not in lens.jacobians:
            raise ValueError(f'No fitted transport from {source} to {target}')
        return lens, target

    def transport(self, residual, source, *, target=None):
        lens, target = self._lens(source, target)
        if residual.shape[-1] != self.d_model:
            raise ValueError('Residual width does not match lens')
        residual = residual.float()
        if source == target:
            return residual
        return lens.transport(residual, source)

    def check_model(self, model):
        if (model.physical_layers, model.recurrent_steps, model.d_model) != (
                self.physical_layers, self.recurrent_steps, self.d_model):
            raise ValueError('Model geometry does not match lens family')

    @torch.no_grad()
    def apply(self, model, prompt, *, sources=None, positions=(-1,), max_seq_len=512,
              use_jacobian=True, target=None):
        """Return (source logits, actual loop-end logits, input IDs).

        Default readouts use each source's own loop end. An explicit target
        enables cross-loop comparisons when that transport was fitted.
        """
        self.check_model(model)
        sources = self.source_layers if sources is None else sorted(set(sources))
        for source in sources:
            self._lens(source, target)
        input_ids = model.encode(prompt, max_length=max_seq_len)
        with ActivationRecorder(model.layers, at=sorted(set(sources) | set(self.lenses))) as recorder:
            model.forward(input_ids)

        def select(source):
            h = recorder.activations[source][0]
            return (h if positions is None else h[list(positions)]).float()

        readouts = {}
        for source in sources:
            h = select(source)
            if use_jacobian:
                h = self.transport(h, source, target=target)
            readouts[source] = model.unembed(h).float().cpu()
        actual = {t: model.unembed(select(t)).float().cpu() for t in self.lenses}
        return readouts, actual, input_ids

    @torch.no_grad()
    def direction(self, model, source, token_id, *, target=None):
        """Normalized J.T @ u, where u is the token's LM-head row."""
        self.check_model(model)
        lens, target = self._lens(source, target)
        weight = model.hf_model.lm_head.weight
        if not 0 <= token_id < weight.shape[0]:
            raise ValueError('Invalid token ID')
        u = weight[token_id].detach().float().cpu()
        d = u if source == target else lens.jacobians[source].T @ u
        norm = d.norm()
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError('Token has a zero or nonfinite pullback')
        return d / norm

    def sites_at_offsets(self, offsets, *, first_loop=0):
        """Matched sites in every remaining loop (first_loop is zero based)."""
        if not 0 <= first_loop < self.recurrent_steps:
            raise ValueError('Invalid first loop')
        plan = source_plan(self.physical_layers, self.recurrent_steps, offsets=offsets)
        sites = [s for t, sources in plan.items() if t // self.physical_layers >= first_loop
                 for s in sources]
        for source in sites:
            self._lens(source, None)
        return sites

    @contextmanager
    def intervene(self, model, token_id, *, sources, positions=(-1,), mode='inject', strength=1.0):
        """Inject or ablate a token direction at selected virtual sites.

        Use sites_at_offsets(..., first_loop=...) to maintain an intervention
        across remaining loops. Each site uses its own fitted direction.
        Strength is an absolute residual-space coefficient for injection.
        Ablation removes the complete projection and ignores strength.
        """
        if mode not in ('inject', 'ablate') or not math.isfinite(strength):
            raise ValueError('Invalid intervention mode or strength')
        sources = sorted(set(sources))
        if not sources:
            raise ValueError('At least one intervention site is required')
        directions = {s: self.direction(model, s, token_id) for s in sources}
        handles = []

        def make_hook(direction):
            def hook(module, args, output):
                d = direction.to(output.device)
                result = output.clone()
                selector = slice(None) if positions is None else list(positions)
                h = output[:, selector, :].float()
                delta = strength * d if mode == 'inject' else -(h @ d).unsqueeze(-1) * d
                result[:, selector, :] = (h + delta).to(output.dtype)
                return result
            return hook

        try:
            for source, direction in directions.items():
                handles.append(model.layers[source].register_forward_hook(make_hook(direction)))
            yield
        finally:
            for handle in handles:
                handle.remove()

    def save(self, directory):
        """Save FP32 matrices without FP16 overflow, plus explicit targets."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for target, lens in self.lenses.items():
            path = directory / f'target-{target}.pt'
            temp = path.with_suffix('.pt.tmp')
            lens.save(str(temp), dtype=torch.float32)
            temp.replace(path)
        manifest = dict(format_version=1, physical_layers=self.physical_layers,
                        recurrent_steps=self.recurrent_steps, d_model=self.d_model,
                        targets=list(self.lenses), metadata=self.metadata)
        temp = directory / 'family.json.tmp'
        temp.write_text(json.dumps(manifest, indent=2) + '\n')
        temp.replace(directory / 'family.json')

    @classmethod
    def load(cls, directory=DEFAULT_FAMILY):
        directory = Path(directory)
        manifest = json.loads((directory / 'family.json').read_text())
        if manifest['format_version'] != 1:
            raise ValueError('Unsupported family format')
        family = cls({t: JacobianLens.load(str(directory / f'target-{t}.pt'))
                      for t in manifest['targets']},
                     physical_layers=manifest['physical_layers'],
                     recurrent_steps=manifest['recurrent_steps'], metadata=manifest['metadata'])
        if family.d_model != manifest['d_model']:
            raise ValueError('Manifest width disagrees with matrices')
        return family
