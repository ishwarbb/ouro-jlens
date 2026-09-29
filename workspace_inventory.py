"""Appendix-E positive pursuit and frozen clean-state subspace removal."""
from contextlib import contextmanager
import hashlib
import math

import torch


def dictionary(weight, jacobian, token_ids, *, chunk=4096):
    """Rows are normalized J.T u; no RMSNorm or softmax in atom selection."""
    rows = []
    for ids in token_ids.split(chunk):
        atoms = weight[ids].float() @ jacobian.float()
        norms = atoms.norm(dim=-1, keepdim=True)
        if not torch.isfinite(norms).all() or (norms <= 0).any():
            raise ValueError('Invalid dictionary atom')
        rows.append(atoms / norms)
    return torch.cat(rows)


def nnls_gram(gram, rhs, initial=None, *, tolerance=2e-6, max_steps=4000):
    """Accelerated projected gradient, stopped by projected KKT residual.

    Inputs use unit state norms so tolerance has a comparable meaning per cell.
    Return convergence diagnostics; callers must not silently accept failures.
    """
    x = torch.zeros_like(rhs) if initial is None else initial.clone()
    y = x.clone()
    lipschitz = gram.abs().sum(-1).max(-1).values.clamp_min(1e-12).unsqueeze(-1)
    t = 1.
    for step in range(max_steps):
        gradient = torch.bmm(gram, y.unsqueeze(-1)).squeeze(-1) - rhs
        nxt = (y - gradient / lipschitz).clamp_min(0)
        tn = (1 + math.sqrt(1 + 4 * t * t)) / 2
        y = nxt + ((t - 1) / tn) * (nxt - x)
        x, t = nxt, tn
        if (step + 1) % 25 == 0:
            gradient = torch.bmm(gram, x.unsqueeze(-1)).squeeze(-1) - rhs
            error = (x - (x - gradient).clamp_min(0)).abs().max(-1).values
            if float(error.max()) <= tolerance:
                return x, dict(steps=step + 1, max_kkt=float(error.max()), converged=True)
    return x, dict(steps=max_steps, max_kkt=float(error.max()), converged=False)


@torch.no_grad()
def pursuit(states, atoms, token_ids, *, k=50, snapshots=(8, 16, 25, 50)):
    if states.ndim != 2 or atoms.ndim != 2 or states.shape[1] != atoms.shape[1]:
        raise ValueError('Expected compatible [cell,hidden] and [atom,hidden] matrices')
    if len(token_ids) != len(atoms) or k < 1 or len(atoms) < k:
        raise ValueError('Invalid dictionary/support size')
    if not torch.isfinite(states).all() or not torch.isfinite(atoms).all():
        raise ValueError('Nonfinite pursuit input')
    norms = states.norm(dim=-1)
    h = states / norms.clamp_min(1e-30).unsqueeze(-1)
    residual = h.clone()
    n, width = h.shape
    support = torch.zeros(n, k, width, device=h.device, dtype=h.dtype)
    chosen = torch.full((n, k), -1, device=h.device, dtype=torch.long)
    selected_indices = torch.zeros(n, k, device=h.device, dtype=torch.long)
    valid = torch.zeros(n, k, device=h.device, dtype=torch.bool)
    coefficients = torch.zeros(n, 0, device=h.device, dtype=h.dtype)
    alive = norms > 0
    history, diagnostics = {}, []
    previous = residual.square().sum(-1)
    for step in range(k):
        scores = residual @ atoms.T
        if step:
            scores.scatter_(1, selected_indices[:, :step], -torch.inf)
        value, index = scores.max(-1)
        alive = alive & (value > 1e-7)
        selected_indices[:, step] = index
        valid[:, step] = alive
        chosen[:, step] = torch.where(alive, token_ids[index], -1)
        support[:, step] = atoms[index] * alive[:, None]
        active = support[:, :step + 1]
        gram = torch.bmm(active, active.transpose(1, 2))
        rhs = torch.bmm(active, h.unsqueeze(-1)).squeeze(-1)
        initial = torch.cat([coefficients, torch.zeros(n, 1, device=h.device, dtype=h.dtype)], -1)
        coefficients, diagnostic = nnls_gram(gram, rhs, initial)
        if not diagnostic['converged']:
            raise RuntimeError(f'NNLS did not converge at support {step + 1}: {diagnostic}')
        diagnostics.append(diagnostic)
        residual = h - torch.bmm(coefficients.unsqueeze(1), active).squeeze(1)
        squared = residual.square().sum(-1)
        if (squared > previous + 2e-5).any():
            raise ArithmeticError('Pursuit reconstruction objective increased')
        previous = squared
        if step + 1 in snapshots or step + 1 == k:
            history[step + 1] = dict(coefficients=(coefficients * norms[:, None]).cpu(),
                residual_fraction=squared.cpu(), normalized_kkt=diagnostic['max_kkt'])
    return dict(token_ids=chosen.cpu(), selected_indices=selected_indices.cpu(),
                valid=valid.cpu(), state_norm=norms.cpu(), snapshots=history,
                max_kkt=max(d['max_kkt'] for d in diagnostics),
                max_refit_steps=max(d['steps'] for d in diagnostics))


def orthonormal_basis(atoms, *, tolerance=1e-6):
    """Rank-aware modified QR with reorthogonalization; zero dependent columns.

    atoms: [cell,k,hidden]; result Q: [cell,hidden,k]. A naive reduced QR can
    add arbitrary directions when selected atoms are dependent.
    """
    q = torch.zeros_like(atoms).transpose(1, 2).contiguous()
    for column in range(atoms.shape[1]):
        v = atoms[:, column].clone()
        if column:
            previous = q[:, :, :column]
            for _ in range(2):
                v -= torch.bmm(previous, torch.bmm(previous.transpose(1, 2), v[:, :, None])).squeeze(-1)
        norm = v.norm(dim=-1)
        q[:, :, column] = torch.where((norm > tolerance)[:, None],
                                      v / norm.clamp_min(tolerance)[:, None], torch.zeros_like(v))
    return q


def remove_subspace(h, q):
    return h - torch.matmul(q, torch.matmul(q.transpose(-1, -2), h.unsqueeze(-1))).squeeze(-1)


def random_atom_indices(size, site, *, count=10, seed=42):
    value = int.from_bytes(hashlib.sha256(f'{seed}:random-atoms:{site}'.encode()).digest()[:8], 'little')
    return torch.randperm(size, generator=torch.Generator().manual_seed(value))[:count]


@contextmanager
def ablation(model, bases, *, sites=None, prefix_only=True):
    """Frozen bases act on original prefill positions, also during continuation."""
    handles = []
    def make_hook(q):
        def hook(module, args, output):
            n = q.shape[0]
            if output.shape[1] < n or (not prefix_only and output.shape[1] != n):
                raise ValueError('Ablation basis/position mismatch')
            result = output.clone()
            result[:, :n] = remove_subspace(output[:, :n].float(), q).to(output.dtype)
            return result
        return hook
    try:
        for site in sorted(bases if sites is None else sites):
            handles.append(model.layers[site].register_forward_hook(make_hook(bases[site])))
        yield
    finally:
        for handle in handles:
            handle.remove()
