"""Loop-structure analysis of the Jacobian lens fitted on Ouro-2.6B.

Produces:
  1. Readout trajectories on probe prompts across all 191 virtual layers,
     organised as physical layer x loop step.
  2. Lens quality curve: KL(model || lens_v) over WikiText, jlens vs logit-lens.
  3. Jacobian geometry: ||J||, ||J-I||, and the 191x191 cosine-similarity map
     (is J organised by loop step or by physical depth?).
  4. Early-exit gate pdf vs per-step lens agreement.
Saves ouro_analysis.json + ouro_jsim.pt and prints a readable summary.
"""

import json

import torch
import torch.nn.functional as F
import transformers

from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from ouro_lens import from_ouro

torch.set_grad_enabled(False)

from ouro_lens import load_ouro
hf, tok = load_ouro()
hf = hf.cuda()
model = from_ouro(hf, tok)
lens = JacobianLens.load("jacobian_lens_ouro-2.6b.pt")
print(model)
print(lens, flush=True)

NP = model.n_physical_layers
R = model.n_ut_steps
V = model.n_layers
SRC = lens.source_layers
results = {}

J_gpu = {v: lens.jacobians[v].cuda() for v in SRC}


def capture(prompt, max_seq_len=512):
    ids = model.encode(prompt, max_length=max_seq_len)
    with ActivationRecorder(model.layers, at=list(range(V))) as rec:
        out = model.forward(ids)
    acts = {v: rec.activations[v].detach()[0] for v in range(V)}  # [seq, d]
    return ids, acts, out


def lens_logits_at(acts, v, pos):
    h = acts[v][pos].float()
    if v in J_gpu:
        h = J_gpu[v] @ h
    return model.unembed(h).float()


# ---------- 1. probe readout trajectories ----------
PROBES = [
    ("boot", "Fact: The currency used in the country shaped like a boot is", -2),
    ("eiffel", "The capital of the country where the Eiffel Tower stands is the city of", -1),
    ("arith", "Question: What is 47 + 38?\nAnswer:", -1),
]
probe_out = {}
for name, prompt, pos in PROBES:
    ids, acts, _ = capture(prompt)
    tok_at = tok.decode([ids[0, pos].item()])
    rows = {}
    for v in range(V):
        ll = lens_logits_at(acts, v, pos)
        top = ll.topk(3).indices.tolist()
        rows[v] = [tok.decode([t]).strip() for t in top]
    probe_out[name] = {"prompt": prompt, "pos_token": tok_at, "rows": rows}
    print(f"\n=== probe {name!r} (position token {tok_at!r}) ===", flush=True)
    for l in range(0, NP, 4):
        cells = " | ".join(
            f"s{s}: " + ",".join(rows[s * NP + l]) for s in range(R) if s * NP + l < V
        )
        print(f"  L{l:2d}  {cells}", flush=True)
results["probes"] = probe_out

# ---------- 2. lens quality curve (KL) over WikiText ----------
with open("prompts_1000.json") as f:
    eval_prompts = json.load(f)[980:1000]

kl_j = torch.zeros(V - 1)
kl_ll = torch.zeros(V - 1)
n_pos = 0
for p in eval_prompts:
    ids, acts, _ = capture(p, max_seq_len=128)
    seq = ids.shape[1]
    positions = list(range(16, seq - 1, 8))
    final = model.unembed(acts[V - 1][positions].float()).float().log_softmax(-1)
    for v in range(V - 1):
        h = acts[v][positions].float()
        lj = model.unembed(h @ J_gpu[v].T).float().log_softmax(-1)
        lll = model.unembed(h).float().log_softmax(-1)
        kl_j[v] += F.kl_div(lj, final, log_target=True, reduction="sum").cpu()
        kl_ll[v] += F.kl_div(lll, final, log_target=True, reduction="sum").cpu()
    n_pos += len(positions)
kl_j /= n_pos
kl_ll /= n_pos
results["kl_jlens"] = kl_j.tolist()
results["kl_logitlens"] = kl_ll.tolist()
print("\n=== KL(model || lens) by virtual layer (mean over WikiText) ===", flush=True)
for s in range(R):
    row = " ".join(
        f"{kl_j[s * NP + l]:5.1f}" for l in range(0, NP, 4) if s * NP + l < V - 1
    )
    print(f"  step {s}: {row}", flush=True)
print("logit-lens at step ends:", [round(kl_ll[min(s * NP + NP - 1, V - 2)].item(), 1) for s in range(R)], flush=True)

# ---------- 3. Jacobian geometry ----------
d = model.d_model
eye = torch.eye(d, device="cuda")
norms, dist_eye = {}, {}
flat = torch.empty(len(SRC), d * d, dtype=torch.float16)
for i, v in enumerate(SRC):
    Jv = J_gpu[v]
    norms[v] = (Jv.norm() / d**0.5).item()
    dist_eye[v] = ((Jv - eye).norm() / d**0.5).item()
    flat[i] = (Jv / Jv.norm()).flatten().half().cpu()
results["J_norm"] = norms
results["J_dist_eye"] = dist_eye

flat_gpu = flat.cuda().float()
sim = flat_gpu @ flat_gpu.T
torch.save({"source_layers": SRC, "sim": sim.cpu()}, "ouro_jsim.pt")

same_layer = []
for l in range(NP):
    vs = [s * NP + l for s in range(R) if s * NP + l in set(SRC)]
    idx = [SRC.index(v) for v in vs]
    if len(idx) > 1:
        vals = [sim[a, b].item() for i, a in enumerate(idx) for b in idx[i + 1:]]
        same_layer.append(sum(vals) / len(vals))
adjacent_depth = [sim[i, i + 1].item() for i in range(len(SRC) - 1)]
results["mean_same_physical_layer_sim"] = sum(same_layer) / len(same_layer)
results["mean_adjacent_virtual_sim"] = sum(adjacent_depth) / len(adjacent_depth)
print(f"\nJ similarity: same physical layer across steps {results['mean_same_physical_layer_sim']:.3f}, "
      f"adjacent virtual layers {results['mean_adjacent_virtual_sim']:.3f}", flush=True)
print("||J-I||/sqrt(d) at step ends:", [round(dist_eye[min(s * NP + NP - 1, V - 2)], 2) for s in range(R)], flush=True)

# ---------- 4. early-exit gate vs lens agreement ----------
gate_pdf_sum = torch.zeros(R)
step_end_kl = torch.zeros(R)
n_tok = 0
for p in eval_prompts[:10]:
    ids, acts, out = capture(p, max_seq_len=128)
    _, hs_list, gate_list = out
    seq = ids.shape[1]
    positions = list(range(16, seq - 1, 8))
    rem = torch.ones(len(positions), device="cuda")
    final = model.unembed(acts[V - 1][positions].float()).float().log_softmax(-1)
    for s in range(R):
        lam = torch.sigmoid(gate_list[s][0, positions, 0].float())
        p_s = lam * rem if s < R - 1 else rem
        rem = rem * (1 - lam)
        gate_pdf_sum[s] += p_s.sum().cpu()
        v = min(s * NP + NP - 1, V - 2)
        h = acts[v][positions].float()
        lj = model.unembed(h @ J_gpu[v].T).float().log_softmax(-1)
        step_end_kl[s] += F.kl_div(lj, final, log_target=True, reduction="sum").cpu()
    n_tok += len(positions)
results["mean_gate_pdf"] = (gate_pdf_sum / n_tok).tolist()
results["mean_step_end_kl"] = (step_end_kl / n_tok).tolist()
print(f"\ngate exit pdf (mean): {[round(x, 3) for x in results['mean_gate_pdf']]}", flush=True)
print(f"KL at step-end layers: {[round(x, 2) for x in results['mean_step_end_kl']]}", flush=True)

with open("ouro_analysis.json", "w") as f:
    json.dump(results, f)
print("\nDONE_ANALYSIS", flush=True)
