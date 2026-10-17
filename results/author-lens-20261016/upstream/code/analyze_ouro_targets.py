"""Analysis of per-loop-target Jacobian lenses on Ouro-2.6B.

1. Readability: for each target t (loop end), top-1 agreement and KL of
   lens_{v->t} against the target's own readout, vs the final-target lens at
   the same source layers. Does targeting the nearest loop end restore
   readable transports from early loops?
2. Weight-tying test: are analogous transports equal across loops,
   J_{l->47} ~ J_{48+l->95} ~ J_{96+l->143} ~ J_{144+l->191}?
3. Probe readouts through own-loop lenses.
Saves ouro_targets_analysis.json.
"""

import json

import torch
import torch.nn.functional as F

from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from ouro_lens import from_ouro, load_ouro

torch.set_grad_enabled(False)

hf, tok = load_ouro()
hf = hf.cuda()
model = from_ouro(hf, tok)
NP, R, V = model.n_physical_layers, model.n_ut_steps, model.n_layers

TARGETS = [47, 95, 143, 191]
lenses = {
    47: JacobianLens.load("jacobian_lens_ouro-2.6b_t47.pt"),
    95: JacobianLens.load("jacobian_lens_ouro-2.6b_t95.pt"),
    143: JacobianLens.load("jacobian_lens_ouro-2.6b_t143.pt"),
    191: JacobianLens.load("jacobian_lens_ouro-2.6b.pt"),
}
J = {t: {v: lenses[t].jacobians[v].cuda() for v in lenses[t].source_layers} for t in TARGETS}
for t in TARGETS:
    print(t, lenses[t], flush=True)
results = {}


def capture(prompt, max_seq_len=512):
    ids = model.encode(prompt, max_length=max_seq_len)
    with ActivationRecorder(model.layers, at=list(range(V))) as rec:
        model.forward(ids)
    return ids, {v: rec.activations[v].detach()[0] for v in range(V)}


# ---------- 1. readability per target ----------
with open("prompts_1000.json") as f:
    eval_prompts = json.load(f)[980:1000]

agree = {t: torch.zeros(t) for t in TARGETS}   # J_{v->t} vs readout at t
kl = {t: torch.zeros(t) for t in TARGETS}
n_pos = 0
for p in eval_prompts:
    ids, acts = capture(p, max_seq_len=128)
    seq = ids.shape[1]
    positions = list(range(16, seq - 1, 8))
    for t in TARGETS:
        tgt_logits = model.unembed(acts[t][positions].float()).float()
        tgt_lsm = tgt_logits.log_softmax(-1)
        tgt_top = tgt_logits.argmax(-1)
        for v in range(t):
            h = acts[v][positions].float()
            lj = model.unembed(h @ J[t][v].T).float()
            agree[t][v] += (lj.argmax(-1) == tgt_top).float().sum().cpu()
            kl[t][v] += F.kl_div(
                lj.log_softmax(-1), tgt_lsm, log_target=True, reduction="sum"
            ).cpu()
    n_pos += len(positions)
for t in TARGETS:
    agree[t] /= n_pos
    kl[t] /= n_pos
results["agree"] = {t: agree[t].tolist() for t in TARGETS}
results["kl"] = {t: kl[t].tolist() for t in TARGETS}

print("\n=== top-1 agreement with each target's own readout ===", flush=True)
print("source v (loop end) | ->t47 | ->t95 | ->t143 | ->t191", flush=True)
for v in (23, 47, 71, 95, 119, 143, 167):
    row = " | ".join(f"{agree[t][v].item():.2f}" if v < t else "  -  " for t in TARGETS)
    print(f"  v{v:3d}: {row}", flush=True)
print("\nsame-loop mid-layer readability (24 layers before target):", flush=True)
print("  " + ", ".join(f"J_{{{t-24}->{t}}}: {agree[t][t-24].item():.2f}" for t in TARGETS), flush=True)
print("one-loop-back readability (48 layers before target):", flush=True)
print("  " + ", ".join(f"J_{{{t-48}->{t}}}: {agree[t][t-48].item():.2f}" for t in TARGETS if t >= 48), flush=True)

# ---------- 2. weight-tying test: analogous transports across loops ----------
def nsim(A, B):
    a, b = A.flatten(), B.flatten()
    return (a @ b / (a.norm() * b.norm())).item()

analog = []       # sim of J_{s*48+l -> (s+1)*48-1} across loops s, per offset l
for l in range(0, 47):
    mats = [J[(s + 1) * NP - 1][s * NP + l] for s in range(R)]
    sims = [nsim(mats[i], mats[j]) for i in range(R) for j in range(i + 1, R)]
    analog.append(sum(sims) / len(sims))
results["analogous_sim_per_offset"] = analog
mean_analog = sum(analog) / len(analog)

# baseline: same lens pairs at mismatched offsets (l vs l+8), and cross-loop mismatched
mismatch = []
for l in range(0, 39):
    mats_a = [J[(s + 1) * NP - 1][s * NP + l] for s in range(R)]
    mats_b = [J[(s + 1) * NP - 1][s * NP + l + 8] for s in range(R)]
    sims = [nsim(mats_a[i], mats_b[j]) for i in range(R) for j in range(R) if i != j]
    mismatch.append(sum(sims) / len(sims))
mean_mismatch = sum(mismatch) / len(mismatch)
results["mean_analogous_sim"] = mean_analog
results["mean_mismatched_sim"] = mean_mismatch
print(f"\nweight-tying: analogous-position J sim across loops {mean_analog:.3f} "
      f"(mismatched-offset baseline {mean_mismatch:.3f})", flush=True)

# norm vs distance-to-target, per target
norm_curves = {}
d = model.d_model
for t in TARGETS:
    norm_curves[t] = [(J[t][t - 1 - k].norm().item() / d**0.5) for k in range(min(t, 48))]
results["norm_vs_dist"] = {t: norm_curves[t] for t in TARGETS}
print("||J||/sqrt(d) at distance 1/8/24/47 from target:", flush=True)
for t in TARGETS:
    c = norm_curves[t]
    print(f"  ->t{t}: {c[0]:.2f} / {c[7]:.2f} / {c[23]:.2f} / {c[46]:.2f}", flush=True)

# ---------- 3. probes through own-loop lenses ----------
PROBES = [
    ("boot", "Fact: The currency used in the country shaped like a boot is", -2),
    ("eiffel", "The capital of the country where the Eiffel Tower stands is the city of", -1),
    ("arith", "Question: What is 47 + 38?\nAnswer:", -1),
]
probe_out = {}
for name, prompt, pos in PROBES:
    ids, acts = capture(prompt)
    rows = {}
    for s in range(R):
        t = (s + 1) * NP - 1
        for l in range(0, NP, 4):
            v = s * NP + l
            h = acts[v][pos].float()
            lj = model.unembed(J[t][v] @ h if v in J[t] else h).float()
            rows[v] = [tok.decode([x]).strip() for x in lj.topk(3).indices.tolist()]
    probe_out[name] = rows
    print(f"\n=== probe {name!r} through OWN-loop-end lens ===", flush=True)
    for l in range(0, NP, 4):
        cells = " | ".join(f"s{s}: " + ",".join(rows[s * NP + l]) for s in range(R))
        print(f"  L{l:2d}  {cells}", flush=True)
results["probes_ownloop"] = probe_out

json.dump(results, open("ouro_targets_analysis.json", "w"))
print("\nDONE_TARGETS_ANALYSIS", flush=True)
