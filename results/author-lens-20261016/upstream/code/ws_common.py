"""Shared harness for the remaining workspace-paper experiments on Ouro and
Qwen: model/lens loading, band readout, clamp-transfer swaps, steering
injection, chat/span utilities.

Readout convention: Ouro uses each loop's own-target lens (t47/t95/t143/t191)
for virtual layers in that loop, band = physical L24-40 per loop.
Interventions clamp loops 2+3+4 (the condition validated on flexible-gen).
Qwen uses the final-target lens, band = L24-48.
Huginn (r=16, 68 virtual layers) uses four trailing-window lenses
(t17/t33/t49/final): mean-J transport reaches ~2 recurrences, so each lens
reads/steers only its own window (s2-3, s6-7, s10-11, s14-15+coda; the
rank-1 windows from the band map). Interventions clamp the last three
windows, the analogue of Ouro's loops-2+3+4 condition.
"""

import torch

from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens

OURO_LENSES = {47: "jacobian_lens_ouro-2.6b_t47.pt",
               95: "jacobian_lens_ouro-2.6b_t95.pt",
               143: "jacobian_lens_ouro-2.6b_t143.pt",
               191: "jacobian_lens_ouro-2.6b.pt"}
QWEN_LENS = "jacobian_lens_qwen3.6-27b.pt"
HUGINN_LENSES = {17: "lens_huginn_ts3_w0_t17.pt",
                 33: "lens_huginn_ts7_w0_t33.pt",
                 49: "lens_huginn_ts11_w0_t49.pt",
                 67: "jacobian_lens_huginn-0125_r16.pt"}
HUGINN_WINDOWS = {17: range(10, 17), 33: range(26, 33),
                  49: range(42, 49), 67: range(58, 67)}
HUGINN_RANGES = {17: range(0, 17), 33: range(17, 33),
                 49: range(33, 49), 67: range(49, 67)}
# Complete 16-lens tiling (a lens at every recurrence end): every source is
# <=1 recurrence (4 virtual layers) from its covering target, well inside
# the measured transport horizon. Opt-in via load_ctx(huginn_full_tiling=
# True) so the validated 4-window intervention suite is unchanged.
HUGINN_LENSES_FULL = {5: "jacobian_lens_huginn_ts0_t5.pt",
                      9: "jacobian_lens_huginn_ts1_t9.pt",
                      13: "jacobian_lens_huginn_ts2_t13.pt",
                      17: "lens_huginn_ts3_w0_t17.pt",
                      21: "jacobian_lens_huginn_ts4_t21.pt",
                      25: "jacobian_lens_huginn_ts5_t25.pt",
                      29: "jacobian_lens_huginn_ts6_t29.pt",
                      33: "lens_huginn_ts7_w0_t33.pt",
                      37: "jacobian_lens_huginn_ts8_t37.pt",
                      41: "jacobian_lens_huginn_ts9_t41.pt",
                      45: "jacobian_lens_huginn_ts10_t45.pt",
                      49: "lens_huginn_ts11_w0_t49.pt",
                      53: "jacobian_lens_huginn_ts12_t53.pt",
                      57: "jacobian_lens_huginn_ts13_t57.pt",
                      61: "jacobian_lens_huginn_ts14_t61.pt",
                      67: "jacobian_lens_huginn-0125_r16.pt"}
HUGINN_WINDOWS_FULL = {t: range(max(t - 4, 0), t) for t in HUGINN_LENSES_FULL}
_FULL_TARGETS = sorted(HUGINN_LENSES_FULL)
HUGINN_RANGES_FULL = {t: range(0 if i == 0 else _FULL_TARGETS[i - 1], t)
                      for i, t in enumerate(_FULL_TARGETS)}


class Ctx:
    pass


def load_ctx(which, full_depth=False, huginn_full_tiling=False):
    ctx = Ctx()
    ctx.which = which
    if which == "ouro":
        from ouro_lens import from_ouro, load_ouro
        hf, tok = load_ouro()
        hf = hf.cuda()
        model = from_ouro(hf, tok)
        NP = model.n_physical_layers
        ctx.fwd_kwargs = {"use_cache": False, "exit_at_step": 3}
        band = list(range(24, 41))
        ctx.read_layers, ctx.int_layers, ctx.Jt = [], [], {}
        for k, (t, path) in enumerate(sorted(OURO_LENSES.items())):
            lens = JacobianLens.load(path)
            step = t // NP
            vs = ([v for v in lens.source_layers if step * NP <= v < t]
                  if full_depth else [step * NP + l for l in band])
            for v in vs:
                ctx.Jt[v] = lens.jacobians[v].float().T.cuda()
            bvs = [step * NP + l for l in band]
            ctx.read_layers += bvs
            if step >= 1:
                ctx.int_layers += bvs
            del lens
        ctx.loop_of = lambda v: v // NP + 1
    elif which == "huginn":
        from huginn_lens import HuginnLensModel, load_huginn
        hf, tok = load_huginn()
        hf = hf.cuda()
        model = HuginnLensModel(hf, tok, num_steps=16)
        ctx.fwd_kwargs = {"use_cache": False,
                          "num_steps": torch.tensor([0, model.num_steps])}
        lenses_map = HUGINN_LENSES_FULL if huginn_full_tiling else HUGINN_LENSES
        windows_map = HUGINN_WINDOWS_FULL if huginn_full_tiling else HUGINN_WINDOWS
        ranges_map = HUGINN_RANGES_FULL if huginn_full_tiling else HUGINN_RANGES
        targets = sorted(lenses_map)
        ctx.read_layers, ctx.int_layers, ctx.Jt = [], [], {}
        for k, t in enumerate(targets):
            lens = JacobianLens.load(lenses_map[t])
            window = list(windows_map[t])
            vs = ([v for v in lens.source_layers if v in ranges_map[t]]
                  if full_depth else window)
            for v in vs:
                ctx.Jt[v] = lens.jacobians[v].float().T.cuda()
            ctx.read_layers += window
            if k >= 1:
                ctx.int_layers += window
            del lens
        ctx.loop_of = lambda v: sum(v >= ranges_map[t].start
                                    for t in targets)
    else:
        import transformers

        import jlens
        from revisions import revision
        rev = revision("Qwen/Qwen3.6-27B")
        hf = transformers.AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3.6-27B", revision=rev, dtype=torch.bfloat16).cuda()
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", revision=rev)
        model = jlens.from_hf(hf, tok)
        ctx.fwd_kwargs = {"use_cache": False}
        lens = JacobianLens.load(QWEN_LENS)
        vs = lens.source_layers if full_depth else list(range(24, 49))
        ctx.Jt = {v: lens.jacobians[v].float().T.cuda() for v in vs}
        ctx.read_layers = list(range(24, 49))
        ctx.int_layers = list(range(24, 49))
        ctx.loop_of = lambda v: 1
        del lens
    ctx.hf, ctx.tok, ctx.model = hf, tok, model
    ctx.U = hf.get_output_embeddings().weight.detach().float().cuda()
    ctx.n_layers = model.n_layers
    return ctx


def first_token(ctx, word):
    return ctx.tok(" " + str(word).strip(), add_special_tokens=False).input_ids[0]


def direction(ctx, v, token_id):
    d = ctx.Jt[v] @ ctx.U[token_id]
    return d / d.norm()


def chat(ctx, messages):
    return ctx.tok.apply_chat_template(messages, tokenize=False,
                                       continue_final_message=True,
                                       enable_thinking=False)


def chat_gen(ctx, messages):
    return ctx.tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)


def encode(ctx, text):
    return ctx.tok(text, return_tensors="pt").input_ids.cuda()


def token_span(ctx, text, sub, last=True):
    """Token index range [a, b) covering substring `sub` of `text`."""
    start = text.rfind(sub) if last else text.find(sub)
    assert start >= 0, f"substring not found: {sub[:50]!r}"
    a = ctx.tok(text[:start], return_tensors="pt").input_ids.shape[1]
    b = ctx.tok(text[: start + len(sub)], return_tensors="pt").input_ids.shape[1]
    n = ctx.tok(text, return_tensors="pt").input_ids.shape[1]
    return max(0, a - 1), min(b + 1, n)


def forward(ctx, ids, layers=None):
    """One recorded forward via the jlens wrapper (matches lens.apply's path;
    a direct hf() call yields different layer-output semantics on Qwen3.6).

    Returns (final_logits[seq, vocab], acts {v: [seq, d]})."""
    layers = ctx.read_layers if layers is None else layers
    final = ctx.n_layers - 1
    rec_at = sorted(set(list(layers) + [final]))
    with ActivationRecorder(ctx.model.layers, at=rec_at) as rec:
        ctx.model.forward(ids)
    acts = {v: rec.activations[v].detach()[0].float() for v in rec_at}
    return ctx.model.unembed(acts[final]).float(), acts


def embeds_forward(ctx, be):
    """Forward from raw input embeddings (ignition's alpha-mixed tokens).

    Raven takes `input_embeds` (needs dummy input_ids for the RoPE length)
    and draws its initial state from randn, so re-seed like the adapter."""
    if ctx.which == "huginn":
        torch.manual_seed(ctx.model.seed)
        dummy = torch.zeros(be.shape[:2], dtype=torch.long, device=be.device)
        return ctx.hf(input_ids=dummy, input_embeds=be, **ctx.fwd_kwargs)
    return ctx.hf(inputs_embeds=be, **ctx.fwd_kwargs)


def min_ranks(ctx, acts, positions, token_ids, layers=None):
    """Min lens rank per token over (band layer, position).

    Returns tensor [len(token_ids)] of best (lowest) rank."""
    layers = ctx.read_layers if layers is None else layers
    tids = torch.tensor(token_ids, device="cuda")
    best = torch.full((len(token_ids),), float(1 << 30), device="cuda")
    for v in layers:
        h = acts[v][positions]
        lg = ctx.model.unembed(h @ ctx.Jt[v]).float()
        tv = lg[:, tids]
        srt, _ = lg.sort(dim=1)
        n_le = torch.searchsorted(srt, tv, right=True)
        r = (lg.shape[1] - n_le + 1).min(0).values
        best = torch.minimum(best, r.float())
    return best


class SwapHooks:
    """Clamp-transfer h <- h - (h.dA)dA + s(h.dA)dB on int_layers over span."""

    def __init__(self, ctx, tok_a, tok_b, span=None, strength=1.0, layers=None):
        self.ctx = ctx
        self.layers = ctx.int_layers if layers is None else layers
        self.dirs = {v: (direction(ctx, v, tok_a), direction(ctx, v, tok_b))
                     for v in self.layers}
        self.span, self.s, self.handles = span, strength, []

    def _hook(self, v):
        da, db = self.dirs[v]
        s, span = self.s, self.span

        def fn(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            sl = slice(None) if span is None else slice(*span)
            hf32 = h[:, sl].float()
            c = hf32 @ da
            hf32 = hf32 - c.unsqueeze(-1) * da + s * c.unsqueeze(-1) * db
            h[:, sl] = hf32.to(h.dtype)

        return fn

    def __enter__(self):
        for v in self.layers:
            self.handles.append(
                self.ctx.model.layers[v].register_forward_hook(self._hook(v)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


class InjectHooks:
    """h[:, span] += strength * sigma_v * d_v on int_layers.

    sigma: {v: mean residual norm over span} from a baseline forward."""

    def __init__(self, ctx, token_id, sigma, span, strength):
        self.ctx = ctx
        self.dirs = {v: direction(ctx, v, token_id) for v in ctx.int_layers}
        self.sigma, self.span, self.s, self.handles = sigma, span, strength, []

    def _hook(self, v):
        d = self.dirs[v] * (self.s * self.sigma[v])
        a, b = self.span

        def fn(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            h[:, a:b] += d.to(h.dtype)

        return fn

    def __enter__(self):
        for v in self.ctx.int_layers:
            self.handles.append(
                self.ctx.model.layers[v].register_forward_hook(self._hook(v)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


NUM_WORDS = {}
_ONES = "zero one two three four five six seven eight nine".split()
_TEENS = ("ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
          "eighteen nineteen").split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
for _i, _w in enumerate(_ONES):
    NUM_WORDS[_i] = _w
for _i, _w in enumerate(_TEENS):
    NUM_WORDS[10 + _i] = _w
for _i, _w in enumerate(_TENS):
    NUM_WORDS[20 + 10 * _i] = _w
for _t in range(20, 100, 10):
    for _o in range(1, 10):
        NUM_WORDS[_t + _o] = f"{NUM_WORDS[_t]}-{_ONES[_o]}"


def number_tokens(n):
    """Surface forms for integer n: digits and number-word."""
    out = [str(n)]
    if n in NUM_WORDS:
        w = NUM_WORDS[n]
        out.append(w)
        if "-" in w:
            out.append(w.split("-")[0])
    return out
