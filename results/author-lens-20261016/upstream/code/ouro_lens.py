"""jlens adapter for looped (universal-transformer) models like ByteDance Ouro.

Ouro applies its whole 48-layer stack total_ut_steps=4 times (final RMSNorm +
early-exit gate at the end of each step; the normed state re-enters the stack).
We unroll: virtual layer v = n_physical*step + layer, so the model presents as
a 192-layer transformer and jlens's fit/apply work unchanged. A physical
block's forward hook fires once per loop step; VirtualBlock dispatches the
step-th firing to the registered hook.
"""

import torch
from jlens.hf import HFLensModel


class _VirtualHandle:
    def __init__(self, handle):
        self._handle = handle

    def remove(self):
        self._handle.remove()


class VirtualBlock:
    def __init__(self, module, step, n_steps):
        self.module = module
        self.step = step
        self.n_steps = n_steps

    def register_forward_hook(self, fn):
        state = {"n": 0}
        step, n_steps = self.step, self.n_steps

        def wrapper(mod, inputs, output):
            if state["n"] % n_steps == step:
                fn(mod, inputs, output)
            state["n"] += 1

        return _VirtualHandle(self.module.register_forward_hook(wrapper))


class OuroLensModel(HFLensModel):
    """Presents an Ouro checkpoint as an unrolled n_physical*n_steps-layer
    LensModel. Virtual layer v <-> (step v // n_physical, layer v % n_physical).
    Target virtual layer n_layers-1 is the step-(R-1) last-block output, i.e.
    exactly the pre-final-norm state the LM head consumes at default inference.
    """

    def __init__(self, hf_model, tokenizer, **kw):
        super().__init__(hf_model, tokenizer, **kw)
        n_steps = int(getattr(hf_model.config, "total_ut_steps", 1) or 1)
        n_phys = len(self.layers)
        self.n_ut_steps = n_steps
        self.n_physical_layers = n_phys
        self.physical_layers = self.layers
        self.layers = [
            VirtualBlock(self.physical_layers[v % n_phys], v // n_phys, n_steps)
            for v in range(n_phys * n_steps)
        ]
        self.n_layers = n_phys * n_steps

    def vl(self, step, layer):
        return step * self.n_physical_layers + layer

    def __repr__(self):
        return (
            f"OuroLensModel({type(self._hf_model).__name__}, "
            f"{self.n_physical_layers} physical x {self.n_ut_steps} steps = "
            f"{self.n_layers} virtual layers, d_model={self.d_model})"
        )


def from_ouro(hf_model, tokenizer, **kw):
    return OuroLensModel(hf_model, tokenizer, **kw)


def _default_rope_init(config=None, device=None, seq_len=None, layer_type=None, **kw):
    base = config.rope_theta
    dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim)
    )
    return inv_freq, 1.0


def load_ouro(model_id=None, dtype=torch.bfloat16):
    """Ouro's remote code targets transformers 4.55; under 5.x the config no
    longer auto-provides pad_token_id and ROPE_INIT_FUNCTIONS lost its
    'default' entry (classic RoPE), so patch both before instantiation."""
    import os
    if model_id is None:
        model_id = os.environ.get("OURO_MODEL_ID", "ByteDance/Ouro-2.6B")
    import transformers
    from transformers import modeling_rope_utils

    from revisions import revision
    rev = revision(model_id)

    modeling_rope_utils.ROPE_INIT_FUNCTIONS.setdefault("default", _default_rope_init)
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    rot_cls = get_class_from_dynamic_module("modeling_ouro.OuroRotaryEmbedding", model_id, revision=rev)
    if not hasattr(rot_cls, "compute_default_rope_parameters"):
        rot_cls.compute_default_rope_parameters = staticmethod(_default_rope_init)
    cfg = transformers.AutoConfig.from_pretrained(model_id, revision=rev, trust_remote_code=True)
    if not hasattr(cfg, "pad_token_id"):
        cfg.pad_token_id = None
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, revision=rev, config=cfg, dtype=dtype, trust_remote_code=True
    )
    tok = transformers.AutoTokenizer.from_pretrained(model_id, revision=rev, trust_remote_code=True)

    import inspect
    import sys

    from transformers.masking_utils import create_causal_mask as _ccm

    _ccm_params = inspect.signature(_ccm).parameters

    def _ccm_compat(**kw):
        if "input_embeds" in kw and "input_embeds" not in _ccm_params:
            kw["inputs_embeds"] = kw.pop("input_embeds")
        return _ccm(**{k: v for k, v in kw.items() if k in _ccm_params})

    mod = sys.modules[type(hf).__module__]
    mod.create_causal_mask = _ccm_compat
    return hf, tok
