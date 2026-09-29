"""CPU correctness checks on the actual Ouro architecture with small random weights."""
import unittest
import torch
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from jlens.hooks import ActivationRecorder
from jlens.fitting import jacobian_for_prompt
from ouro_lens import OuroLensModel


class TinyTokenizer:
    bos_token_id = 0
    def encode(self, text, add_special_tokens=False):
        return [1 + ord(c) % 30 for c in text]


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        config = AutoConfig.from_pretrained('model', trust_remote_code=True)
        config.hidden_size = 16
        config.intermediate_size = 32
        config.num_hidden_layers = 2
        config.num_attention_heads = 2
        config.num_key_value_heads = 2
        config.head_dim = 8
        config.layer_types = ['full_attention'] * 2
        config.vocab_size = 32
        config.total_ut_steps = 3
        config._attn_implementation = 'eager'
        cls = get_class_from_dynamic_module(config.auto_map['AutoModelForCausalLM'], 'model')
        self.hf = cls(config).float().eval()
        self.model = OuroLensModel(self.hf, TinyTokenizer())
        self.ids = self.model.encode('a small sequence for a gradient test', max_length=24)

    def tearDown(self):
        self.model.close()

    def test_all_passes_and_readout(self):
        with torch.no_grad(), ActivationRecorder(self.model.layers, at=range(6)) as recorder:
            self.model.forward(self.ids)
        self.assertEqual(self.model.last_site_order, list(range(6)))
        self.assertEqual(len({id(t) for t in recorder.activations.values()}), 6)
        with torch.no_grad():
            expected = self.hf(self.ids, use_cache=False, exit_at_step=2).logits
            actual = self.model.unembed(recorder.activations[5])
        torch.testing.assert_close(actual, expected)
        # First and second pass outputs of the same physical block are not aliases.
        self.assertIsNot(recorder.activations[1], recorder.activations[3])
        with torch.no_grad():
            self.model.forward(self.ids)
        self.assertEqual(self.model.last_site_order, list(range(6)))

    def test_cross_pass_gradient_and_finite_difference(self):
        with ActivationRecorder(self.model.layers, at=[1, 3, 5], start_graph_at=1) as recorder:
            self.model.forward(self.ids)
            h = recorder.activations[1]
            target = recorder.activations[5]
            objective = target[0, -1, 0]
            grad = torch.autograd.grad(objective, h)[0]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.norm().item(), 0)
        direction = torch.randn_like(h)
        direction /= direction.norm()
        expected = (grad * direction).sum()
        observed = []
        epsilon = 0.002
        for sign in [-1, 1]:
            hook = self.model.layers[1].register_forward_hook(lambda m, a, o, sign=sign: o + sign * epsilon * direction)
            try:
                with torch.no_grad(), ActivationRecorder(self.model.layers, at=[5]) as recorder:
                    self.model.forward(self.ids)
                    observed.append(recorder.activations[5][0, -1, 0].item())
            finally:
                hook.remove()
        actual = (observed[1] - observed[0]) / (2 * epsilon)
        self.assertAlmostEqual(actual, expected.item(), delta=max(0.002, abs(expected.item()) * 0.03))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.hf.parameters()))

    def test_reference_estimator_dimension_batching(self):
        kwargs = dict(source_layers=[1, 3, 4], max_seq_len=24, skip_first=2)
        one, _, _ = jacobian_for_prompt(self.model, 'a small sequence for a gradient test', dim_batch=1, **kwargs)
        two, _, _ = jacobian_for_prompt(self.model, 'a small sequence for a gradient test', dim_batch=2, **kwargs)
        for site in one:
            torch.testing.assert_close(one[site], two[site], rtol=0.002, atol=0.0001)

    def test_close_removes_physical_hooks(self):
        self.model.close()
        self.assertTrue(all(len(block._forward_hooks) == 0 for block in self.hf.model.layers))
        with self.assertRaises(RuntimeError):
            self.model.forward(self.ids)


if __name__ == '__main__':
    unittest.main()
