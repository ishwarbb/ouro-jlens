"""Mathematical and recurrence checks on the real, small Ouro architecture."""
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import json

import torch
from jlens import JacobianLens
from jlens.fitting import jacobian_for_prompt
from jlens.hooks import ActivationRecorder

from ouro_lens_family import OuroLensFamily, TargetModel, source_plan
import test_ouro_lens


class FamilyTests(unittest.TestCase):
    setUp = test_ouro_lens.AdapterTests.setUp
    tearDown = test_ouro_lens.AdapterTests.tearDown

    def family(self):
        # Nonsymmetric maps catch J vs J.T orientation mistakes.
        return OuroLensFamily(
            {t: JacobianLens({s: torch.randn(16, 16) for s in range(t)},
                              n_prompts=2, d_model=16) for t in (1, 3, 5)},
            physical_layers=2, recurrent_steps=3)

    def test_prefix_states_and_loop_logits_match_full_forward(self):
        with torch.no_grad(), ActivationRecorder(self.model.layers, at=range(6)) as recorder:
            self.model.forward(self.ids)
        for target in range(6):
            with torch.no_grad():
                stopped = self.model.forward(self.ids, stop_at=target)
            torch.testing.assert_close(stopped, recorder.activations[target], rtol=0, atol=0)
            self.assertEqual(self.model.last_site_order, list(range(target + 1)))
        for step, target in enumerate((1, 3, 5)):
            with torch.no_grad():
                logits = self.hf(self.ids, use_cache=False, exit_at_step=step).logits
                readout = self.model.unembed(recorder.activations[target])
            torch.testing.assert_close(logits, readout, rtol=0, atol=0)
        with torch.no_grad(), ActivationRecorder(self.model.layers, at=range(6)) as repeat:
            self.model.forward(self.ids)
        for site in range(6):
            torch.testing.assert_close(repeat.activations[site], recorder.activations[site], rtol=0, atol=0)

    def test_intermediate_jacobians_match_full_graph_and_dimension_batching(self):
        kwargs = dict(source_layers=[0, 1, 2], target_layer=3, max_seq_len=24, skip_first=2)
        full, _, _ = jacobian_for_prompt(self.model, 'a small gradient sequence', dim_batch=1, **kwargs)
        prefix, _, _ = jacobian_for_prompt(TargetModel(self.model, 3), 'a small gradient sequence',
                                           dim_batch=3, **kwargs)
        for source in full:
            torch.testing.assert_close(full[source], prefix[source], rtol=0.002, atol=0.0001)
            self.assertGreater(prefix[source].norm().item(), 0)
        self.assertTrue(all(p.grad is None for p in self.hf.parameters()))

    def test_intermediate_cross_boundary_finite_difference(self):
        with ActivationRecorder(self.model.layers, at=[1, 3], start_graph_at=1) as recorder:
            h = self.model.forward(self.ids, stop_at=3)
            source = recorder.activations[1]
            gradient = torch.autograd.grad(h[0, -1, 2], source)[0]
        d = torch.randn_like(source)
        d /= d.norm()
        values = []
        epsilon = 0.002
        for sign in (-1, 1):
            handle = self.model.layers[1].register_forward_hook(
                lambda m, a, o, sign=sign: o + sign * epsilon * d)
            try:
                with torch.no_grad():
                    values.append(self.model.forward(self.ids, stop_at=3)[0, -1, 2].item())
            finally:
                handle.remove()
        observed = (values[1] - values[0]) / (2 * epsilon)
        expected = (gradient * d).sum().item()
        self.assertAlmostEqual(observed, expected, delta=max(0.002, abs(expected) * 0.03))

    def test_routing_transport_and_pullback(self):
        family = self.family()
        self.assertEqual([family.target_for(s) for s in range(6)], [1, 1, 3, 3, 5, 5])
        self.assertEqual(family.source_layers, [0, 2, 4])
        self.assertEqual(family.sites_at_offsets([1], first_loop=1), [2, 4])
        h = torch.randn(2, 16)
        torch.testing.assert_close(family.transport(h, 2), h @ family.lenses[3].jacobians[2].T)
        torch.testing.assert_close(family.transport(h, 3), h)
        torch.testing.assert_close(family.transport(h, 2, target=5), h @ family.lenses[5].jacobians[2].T)
        u = self.hf.lm_head.weight[7].float()
        expected = family.lenses[3].jacobians[2].T @ u
        expected /= expected.norm()
        torch.testing.assert_close(family.direction(self.model, 2, 7), expected)
        with self.assertRaises(ValueError):
            family.transport(h, 4, target=3)

    def test_serialization_preserves_large_values_and_targets(self):
        family = self.family()
        family.lenses[3].jacobians[0][0, 1] = 1e6  # FP16 would overflow.
        with tempfile.TemporaryDirectory() as directory:
            family.save(directory)
            restored = OuroLensFamily.load(directory)
        for target, lens in family.lenses.items():
            for source, matrix in lens.jacobians.items():
                torch.testing.assert_close(matrix, restored.lenses[target].jacobians[source], rtol=0, atol=0)

    def test_apply_uses_own_loop_and_actual_loop_baselines(self):
        family = self.family()
        readouts, actual, ids = family.apply(self.model, 'small prompt', sources=[0, 2, 4])
        with torch.no_grad(), ActivationRecorder(self.model.layers, at=range(6)) as recorder:
            self.model.forward(ids)
        for source, logits in readouts.items():
            h = recorder.activations[source][0, [-1]].float()
            expected = self.model.unembed(family.transport(h, source))
            torch.testing.assert_close(logits, expected)
        for target, logits in actual.items():
            expected = self.model.unembed(recorder.activations[target][0, [-1]])
            torch.testing.assert_close(logits, expected)

    def test_intervention_position_scope_ablation_and_cleanup(self):
        family = self.family()
        with torch.no_grad():
            baseline = self.model.forward(self.ids, stop_at=2)
        with family.intervene(self.model, 7, sources=[2], strength=0), torch.no_grad():
            zero = self.model.forward(self.ids, stop_at=2)
        torch.testing.assert_close(zero, baseline, rtol=0, atol=0)
        with family.intervene(self.model, 7, sources=[2], strength=2), torch.no_grad():
            edited = self.model.forward(self.ids, stop_at=2)
        direction = family.direction(self.model, 2, 7)
        torch.testing.assert_close(edited[:, :-1], baseline[:, :-1], rtol=0, atol=0)
        torch.testing.assert_close(edited[0, -1] - baseline[0, -1], 2 * direction)
        with family.intervene(self.model, 7, sources=[2], mode='ablate'), torch.no_grad():
            ablated = self.model.forward(self.ids, stop_at=2)
        self.assertLess(abs((ablated[0, -1] @ direction).item()), 1e-5)
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            with family.intervene(self.model, 7, sources=[0, 2, 4]):
                raise RuntimeError('intentional')
        self.assertTrue(all(not tap._forward_hooks for tap in self.model.layers))

    def test_checkpointing_reenabled_is_rejected(self):
        self.hf.gradient_checkpointing_enable()
        with self.assertRaisesRegex(RuntimeError, 'checkpointing disabled'):
            self.model.forward(self.ids)

    def test_source_plans(self):
        plan = source_plan(48, 4)
        self.assertEqual(list(plan), [47, 95, 143, 191])
        self.assertEqual(len(plan[191]), 191)
        local = source_plan(48, 4, scope='within-loop')
        self.assertEqual(local[95], list(range(48, 95)))
        self.assertEqual(source_plan(48, 4, offsets=[11, 23])[95], [72, 84])
        for offsets in ([0], [48], []):
            with self.assertRaises(ValueError):
                source_plan(48, 4, offsets=offsets)

    def test_runner_resume_and_corpus_mismatch(self):
        from fit_ouro_family import ROOT, fit_family
        prompts = ['a sufficiently long prompt for the first sample',
                   'a different sufficiently long text for the second sample']
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            corpus = directory / 'corpus.json'
            corpus.write_text(json.dumps(prompts))
            args = SimpleNamespace(model=ROOT / 'model', corpus=corpus,
                                   output_dir=directory / 'fit', scope='within-loop',
                                   offsets=None, max_seq_len=24, skip_first=2,
                                   dim_batch=3, checkpoint_every=1, device='cpu')
            # Avoid hashing the production 5 GB weights in a tiny-model test.
            with patch('fit_ouro_family.weight_identity', return_value={'tiny': 'fixture'}):
                first = fit_family(self.model, prompts, args)
                with patch('jlens.fitting.jacobian_for_prompt',
                           side_effect=AssertionError('Completed fit should resume without gradients')):
                    resumed = fit_family(self.model, prompts, args)
                for target in first.lenses:
                    for source, matrix in first.lenses[target].jacobians.items():
                        torch.testing.assert_close(matrix, resumed.lenses[target].jacobians[source],
                                                   rtol=0, atol=0)
                corpus.write_text(json.dumps(prompts + ['changed corpus']))
                with self.assertRaisesRegex(ValueError, 'Resume manifest differs'):
                    fit_family(self.model, prompts, args)


if __name__ == '__main__':
    unittest.main()
