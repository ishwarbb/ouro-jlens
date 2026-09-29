import itertools
import unittest
from types import SimpleNamespace

import torch
from workspace_inventory import (dictionary, nnls_gram, pursuit, orthonormal_basis,
                                 remove_subspace, random_atom_indices, ablation)
from workspace_validated import encode_prompt, first_word, word_atom, ValidatedIO, prefix_transfer


class InventoryTests(unittest.TestCase):
    def test_nnls_against_independent_exhaustive_active_set_solver(self):
        torch.manual_seed(14)
        atoms = torch.randn(4, 5, 8, dtype=torch.double)
        atoms /= atoms.norm(dim=-1, keepdim=True)
        states = torch.randn(4, 8, dtype=torch.double)
        gram = atoms @ atoms.transpose(-1, -2)
        rhs = (atoms @ states.unsqueeze(-1)).squeeze(-1)
        actual, info = nnls_gram(gram, rhs, tolerance=1e-9)
        self.assertTrue(info['converged'])
        for batch in range(4):
            best = float(states[batch].square().sum())
            for size in range(1, 6):
                for support in itertools.combinations(range(5), size):
                    a = atoms[batch, list(support)].T
                    c = torch.linalg.lstsq(a, states[batch]).solution
                    if (c >= 0).all():
                        best = min(best, float((states[batch] - a @ c).square().sum()))
            self.assertAlmostEqual(float((states[batch] - actual[batch] @ atoms[batch]).square().sum()), best, places=7)

    def test_dictionary_orientation_and_implicit_scores(self):
        torch.manual_seed(3)
        u = torch.randn(13, 7); j = torch.randn(7, 7); h = torch.randn(3, 7)
        ids = torch.tensor([1, 3, 6, 10])
        d = dictionary(u, j, ids, chunk=2)
        expected = torch.stack([j.T @ u[t] for t in ids])
        expected /= expected.norm(dim=-1, keepdim=True)
        torch.testing.assert_close(d, expected)
        implicit = ((h @ j.T) @ u[ids].T) / (u[ids] @ j).norm(dim=-1)
        torch.testing.assert_close(h @ d.T, implicit)

    def test_exact_orthogonal_reconstruction_and_zero_state(self):
        atoms = torch.eye(5)
        states = torch.tensor([[3., 2., 0., 0., 1.], [0., 0., 0., 0., 0.]])
        result = pursuit(states, atoms, torch.arange(5), k=5, snapshots=(1, 3, 5))
        self.assertEqual(result['token_ids'][0, :3].tolist(), [0, 1, 4])
        self.assertEqual(int(result['valid'][1].sum()), 0)
        self.assertLess(float(result['snapshots'][3]['residual_fraction'][0]), 1e-10)
        self.assertEqual(float(result['snapshots'][5]['residual_fraction'][1]), 0)
        torch.testing.assert_close(result['snapshots'][3]['coefficients'][0], torch.tensor([3., 2., 1.]))

    def test_correlated_dictionary_scalar_batch_agreement(self):
        torch.manual_seed(22)
        atoms = torch.randn(12, 9); atoms /= atoms.norm(dim=-1, keepdim=True)
        h = torch.randn(3, 9)
        batch = pursuit(h, atoms, torch.arange(12), k=7)
        for i in range(3):
            single = pursuit(h[i:i+1], atoms, torch.arange(12), k=7)
            self.assertEqual(batch['token_ids'][i].tolist(), single['token_ids'][0].tolist())
            torch.testing.assert_close(batch['snapshots'][7]['coefficients'][i], single['snapshots'][7]['coefficients'][0], atol=3e-5, rtol=3e-5)

    def test_rank_deficiency_does_not_erase_unrelated_dimensions(self):
        a = torch.tensor([[[1., 0., 0.], [1., 0., 0.], [-1., 0., 0.]]])
        q = orthonormal_basis(a)
        self.assertEqual(int((q.norm(dim=1) > .5).sum()), 1)
        h = torch.tensor([[2., 3., 4.]])
        expected = torch.tensor([[0., 3., 4.]])
        torch.testing.assert_close(remove_subspace(h, q), expected)
        torch.testing.assert_close(remove_subspace(expected, q), expected)

    def test_frozen_basis_hook_and_cleanup(self):
        model = SimpleNamespace(layers=[torch.nn.Identity()])
        q = torch.tensor([[[1.], [0.]]])
        x = torch.tensor([[[2., 3.], [4., 5.]]])
        with ablation(model, {0:q}):
            torch.testing.assert_close(model.layers[0](x), torch.tensor([[[0., 3.], [4., 5.]]]))
        torch.testing.assert_close(model.layers[0](x), x)

    def test_random_control_fixed_by_layer(self):
        torch.testing.assert_close(random_atom_indices(100, 72), random_atom_indices(100, 72))
        self.assertEqual(len(set(random_atom_indices(100, 72).tolist())), 10)
        self.assertFalse(torch.equal(random_atom_indices(100, 72), random_atom_indices(100, 73)))


class Tokenizer:
    bos_token_id = 1
    all_special_ids = [1, 9]
    def encode(self, text, **kwargs):
        return [2, 3] if text == 'text' else [1, 2, 3]


class ValidationTests(unittest.TestCase):
    def test_bos_once_and_no_truncation(self):
        for text in ['text', 'already-bos']:
            p = encode_prompt(Tokenizer(), text)
            self.assertEqual(p.ids, (1, 2, 3))
            self.assertEqual(p.record()['seq_ids'], [1, 2, 3])
        with self.assertRaises(ValueError):
            encode_prompt(Tokenizer(), 'text', limit=2)

    def test_completed_words_not_prefixes(self):
        self.assertIsNone(first_word('"New'))
        self.assertEqual(first_word('"New York'), 'new')
        self.assertEqual(first_word('Parisian.'), 'parisian')
        self.assertNotEqual(first_word('Parisian.'), 'paris')
        self.assertEqual(first_word('Paris', ended=True), 'paris')
        self.assertIsNone(first_word('" : '))

    def test_declared_dictionary_filter(self):
        for text in [' tree', 'école', '東京', 'ice-cream', "don't"]:
            self.assertTrue(word_atom(text), text)
        for text in [' ', '123', 'tree.', 'two words', '�', '<end>', 'foo_bar']:
            self.assertFalse(word_atom(text), text)

    def test_generation_appends_exact_ids(self):
        tok = Tokenizer()
        tok.decode = lambda ids, **kw: ''.join({4:'Par', 5:'is', 6:'.'}[i] for i in ids)
        io = object.__new__(ValidatedIO); io.tokenizer = tok
        seen = []
        def forward(ids):
            seen.append(tuple(ids)); logits = torch.full((10,), -10.)
            logits[{0:4, 1:5, 2:6}[len(ids)-2]] = 10.
            return logits, {}
        io.forward = forward
        result = io.generate((1, 7), limit=5, stop_word=True)
        self.assertEqual(seen, [(1, 7), (1, 7, 4), (1, 7, 4, 5)])
        self.assertEqual(result['token_ids'], [4, 5, 6])
        self.assertEqual(result['complete_first_word'], 'paris')

    def test_sequence_score_uses_full_continuation(self):
        io=object.__new__(ValidatedIO)
        io.target=lambda word:dict(continuation_ids=[4,5])
        seen=[]
        def forward(ids):
            seen.append(tuple(ids));return torch.zeros(10),{}
        io.forward=forward
        score=io.sequence_score((1,7),'Paris',initial_logits=torch.zeros(10))
        self.assertEqual(seen,[(1,7,4)])
        self.assertEqual(score['token_ids'],[4,5])
        self.assertAlmostEqual(score['log_probability'],-2*float(torch.tensor(10.).log()))

    def test_transfer_and_random_control_are_matched_on_frozen_positions(self):
        model=SimpleNamespace(layers=[torch.nn.Identity()])
        bank=SimpleNamespace(get=lambda site,token:torch.eye(3)[token],
                             random=lambda site,key:torch.tensor([0.,0.,1.]))
        h=torch.tensor([[[2.,3.,4.],[5.,6.,7.]]])
        with prefix_transfer(model,bank,0,1,[0],1):changed=model.layers[0](h)
        with prefix_transfer(model,bank,0,1,[0],1,random_key='test'):random=model.layers[0](h)
        torch.testing.assert_close(changed[0,0],torch.tensor([0.,5.,4.]))
        torch.testing.assert_close(changed[:,1],h[:,1])
        torch.testing.assert_close(random[:,1],h[:,1])
        torch.testing.assert_close((changed-h).norm(dim=-1),(random-h).norm(dim=-1))
        torch.testing.assert_close(model.layers[0](h),h)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
