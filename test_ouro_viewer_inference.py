"""Cached recurrence must agree with full forwards, including prompt-only edits."""
from contextlib import nullcontext
import unittest
import torch
from jlens.hooks import ActivationRecorder
import test_ouro_lens
from ouro_viewer_inference import CachedDecoder,prefix_edit
from workspace_validated import prefix_transfer


class CachedInferenceTests(unittest.TestCase):
    def setUp(self):
        self.fixture=test_ouro_lens.AdapterTests();self.fixture.setUp();self.model=self.fixture.model

    def tearDown(self):self.fixture.tearDown()

    def compare(self,edited=False,erase=False):
        ids=self.fixture.ids[0].tolist();prefix=9
        directions={}
        for site in [1,3]:
            for token in [4,5]:
                vector=torch.randn(self.model.d_model);directions[site,token]=vector/vector.norm()
        class Bank:
            def get(self,site,token):return directions[site,token]
        bank=Bank();decoder=CachedDecoder(self.model)
        expected=[]
        with torch.inference_mode():
            for end in [4,9,10,11]:
                # Prefix edits during prefill apply only to available prompt positions.
                ctx=prefix_edit(self.model,bank,4,None if erase else 5,[1,3],prefix,lambda:0) if edited else nullcontext()
                with ctx,ActivationRecorder(self.model.layers,at=range(self.model.n_layers)) as recorder:
                    self.model.forward(torch.tensor([ids[:end]]))
                logits=self.model.unembed(recorder.activations[self.model.n_layers-1])[0,-1]
                expected.append((logits.clone(),torch.stack([recorder.activations[s][0].clone() for s in range(self.model.n_layers)])))
            ctx=prefix_edit(self.model,bank,4,None if erase else 5,[1,3],prefix,lambda:decoder.position) if edited else nullcontext()
            with ctx,decoder:
                start=0
                for end,(logits,states) in zip([4,9,10,11],expected):
                    actual,captured=decoder.forward(ids[start:end])
                    torch.testing.assert_close(actual,logits,atol=2e-6,rtol=2e-5)
                    torch.testing.assert_close(captured,states[:,start:end],atol=2e-6,rtol=2e-5)
                    self.assertEqual(decoder.position,end)
                    for layer in decoder.cache.layers:self.assertEqual(layer.keys.shape[-2],end)
                    start=end
        self.assertTrue(all(not layer._forward_hooks for layer in self.model.layers))
        self.assertIsNone(decoder.cache)

    def test_chunked_prefill_and_decode_all_recurrent_sites(self):self.compare()
    def test_transfer_cached_at_original_positions(self):self.compare(edited=True)
    def test_erase_cached_at_original_positions(self):self.compare(edited=True,erase=True)

    def test_transfer_matches_frozen_intervention(self):
        vector=torch.randn(2,self.model.d_model);vector/=vector.norm(dim=1,keepdim=True)
        class Bank:
            def get(self,site,token):return vector[token]
        with torch.inference_mode():
            with prefix_transfer(self.model,Bank(),0,1,[1,3],9),ActivationRecorder(self.model.layers,at=[5]) as r:
                self.model.forward(self.fixture.ids)
            expected=r.activations[5].clone()
            with prefix_edit(self.model,Bank(),0,1,[1,3],9,lambda:0),ActivationRecorder(self.model.layers,at=[5]) as r:
                self.model.forward(self.fixture.ids)
            torch.testing.assert_close(r.activations[5],expected)

    def test_failure_releases_hooks_and_cache(self):
        with self.assertRaisesRegex(RuntimeError,'stop'),CachedDecoder(self.model) as decoder:
            decoder.forward([1,2]);raise RuntimeError('stop')
        self.assertTrue(all(not layer._forward_hooks for layer in self.model.layers))
        self.assertIsNone(decoder.cache)
        self.assertFalse(self.model._active)


if __name__=='__main__':unittest.main()
