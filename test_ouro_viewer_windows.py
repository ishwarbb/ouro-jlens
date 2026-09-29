"""Late-token windows preserve exact scores and old residual files remain readable."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import torch
from ouro_viewer import Store,write_json
from ouro_viewer_engine import Engine


class WindowReadoutTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.store=Store(Path(self.temp.name))
        torch.manual_seed(19);self.states={s:torch.randn(600,4) for s in [0,1]};self.weight=torch.randn(32,4)
        self.engine=Engine.__new__(Engine);self.engine.store=self.store
        self.engine.model=SimpleNamespace(hf_model=SimpleNamespace(lm_head=SimpleNamespace(weight=self.weight)),
                                          unembed=lambda h:h@self.weight.T)
        self.engine.matrix=lambda site:torch.eye(4)*1.1 if site==0 else None
        self.engine.family=SimpleNamespace(target_for=lambda site:1)
        self.engine.io=SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda ids:str(ids[0])),
                                      vocab=SimpleNamespace(candidates=lambda word:[2,3]))
        self.run=dict(id='long',title='test',kind='live',created_at='',input={},generation={},layers=[],tokens=list(range(600)),vocabulary={})
        self.store.save(self.run)
        torch.save(dict(states=self.states,seq_ids=list(range(600))),self.store.path('long')/'states.pt')

    def tearDown(self):self.temp.cleanup()

    def test_legacy_and_per_site_storage_late_readouts(self):
        expected=self.engine.readouts('long',512,600,lambda *a,**kw:None)
        self.assertEqual(expected['window'],dict(start=512,end=600,size=512,total=600))
        for row in expected['layers']:
            site=row['site'];self.assertEqual(row['position_start'],512)
            h=self.states[site][512:]* (1.1 if site==0 else 1)
            probs,ids=(h@self.weight.T).softmax(-1).topk(8,dim=-1)
            self.assertEqual(row['jlens']['ids'],ids.tolist())
            torch.testing.assert_close(torch.tensor(row['jlens']['probabilities']),probs,atol=1e-7,rtol=1e-5)
        directory=self.store.path('long')/'states';directory.mkdir()
        for site,state in self.states.items():torch.save(state,directory/f'site-{site:03d}.pt')
        write_json(directory/'index.json',dict(sites=[0,1]))
        (self.store.path('long')/'states.pt').unlink()
        actual=self.engine.window(dict(run_id='long',start=512),lambda *a,**kw:None)
        self.assertEqual(actual,expected)
        self.assertEqual(self.store.window('long',512),expected)
        self.assertEqual(list(self.engine.states('long',1)),[1])

    def test_late_exact_rank_and_probability_and_cache(self):
        payload=dict(run_id='long',word='word',start=512)
        result=self.engine.trace(payload,lambda *a,**kw:None)
        self.assertEqual(result['position_start'],512)
        for site in [0,1]:
            logits=self.states[site][512:]@self.weight.T
            expected=(logits>logits[:,[2,3]].max(-1).values[:,None]).sum(-1)+1
            self.assertEqual(result['layers'][str(site)]['logit']['ranks'],expected.tolist())
            torch.testing.assert_close(torch.tensor(result['layers'][str(site)]['logit']['probabilities']),logits.softmax(-1)[:,[2,3]].sum(-1),atol=1e-7,rtol=1e-5)
        def fail(*a,**kw):raise AssertionError('Cached trace was recomputed')
        self.assertEqual(self.engine.trace(payload,fail),result)
        with self.assertRaises(ValueError):self.engine.window(dict(run_id='long',start=1024),fail)


if __name__=='__main__':unittest.main()
