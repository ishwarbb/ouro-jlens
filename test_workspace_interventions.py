import unittest
import torch
from workspace_interventions import swap_delta,transfer_delta,score_tokens


class InterventionTests(unittest.TestCase):
    def test_swap_matches_pseudoinverse_for_correlated_directions(self):
        torch.manual_seed(42)
        a=torch.randn(32,dtype=torch.float64);a/=a.norm()
        b=0.8*a+0.2*torch.randn(32,dtype=torch.float64);b/=b.norm()
        h=torch.randn(2,7,32,dtype=torch.float64)
        v=torch.stack([a,b],dim=1)
        coordinates=h@torch.linalg.pinv(v).T
        expected=(coordinates.flip(-1)-coordinates)@v.T
        delta=swap_delta(h,a,b)
        torch.testing.assert_close(delta,expected)
        torch.testing.assert_close((h+delta)@a,h@b)
        torch.testing.assert_close((h+delta)@b,h@a)
        torch.testing.assert_close((h+delta).norm(dim=-1),h.norm(dim=-1))
        torch.testing.assert_close(swap_delta(h,a,a),torch.zeros_like(h))

    def test_transfer_has_declared_projection(self):
        h=torch.tensor([[2.,3.,4.]])
        a=torch.tensor([1.,0.,0.]);b=torch.tensor([0.,1.,0.])
        torch.testing.assert_close(h+transfer_delta(h,a,b),torch.tensor([[0.,5.,4.]]))

    def test_rank_does_not_confuse_ties_with_actual_top1(self):
        logits=torch.tensor([3.,3.,1.,0.,-1.,-2.])
        result=score_tokens(logits,[1])
        self.assertEqual(result['rank'],1)
        self.assertFalse(result['top1'])
        self.assertTrue(score_tokens(logits,[0])['top1'])
        self.assertFalse(score_tokens(logits,[])['eligible'])


if __name__=='__main__':unittest.main()
