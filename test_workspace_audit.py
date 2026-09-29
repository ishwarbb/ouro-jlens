import unittest
import torch
from workspace_audit import delta_for, conditions, tasks
from workspace_audit_analysis import hit, wilson


class AuditTests(unittest.TestCase):
    def test_transfer_is_not_coordinate_exchange(self):
        h=torch.tensor([[2.,3.,4.]])
        a=torch.tensor([1.,0.,0.]);b=torch.tensor([0.,1.,0.])
        torch.testing.assert_close(h+delta_for(h,a,b,'transfer'),torch.tensor([[0.,5.,4.]]))
        torch.testing.assert_close(h+delta_for(h,a,b,'swap'),torch.tensor([[3.,2.,4.]]))
        torch.testing.assert_close(h+delta_for(h,a,b,'replace_target'),torch.tensor([[0.,2.,4.]]))

    def test_dual_clamp_enforces_constraints_for_correlated_directions(self):
        a=torch.tensor([1.,0.,0.]);b=torch.tensor([.6,.8,0.]);h=torch.tensor([[2.,3.,4.],[-1.,2.,0.]])
        after=h+delta_for(h,a,b,'dual_clamp')
        torch.testing.assert_close(after@a,torch.zeros(2),atol=1e-6,rtol=0)
        torch.testing.assert_close(after@b,h@a)
        torch.testing.assert_close(after[:,2],h[:,2])

    def test_all_released_affected_items_and_loop_subsets(self):
        primary=[t for t in tasks() if t['profile']=='primary']
        expected={'flexible':64,'probe':90,'report':14,'language':8,'linecount':11}
        for family,n in expected.items():self.assertEqual(sum(t['family']==family for t in primary),n)
        self.assertEqual(len({t['id'] for t in tasks()}),len(tasks()))
        for key in ['loops_2','loops_3','loops_4','loops_23','loops_24','loops_34','transfer']:
            self.assertIn(key,conditions())

    def test_readout_does_not_leak_final_loop_into_cross_loop_comparison(self):
        reads={'168:own':{'x':{'hit5':True}},'72:own':{'x':{'hit5':False}}}
        self.assertFalse(hit(reads,'x',loops=(2,3)))
        self.assertTrue(hit(reads,'x',loops=(2,3,4)))
        self.assertFalse(hit(reads,'x',mode='final'))

    def test_wilson_zero_and_full_are_bounded(self):
        for k in [0,8]:
            lo,hi=wilson(k,8);self.assertGreaterEqual(lo,-1e-12);self.assertLessEqual(hi,1+1e-12)


if __name__=='__main__':unittest.main()
