import unittest
from statistics_deception_run3 import paired_endpoint, holm_three


class PairedTests(unittest.TestCase):
    def test_missing_is_not_a_zero_effect(self):
        self.assertIsNone(paired_endpoint([])['mean'])
        self.assertEqual(paired_endpoint([0]*12)['p_two_sided'],1)
        self.assertEqual(holm_three([.01,None,.03]),[.03,None,.06])

    def test_exact_small_sample_and_direction(self):
        self.assertEqual(paired_endpoint([1,1])['p_two_sided'],.5)
        self.assertEqual(paired_endpoint([1]*12)['p_two_sided'],2/4096)
        self.assertFalse(paired_endpoint([-1]*12)['positive'])
        self.assertEqual(paired_endpoint([1,-1])['mean'],0)


if __name__=='__main__':unittest.main()
