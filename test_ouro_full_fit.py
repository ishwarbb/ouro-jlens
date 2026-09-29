"""Check that row partitioning preserves the reference estimator."""
import unittest
import torch
from jlens.fitting import jacobian_for_prompt
import test_ouro_lens
from ouro_full_fit import row_jacobian


class FullFitTests(unittest.TestCase):
    setUp = test_ouro_lens.AdapterTests.setUp
    tearDown = test_ouro_lens.AdapterTests.tearDown

    def test_rows_and_partition_match_reference(self):
        prompt = 'a sufficiently long sequence for Jacobian validation'
        kwargs = dict(dim_batch=3, max_seq_len=24, skip_first=2)
        reference, length, count = jacobian_for_prompt(self.model, prompt,
            [0, 1, 2], target_layer=3, **kwargs)
        whole, got_length, got_count = row_jacobian(self.model, prompt, [0, 1, 2], 3, **kwargs)
        self.assertEqual((got_length, got_count), (length, count))
        for index, source in enumerate([0, 1, 2]):
            torch.testing.assert_close(whole[index], reference[source], rtol=0, atol=0)
        pieces = [row_jacobian(self.model, prompt, [0, 1, 2], 3,
                  row_start=a, row_end=b, **kwargs)[0] for a,b in [(0,7),(7,16)]]
        torch.testing.assert_close(torch.cat(pieces,dim=1), whole, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
