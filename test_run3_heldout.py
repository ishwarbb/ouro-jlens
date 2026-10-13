import inspect
from pathlib import Path
import tempfile
import unittest

import analyze_deception_run3_heldout as analysis
import replay_deception_run3 as original
import replay_deception_run3_heldout as heldout
from run_deception_run3 import dump


class FrozenAnalysisTests(unittest.TestCase):
    def test_heldout_uses_identical_registered_window_function(self):
        self.assertEqual(inspect.getsource(original.windows),inspect.getsource(heldout.windows))

    def test_missing_and_incomplete_do_not_change_attempted_denominator(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);rows={}
            for record in ('handoff-04','handoff-05','evidence-04','evidence-05'):
                for condition in ('accuracy','aligned','conflict','conflict_verified'):
                    for sample in range(6):
                        case=f'{record}-{condition}-r{sample}'
                        candidate=condition=='conflict' and sample==0
                        rows[case]=dict(primary_candidate=candidate,complete=sample<2,
                            label=analysis.PRIMARY if candidate else 'truthful' if sample<2 else 'incomplete',
                            borderline=record=='handoff-04' and candidate)
                        dump(root/'cases'/case/'generation.json',dict(record_id=record,condition=condition))
            result=analysis.stats(root,{'cases':rows})
            self.assertAlmostEqual(result['endpoints'][0]['mean'],1/6)
            self.assertEqual(result['records'][0]['complete_only_rates']['conflict'],.5)
            self.assertIsNone(result['endpoints'][2]['mean'])
            self.assertIsNone(result['endpoints'][2]['p_holm'])
            self.assertFalse(result['causal_stage_eligible'])
            self.assertTrue(result['replication_passes'])
            sensitivity=analysis.stats(root,{'cases':rows},exclude_borderline=True)
            self.assertAlmostEqual(sensitivity['endpoints'][0]['mean'],1/8)
            self.assertFalse(sensitivity['replication_passes'])


if __name__=='__main__':unittest.main()
