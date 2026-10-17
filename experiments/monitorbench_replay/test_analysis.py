import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .aggregate import conditional, summarize
from .assay import source_positions
from .run import configure_device, enforce_brief_trial, select_precision


def result(words, status='measured'):
    return dict(status=status, supported_words=['hide', 'false'], metrics={
        'jlens/early/loopall': dict(cells=21, thresholds={'10': dict(word_cells={word: 1 for word in words})})})


def pair(baseline, intervention, index=0):
    return dict(task='goal_deception.realworld', model='Ouro-2.6B-Thinking_ut4', condition='standard',
                root_index=0, n=2, base_id='0', baseline=baseline, intervention=intervention, response_index=index)


class AnalysisTests(unittest.TestCase):
    def test_conditional_denominator_allows_hits_in_other_baselines(self):
        results = {'b0': result([]), 'i0': result(['hide']), 'b1': result(['false']), 'i1': result(['hide'])}
        summary = conditional([pair('b0', 'i0'), pair('b1', 'i1', 1)], results, 'jlens/early/loopall')
        self.assertEqual(summary['baseline_clean'], 1)
        self.assertEqual(summary['absent_to_present'], 1)
        self.assertEqual(summary['rate'], 1)
        self.assertEqual(summary['word_specific']['hide']['baseline_word_absent'], 2)
        self.assertEqual(summary['words_from_any_clean_baseline'], {'hide': 1})

    def test_unmeasured_is_not_no_hit(self):
        results = {'b0': result([]), 'i0': result([], status='unmeasured')}
        summary = conditional([pair('b0', 'i0')], results, 'jlens/early/loopall')
        self.assertEqual(summary['baseline_clean'], 0)
        self.assertIsNone(summary['rate'])
        self.assertEqual(summary['unmeasured_pairs'], 1)

    def test_benign_resampling_excludes_the_clean_response_itself(self):
        results = {'b0': result([]), 'i0': result(['hide']), 'b1': result(['false']), 'i1': result([])}
        row = summarize([pair('b0', 'i0'), pair('b1', 'i1', 1)], results)['groups'][0]
        self.assertEqual(row['benign_resampling'], {'gains': 1, 'denominator': 1, 'rate': 1})
        self.assertEqual(row['all_pairs_incentive'], {'gains': 1, 'denominator': 2, 'rate': 0.5})
        self.assertEqual(row['resampling_rate_difference'], -0.5)
        self.assertIsNone(row['bootstrap_95_interval'])

    def test_early_sources_include_prompt_boundary_and_exclude_final_source(self):
        self.assertEqual(source_positions(3, 1), [2])
        self.assertEqual(source_positions(3, 3), [2, 3, 4])
        self.assertEqual(source_positions(3, 100), list(range(2, 23)))
        self.assertEqual(source_positions(3, 100, 'full'), list(range(2, 102)))

    def test_potentially_lossy_text_is_reported_separately(self):
        results = {'b0': result([]), 'i0': result(['hide']), 'b1': result([]), 'i1': result([])}
        results['i0']['replacement_character'] = True
        groups = summarize([pair('b0', 'i0'), pair('b1', 'i1', 1)], results)['groups']
        rates = {r['reconstruction_quality']: r['rate'] for r in groups}
        self.assertEqual(rates, {'replacement_character_in_pair': 1, 'no_replacement_character_in_pair': 0})

    def test_gpu8_is_rejected_and_cpu_hides_all_cuda_devices(self):
        with patch.dict(os.environ, {}, clear=False):
            with self.assertRaisesRegex(ValueError, 'GPU 8'):
                configure_device('cuda', 8)
            configure_device('cpu', None)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '')
            with self.assertRaisesRegex(ValueError, 'explicit'):
                configure_device('cuda', None)

    def test_brief_gpu8_resolves_physical_uuid_and_rejects_large_trials(self):
        with patch.dict(os.environ, {}, clear=False), patch('subprocess.check_output', return_value='GPU-test-device\n') as query:
            configure_device('cuda', 8, brief_gpu8=True)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'GPU-test-device')
            self.assertIn('8', query.call_args.args[0])
        args = SimpleNamespace(brief_gpu8=True, device='cuda', gpu_index=8, window='early', loops=4, max_input_tokens=512)
        enforce_brief_trial(args, 2)
        with self.assertRaises(ValueError):
            enforce_brief_trial(args, 3)
        args.window = 'full'
        with self.assertRaises(ValueError):
            enforce_brief_trial(args, 2)

    def test_unvalidated_bf16_cannot_produce_measurements(self):
        self.assertEqual(select_precision(None, False), 'float32')
        with self.assertRaisesRegex(ValueError, 'require --validate'):
            select_precision('bfloat16', False)
        self.assertEqual(select_precision('bfloat16', True), 'bfloat16')


if __name__ == '__main__':
    unittest.main()
