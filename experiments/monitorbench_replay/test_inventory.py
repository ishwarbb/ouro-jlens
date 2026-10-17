"""Checks that a replay inventory preserves samples and rejects false pairs."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.monitorbench_replay.inventory import inventory, render


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'results'
        self.folder = self.root / 'Ouro-2.6B-Thinking_ut4' / 'goal_deception.realworld' / 'inference_results'
        self.folder.mkdir(parents=True)
        self.out = Path(self.temp.name) / 'inventory'
        self.out.mkdir()

    def row(self, condition, base='0', responses=None, loops=4, target='yes'):
        return dict(task_id=f'{base}_{condition}', messages=[dict(role='user', content=f'é {condition}')],
                    target=target, metadata=dict(original_question='same scenario', decision_critical_factor='cue',
                                                 ouro_total_ut_steps=loops),
                    response=responses if responses is not None else ['same output', 'same output'])

    def write(self, condition, rows=None, n=2, suffix=''):
        path = self.folder / f'evaluated_llm_{condition}_n={n}{suffix}.jsonl'
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in (rows or [self.row(condition)])))
        return path

    def scan(self):
        return inventory([self.root], self.out)

    def test_identical_sampled_text_is_retained_and_copies_are_excluded(self):
        self.write('original')
        self.write('standard')
        self.write('standard', suffix='.tested')
        shard = self.folder / 'temp_output_n=2'
        shard.mkdir()
        (shard / 'shard_1.jsonl').write_text(json.dumps(self.row('standard')))
        summary, _, pairs = self.scan()
        self.assertEqual(summary['canonical_files'], 2)
        self.assertEqual(summary['totals']['rollouts'], 4)
        self.assertEqual(pairs[0]['index_pairs'], [0, 1])

    def test_n_strata_are_kept_separate(self):
        for condition in ['original', 'standard']:
            self.write(condition)
            self.write(condition, [self.row(condition, responses=['a', 'b', 'c'])], n=3)
        _, _, pairs = self.scan()
        self.assertEqual(sorted((p['filename_rollouts'], len(p['index_pairs'])) for p in pairs), [(2, 2), (3, 3)])

    def test_missing_original_is_not_manufactured(self):
        self.write('standard')
        summary, _, pairs = self.scan()
        self.assertEqual(pairs, [])
        self.assertEqual(summary['unpaired_reasons'], {'missing_original': 1})

    def test_changed_target_does_not_pair(self):
        self.write('original')
        self.write('standard', [self.row('standard', target='no')])
        summary, _, pairs = self.scan()
        self.assertEqual(pairs, [])
        self.assertEqual(summary['unpaired_reasons'], {'pairing_metadata_changed': 1})

    def test_duplicate_original_ids_are_ambiguous(self):
        self.write('original', [self.row('original'), self.row('original')])
        self.write('standard')
        summary, _, pairs = self.scan()
        self.assertEqual(pairs, [])
        self.assertEqual(summary['unpaired_reasons'], {'ambiguous_duplicate_records': 1})

    def test_invalid_depth_is_retained_but_not_used_in_pairs(self):
        self.write('original')
        self.write('standard', [self.row('standard', loops=8)])
        summary, _, pairs = self.scan()
        self.assertEqual(summary['totals']['rollouts'], 4)
        self.assertEqual(pairs[0]['index_pairs'], [])
        self.assertEqual(summary['issues'][0]['kind'], 'loop_mismatch')

    def test_empty_and_nontext_responses_are_not_negative_readouts(self):
        self.write('original')
        self.write('standard', [self.row('standard', responses=['', None])])
        summary, _, pairs = self.scan()
        self.assertEqual(summary['totals']['rollouts'], 4)
        self.assertEqual(pairs[0]['index_pairs'], [])

    def test_offsets_and_lossy_text_flags_are_recoverable(self):
        path = self.write('original', [self.row('original'), self.row('original', base='1', responses=['lossy \ufffd', 'ok'])])
        self.scan()
        records = [json.loads(line) for line in (self.out / 'prompts.jsonl').read_text().splitlines()]
        self.assertEqual(records[1]['replacement_character_response_indices'], [0])
        with path.open('rb') as stream:
            stream.seek(records[1]['byte_offset'])
            self.assertEqual(json.loads(stream.readline())['task_id'], '1_original')

    def test_custom_inventory_reports_link_to_their_own_artifacts(self):
        self.write('original')
        self.write('standard')
        summary, files, pairs = self.scan()
        (self.out / 'protocol.json').write_text(json.dumps(dict(replay={}, measurement={}, estimands={}, execution_order=[])))
        report = render(summary, files, pairs, self.out)
        self.assertEqual(report, self.out / 'index.html')
        self.assertIn('href="./protocol.json"', report.read_text())
        alternate = Path(self.temp.name) / 'reports' / 'plan.html'
        render(summary, files, pairs, self.out, alternate)
        self.assertIn('href="../inventory/protocol.json"', alternate.read_text())


if __name__ == '__main__':
    unittest.main()
