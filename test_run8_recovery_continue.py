"""Only report publication cadence changes in the continuation adapter."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import report_deception_run8 as report
import run8_recovery_continue as continuation
from run_deception_run4 import dump


class ContinuationCadenceTests(unittest.TestCase):
    def test_reused_core_and_first_benchmark_do_not_rebuild_gallery(self):
        renderer = Mock()
        with patch.object(report, 'render', renderer), continuation.report_cadence(initial=1):
            for extra in (None, dict(stage='measurement'), dict(stage='calibration'),
                          dict(stage='disjoint_validation'), dict(stage='knowledge'),
                          dict(main_saved=96, benchmark_saved=0), dict(benchmark_saved=1)):
                self.assertIsNone(report.render(Path('/unused'), extra=extra, tokenizer=None))
        renderer.assert_not_called()

    def test_sixteen_fresh_cases_publish_and_final_status_always_publishes(self):
        renderer = Mock(return_value='published')
        selected = []
        with patch.object(report, 'render', renderer), continuation.report_cadence(initial=1):
            for count in range(2, 289):
                result = report.render(Path('/unused'), extra=dict(benchmark_saved=count), tokenizer='same-tokenizer')
                if result == 'published':
                    selected.append(count)
            self.assertEqual(report.render(Path('/unused'), extra=dict(benchmark_saved=288, terminal=True)), 'published')
        self.assertEqual(selected, list(range(17, 274, 16)) + [288])
        self.assertEqual(renderer.call_count, len(selected) + 1)
        self.assertEqual(renderer.call_args.kwargs['extra'], dict(benchmark_saved=288, terminal=True))
        self.assertTrue(all(call.kwargs.get('tokenizer') == 'same-tokenizer'
                            for call in renderer.call_args_list[:-1]))

    def test_renderer_failure_propagates_and_context_restores_original(self):
        renderer = Mock(side_effect=RuntimeError('publication failed'))
        with patch.object(report, 'render', renderer):
            with self.assertRaisesRegex(RuntimeError, 'publication failed'):
                with continuation.report_cadence(initial=1):
                    report.render(Path('/unused'), extra=dict(benchmark_saved=17))
            self.assertIs(report.render, renderer)

    def test_explicit_report_mode_bypasses_cadence(self):
        renderer = Mock(return_value='immediate report')

        def base_execute(root, manifest, mode):
            self.assertEqual(mode, 'report')
            return report.render(root, extra=dict(benchmark_saved=1))

        with patch.object(report, 'render', renderer), \
                patch.object(continuation, 'read', return_value={}), \
                patch.object(continuation, 'validate_policy', return_value=dict(initial=1, every=16, total=288)), \
                patch.object(continuation.base, 'execute', side_effect=base_execute):
            self.assertEqual(continuation.execute(Path('/unused'), Path('/manifest'), mode='report'), 'immediate report')
        renderer.assert_called_once()

    def test_policy_matches_bound_inventory_plan_and_code_bindings(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            root = project / 'results/run8'
            jobs = [dict(id=f'benchmark-{i}') for i in range(288)]
            dump(root / 'plan/tasks.json', dict(benchmark=jobs))
            inventory = project / 'inventory.json'
            dump(inventory, dict(files={'results/run8/cases/benchmark-0/episode.json': 'bound-hash'}))
            manifest = dict(adapter_version=continuation.base.ADAPTER_VERSION,
                continuation_adapter_version=continuation.CONTINUATION_VERSION,
                reporting_every=16, saved_benchmark_episodes=1,
                code_sha256={'run8_recovery_continue.py': 'bound', 'test_run8_recovery_continue.py': 'bound'},
                source_inventory=dict(path='inventory.json'))
            with patch.object(continuation.base, 'ROOT', project):
                self.assertEqual(continuation.validate_policy(root, manifest), dict(initial=1, every=16, total=288))
                for overrides in (dict(saved_benchmark_episodes=2), dict(reporting_every=1),
                                  dict(reporting_every=True), dict(continuation_adapter_version='unknown'),
                                  dict(code_sha256={})):
                    with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                        continuation.validate_policy(root, dict(manifest, **overrides))


if __name__ == '__main__':
    unittest.main()
