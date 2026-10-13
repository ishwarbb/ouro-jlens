"""Regression checks for the observed Run 8 crash and evidence-preserving resume."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import report_deception_run7 as report
import run8_recovery as recovery
from prepare_deception_run4 import digest, read, sha
from run_deception_run4 import dump


class RendererRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.case = Path(self.temp.name) / 'cases/benchmark'
        self.case.mkdir(parents=True)

    def generation(self, turn):
        path = self.case / f'turn-{turn}/generation.json'
        dump(path, dict(id='benchmark', phase='benchmark', turn=turn, complete=True,
                       messages=[dict(role='user', content='Choose a task action.')],
                       raw_text='CHOICE: A', assistant_history_text='CHOICE: A', artifacts={}))
        return path

    def mocked_readout(self, path, tokenizer):
        return dict(turn=read(path)['turn'], available=False, generation_sha256=sha(path))

    def test_numbered_paths_sort_numerically_and_diagnostics_remain_separate(self):
        for name in ('tool-10.json', 'tool-2.json', 'tool-result.json', 'tool-2-extra.json', 'tool--1.json'):
            dump(self.case / name, {})
        self.assertEqual([p.name for p in recovery.numbered_tool_paths(self.case)],
                         ['tool-2.json', 'tool-10.json'])
        details = ''.join(recovery.diagnostic_tool_details(self.case))
        for name in ('tool-result.json', 'tool-2-extra.json', 'tool--1.json'):
            self.assertIn(name, details)

    def test_actual_crash_then_corrected_view_retains_evidence_and_numbered_order(self):
        self.generation(0)
        dump(self.case / 'tool-10.json', dict(result=dict(observed='ten')))
        dump(self.case / 'tool-2.json', dict(result=dict(observed='two')))
        dump(self.case / 'tool-result.json', dict(observed_result='DIAGNOSTIC_RESULT', artifact=dict(a=42)))
        with patch.object(report, 'explorer_data', self.mocked_readout):
            with self.assertRaisesRegex(ValueError, 'result'):
                report._case_view(self.case, None, None)
            original = report._case_view
            with recovery.preserve_renderer():
                receipt = report._case_view(self.case, None, None)
            self.assertIs(report._case_view, original)
        html = (self.case / 'view.html').read_text()
        self.assertLess(html.index('Actual tool event tool-2'), html.index('Actual tool event tool-10'))
        self.assertIn('Saved tool evidence tool-result.json (not a numbered event)', html)
        self.assertIn('DIAGNOSTIC_RESULT', html)
        self.assertIn('href="tool-result.json"', html)
        self.assertEqual(receipt['renderer_version'], recovery.ADAPTER_VERSION)

    def test_diagnostic_change_invalidates_derived_cache_for_partial_case(self):
        self.generation(0)
        dump(self.case / 'tool-result.json', dict(observed_result='first'))
        with patch.object(report, 'explorer_data', self.mocked_readout), recovery.preserve_renderer():
            first = report._case_view(self.case, None, None)
            dump(self.case / 'tool-result.json', dict(observed_result='second'))
            second = report._case_view(self.case, None, None)
        self.assertNotEqual(first['source_digest'], second['source_digest'])
        self.assertNotEqual(first['view_sha256'], second['view_sha256'])
        self.assertIn('second', (self.case / 'view.html').read_text())

    def test_saved_diagnostic_artifact_still_rejects_tampering(self):
        self.generation(0)
        dump(self.case / 'tool-result.json', dict(observed_result='first'))
        episode = dict(id='benchmark', phase='benchmark', complete=True,
                       artifacts={'tool-result.json': sha(self.case / 'tool-result.json')})
        dump(self.case / 'episode.json', episode)
        dump(self.case / 'tool-result.json', dict(observed_result='tampered'))
        with patch.object(report, 'explorer_data', self.mocked_readout), recovery.preserve_renderer():
            with self.assertRaisesRegex(AssertionError, 'Changed episode artifact'):
                report._case_view(self.case, episode, None)

    def test_diagnostic_untrusted_content_is_escaped(self):
        dump(self.case / 'tool-result.json', dict(observed_result='<script>window.BAD=1</script>'))
        details = ''.join(recovery.diagnostic_tool_details(self.case))
        self.assertNotIn('<script>', details)
        self.assertIn('&lt;script&gt;', details)

    def test_renderer_adapter_restores_original_on_exception(self):
        original = report._case_view
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with recovery.preserve_renderer():
                raise RuntimeError('fixture')
        self.assertIs(report._case_view, original)

    def test_regular_case_keeps_original_cache_without_rebuilding_readouts(self):
        self.generation(0)
        dump(self.case / 'tool-0.json', dict(result=dict(observed_result='ordinary')))
        with patch.object(report, 'explorer_data', self.mocked_readout):
            original_receipt = report._case_view(self.case, None, None)
        before = {p: p.read_bytes() for p in (self.case / 'view.html', self.case / 'view-provenance.json')}
        self.assertEqual(original_receipt['renderer_version'], report.REPORT_VERSION)
        with patch.object(report, 'explorer_data', side_effect=AssertionError('unaffected case rebuilt')), recovery.preserve_renderer():
            reused_receipt = report._case_view(self.case, None, None)
        self.assertEqual(reused_receipt, original_receipt)
        self.assertEqual({p: p.read_bytes() for p in before}, before)


class ResumeEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.root = self.project / 'results/run8'
        self.generation = self.root / 'cases/job/turn-0/generation.json'
        self.generation.parent.mkdir(parents=True)
        state = self.generation.parent / 'states.pt'
        state.write_bytes(b'saved exact residuals')
        dump(self.generation, dict(artifacts={'states.pt': sha(state)}))
        self.episode = self.generation.parent.parent / 'episode.json'
        dump(self.episode, dict(artifacts={'turn-0/generation.json': sha(self.generation),
                                        'turn-0/states.pt': sha(state)}))
        self.inventory = dict(files={str(p.relative_to(self.project)): sha(p)
             for p in (self.generation, state, self.episode)}, execution_sha256='original')
        self.root_patch = patch.object(recovery, 'ROOT', self.project)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def test_inventory_covers_saved_generations_episodes_and_source_artifacts(self):
        result = recovery.verify_source_inventory(self.root, self.inventory)
        self.assertEqual(result, dict(source_files=3, required_source_files=3))

    def test_missing_referenced_source_is_rejected(self):
        broken = copy.deepcopy(self.inventory)
        broken['files'].pop(str((self.generation.parent / 'states.pt').relative_to(self.project)))
        with self.assertRaisesRegex(ValueError, 'artifact missing or mismatched'):
            recovery.verify_source_inventory(self.root, broken)

    def test_changed_preserved_source_is_rejected(self):
        (self.generation.parent / 'states.pt').write_bytes(b'changed residuals')
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            recovery.verify_source_inventory(self.root, self.inventory)

    def test_orphan_artifacts_require_explicit_recovery_and_empty_dir_is_safe(self):
        orphan = self.root / 'cases/job/turn-1'
        orphan.mkdir()
        recovery.require_no_orphan_generations(self.root)
        (orphan / 'states.pt').write_bytes(b'unsaved draw')
        with self.assertRaisesRegex(ValueError, 'Unsaved generation artifacts'):
            recovery.require_no_orphan_generations(self.root)

    def test_generation_guard_detects_a_reroll_of_saved_output(self):
        import run7_generation
        before = sha(self.generation)

        def erroneous_reroll(*args, **kwargs):
            dump(self.generation, dict(raw_text='rerolled output'))
            return read(self.generation), self.generation

        with patch.object(run7_generation, 'generate', erroneous_reroll):
            with recovery.guard_saved_generations(self.root, self.inventory):
                with self.assertRaisesRegex(ValueError, 'changed during reuse'):
                    run7_generation.generate(self.root, dict(id='job'), 0)
        self.assertNotEqual(sha(self.generation), before)

    def test_generation_guard_reuses_saved_output_unchanged(self):
        import run7_generation
        import run_deception_run8
        before = self.generation.read_bytes()

        def reuse(*args, **kwargs):
            return read(self.generation), self.generation

        with patch.object(run7_generation, 'generate', reuse):
            with recovery.guard_saved_generations(self.root, self.inventory):
                run_deception_run8.generate(self.root, dict(id='job'), 0)
        self.assertEqual(self.generation.read_bytes(), before)

    def test_project_binding_cannot_escape(self):
        with self.assertRaisesRegex(ValueError, 'escapes the project'):
            recovery.project_path('../outside')

    def test_original_benchmark_resume_skips_model_generation_and_scoring(self):
        import run_deception_run8 as run8
        job = dict(id='job', phase='benchmark')
        episode = read(self.episode)
        episode.update(job_sha256=digest(job), revision_execution_sha256='original')
        dump(self.episode, episode)
        before = {p: p.read_bytes() for p in (self.episode, self.generation, self.generation.parent / 'states.pt')}
        with patch.object(run8, 'generate', side_effect=AssertionError('saved episode rerolled')):
            actual = run8.benchmark_episode(self.root, job, None, None, None, 'original', None, 'greedy')
        self.assertEqual(actual, episode)
        self.assertEqual({p: p.read_bytes() for p in before}, before)

    def test_original_benchmark_resume_rejects_a_changed_source_artifact(self):
        import run_deception_run8 as run8
        job = dict(id='job', phase='benchmark')
        episode = read(self.episode)
        episode.update(job_sha256=digest(job), revision_execution_sha256='original')
        dump(self.episode, episode)
        (self.generation.parent / 'states.pt').write_bytes(b'changed residuals')
        with self.assertRaises(AssertionError):
            run8.benchmark_episode(self.root, job, None, None, None, 'original', None, 'greedy')


class NumericalReceiptTests(unittest.TestCase):
    def test_reuse_retains_original_receipt_bytes_and_does_not_call_checker(self):
        import analyze_evaluation_revisions as analysis
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = dict(cases=['fixed-spec'], minimum_state_cosine=.99)
            dump(root / 'plan/numerical-checks.json', spec)
            receipt = dict(status='passed', revision_execution_sha256='original', tolerances=spec,
                           checks=[dict(passed=True) for _ in range(11)])
            dump(root / 'numerical-verification.json', receipt)
            path = root / 'numerical-verification.json'
            before = path.read_bytes()
            original = analysis.numerical_validation
            with recovery.preserve_numerical_receipt(root, 'original', sha(path)):
                self.assertEqual(analysis.numerical_validation(root, None, None, 'original'), receipt)
            self.assertIs(analysis.numerical_validation, original)
            self.assertEqual(path.read_bytes(), before)

    def test_wrong_execution_and_failed_numerical_receipt_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = dict(cases=['fixed-spec'])
            dump(root / 'plan/numerical-checks.json', spec)
            receipt = dict(status='passed', revision_execution_sha256='different', tolerances=spec,
                           checks=[dict(passed=True) for _ in range(11)])
            path = root / 'numerical-verification.json'
            dump(path, receipt)
            with self.assertRaisesRegex(ValueError, 'eleven passed production checks'):
                recovery.verify_numerical_receipt(root, 'original', sha(path))
            receipt.update(revision_execution_sha256='original', status='failed')
            dump(path, receipt)
            with self.assertRaisesRegex(ValueError, 'eleven passed production checks'):
                recovery.verify_numerical_receipt(root, 'original', sha(path))


if __name__ == '__main__':
    unittest.main()
