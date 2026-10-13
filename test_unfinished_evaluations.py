"""Meaningful guards for durable completion queue dependencies and deduplication."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import queue_unfinished_evaluations as queue


class QueueTests(unittest.TestCase):
    def test_original_failed_gate_cannot_release_generation(self):
        self.assertEqual(queue.phase_state({}, {'status': 'failed'}), 'not_eligible_pilot_gate_failed')

    def test_passed_pilot_does_not_replace_scientific_lock(self):
        self.assertEqual(queue.phase_state({}, {'status': 'passed'}, False), 'queued_awaiting_review_and_analysis_lock')

    def test_saved_generations_still_need_factual_reviews(self):
        summary = {'generation_complete': True, 'factual_reviews_pending': 2}
        self.assertEqual(queue.phase_state(summary, {'status': 'passed'}), 'awaiting_factual_review')
        self.assertFalse(queue.analysis_ready(summary))
        self.assertFalse(queue.analysis_ready({'factual_reviews_pending': 0}))

    def test_queue_binding_changes_stop_followups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'frozen.json'
            path.write_text('{}')
            manifest = {'bindings': {'frozen.json': queue.sha(path)}}
            with patch.object(queue, 'ROOT', root):
                queue.verify(manifest)
                path.write_text('{"changed": true}')
                with self.assertRaisesRegex(ValueError, 'Queue binding changed'):
                    queue.verify(manifest)

    def test_cpu_followup_is_deduplicated_and_bound_to_reviews(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run4 = root / 'run4'; run4.mkdir()
            output = root / 'queue'; output.mkdir()
            queue.write(run4 / 'pilot-summary.json', {'generation_complete': True, 'factual_reviews_pending': 0})
            queue.write(run4 / 'fact-reviews.json', {'case': {'reported_status': 'FAILURE'}})
            queue.write(run4 / 'pilot-analysis.json', {'status': 'factual_analysis_complete'})
            manifest = {'items': [{'status': 'queued_cpu_followup', 'phase': 'pilot', 'command': ['cpu-analysis']}]}
            with patch.object(queue, 'RUN4', run4), patch.object(queue, 'ROOT', root), patch.object(queue.subprocess, 'run') as command:
                queue.run_cpu_followups(output, manifest)
                queue.run_cpu_followups(output, manifest)
                self.assertEqual(command.call_count, 1)
                self.assertEqual(command.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
                queue.write(run4 / 'fact-reviews.json', {'case': {'reported_status': 'SUCCESS'}})
                queue.run_cpu_followups(output, manifest)
                self.assertEqual(command.call_count, 2)

    def test_review_ingestion_deduplicates_immutable_source_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'review-submissions/reviewer-a/partial.json'
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps({'packet_sha256': 'packet', 'cases': []}))
            output = root / 'queue'; output.mkdir()
            with patch.object(queue, 'RUN8', root), patch.object(queue, 'ROOT', root), patch('run8_review.ingest_reviews', return_value={'studies': {}}) as ingest:
                self.assertEqual(queue.collect_submissions(output), 1)
                self.assertEqual(queue.collect_submissions(output), 1)
                self.assertEqual(ingest.call_count, 1)


if __name__ == '__main__':
    unittest.main()
