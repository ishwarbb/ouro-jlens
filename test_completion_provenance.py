import unittest
from unittest.mock import patch
import completion_review_provenance as provenance
import run8_review


class ProvenanceTests(unittest.TestCase):
    def test_historical_inventory_preserves_original_job_order(self):
        cases = [{'id': 'a', 'generations': [{'sha256': 'hash-a'}]},
                 {'id': 'b', 'generations': [{'sha256': 'hash-b'}]}]
        value = provenance.historical_inventory({'cases': cases}, [{'id': 'b'}, {'id': 'other'}, {'id': 'a'}])
        import json
        self.assertEqual([row['id'] for row in json.loads(value)], ['b', 'a'])

    def test_missing_historical_job_is_rejected(self):
        with self.assertRaises(ValueError):
            provenance.historical_inventory({'cases': [{'id': 'a', 'generations': [{'sha256': 'hash-a'}]}]}, [])

    def test_only_documented_derived_source_is_adapted(self):
        packet = {'source_files': {provenance.EVOLVED_SOURCE: 'old', 'raw.json': 'raw'}}
        original_inventory = dict(packet['source_files'])
        with patch.object(run8_review, 'verify_sources', return_value=True) as original, patch.object(provenance, 'verify_evolution') as check:
            with provenance.preserve_handoff_sources({'historical_sha256': 'old'}):
                run8_review.verify_sources(packet)
            check.assert_called_once()
            self.assertEqual(original.call_args.args[0]['source_files'], {'raw.json': 'raw'})
            self.assertEqual(packet['source_files'], original_inventory)
            self.assertIs(run8_review.verify_sources, original)

    def test_wrong_historical_hash_is_rejected(self):
        packet = {'source_files': {provenance.EVOLVED_SOURCE: 'other'}}
        with patch.object(provenance, 'verify_evolution') as check:
            with provenance.preserve_handoff_sources({'historical_sha256': 'old'}):
                with self.assertRaises(ValueError):
                    run8_review.verify_sources(packet)
            check.assert_not_called()

    def test_failed_evolution_proof_does_not_bypass_original_validation(self):
        packet = {'source_files': {provenance.EVOLVED_SOURCE: 'old'}}
        with patch.object(run8_review, 'verify_sources', return_value=True) as original, patch.object(provenance, 'verify_evolution', side_effect=ValueError('changed annotation')):
            with provenance.preserve_handoff_sources({'historical_sha256': 'old'}):
                with self.assertRaises(ValueError):
                    run8_review.verify_sources(packet)
            original.assert_not_called()
            self.assertIs(run8_review.verify_sources, original)


if __name__ == '__main__':
    unittest.main()
