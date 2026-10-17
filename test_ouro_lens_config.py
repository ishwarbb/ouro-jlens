"""Changing the default must never mix lenses inside saved viewer runs."""
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch

import ouro_lens_config as config


class LensSelectionTests(TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.author = Path(self.directory.name) / 'author'
        self.legacy = Path(self.directory.name) / 'legacy'
        self.hashes = {}
        for path in (self.author, self.legacy):
            path.mkdir()
            data = path.name.encode()
            (path / 'family.json').write_bytes(data)
            self.hashes[path] = hashlib.sha256(data).hexdigest()
        for name, value in [('DEFAULT_FAMILY', self.author), ('LEGACY_FAMILY', self.legacy)]:
            context = patch.object(config, name, value)
            context.start()
            self.addCleanup(context.stop)

    def test_saved_runs_keep_their_own_family(self):
        for path in (self.author, self.legacy):
            run = dict(provenance=dict(family_sha256=self.hashes[path]))
            self.assertEqual(config.family_for_run(run), path)

    def test_historical_research_import_uses_original_fit(self):
        run = dict(kind='research', provenance=dict(source='workspace-inventory-v2'))
        self.assertEqual(config.family_for_run(run), self.legacy)

    def test_missing_or_changed_provenance_cannot_fall_back_to_default(self):
        for run in [{}, dict(provenance=dict(family_sha256='unknown'))]:
            with self.assertRaises(ValueError):
                config.family_for_run(run)
        (self.author / 'family.json').write_text('changed')
        with self.assertRaises(ValueError):
            config.family_for_run(dict(provenance=dict(family_sha256=self.hashes[self.author])))

    def test_viewer_switches_between_saved_followups_and_new_work(self):
        from types import SimpleNamespace
        import torch
        from ouro_viewer_engine import Engine
        legacy = dict(provenance=dict(family_sha256=self.hashes[self.legacy]), input={})
        engine = Engine.__new__(Engine)
        engine.store = SimpleNamespace(get=lambda run_id: legacy)
        engine.preference = 'cpu'
        engine.model = SimpleNamespace(hf_model=SimpleNamespace(lm_head=SimpleNamespace(weight=torch.zeros(1))))
        selected = []
        engine.select_family = selected.append
        engine.trace = lambda *args: 'saved trace'
        engine.window = lambda *args: 'saved window'
        engine.analyze = lambda *args, **kwargs: 'new analysis'
        with patch('ouro_viewer_engine.FAMILY', self.author):
            for kind in ('trace', 'analyze', 'window', 'intervene'):
                engine.execute(dict(kind=kind, run_id='saved'), lambda *args: None)
        self.assertEqual(selected, [self.legacy, self.author, self.legacy, self.author])


if __name__ == '__main__':
    main()
