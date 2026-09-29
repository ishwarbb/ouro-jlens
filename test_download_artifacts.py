"""Release restoration must preserve bytes, reject corruption, and confine paths."""
import hashlib
import io
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from download_artifacts import restore


class ArtifactTests(unittest.TestCase):
    def bundle(self, root, name='results/example/tensor.pt', data=b'original tensor bytes' * 100):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w:gz') as archive:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        payload = stream.getvalue()
        cut = len(payload) // 3
        parts = []
        for i, chunk in enumerate([payload[:cut], payload[cut:cut*2], payload[cut*2:]]):
            path = root / f'example.part-{i}'
            path.write_bytes(chunk)
            parts.append(dict(name=path.name, bytes=len(chunk), sha256=hashlib.sha256(chunk).hexdigest()))
        return dict(parts=parts, files=[dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())]), data

    def test_split_archive_and_identical_existing_file(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); bundle, data = self.bundle(root); output = root / 'output'
            restore(bundle, root, output)
            self.assertEqual((output / 'results/example/tensor.pt').read_bytes(), data)
            restore(bundle, root, output)

    def test_corruption_rejected_before_extracting(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); bundle, _ = self.bundle(root); output = root / 'output'
            part = root / bundle['parts'][0]['name']
            part.write_bytes(b'x' * part.stat().st_size)
            with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
                restore(bundle, root, output)
            self.assertFalse(output.exists())

    def test_conflicting_existing_file_preserved(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); bundle, _ = self.bundle(root); output = root / 'output'
            target = output / 'results/example/tensor.pt'
            target.parent.mkdir(parents=True); target.write_bytes(b'local work')
            with self.assertRaises(FileExistsError):
                restore(bundle, root, output)
            self.assertEqual(target.read_bytes(), b'local work')

    def test_archive_cannot_escape_destination(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); bundle, _ = self.bundle(root, name='../outside.pt')
            with self.assertRaisesRegex(ValueError, 'Unexpected archive member'):
                restore(bundle, root, root / 'output')
            self.assertFalse((root / 'outside.pt').exists())

    def test_wrong_extracted_hash_leaves_no_partial_file(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); bundle, _ = self.bundle(root); output = root / 'output'
            bundle['files'][0]['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'Extracted file checksum mismatch'):
                restore(bundle, root, output)
            self.assertFalse(list(output.rglob('*.pt')))
            self.assertFalse(list(output.rglob('*.tmp')))


if __name__ == '__main__':
    unittest.main()
