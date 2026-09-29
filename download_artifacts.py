"""Download and restore checksum-verified tensors from this repository's release.

Python's standard library handles extraction. Private release downloads use
the authenticated GitHub CLI; --from-dir can restore an offline copy instead.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


class Parts(io.RawIOBase):
    """Expose ordered archive parts as one sequential byte stream."""
    def __init__(self, paths):
        super().__init__()
        self.paths = iter(paths)
        self.current = None

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    return 0
                self.current = path.open('rb')
            count = self.current.readinto(buffer)
            if count:
                return count
            self.current.close()
            self.current = None

    def close(self):
        if self.current is not None:
            self.current.close()
        super().close()


def restore(bundle, directory, destination):
    paths = []
    for part in bundle['parts']:
        if Path(part['name']).name != part['name']:
            raise ValueError('Invalid archive part name')
        path = directory / part['name']
        if path.stat().st_size != part['bytes'] or digest(path) != part['sha256']:
            raise ValueError(f'Checksum mismatch: {path}. Remove the corrupt part and retry.')
        paths.append(path)
    expected = {entry['path']: entry for entry in bundle['files']}
    seen = set()
    destination = destination.resolve()
    with Parts(paths) as raw, io.BufferedReader(raw) as joined, tarfile.open(fileobj=joined, mode='r|gz') as archive:
        for member in archive:
            relative = PurePosixPath(member.name)
            if (not member.isfile() or member.name not in expected or member.name in seen
                    or relative.is_absolute() or '..' in relative.parts):
                raise ValueError(f'Unexpected archive member: {member.name}')
            record = expected[member.name]
            if member.size != record['bytes']:
                raise ValueError(f'Archive size mismatch: {member.name}')
            path = destination.joinpath(*relative.parts)
            if not path.resolve().is_relative_to(destination):
                raise ValueError(f'Archive member escapes destination: {member.name}')
            seen.add(member.name)
            if path.exists():
                if path.stat().st_size == record['bytes'] and digest(path) == record['sha256']:
                    continue
                raise FileExistsError(f'Preserving different existing file: {path}')
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as out:
                    temporary = Path(out.name)
                    value = hashlib.sha256()
                    stream = archive.extractfile(member)
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                        out.write(block)
                        value.update(block)
                if value.hexdigest() != record['sha256']:
                    raise ValueError(f'Extracted file checksum mismatch: {member.name}')
                temporary.replace(path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            print('Restored', member.name, flush=True)
    if seen != set(expected):
        raise ValueError('Archive is missing expected files')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', action='append', choices=['lenses', 'inventories', 'viewer', 'pilot', 'all'], required=True)
    parser.add_argument('--from-dir', type=Path, help='Use downloaded parts from this directory without network access')
    parser.add_argument('--destination', type=Path, default=ROOT)
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'artifacts.json').read_text())
    names = list(manifest['bundles']) if 'all' in args.bundle else list(dict.fromkeys(args.bundle))
    directory = args.from_dir or ROOT / '.artifacts' / manifest['release']
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        bundle = manifest['bundles'][name]
        if args.from_dir is None:
            if shutil.which('gh') is None:
                parser.error('Install GitHub CLI and run gh auth login, or use --from-dir with downloaded release parts.')
            command = ['gh', 'release', 'download', manifest['release'], '--repo', manifest['repository'],
                       '--dir', str(directory), '--skip-existing']
            for part in bundle['parts']:
                command.extend(['--pattern', part['name']])
            subprocess.run(command, check=True)
        print('Verifying and restoring', name, flush=True)
        restore(bundle, directory, args.destination)


if __name__ == '__main__':
    main()
