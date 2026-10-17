"""Restore the four pinned author Ouro banks using their recorded SHA-256 hashes."""
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import urlopen

from ouro_lens_config import DEFAULT_FAMILY


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    metadata = json.loads((DEFAULT_FAMILY / 'family.json').read_text())['metadata']
    root = DEFAULT_FAMILY.parent / 'upstream'
    for target, entry in metadata['files'].items():
        path = root / entry['file']
        if path.exists():
            if path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
                raise ValueError(f'Existing author bank differs from the pinned artifact: {path}')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.pt.download')
            url = f'https://huggingface.co/{metadata["repo"]}/resolve/{metadata["revision"]}/{entry["file"]}'
            print(f'Downloading target {target}', flush=True)
            try:
                with urlopen(url, timeout=120) as response, temporary.open('wb') as output:
                    shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
                if temporary.stat().st_size != entry['bytes'] or sha(temporary) != entry['sha256']:
                    raise ValueError(f'Author bank checksum mismatch: {path}')
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        link = DEFAULT_FAMILY / f'target-{target}.pt'
        if link.is_symlink():
            if link.resolve() != path.resolve():
                raise ValueError(f'Existing lens link has a different destination: {link}')
        elif link.exists():
            if sha(link) != entry['sha256']:
                raise ValueError(f'Existing target differs from the author bank: {link}')
        else:
            link.symlink_to(Path('../upstream') / entry['file'])
        print(f'Author target {target}: verified', flush=True)


if __name__ == '__main__':
    main()
