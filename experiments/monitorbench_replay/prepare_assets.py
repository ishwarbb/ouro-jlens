"""Download and verify the pinned Thinking checkpoint and four author banks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from .common import DEFAULT_ASSETS, LENS_ID, LENS_REVISION, MODEL_ID, MODEL_REVISION, save, sha


def tree(repo, revision, folder=''):
    url = f'https://huggingface.co/api/models/{repo}/tree/{revision}'
    if folder:
        url += '/' + folder
    with urlopen(url + '?recursive=true', timeout=60) as stream:
        return json.load(stream)


def fetch(entry, repo, revision, destination):
    target = destination / entry['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_sha = (entry.get('lfs') or {}).get('oid')

    def verify(path):
        if path.stat().st_size != entry['size']:
            raise ValueError(f'Incorrect file size: {path}')
        actual_sha = sha(path)
        if expected_sha:
            if actual_sha != expected_sha:
                raise ValueError(f'Incorrect SHA256: {path}')
        else:
            data = path.read_bytes()
            blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            if blob != entry['oid']:
                raise ValueError(f'Incorrect Git blob identity: {path}')
        return actual_sha

    if target.exists():
        actual = verify(target)
    else:
        print(f'Downloading {repo}/{entry["path"]} ({entry["size"]:,} bytes)', flush=True)
        temporary = target.with_suffix(target.suffix + '.download')
        with urlopen(f'https://huggingface.co/{repo}/resolve/{revision}/{entry["path"]}', timeout=60) as source, temporary.open('wb') as output:
            while chunk := source.read(8 * 1024 * 1024):
                output.write(chunk)
        actual = verify(temporary)
        temporary.replace(target)
    print(f'Verified {entry["path"]}', flush=True)
    return dict(path=str(target.resolve()), bytes=entry['size'], sha256=actual, remote_path=entry['path'], repo=repo, revision=revision)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=DEFAULT_ASSETS)
    args = parser.parse_args()
    model_names = {'config.json', 'configuration_ouro.py', 'modeling_ouro.py', 'model.safetensors',
                   'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json', 'merges.txt'}
    model_entries = [e for e in tree(MODEL_ID, MODEL_REVISION) if e['path'] in model_names]
    lens_entries = [e for e in tree(LENS_ID, LENS_REVISION, 'lenses/ouro_thinking') if e['path'].endswith('.pt')]
    if len(model_entries) != len(model_names) or len(lens_entries) != 4:
        raise ValueError('Unexpected author asset inventory')
    work = [(e, MODEL_ID, MODEL_REVISION, args.assets / 'model') for e in model_entries]
    work += [(e, LENS_ID, LENS_REVISION, args.assets / 'author') for e in lens_entries]
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(lambda job: fetch(*job), work))
    save(args.assets / 'manifest.json', dict(model=MODEL_ID, model_revision=MODEL_REVISION,
         lens=LENS_ID, lens_revision=LENS_REVISION, files=records))
    print(f'Assets ready: {args.assets}', flush=True)


if __name__ == '__main__':
    main()
