"""Pinned identities and filesystem helpers; imports no inference framework."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
DEFAULT_ASSETS = ROOT / '.artifacts/monitorbench-thinking'
DEFAULT_INVENTORY = ROOT / 'results/monitorbench-replay-plan-20261016'
MODEL_ID = 'ByteDance/Ouro-2.6B-Thinking'
MODEL_REVISION = 'f1edd81e7ac41355db670500ceaf204e0f73af68'
LENS_ID = 'zigzag9939/jlens-looped-transformers'
LENS_REVISION = 'dc9e1b928a2d14bbb09e914525e39c258a489b05'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def read_jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
