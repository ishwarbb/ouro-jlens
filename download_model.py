"""Download the pinned Ouro checkpoint, or only its code for CPU unit tests."""
import argparse
import json
from pathlib import Path


def main():
    from huggingface_hub import snapshot_download
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-only', action='store_true', help='Skip weights; sufficient for tiny-architecture CPU tests')
    args = parser.parse_args()
    model = json.loads((root / 'model_revision.json').read_text())
    patterns = ['*.json', '*.py', '*.model', '*.txt', '*.jinja', 'LICENSE*', 'README.md']
    if not args.code_only:
        patterns.append('*.safetensors')
    snapshot_download(model['model'], revision=model['revision'], local_dir=root / 'model', allow_patterns=patterns)


if __name__ == '__main__':
    main()
