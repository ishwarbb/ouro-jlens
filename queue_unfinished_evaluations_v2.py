"""Completion queue with explicit provenance for the authorized Run4 handoff."""
from pathlib import Path
import queue_unfinished_evaluations as queue
import completion_review_provenance as provenance

original_prepare = queue.prepare
original_collect = queue.collect_submissions


def prepare(output):
    exists = (output / 'queue.json').exists()
    manifest = original_prepare(output)
    if not exists:
        manifest['successor_of'] = 'results/evaluation-completion-20261012'
        for path in (Path(__file__), Path(provenance.__file__), queue.ROOT / 'test_completion_provenance.py'):
            manifest['bindings'][str(path.relative_to(queue.ROOT))] = queue.sha(path)
        manifest['source_evolution_proof'] = str((provenance.OUT / 'review-source-evolution.json').relative_to(queue.ROOT))
        manifest['bindings'][manifest['source_evolution_proof']] = queue.sha(queue.ROOT / manifest['source_evolution_proof'])
        queue.write(output / 'queue.json', manifest)
    return manifest


def collect(output):
    proof = queue.read(provenance.OUT / 'review-source-evolution.json')
    with provenance.preserve_handoff_sources(proof):
        return original_collect(output)


if __name__ == '__main__':
    provenance.prepare_archive()
    queue.OUT = queue.ROOT / 'results/evaluation-completion-20261012-v2'
    queue.prepare = prepare
    queue.collect_submissions = collect
    queue.main()
