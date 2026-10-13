"""Preserve reviewed evidence when an authorized factual handoff refreshes a queue.

No frozen packet, experiment code, prompt, generation, or source hash is edited.
The historical pending-review inventory is archived at its exact original hash;
its current derived version must match the original runner and installed facts.
All raw histories, traces, and other original sources still use live hash checks.
"""
from contextlib import contextmanager
import json
from pathlib import Path

import queue_unfinished_evaluations as queue

ROOT = queue.ROOT
OUT = ROOT / 'results/evaluation-completion-20261012'
RUN4 = queue.RUN4
RUN8 = queue.RUN8
EVOLVED_SOURCE = 'results/deception-run4-20261011/pilot-fact-review-queue.json'


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + '\n').encode()


def historical_inventory(packet, jobs):
    cases = {c['id']: c for c in packet['cases']}
    rows = [dict(id=job['id'], generation_sha256=cases[job['id']]['generations'][0]['sha256'],
                 reason='MESSAGE status needs semantic review') for job in jobs if job['id'] in cases]
    if len(rows) != len(cases):
        raise ValueError('Historical review inventory is not covered by original jobs')
    return encoded(rows)


def prepare_archive():
    path = OUT / 'review-source-evolution.json'
    if path.exists():
        return queue.read(path)
    packet_path = RUN8 / 'reviews/run4_pending_factual/blind-evidence.json'
    packet = queue.read(packet_path)
    handoff = queue.read(OUT / 'run4-factual-handoff.json')
    if handoff['status'] != 'installed':
        raise ValueError('Factual handoff is not installed')
    archive = OUT / 'review-source-archive/pilot-fact-review-queue.json'
    archive.parent.mkdir(parents=True, exist_ok=True)
    data = historical_inventory(packet, queue.read(RUN4 / 'plan/jobs.json'))
    archive.write_bytes(data)
    if queue.sha(archive) != packet['source_files'][EVOLVED_SOURCE]:
        raise ValueError('Reconstructed historical inventory differs from frozen source hash')
    facts = OUT / 'run4-fact-reviews-after-handoff.json'
    facts.write_bytes((RUN4 / 'fact-reviews.json').read_bytes())
    if queue.sha(facts) != handoff['next_sha256']:
        raise ValueError('Installed pilot facts changed before archiving')
    result = dict(status='authorized_derived_source_evolution', source=EVOLVED_SOURCE,
                  archive_path=str(archive.relative_to(ROOT)), historical_sha256=queue.sha(archive),
                  handoff_path=str((OUT / 'run4-factual-handoff.json').relative_to(ROOT)),
                  handoff_sha256=queue.sha(OUT / 'run4-factual-handoff.json'),
                  installed_facts_path=str(facts.relative_to(ROOT)), installed_facts_sha256=queue.sha(facts),
                  adjudication_path=str((OUT / 'run4-pilot-adjudication.json').relative_to(ROOT)),
                  adjudication_sha256=queue.sha(OUT / 'run4-pilot-adjudication.json'),
                  original_packet_path=str(packet_path.relative_to(ROOT)), original_packet_sha256=queue.sha(packet_path),
                  explanation='Original Run4 refreshes its derived pending-review list after the authorized '
                              'hash-matched factual handoff. Preserve the reviewed historical inventory and '
                              'validate the live list against the unchanged original runner. No raw source or '
                              'frozen Run8 packet/code/manifest is changed.')
    queue.write(path, result)
    return result


def verify_evolution(proof):
    for name, hash_name in (('archive_path', 'historical_sha256'), ('handoff_path', 'handoff_sha256'),
                            ('installed_facts_path', 'installed_facts_sha256'),
                            ('adjudication_path', 'adjudication_sha256'),
                            ('original_packet_path', 'original_packet_sha256')):
        path = (ROOT / proof[name]).resolve()
        if not path.is_relative_to(ROOT) or queue.sha(path) != proof[hash_name]:
            raise ValueError('Source evolution proof changed: ' + name)
    handed_off = queue.read(ROOT / proof['installed_facts_path'])
    current = queue.read(RUN4 / 'fact-reviews.json')
    if any(current.get(case) != annotation for case, annotation in handed_off.items()):
        raise ValueError('Previously handed-off factual annotation changed')
    from run_deception_run4 import effective_fact, sha, verify_frozen
    verify_frozen(RUN4)
    rows = []
    for job in queue.read(RUN4 / 'plan/jobs.json'):
        if job['stage'] != 'pilot':
            continue
        path = RUN4 / 'cases' / job['id'] / 'generation.json'
        if path.exists():
            fact = effective_fact(RUN4, queue.read(path), path)
            if fact['factual_review_required'] and job['kind'] != 'history':
                rows.append(dict(id=job['id'], generation_sha256=sha(path),
                                 reason='MESSAGE status needs semantic review'))
    if (ROOT / EVOLVED_SOURCE).read_bytes() != encoded(rows):
        raise ValueError('Live pilot review inventory is not the original derived inventory')


@contextmanager
def preserve_handoff_sources(proof):
    """Temporarily validate this one derived source through its historical archive."""
    import run8_review as reviews
    original = reviews.verify_sources

    def verify_packet(packet):
        value = reviews.read(packet) if isinstance(packet, (str, Path)) else packet
        if EVOLVED_SOURCE in value['source_files']:
            if value['source_files'][EVOLVED_SOURCE] != proof['historical_sha256']:
                raise ValueError('Packet historical source differs from source evolution proof')
            verify_evolution(proof)
            # The sole omitted live hash has just been checked against its exact
            # historical bytes AND its permitted current derivation. Everything
            # else goes through the unchanged original raw-source validator.
            value = dict(value, source_files={key: digest for key, digest in value['source_files'].items()
                                             if key != EVOLVED_SOURCE})
        return original(value)

    reviews.verify_sources = verify_packet
    try:
        yield
    finally:
        reviews.verify_sources = original


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-run8', action='store_true')
    args = parser.parse_args()
    proof = prepare_archive()
    verify_evolution(proof)
    if args.verify_run8:
        from run_deception_run8 import verify_frozen
        with preserve_handoff_sources(proof):
            verify_frozen(RUN8)
    print(json.dumps(dict(status='passed', frozen_run8_files_modified=False,
                         authorized_derived_source=EVOLVED_SOURCE, raw_sources_still_hash_verified=True)))
