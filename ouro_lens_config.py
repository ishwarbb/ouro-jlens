"""Lens selection for new work and provenance-preserving saved-run readouts."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_FAMILY = ROOT / 'results/author-lens-20261016/family'
LEGACY_FAMILY = ROOT / 'results/jlens-family-1000'


def family_for_run(run):
    """Resolve an existing run's recorded family, never today's default.

    Historical research imports predate family_sha256 and used the local fit.
    A recorded but unknown hash is an error rather than a silent substitution.
    """
    provenance = run.get('provenance', {})
    expected = provenance.get('family_sha256')
    if not expected:
        if run.get('kind') == 'research' and provenance.get('source') == 'workspace-inventory-v2':
            return LEGACY_FAMILY
        raise ValueError('Saved run has no lens identity; its readouts cannot be recomputed safely')
    for family in (DEFAULT_FAMILY, LEGACY_FAMILY):
        manifest = family / 'family.json'
        if manifest.is_file() and hashlib.sha256(manifest.read_bytes()).hexdigest() == expected:
            return family
    raise ValueError('The lens recorded for this saved run is unavailable or has changed')
