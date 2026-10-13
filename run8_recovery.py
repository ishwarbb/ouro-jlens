"""Versioned recovery for Run 8's diagnostic tool filename report failure.

The original frozen runner, prompts, cases and execution manifest stay intact.
This adapter changes the renderer in memory: only tool-<integer>.json is a
numbered event; other tool JSON remains hashed and is displayed as evidence.
Saved generations and the already-passed numerical receipt are reused. A new,
external manifest binds this adapter, the preserved source inventory and the
authorized Run 4 derived-review-inventory evolution proof.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import inspect
from pathlib import Path
import re
import traceback

from prepare_deception_run4 import read, sha
from prepare_deception_run6 import ROOT
from run_deception_run4 import dump, now

ADAPTER_VERSION = 'run8-recovery-v1'
NUMBERED_TOOL = re.compile(r'tool-([0-9]+)\.json\Z')


def numbered_tool_paths(directory):
    """Keep actual integer event names in chronological numeric order."""
    paths = [p for p in directory.glob('tool-*.json') if NUMBERED_TOOL.fullmatch(p.name)]
    return sorted(paths, key=lambda p: int(NUMBERED_TOOL.fullmatch(p.name).group(1)))


def diagnostic_tool_details(directory):
    """Retain nonnumbered results without inventing a transcript turn."""
    from report_deception_run7 import _json_details
    return [_json_details('Saved tool evidence ' + p.name + ' (not a numbered event)', read(p))
            for p in sorted(directory.glob('tool-*.json')) if not NUMBERED_TOOL.fullmatch(p.name)]


def corrected_case_view(original):
    """Apply two guarded changes to the original function, without editing it.

    Its source digest still hashes every tool JSON, including tool-result.json.
    All generation, workspace and readout verification remains original code.
    A separate renderer version invalidates only derived HTML caches.
    """
    tree = ast.parse(inspect.getsource(original))
    function = tree.body[0]
    replacements = insertions = 0
    for index, statement in enumerate(function.body):
        if (isinstance(statement, ast.Assign) and len(statement.targets) == 1 and
                isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == 'tool_paths'):
            expected = ast.parse("tool_paths = sorted(directory.glob('tool-*.json'), key=lambda p: int(p.stem.split('-')[-1]))").body[0]
            if ast.dump(statement, include_attributes=False) != ast.dump(expected, include_attributes=False):
                raise ValueError('Frozen renderer tool path expression differs from the reviewed recovery')
            statement.value = ast.parse('numbered_tool_paths(directory)', mode='eval').body
            replacements += 1
        if (isinstance(statement, ast.For) and isinstance(statement.iter, ast.Name) and
                statement.iter.id == 'tool_paths'):
            function.body.insert(index + 1, ast.parse('body.extend(diagnostic_tool_details(directory))').body[0])
            insertions += 1
            break
    if (replacements, insertions) != (1, 1):
        raise ValueError('Frozen renderer no longer matches the two reviewed recovery sites')
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__, numbered_tool_paths=numbered_tool_paths,
                     diagnostic_tool_details=diagnostic_tool_details, REPORT_VERSION=ADAPTER_VERSION)
    exec(compile(tree, str(Path(__file__).resolve()) + ':renderer-adapter', 'exec'), namespace)
    return namespace[original.__name__]


@contextmanager
def preserve_renderer():
    import report_deception_run7 as report
    original = report._case_view
    corrected = corrected_case_view(original)

    def render_case(directory, episode, tokenizer):
        # Keep the original cache identity for regular tool traces. Only
        # diagnostic filenames exercise the corrected parser and new cache.
        if any(not NUMBERED_TOOL.fullmatch(p.name) for p in directory.glob('tool-*.json')):
            return corrected(directory, episode, tokenizer)
        return original(directory, episode, tokenizer)

    report._case_view = render_case
    try:
        yield
    finally:
        report._case_view = original


def project_path(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('Recovery binding escapes the project: ' + str(relative))
    return path


def require_no_orphan_generations(root):
    """An interrupted unsaved draw needs separate handling, never a reroll."""
    for directory in (root / 'cases').glob('*/turn-*'):
        if directory.is_dir() and not (directory / 'generation.json').exists() and any(directory.iterdir()):
            raise ValueError('Unsaved generation artifacts require explicit recovery: ' + str(directory))


def verify_source_inventory(root, inventory):
    """Require coverage of every saved episode, generation and source artifact."""
    files = inventory['files']
    required = {}
    for pattern in ('*/episode.json', '*/turn-*/generation.json'):
        for path in (root / 'cases').glob(pattern):
            relative = str(path.relative_to(ROOT))
            if relative not in files:
                raise ValueError('Existing evidence absent from recovery inventory: ' + relative)
            required[relative] = files[relative]
            for name, expected in read(path).get('artifacts', {}).items():
                artifact = (path.parent / name).resolve()
                if not artifact.is_relative_to((root / 'cases').resolve()):
                    raise ValueError('Saved artifact escapes the case root: ' + name)
                relative_artifact = str(artifact.relative_to(ROOT))
                if files.get(relative_artifact) != expected:
                    raise ValueError('Saved artifact missing or mismatched in recovery inventory: ' + relative_artifact)
                required[relative_artifact] = expected
    for relative, expected in files.items():
        if sha(project_path(relative)) != expected:
            raise ValueError('Preserved recovery evidence changed: ' + relative)
    require_no_orphan_generations(root)
    return dict(source_files=len(files), required_source_files=len(required))


def verify_numerical_receipt(root, execution_hash, expected_hash):
    path = root / 'numerical-verification.json'
    if sha(path) != expected_hash:
        raise ValueError('Preserved production numerical receipt changed')
    receipt = read(path)
    if (receipt['status'] != 'passed' or receipt['revision_execution_sha256'] != execution_hash or
            receipt['tolerances'] != read(root / 'plan/numerical-checks.json') or
            len(receipt['checks']) != 11 or not all(check['passed'] for check in receipt['checks'])):
        raise ValueError('Recovery requires the original eleven passed production checks')
    return receipt


def verify_manifest(root, path):
    manifest = read(path)
    if manifest['adapter_version'] != ADAPTER_VERSION or project_path(manifest['run8_root']) != root.resolve():
        raise ValueError('Recovery adapter or Run 8 root differs')
    execution_hash = sha(root / 'execution.json')
    if execution_hash != manifest['execution_sha256']:
        raise ValueError('Original frozen Run 8 execution differs')
    for required in ('run8_recovery.py', 'test_run8_recovery.py'):
        if sha(ROOT / required) != manifest['code_sha256'][required]:
            raise ValueError('Recovery code binding differs: ' + required)
    for name, expected in manifest['code_sha256'].items():
        if sha(project_path(name)) != expected:
            raise ValueError('Recovery code binding differs: ' + name)
    proof_binding = manifest['source_evolution_proof']
    proof_path = project_path(proof_binding['path'])
    if sha(proof_path) != proof_binding['sha256']:
        raise ValueError('Derived source evolution proof binding differs')
    proof = read(proof_path)
    import completion_review_provenance as provenance
    provenance.verify_evolution(proof)
    source_binding = manifest['source_inventory']
    inventory_path = project_path(source_binding['path'])
    if sha(inventory_path) != source_binding['sha256']:
        raise ValueError('Preserved source inventory binding differs')
    inventory = read(inventory_path)
    if inventory['execution_sha256'] != execution_hash:
        raise ValueError('Preserved source inventory execution differs')
    coverage = verify_source_inventory(root, inventory)
    numerical = verify_numerical_receipt(root, execution_hash, manifest['numerical_verification_sha256'])
    from run_deception_run8 import verify_frozen
    with provenance.preserve_handoff_sources(proof):
        verify_frozen(root)
    return manifest, proof, numerical, coverage


@contextmanager
def preserve_numerical_receipt(root, execution_hash, expected_hash):
    import analyze_evaluation_revisions as analysis
    original = analysis.numerical_validation

    def reused(bundle, model, lens, actual_execution_hash):
        if Path(bundle).resolve() != root.resolve() or actual_execution_hash != execution_hash:
            raise ValueError('Production receipt reused for a different execution')
        return verify_numerical_receipt(root, execution_hash, expected_hash)

    analysis.numerical_validation = reused
    try:
        yield
    finally:
        analysis.numerical_validation = original


@contextmanager
def guard_saved_generations(root, inventory):
    """Verify saved draws before and after reuse across every imported alias."""
    import run7_generation as generation
    import run_deception_run7 as run7
    import run_deception_run8 as run8
    modules = (generation, run7, run8)
    originals = [module.generate for module in modules]
    original = generation.generate
    preserved = {project_path(name): value for name, value in inventory['files'].items()
                 if name.endswith('/generation.json')}

    def reuse(*args, **kwargs):
        job, turn = args[1:3]
        path = (root / 'cases' / job['id'] / f'turn-{turn}' / 'generation.json').resolve()
        if path in preserved and sha(path) != preserved[path]:
            raise ValueError('Saved generation changed before reuse: ' + str(path))
        value = original(*args, **kwargs)
        if path in preserved and sha(path) != preserved[path]:
            raise ValueError('Saved generation changed during reuse: ' + str(path))
        return value

    for module in modules:
        module.generate = reuse
    try:
        yield
    finally:
        for module, function in zip(modules, originals):
            module.generate = function


def execute(root, manifest_path, mode='run'):
    root = root.resolve()
    manifest, proof, numeric, coverage = verify_manifest(root, manifest_path)
    import completion_review_provenance as provenance
    inventory = read(project_path(manifest['source_inventory']['path']))
    with provenance.preserve_handoff_sources(proof), preserve_renderer(), \
            preserve_numerical_receipt(root, manifest['execution_sha256'], manifest['numerical_verification_sha256']), \
            guard_saved_generations(root, inventory):
        if mode == 'verify':
            return dict(status='passed', adapter_version=ADAPTER_VERSION, production_checks=len(numeric['checks']), **coverage)
        if mode == 'report':
            from transformers import AutoTokenizer
            from report_deception_run8 import render
            extra = read(root / 'run8-summary.json') if (root / 'run8-summary.json').exists() else None
            render(root, extra=extra, tokenizer=AutoTokenizer.from_pretrained(ROOT / 'model', trust_remote_code=True))
        else:
            from run_deception_run8 import run
            run(root)
    return dict(status='finished', adapter_version=ADAPTER_VERSION, mode=mode, **coverage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    if args.verify_only and args.report_only:
        parser.error('Select one mode')
    try:
        result = execute(args.output, args.manifest, 'verify' if args.verify_only else 'report' if args.report_only else 'run')
        print(result)
    except BaseException as error:
        dump(args.output / 'recovery-status.json', dict(utc=now(), stage='stopped_at_error',
             adapter_version=ADAPTER_VERSION, error=repr(error)))
        traceback.print_exc()
        raise
