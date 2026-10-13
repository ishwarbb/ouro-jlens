"""Run the unchanged Run8 scientific functions on an explicitly relocated GPU."""
import argparse
from contextlib import contextmanager
import inspect
from pathlib import Path
import os

from prepare_deception_run4 import read,sha
from evaluation_revisions import immutable
import run8_recovery as base
import run8_recovery_continue as continuation


def relocated_runner(original, gpu):
    source=inspect.getsource(original)
    replacements={
        "cfg=execution['resources'];plan=read(root/'plan/tasks.json')":
            "cfg={**execution['resources'],'gpu':relocation_gpu};plan=read(root/'plan/tasks.json')",
        "ROOT/'results/.gpu-evaluation-2-run8.lock'":
            "ROOT/f'results/.gpu-evaluation-{relocation_gpu}-run8.lock'",
        "immutable(root/'environment.json',dict":
            "relocation_environment(root,dict",
        "GPU2 lacks registered free memory":"Relocated GPU lacks registered free memory",
    }
    for old,new in replacements.items():
        if source.count(old)!=1:raise ValueError('Frozen relocation site differs: '+old)
        source=source.replace(old,new)

    def environment(root,value):
        expected={**read(root/'environment.json'),'cuda_visible_devices':gpu}
        if value!=expected:raise ValueError('Relocation changes software, hardware type, precision or scientific execution')
        immutable(root/f'environment-gpu{gpu}-20261013.json',value)
    namespace=dict(original.__globals__,relocation_gpu=gpu,relocation_environment=environment)
    exec(compile(source,str(Path(__file__).resolve())+':device-adapter','exec'),namespace)
    return namespace[original.__name__]


@contextmanager
def relocated_numerics(root,execution_hash,receipt_hash):
    """Recheck all11 production checks on GPU1, preserving the original receipt."""
    import analyze_evaluation_revisions as analysis
    original=analysis.numerical_validation
    original_dump=analysis.dump

    def routed_dump(path,value):
        if Path(path).resolve()==(root/'numerical-verification.json').resolve():
            path=root/'numerical-verification-gpu1-20261013.json'
            value=dict(value,physical_gpu='1',original_verification_sha256=receipt_hash)
        return original_dump(path,value)
    analysis.dump=routed_dump
    try:
        yield
    finally:
        analysis.numerical_validation=original
        analysis.dump=original_dump


@contextmanager
def device_context(gpu):
    import run_deception_run8 as run8
    original=run8.run
    numeric_context=base.preserve_numerical_receipt
    run8.run=relocated_runner(original,gpu)
    base.preserve_numerical_receipt=relocated_numerics
    try:
        yield
    finally:
        run8.run=original
        base.preserve_numerical_receipt=numeric_context


def execute(root,manifest_path):
    manifest=read(manifest_path)
    if manifest.get('relocation_version')!='run8-gpu1-20261013':raise ValueError('Unbound device relocation')
    gpu=manifest['physical_gpu']
    if gpu!='1' or os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu:raise ValueError('GPU1 relocation requires CUDA_VISIBLE_DEVICES=1')
    for name in ('run8_relocate_gpu.py','test_run8_relocate_gpu.py'):
        if sha(base.ROOT/name)!=manifest['code_sha256'][name]:raise ValueError('Relocation code changed')
    if sha(root/'environment.json')!=manifest['original_environment_sha256']:raise ValueError('Original environment receipt changed')
    with device_context(gpu):
        return continuation.execute(root,manifest_path)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    args=p.parse_args()
    try:
        print(execute(args.output.resolve(),args.manifest))
    except BaseException as error:
        from run_deception_run4 import dump,now
        dump(args.output/'recovery-status.json',dict(utc=now(),stage='stopped_at_error',gpu='1',error=repr(error)))
        raise
