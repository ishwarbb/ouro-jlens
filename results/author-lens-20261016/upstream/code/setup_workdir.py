"""Build a flat run directory in the layout the experiments were run in.

Every script here loads lenses, prompts and the jacobian-lens experiment data
by bare relative path from the current directory (e.g.
"jacobian_lens_ouro-2.6b_t47.pt", "jacobian-lens/data/experiments/..."), and
writes its outputs there too. This repo stores lenses under lenses/<model>/,
so this script symlinks everything into one directory to run from.

The Ouro-2.6B-Thinking lenses share filenames with the base Ouro lenses, so
they need their own run directory (--thinking), with OURO_MODEL_ID set.

Lenses are symlinked (read-only inputs); results are copied, because the
scripts rewrite their outputs by bare filename and would otherwise write
through a link into this repo. To refit lenses, use --no-lenses: the merge
scripts save jacobian_lens_*.pt by bare filename too.

Usage:
  python code/setup_workdir.py RUN_DIR --jlens PATH/TO/jacobian-lens
                               [--thinking] [--with-results] [--no-lenses]

  --jlens         a git checkout of https://github.com/anthropics/jacobian-lens
                  (at the commit in revisions.JLENS_COMMIT)
  --thinking      link the Ouro-2.6B-Thinking lenses instead of the base
                  Qwen/Ouro/Huginn ones
  --with-results  also copy in the released results, so analysis/plot
                  steps can run without regenerating them on a GPU
  --no-lenses     link no lenses (for refitting them from scratch)
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from revisions import JLENS_COMMIT  # noqa: E402


def link(src, dst):
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        raise SystemExit(f"refusing to overwrite {dst}")
    dst.symlink_to(src.resolve())


def copy(src, dst):
    if dst.is_symlink() or dst.exists():
        raise SystemExit(f"refusing to overwrite {dst}")
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--jlens", required=True)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--with-results", action="store_true")
    ap.add_argument("--no-lenses", action="store_true")
    args = ap.parse_args()

    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)

    jlens = Path(args.jlens)
    if not (jlens / "data" / "experiments").is_dir():
        raise SystemExit(f"{jlens} is not a jacobian-lens checkout "
                         "(no data/experiments/)")
    try:
        head = subprocess.run(["git", "-C", str(jlens), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    if head != JLENS_COMMIT:
        print(f"WARNING: jacobian-lens is at {head}, the runs used {JLENS_COMMIT}")
    link(jlens, run / "jacobian-lens")

    for f in sorted((ROOT / "code").iterdir()):
        if f.suffix in (".py", ".sh") and f.name != "setup_workdir.py":
            link(f, run / f.name)

    models = ["ouro_thinking"] if args.thinking else ["qwen", "ouro", "huginn"]
    if args.no_lenses:
        models = []
    for m in models:
        for f in sorted((ROOT / "lenses" / m).glob("*.pt")):
            link(f, run / f.name)

    if args.with_results:
        res = ROOT / "results" / ("thinking" if args.thinking else "")
        for f in sorted(res.iterdir()):
            if f.name in ("thinking", "logs"):
                continue
            copy(f, run / f.name)

    print(f"run directory ready: {run}")
    if args.thinking:
        print("before running, export OURO_MODEL_ID=ByteDance/Ouro-2.6B-Thinking")
    print("generate the fitting prompts there first: python prep_prompts.py")


if __name__ == "__main__":
    main()
