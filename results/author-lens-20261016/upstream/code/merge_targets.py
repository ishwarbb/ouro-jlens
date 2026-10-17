import glob

from jlens.lens import JacobianLens

for t in (47, 95, 143):
    paths = sorted(glob.glob(f"lens_ouro_t{t}_w*.pt"))
    print(t, paths)
    merged = JacobianLens.merge([JacobianLens.load(p) for p in paths])
    print(merged)
    merged.save(f"jacobian_lens_ouro-2.6b_t{t}.pt")
print("DONE_MERGE_TARGETS")
