import glob

from jlens.lens import JacobianLens

paths = sorted(glob.glob("lens_ouro_w*.pt"))
print(paths)
lenses = [JacobianLens.load(p) for p in paths]
merged = JacobianLens.merge(lenses)
print(merged)
merged.save("jacobian_lens_ouro-2.6b.pt")
print("DONE_MERGE")
