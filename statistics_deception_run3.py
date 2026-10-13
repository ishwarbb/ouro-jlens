"""Prospectively frozen record-cluster statistics for Run 3 held-out data."""
import itertools
import numpy as np


def paired_endpoint(differences, *, seed=20261009, bootstraps=50000):
    # Missing pairs must be omitted by ID upstream, never converted to zeros.
    values=np.asarray(differences,dtype=float)
    assert np.isfinite(values).all()
    n=len(values)
    if not n:
        return dict(n=0,mean=None,p_two_sided=None,ci95=None,positive=False,zeros=0)
    assert n<=12
    mean=float(values.mean())
    means=np.array([np.mean(values*np.asarray(signs)) for signs in itertools.product([-1,1],repeat=n)])
    p=float(np.mean(np.abs(means)>=abs(mean)-1e-12))
    rng=np.random.default_rng(seed)
    boot=values[rng.integers(0,n,size=(bootstraps,n))].mean(1)
    return dict(n=n,mean=mean,p_two_sided=p,ci95=np.quantile(boot,[.025,.975]).tolist(),
                positive=mean>0,zeros=int(np.count_nonzero(values==0)))


def holm_three(pvalues):
    assert len(pvalues)==3
    # A missing endpoint cannot pass; retain its family slot as p=1 for the
    # multiplicity adjustment, while returning missing in its report field.
    finite=[1.0 if p is None else float(p) for p in pvalues]
    order=sorted(range(3),key=lambda i:finite[i]);out=[None]*3;previous=0.0
    for rank,i in enumerate(order):
        previous=max(previous,min(1.0,(3-rank)*finite[i]))
        if pvalues[i] is not None:out[i]=previous
    return out
