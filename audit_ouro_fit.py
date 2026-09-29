"""Read-only full merge verification and float64 half/full convergence audit."""
import json
from pathlib import Path
import statistics
import time
import torch
from fit_ouro_family import ROOT, digest, save_json


def main():
    torch.set_num_threads(2)
    base=ROOT/'results/jlens-family-1000';out=ROOT/'results/workspace-audit'
    manifest=json.loads((base/'full-manifest.json').read_text());fingerprint=digest(base/'full-manifest.json')
    start=time.time();results={};checked=0
    for target in manifest['targets']:
        merged=torch.load(base/f'target-{target}.pt',weights_only=True,mmap=True,map_location='cpu')
        assert merged['n_prompts']==1000 and merged['d_model']==2048
        sites=list(range(target-47+24,target-47+41))
        sums={name:torch.zeros(len(sites),dtype=torch.float64) for name in ['xx','yy','xy','diff']}
        for rank in range(8):
            lo,hi=rank*256,(rank+1)*256;folder=base/'workers'/f'worker-{rank}'
            full=torch.load(folder/f'target-{target}-sum.pt',weights_only=True,mmap=True,map_location='cpu')
            half=torch.load(folder/f'target-{target}-half.pt',weights_only=True,mmap=True,map_location='cpu')
            for data,n in [(full,1000),(half,500)]:
                assert data['n_done']==n
                assert data['header']==dict(manifest_sha256=fingerprint,target=target,row_start=lo,row_end=hi,rank=rank)
            for site in range(target):
                expected=full['sum'][site]/1000
                assert torch.equal(merged['J'][site][lo:hi],expected),(target,rank,site)
                assert torch.isfinite(expected).all()
                checked+=1
            x=(full['sum'][sites]/1000).double().flatten(1)
            y=(half['sum'][sites]/500).double().flatten(1)
            sums['xx']+=x.square().sum(1);sums['yy']+=y.square().sum(1)
            sums['xy']+=(x*y).sum(1);sums['diff']+=(x-y).square().sum(1)
            del full,half,x,y
        cosine=sums['xy']/(sums['xx']*sums['yy']).sqrt()
        relative=(sums['diff']/sums['xx']).sqrt()
        assert ((cosine>=-1)&(cosine<=1)).all()
        results[target]=dict(sites=sites,cosine=cosine.tolist(),relative_change=relative.tolist(),
            cosine_min=float(cosine.min()),cosine_median=statistics.median(cosine.tolist()),
            relative_change_max=float(relative.max()),relative_change_median=statistics.median(relative.tolist()))
        print('Verified target',target,'all sources; convergence',results[target]['cosine_min'],flush=True)
        del merged
    save_json(out/'fit-audit.json',dict(checked_row_blocks=checked,matrices=checked//8,merge_exact=True,
        finite=True,convergence_precision='float64 reductions over original FP32 matrices',convergence=results,
        seconds=time.time()-start,code_sha256=digest(Path(__file__))))


if __name__=='__main__':main()
