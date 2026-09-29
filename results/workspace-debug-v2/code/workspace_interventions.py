"""Causal operators and token scoring for the frozen Ouro workspace tests."""
from contextlib import contextmanager
import hashlib
import torch


def swap_delta(h, source, destination):
    """Exchange coordinates in two unit directions, preserving the orthogonal component.

    This reflection equals V @ (swap(pinv(V) @ h) - pinv(V) @ h)
    for V=[source,destination]. Unit normalization is required.
    """
    difference = destination-source
    squared = difference.square().sum()
    if squared < 1e-12:
        return torch.zeros_like(h)
    return -2*(h@difference).unsqueeze(-1)*difference/squared


def transfer_delta(h, source, destination):
    """Sensitivity operator: remove the source projection and transfer it."""
    return (h@source).unsqueeze(-1)*(destination-source)


def score_tokens(logits, candidates):
    """Exact normalized-token match; rank uses strict greater-than for ties."""
    if not candidates:
        return dict(eligible=False,rank=None,top1=False,top5=False,probability=None)
    logits=logits.float()
    values=logits[list(candidates)]
    best=values.max()
    return dict(eligible=True,rank=int((logits>best).sum().item())+1,
        top1=int(logits.argmax()) in candidates,
        top5=bool(torch.isin(logits.topk(5).indices,torch.tensor(candidates,device=logits.device)).any()),
        probability=float(logits.softmax(-1)[list(candidates)].sum()))


class Vocabulary:
    def __init__(self,tokenizer):
        self.tokenizer=tokenizer
        self.by_word={}
        for index in range(len(tokenizer)):
            text=tokenizer.decode([index]).strip().casefold()
            if text and index not in tokenizer.all_special_ids:
                self.by_word.setdefault(text,[]).append(index)

    def candidates(self,word):
        return self.by_word.get(word.strip().casefold(),[])

    def direction_token(self,word, *, leading_space=True):
        variants=[' '+word,' '+word.lower(),word,word.lower()]
        if not leading_space:
            variants=[word,word.lower(),' '+word,' '+word.lower()]
        for text in variants:
            ids=self.tokenizer.encode(text,add_special_tokens=False)
            if len(ids)==1 and ids[0] in self.candidates(word):
                return ids[0]
        candidates=self.candidates(word)
        return candidates[0] if candidates else None


class Directions:
    def __init__(self,model,family):
        self.model,self.family=model,family
        self.cache={}
        self.matrices={}

    def get(self,site,token,kind='own'):
        key=(site,token,kind)
        if key in self.cache:
            return self.cache[key]
        weight=self.model.hf_model.lm_head.weight
        u=weight[token].detach().float()
        if kind=='raw':
            result=u
        else:
            target=self.family.target_for(site) if kind=='own' else self.model.n_layers-1
            matrix_key=(site,target)
            if matrix_key not in self.matrices:
                self.matrices[matrix_key]=self.family.lenses[target].jacobians[site].to(weight.device)
            result=self.matrices[matrix_key].T@u
        norm=result.norm()
        if not torch.isfinite(norm) or norm<=0:
            raise ValueError('Invalid causal direction')
        result=result/norm
        self.cache[key]=result
        return result

    def random(self,site,key):
        seed=int.from_bytes(hashlib.sha256(f'42:{site}:{key}'.encode()).digest()[:8],'little')
        gen=torch.Generator(device='cpu').manual_seed(seed)
        result=torch.randn(self.model.d_model,generator=gen)
        return (result/result.norm()).to(self.model.hf_model.lm_head.weight.device)


@contextmanager
def intervention(model,bank,*,sites,source=None,destination=None,mode='swap',kind='own',
                 strength=1.0,positions=None,scales=None,random_key='control'):
    handles=[]
    if mode not in ('swap','transfer','random_matched','ablate','inject','random_inject'):
        raise ValueError('Unknown operator')
    # Prepare before installing hooks so validation failure leaves no partial edit.
    prepared={}
    for site in sites:
        first=bank.get(site,source,kind) if source is not None else None
        second=bank.get(site,destination,kind) if destination is not None else None
        random=bank.random(site,random_key) if mode.startswith('random') else None
        prepared[site]=(first,second,random)
    def make_hook(site):
        first,second,random=prepared[site]
        selector=slice(None) if positions is None else list(positions)
        def hook(module,args,output):
            h=output[:,selector,:].float()
            if mode=='swap':delta=swap_delta(h,first,second)
            elif mode=='transfer':delta=transfer_delta(h,first,second)
            elif mode=='random_matched':
                delta=swap_delta(h,first,second).norm(dim=-1,keepdim=True)*random
            elif mode=='ablate':delta=-(h@first).unsqueeze(-1)*first
            else:
                if scales is None:raise ValueError('Injection requires clean activation scales')
                delta=scales[site]*(second if mode=='inject' else random)
            result=output.clone()
            result[:,selector,:]=(h+strength*delta).to(output.dtype)
            return result
        return hook
    try:
        for site in sites:
            handles.append(model.layers[site].register_forward_hook(make_hook(site)))
        yield
    finally:
        for handle in handles:handle.remove()
