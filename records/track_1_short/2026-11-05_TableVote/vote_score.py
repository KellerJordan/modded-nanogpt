"""Score the plain vote on one trained network. Usage: vote_score.py RUN [K6,K10,K20]. Constants are fit only on shard-900 positions.

The vote treats the network as a prior and the table's counts as evidence: for each match length in turn,
P = (hits + k * P) / (total + k). A context with no match leaves P unchanged. Nothing here runs on the clock.
"""
import json,sys
from pathlib import Path
import numpy as np,torch
root=Path('/workspace/cpu-single-bridge');B=131072;NV,NX=80,25;d='cuda'
run=root/f'virtual_multiscale/results/{sys.argv[1]}'
nll=np.concatenate([np.load(run/f'validation_rank0_batch{i}.npz')['nll'] for i in range(NV+NX)]).astype(np.float64)
L=np.concatenate([np.load(run/f'validation_rank0_batch{i}.npz')['level_counts'] for i in range(NV+NX)]).astype(np.float64)
net=torch.from_numpy(np.exp(-nll)).to(d);n=torch.from_numpy(L[:,:3]).to(d);h=torch.from_numpy(L[:,3:]).to(d);val=torch.arange(len(net),device=d)<NV*B;fit=~val
def vote(k,mask=None):
    P=net if mask is None else net[mask];N=n if mask is None else n[mask];H=h if mask is None else h[mask]
    for j in range(3):P=(H[:,j]+k[j]*P)/(N[:,j]+k[j])
    return P
log_k=torch.zeros(3,device=d,dtype=torch.float64,requires_grad=True);o=torch.optim.Adam([log_k],lr=.05)
for _ in range(600):
    o.zero_grad();l=-torch.log(vote(log_k.exp(),fit)).mean();l.backward();o.step()
own=log_k.exp().detach()
r=dict(run=sys.argv[1],network=-torch.log(net[val]).mean().item(),own_constants=own.tolist(),vote_own=-torch.log(vote(own,val)).mean().item(),match_rate=[(n[val][:,j]>0).double().mean().item() for j in range(3)],shard900_network=-torch.log(net[fit]).mean().item())
if len(sys.argv)>2:
    k=torch.tensor([float(a) for a in sys.argv[2].split(',')],device=d,dtype=torch.float64);r.update(fixed_constants=k.tolist(),vote_fixed=-torch.log(vote(k,val)).mean().item())
# Reference: the same voters with weights learned from evidence sizes, to see what plainness costs.
X=torch.log1p(n).float();mu,sd=X[fit].mean(0),X[fit].std(0).clamp_min(.1);Z=(X-mu)/sd
torch.manual_seed(0);g=torch.nn.Sequential(torch.nn.Linear(3,64),torch.nn.GELU(),torch.nn.Linear(64,4)).to(d);o=torch.optim.AdamW(g.parameters(),lr=2e-3);ti=torch.nonzero(fit)[:,0]
E=torch.cat([net[:,None],torch.where(n>0,h/n.clamp_min(1),net[:,None])],1)
for _ in range(1500):
    j=ti[torch.randint(len(ti),(4096,),device=d)];l=-torch.log((torch.softmax(g(Z[j]),1).double()*E[j]).sum(1).clamp_min(1e-300)).mean();o.zero_grad();l.backward();o.step()
with torch.no_grad():
    s=torch.nonzero(val)[:,0];r['fitted_gate_reference']=sum(-torch.log((torch.softmax(g(Z[c]),1).double()*E[c]).sum(1).clamp_min(1e-300)).sum().item() for c in s.split(1<<19))/len(s)
print(json.dumps(r),flush=True)
