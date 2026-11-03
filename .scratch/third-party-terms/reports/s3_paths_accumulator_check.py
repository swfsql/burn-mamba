# Scalar check of TRI/mamba3_siso_fwd.py: the kernel's final acc_ssm_states equals
# the recurrence state at a call boundary, and differs inside the call.
import numpy as np
rng = np.random.default_rng(0)
def coeffs(T):
    return rng.uniform(.2,1.5,T), rng.uniform(.05,.95,T), -rng.uniform(.1,2.,T), rng.normal(size=T), rng.normal(size=T)
def recur(h, pk, pv, dt, lam, A, k, v):
    hs=[]
    for t in range(len(dt)):
        a=np.exp(A[t]*dt[t])
        h = a*h + (1-lam[t])*dt[t]*a*pk*pv + lam[t]*dt[t]*k[t]*v[t]
        pk,pv = k[t],v[t]; hs.append(h)
    return np.array(hs)
def kernel(h, pk, pv, dt, lam, A, k, v, L):
    # lines 345-352 (seed), 276-287 (scale), 421-425 (chunk update)
    T=len(dt); acc=h+pv*pk*dt[0]*(1-lam[0]); da=A*dt; accs=[]
    for c in range(0,T,L):
        idx=np.arange(c,min(c+L,T))
        sg=np.array([dt[i+1]*(1-lam[i+1]) if i+1<T else 0.0 for i in idx])
        scale=sg+dt[idx]*lam[idx]
        cs=np.cumsum(da[idx]); last=cs[-1]; rev=last-cs
        acc=acc*np.exp(last)+np.sum(k[idx]*scale*v[idx]*np.exp(rev))
        accs.append(acc)
    return acc,np.array(accs)
T,L=12,4
dt,lam,A,k,v=coeffs(T)
h0,pk0,pv0=0.7,0.3,-0.4
hs=recur(h0,pk0,pv0,dt,lam,A,k,v)
acc,accs=kernel(h0,pk0,pv0,dt,lam,A,k,v,L)
print("final acc - final state:",acc-hs[-1])
print("chunk-end acc - state  :",accs-hs[L-1::L])
