import sys
import netket as nk
import matplotlib.pyplot as plt
from netket.operator.spin import sigmax,sigmaz
from scipy.sparse.linalg import eigsh
import jax.numpy as jnp
import flax
import jax
import flax.linen as nn 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

## Define the parameters
parameters=sys.argv
n_par=len(parameters)
print(parameters)
parameters=[int(parameters[x]) for x in range(1,n_par)]
#...
print(parameters)
N=parameters[0]
Gamma=parameters[1]*(-0.01)
V=-1.0
n_sample=parameters[3]
print(n_sample)
n_run=parameters[4]
n_mean=parameters[5]
each=bool(parameters[6])
k=parameters[7]
L=N

hi=nk.hilbert.Spin(s=1 / 2,N=L)
## Define the Hamiltonian

def Ham(Gamma,V):
    H=sum([Gamma*sigmax(hi,i) for i in range(N)])
    H+=sum([V*sigmaz(hi,i)*sigmaz(hi,(i+1)%N) for i in range(N)])
    return H
    
H=Ham(Gamma,V)
sp_h=H.to_sparse()
eig_vals, eig_vecs = eigsh(sp_h,k=1,which="SA")
print("eig vals: ",eig_vals)


def change_to_int(x):
    Aux=jnp.array([2**(N-1-i) for i in range(N)])
    Z=jnp.array(jnp.mod(1+x,3)/2,int)
    return np.sum(Aux*Z,axis=-1)


## Define the variational Ansatz




class MF(nn.Module):
    @nn.compact # What is this ?
    def __call__(self,x): #x.shape(Nsamples,L)
        lam= self.param("lambda", nn.initializers.normal(),(1,),float)
        p= nn.log_sigmoid(lam*x) ## How does the initializers work?
        return 0.5*jnp.sum(p,axis=-1) 

class EWF(nn.Module):
    eig_vec:tuple
    def setup(self):
        self.aux=jnp.array(self.eig_vec)
        self.j1=self.param("j1", nn.initializers.normal(),(1,),float)
    
    def __call__(self,x):
        indices = change_to_int(x)
        A = [self.aux[idx] for idx in indices]
        return jnp.log(jnp.array(A))
        
class JasShort(nn.Module):
    @nn.compact
    def __call__(self,x):
        j1=self.param("j1", nn.initializers.normal(),(1,),float)
        j2=self.param("j2", nn.initializers.normal(),(1,),float)
        ## Nearest neighbor correlations
        corr1=x*jnp.roll(x,-1,axis=-1)
        corr2=x*jnp.roll(x,-2,axis=-1)
        return jnp.sum(j1*corr1+j2*corr2,axis=-1)

class FFN(nn.Module):
    alpha : int = 1
    @nn.compact
    def __call__(self, x):
        dense = nn.Dense(features=self.alpha * x.shape[-1])
        y = dense(x)
        y = nn.relu(y)
        return jnp.sum(y, axis=-1)

### Define the functions to be used:

def min_d(vr,eps):
    for i in range(len(vr)):
        if np.abs(vr[i])<eps:
            return i
    
def S_PCA(D,eps,V,h,exact=False):
    pca=PCA()
    lenght=len(D)
    if np.abs(h)<0.9 and not exact:
        D[:int(lenght/2),:]=(-1)*D[:int(lenght/2),:]
        
    E=pca.fit(D)
    vr=E.explained_variance_ratio_ 
    Sin=-vr[0:min_d(vr,eps)]*np.log(vr[0:min_d(vr,eps)])
    return np.sum(Sin)
    
def M(D,Nsamples,L):
    return np.sum(D)/(Nsamples*L)

def SiSj(D,Nsamples,L):
    E= np.array([D[:,0]*D[:,i] for i in range(1,L)])
    return np.sum(E,axis=1)/(Nsamples)


def v_state(model,n_sample,hi,n_run,L,V,h,each=False):
    sampler = nk.sampler.MetropolisLocal(hi) #Sampler in the Hilbert Space
    vstate= nk.vqs.MCState(sampler,model,n_samples=n_sample) ;
    optimizer= nk.optimizer.Sgd(learning_rate=0.05) ;
    gs=nk.driver.VMC(H, optimizer, variational_state=vstate,preconditioner=nk.optimizer.SR(diag_shift=0.1)) ;
    log=nk.logging.RuntimeLog() ;
    
    energy_history=np.zeros(n_run)
    S_hist=np.zeros(n_run)
    m=np.zeros(n_run)
    eps=10**(-6)
    
    if each==False:
        s_is_j=np.zeros(L-1)
        it=0
            #gs.run(n_iter=1,out=None,show_progress=False) ;
        for step in gs.iter(n_run):
            A=np.array(vstate.samples).reshape((n_sample,L))
            E= vstate.expect(H);
            S_hist[it]=S_PCA(A,eps,V,h)
            m[it]=M(A,n_sample,L)
            energy_history[it]=E.mean.real
            it+=1          
        s_is_j=SiSj(A,n_sample,L)
    else:
        s_is_j=np.zeros((n_run,L-1))
        #for i in range(n_run):
        it=0
        for step in gs.iter(n_run):
            #gs.run(n_iter=1,out=None,show_progress=False) ;
            A=np.array(vstate.samples).reshape((n_sample,L))
            E= vstate.expect(H);
            S_hist[it]=S_PCA(A,eps,V,h)
            m[it]=M(A,n_sample,L)
            energy_history[it]=E.mean.real
            s_is_j[it]=SiSj(A,n_sample,L)
            it+=1
    return s_is_j,m,energy_history,S_hist


def v_state_steady(model,n_sample,hi,n_run,L,dh,Nh,each=False):
    sampler = nk.sampler.MetropolisLocal(hi) #Sampler in the Hilbert Space
    vstate= nk.vqs.MCState(sampler,model,n_samples=n_sample) ;
    optimizer= nk.optimizer.Sgd(learning_rate=0.05) ;
    log=nk.logging.RuntimeLog() ;    
    energy_history=np.zeros(Nh)
    S_hist=np.zeros(Nh)
    m=np.zeros(Nh)
    Hts=[Ham(Gamma+dh*i,V) for i in range(Nh)]
    gs=nk.driver.VMC(Hts[0], optimizer, variational_state=vstate,preconditioner=nk.optimizer.SR(diag_shift=0.1))
    eps=10**(-6)
    
    if each==False:
        for i in range(Nh):
            gs._ham=Hts[i];
            gs.advance(n_run) ;
            A=np.array(vstate.samples).reshape((n_sample,L))
            E= vstate.expect(Hts[i]);
            S_hist[i]=S_PCA(A,eps,V,Gamma+dh*i)
            m[i]=M(A,n_sample,L)
            energy_history[i]=E.mean.real
        return m,energy_history,S_hist
    else:
        s_is_j=np.zeros((Nh,L-1))
        for i in range(Nh):
            gs._ham=Hts[i];
            gs.advance(n_run);
            A=np.array(vstate.samples).reshape((n_sample,L))
            E= vstate.expect(Hts[i]);
            S_hist[i]=S_PCA(A,eps,V,Gamma+dh*i)
            m[i]=M(A,n_sample,L)
            energy_history[i]=E.mean.real
            s_is_j[i]=SiSj(A,n_sample,L)
    return s_is_j,m,energy_history,S_hist

def Exact_Calculation(n_sample,n_run,n_mean,L,eig_st,corr=True):
    sampler = nk.sampler.MetropolisLocal(hi)
    eps=10**(-6)
    S_exact=np.zeros((n_mean))
    m_exact=np.zeros((n_mean))
    if corr==True:
        s_is_j_exact=np.zeros((n_mean,L-1))
    aux=tuple(eig_st)
    
    for i in range(n_mean):
        vstate_exact = nk.vqs.MCState(sampler, EWF(eig_vec=aux), n_samples=n_sample)
        A_exact=np.array(vstate_exact.samples).reshape((n_sample,L))
        S_exact[i]=S_PCA(A_exact,eps,1,1,True)
        m_exact[i]=M(A_exact,n_sample,L)
        if corr==True:
            s_is_j_exact[i,:]=SiSj(A_exact,n_sample,L)

    S_exact=np.mean(S_exact)
    m_exact=np.mean(m_exact)
    
    if corr==True:
        var_sisj_exact=np.var(s_is_j_exact,axis=0)
        s_is_j_exact=np.mean(s_is_j_exact,axis=0)    
        return S_exact,m_exact,var_sisj_exact,s_is_j_exact
    else:
        return S_exact,m_exact
                                      

def Exact_Calculation_steady(n_sample,n_run,n_mean,L,Nh,dh):
    sampler = nk.sampler.MetropolisLocal(hi)
    eps=10**(-6)
    S_exact=np.zeros((Nh,n_mean))
    m_exact=np.zeros((Nh,n_mean))
    s_is_j_exact=np.zeros((Nh,n_mean,L-1))
    eig_vals=np.zeros(Nh)
    Hts=[Ham(Gamma,V+dh*j) for j in range(Nh)] 
    for j in range(Nh):
        sp_h=Hts[j].to_sparse()
        eig_vals[j], eig_vecs = eigsh(sp_h,k=1,which="SA")

        for i in range(n_mean):
            vstate_exact = nk.vqs.MCState(sampler, EWF(eig_vec=tuple(eig_vecs[:,0])), n_samples=n_sample)
            A_exact=np.array(vstate_exact.samples).reshape((n_sample,L))
            S_exact[j,i]=S_PCA(A_exact,eps)
            m_exact[j,i]=M(A_exact,n_sample,L)        
            s_is_j_exact[j,i,:]=SiSj(A_exact,n_sample,L)

    S_exact=np.mean(S_exact,axis=1)
    m_exact=np.mean(m_exact,axis=1)
    var_sisj_exact=np.var(s_is_j_exact,axis=1)
    s_is_j_exact=np.mean(s_is_j_exact,axis=1)
        
    return eig_vals,S_exact,m_exact,var_sisj_exact,s_is_j_exact 



