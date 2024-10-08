import sys
import netket as nk
import matplotlib.pyplot as plt
from netket.operator.spin import sigmax,sigmaz,identity
from scipy.sparse.linalg import eigsh
import jax.numpy as jnp
import flax
import jax
import flax.linen as nn 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import random as rn

## Define the parameters
#parameters=sys.argv
#n_par=len(parameters)
#print(parameters)
#parameters=[int(parameters[x]) for x in range(1,n_par)]
#...
#print(parameters)
#N=parameters[0]
#Gamma=parameters[1]*(-0.01)
#V=-1.0
#n_sample=parameters[2]
#print(n_sample)
#n_run=parameters[3]
#n_mean=parameters[4]
#each=bool(parameters[5])
#k=parameters[6]
#L=N

#hi=nk.hilbert.Spin(s=1 / 2,N=L)
## Define the Hamiltonian

def Ham(Gamma,V,L,hi):
    H=sum([Gamma*sigmax(hi,i) for i in range(L)])
    H+=sum([V*sigmaz(hi,i)*sigmaz(hi,(i+1)%L) for i in range(L)])
    return H

def Diag(H,eigv=False):
    sp_h=H.to_sparse()
    eig_vals, eig_vecs = eigsh(sp_h,k=1,which="SA")
    if eigv==False:
        return eig_vecs
    else:
        return eig_vals,eig_vecs
    

def break_sym(eps):
    Vin=sum([rn.random()*eps*sigmaz(hi,i)+rn.random()*eps/2.0*identity(hi) for i in range(N)])
    return Vin
    



#H=Ham(Gamma,V,L,hi)
#sp_h=H.to_sparse()
#eig_vals, eig_vecs = eigsh(sp_h,k=1,which="SA")
#print("eig vals: ",eig_vals)


def change_to_int(x,L):
    Aux=jnp.array([2**(L-1-i) for i in range(L)])
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
    L:float
    def setup(self):
        self.aux=jnp.array(self.eig_vec)
        self.j1=self.param("j1", nn.initializers.normal(),(1,),float)
    
    def __call__(self,x):
        indices = change_to_int(x,self.L)
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
    
def S_PCA(D,eps,V,h,exvar=False):
    pca=PCA()
    lenght=len(D)
    if np.abs(h)<1.5:
        D[:int(lenght/2),:]=(-1)*D[:int(lenght/2),:]        
    E=pca.fit(D)
    vr=E.explained_variance_ratio_ 
    Sin=-vr[0:min_d(vr,eps)]*np.log(vr[0:min_d(vr,eps)])
    if exvar==False:
        return np.sum(Sin)
    else:
        return [np.sum(Sin),vr]
    

def S_PCA_WF(D,eps):
    pca=PCA()
    E=pca.fit(D)
    vr=E.explained_variance_ratio_ 
    Sin=-vr[0:min_d(vr,eps)]*np.log(vr[0:min_d(vr,eps)])
    return np.sum(Sin),np.array(vr)

def M(D,Nsamples,L):
    return np.sum(D)/(Nsamples*L)

def SiSj(D,Nsamples,L):
    E= np.array([D[:,0]*D[:,i] for i in range(1,L)])
    return np.sum(E,axis=1)/(Nsamples)



def v_state(model,n_sample,hi,n_run,L,V,h,q,H,each=False,var=False):
    sampler = nk.sampler.MetropolisLocal(hi) #Sampler in the Hilbert Space
    vstate= nk.vqs.MCState(sampler,model,n_samples=n_sample) ;
    optimizer= nk.optimizer.Sgd(learning_rate=0.05) ;
    gs=nk.driver.VMC(H, optimizer, variational_state=vstate,preconditioner=nk.optimizer.SR(diag_shift=0.1)) ;
    log=nk.logging.RuntimeLog() ;
    
    energy_history=np.zeros(n_run)
    S_hist=np.zeros(n_run)
    Kback=np.zeros(n_run)
    eps=10**(-6)
    it=0

    if var==True:
        ex_var=np.zeros((n_run,L))
  
        for step in gs.iter(n_run):
            A=np.array(vstate.samples).reshape((n_sample,L))
            E= vstate.expect(H);
            
            S_hist[it],ex_var[it,:]=S_PCA(A,eps,V,h,exvar=var)
            Kback[it]=PDF(A,q,L)
            energy_history[it]=E.mean.real
            it+=1
        return Kback,energy_history,S_hist,ex_var
    else:
        for step in gs.iter(n_run):
            A=np.array(vstate.samples).reshape((n_sample,L))
            E= vstate.expect(H);
            S_hist[it]=S_PCA(A,eps,V,h,exvar=var)
            Kback[it]=PDF(A,q,L)
            energy_history[it]=E.mean.real
            it+=1
            
        return Kback,energy_history,S_hist




def v_state_steady(model,n_sample,hi,n_run,L,Gamma,dh,Nh,Hts,Eigvs,each=False,exvar=False):
    sampler = nk.sampler.MetropolisLocal(hi) #Sampler in the Hilbert Space
    vstate= nk.vqs.MCState(sampler,model,n_samples=n_sample) ;
    optimizer= nk.optimizer.Sgd(learning_rate=0.05) ;
    log=nk.logging.RuntimeLog() ;    
    energy_history=np.zeros(Nh)
    S_hist=[0 for i in range(Nh)]
    Kback=np.zeros(Nh)
    m=np.zeros(Nh)
    #Hts=[Ham(Gamma+dh*i,-1.0,L,hi) for i in range(Nh)]
    gs=nk.driver.VMC(Hts[0], optimizer, variational_state=vstate,preconditioner=nk.optimizer.SR(diag_shift=0.1))
    eps=10**(-6)
    
    if each==False:
        for i in range(Nh):
            gs._ham=Hts[i];
            gs.advance(n_run) ;
            A=np.array(vstate.samples).reshape((n_sample,L))
            E= vstate.expect(Hts[i]);
            Kback[i]=PDF(A,(Eigvs[i][:,0])**2,L)
            S_hist[i]=S_PCA(A,eps,-1.0,Gamma+dh*i,exvar)
            m[i]=M(A,n_sample,L)
            energy_history[i]=E.mean.real
        return m,energy_history,Kback,S_hist

    else:
        s_is_j=np.zeros((Nh,L-1))
        for i in range(Nh):
            gs._ham=Hts[i];
            gs.advance(n_run);
            A=np.array(vstate.samples).reshape((n_sample,L))
            E= vstate.expect(Hts[i]);
            Kback[i]=PDF(A,(Eigvs[i][:,0])**2,L)
            S_hist[i]=S_PCA(A,eps,-1.0,Gamma+dh*i,exvar)
            m[i]=M(A,n_sample,L)
            energy_history[i]=E.mean.real
            s_is_j[i]=SiSj(A,n_sample,L)
    return s_is_j,m,energy_history,Kback,S_hist



def Exact_Calculation(n_sample,n_run,n_mean,L,eig_st,hi):
    sampler = nk.sampler.MetropolisLocal(hi)
    eps=10**(-6)
    S_exact=np.zeros((n_mean))
    Pv=np.zeros((n_mean,L))
    aux=tuple(eig_st)

    for i in range(n_mean):
        vstate_exact = nk.vqs.MCState(sampler, EWF(eig_vec=aux,L=L), n_samples=n_sample)
        A_exact=np.array(vstate_exact.samples).reshape((n_sample,L))
        S_exact[i],Pv[i]=S_PCA_WF(A_exact,eps)
        
  
    S_error=np.std(S_exact)/(np.sqrt(n_sample))
    S_exact=np.mean(S_exact)
    Pv=np.mean(Pv,axis=0)
    
    kback_exact=np.sum(np.array([(x**2)*np.log(x**2) for x in eig_st]))
    return S_error,S_exact,kback_exact,Pv
                                      

def Exact_Calculation_steady(n_sample,n_run,n_mean,L,Nh,dh):
    sampler = nk.sampler.MetropolisLocal(hi)
    eps=10**(-6)
    S_exact=np.zeros((Nh,n_mean))
    m_exact=np.zeros((Nh,n_mean))
    s_is_j_exact=np.zeros((Nh,n_mean,L-1))
    eig_vals=np.zeros(Nh)
    Hts=[Ham(Gamma,V+dh*j,L,hi) for j in range(Nh)] 
    for j in range(Nh):
        sp_h=Hts[j].to_sparse()
        eig_vals[j], eig_vecs = eigsh(sp_h,k=1,which="SA")

        for i in range(n_mean):
            vstate_exact = nk.vqs.MCState(sampler, EWF(eig_vec=tuple(eig_vecs[:,0])), n_samples=n_sample)
            A_exact=np.array(vstate_exact.samples).reshape((n_sample,L))

            S_exact[j,i]=S_PCA(A_exact,eps)
            m_exact[j,i]=M(A_exact,n_sample,L)        
            s_is_j_exact[j,i,:]=SiSj(A_exact,n_sample,L)

    S_error=np.std(S_exact,axis=1)
    S_exact=np.mean(S_exact,axis=1)
    m_error=np.std(m_exact,axis=1)
    m_exact=np.mean(m_exact,axis=1)
    var_sisj_exact=np.var(s_is_j_exact,axis=1)
    s_is_j_exact=np.mean(s_is_j_exact,axis=1)
        
    return eig_vals,S_exact,m_exact,var_sisj_exact,s_is_j_exact 


def PDF(x,q,L):
    aux=change_to_int(x,L)
    aux=np.sort(aux)
    size=len(aux)
    aux_1=aux.tolist()
    aux_2=list(set(aux_1))
    aux_3=[aux_1.count(xx)/(size*1.0) for xx in aux_2]
    kback=sum([q[aux_2[ii]]*(np.log(q[aux_2[ii]])-np.log(aux_3[ii])) for ii in range(len(aux_2))])
    return kback

    



