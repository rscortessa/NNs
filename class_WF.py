import matplotlib.pyplot as plt
import numpy as np
import netket as nk
from typing import TextIO
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
from math import log,sqrt,pow,exp,lgamma,pi                                                                                           
from sklearn.neighbors import NearestNeighbors
from TId import sets,neighbors,n_points,Volume_ratio,func,roots


def min_d(vr,eps):
    for i in range(len(vr)):
        if np.abs(vr[i])<eps:
            return i

def Ham(Gamma,V,L,hi):                                                                                                                 
    H=sum([Gamma*sigmax(hi,i) for i in range(L)])                                                                                      
    H+=sum([V*sigmaz(hi,i)*sigmaz(hi,(i+1)%L) for i in range(L)])                                                                      
    return H    

def Ham_W(Gamma,V,L,W,hi):
    H=sum([Gamma*sigmax(hi,i%L+int(i/L)*L) for i in range(L*W)])
    H+=sum([V*(sigmaz(hi,i%L+int(i/L)*L)*sigmaz(hi,(i+1)%L+int(i/L)*L)+(W>1)*sigmaz(hi,i%L+int(i/L)*L)*sigmaz(hi,i%L+((int(i/L)+1)%W)*L)) for i in range(L*W)])
    return H



def Diag(H,eigv=False):
    sp_h=H.to_sparse()
    eig_vals, eig_vecs = eigsh(sp_h,k=1,which="SA")
    if eigv==False:
        return eig_vecs
    else:
        return eig_vals,eig_vecs


def S_PCA(D,eps,exvar):                                                                                                                   
    pca=PCA()                                                                                                                          
    E=pca.fit(D)                                                                                                                       
    vr=E.explained_variance_ratio_                                                                                                     
    Sin=-vr[0:min_d(vr,eps)]*np.log(vr[0:min_d(vr,eps)])
    if exvar==False:                                                                                                                   
        return np.sum(Sin)                                                                                                             
    else:                                                                                                                              
        return [np.sum(Sin),vr]  


def Id(data,eps):                                                                                                                
    lenght=len(data)                                                                                                                   
    data=np.column_stack((data,np.random.rand(lenght)))                                                                                
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(data)                                                                 
    distances, indices = nbrs.kneighbors(data)                                                                                         
    Nele=distances.shape[0]                                                                                                            
    aux=distances[:,2]/distances[:,1]                                                                                                  
    aux[np.isnan(aux)]=1.0                                                                                                             
    aux[np.isinf(aux)]=1.0                                                                                                             
    aux[aux<eps]=1.0                                                                                                                                                                                                                                      
    dim=Nele/(np.sum(np.log(aux)))                                                                                                                                                                                    
    return dim   




class WF:
    L:int
    N_sample:int
    H_space: nk.hilbert.Spin
    user_H:nk.operator.LocalOperator                                                                                                                                   
    user_state:nk.vqs.MCState
    user_sampler:nk.sampler.MetropolisLocal
    user_optimizer:nk.optimizer.Sgd
    user_driver:nk.driver.VMC
    
    def __init__(self,L,model,H,N_samples):
        self.L=L
        self.H=H
        self.N_sample=N_samples
        self.H_space=nk.hilbert.Spin(s=1/2,N=L)
        self.user_sampler=nk.sampler.MetropolisLocal(self.H_space)
        self.user_state=nk.vqs.MCState(self.user_sampler,model,n_samples=N_samples)        
        self.user_optimizer=nk.optimizer.Sgd(learning_rate=0.05)
        self.user_driver=nk.driver.VMC(self.H, self.user_optimizer, variational_state=self.user_state,preconditioner=nk.optimizer.SR(diag_shift=0.1))
    def sampling(self):
        return np.array(self.user_state.samples).reshape((self.N_sample,self.L))
    def advance(self,n_run):
        self.user_driver.advance(n_run)
    def iteration(self,n_run):
        return self.user_driver.iter(n_run)
    def change_H(self,H):
        self.H=H
        self.user_driver._ham=H
    def change_state(self,new_state):
        self.user_state=new_state
        self.user_driver=nk.driver.VMC(self.H, self.user_optimizer, variational_state=self.user_state,preconditioner=nk.optimizer.SR(diag_shift=0.1))
        
    def compute_PCA(self,eps,exvar=False,A=None):
        if A is None:
            A=self.sampling()
        return S_PCA(A,eps,exvar)
    def compute_ID(self,eps,A=None):
        if A is None:
            A=self.sampling()
        return Id(A,eps)
    
    def compute_3ID(self,t1,t2,cutoff,eps,A=None,neigh=None,weights=None,states=None):
        if A is None:
            A=self.sampling()
        if  weights or states is None:
            states,weights=sets(A)
        if neigh is None:
            neigh=neighbors(states)
            
        V=Volume_ratio(neigh,weights,t1,t2)
        min=roots(V,cutoff,100,eps,func,t1,t2)
        Ymin=func(min,t1,t2)-V
       
        return min,Ymin,V
        
    def compute_E(self):
        E=self.user_state.expect(self.H)
        return E.mean.real



    
class publisher:
    filename:str
    variables:list
    file:TextIO
    def __init__(self,name_var,var,variables):
        ms=len(var)
        self.variables=variables
        self.filename=""
        for i in range(ms):
            self.filename+=name_var[i]+str(var[i]) 
        self.filename+="".join(variables)
    def create(self):
        a=""
        self.file=open(self.filename+".txt","w")
        for i in range(len(self.variables)):
            a+="d"+self.variables[i]+"\t "+self.variables[i]+"\t "
        self.file.write(a+"\n") 
    def write(self,val_variables):
        self.file.write("\t".join(str(val) for val in val_variables) + "\n")
    def close(self):
        self.file.close()
