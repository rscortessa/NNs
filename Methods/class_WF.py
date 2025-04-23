import matplotlib.pyplot as plt
import numpy as np
import netket as nk
from typing import TextIO
import sys                                                                                                                            
import netket as nk                                                                                                                   
import matplotlib.pyplot as plt                                                                                                       
from netket.operator.spin import sigmax,sigmaz,sigmay,identity,sigmam,sigmap                                                                               
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
from Methods.TId import sets,neighbors,n_points,Volume_ratio,func,roots
from netket.hilbert import constraint
import equinox as eqx 
from functools import reduce

def min_d(vr,eps):
    for i in range(len(vr)):
        if np.abs(vr[i])<eps:
            return i

#NOT PERIODIC HAMILTONIAN:


def rotated_sigmax(angle):
    r_sigmax = np.array([[0, 1], [1, 0]])
    r_sigmaz = np.array([[1, 0], [0, -1]])
    return np.cos(angle)*r_sigmax+np.sin(angle)*r_sigmaz


def rotated_sigmaz(angle):
    r_sigmax = np.array([[0, 1], [1, 0]])
    r_sigmaz = np.array([[1, 0], [0, -1]])
    return np.cos(angle)*r_sigmaz-np.sin(angle)*r_sigmax

def isigmay():
    return np.array([[0,1],[-1,0]])

def rotated_IsingModel(angle,Gamma,L,hi):
     # Initialize Hamiltonian as a LocalOperator
    pseudo_sigma_x=rotated_sigmax(angle)
    pseudo_sigma_z=rotated_sigmaz(angle)
    H = nk.operator.LocalOperator(hi)

    # Add 2 body- interactions
    for i in range(L - 1):
        H -= nk.operator.LocalOperator(hi, np.kron(pseudo_sigma_z,pseudo_sigma_z), [i, i+1])
    # Add single body term
    for i in range(L):
        H -= Gamma * nk.operator.LocalOperator(hi,pseudo_sigma_x,[i])
    return H
def rotated_BROKEN_Z2IsingModel(angle,Gamma,L,hi):
     # Initialize Hamiltonian as a LocalOperator
    pseudo_sigma_x=rotated_sigmax(angle)
    pseudo_sigma_z=rotated_sigmaz(angle)
    H = nk.operator.LocalOperator(hi)
    eps=10**(-4)
    # Add 2 body- interactions
    for i in range(L - 1):
        H -= nk.operator.LocalOperator(hi, np.kron(pseudo_sigma_z,pseudo_sigma_z), [i, i+1])
    # Add single body term
    for i in range(L):
        H -= Gamma * nk.operator.LocalOperator(hi,pseudo_sigma_x,[i])-eps*nk.operator.LocalOperator(hi,pseudo_sigma_z,[i])
    return H



def rotated_TWOIsingModel(angle,Gamma,L,hi):
     # Initialize Hamiltonian as a LocalOperator
    pseudo_sigma_x=rotated_sigmax(angle)
    pseudo_sigma_z=rotated_sigmaz(angle)
    H = nk.operator.LocalOperator(hi)

    # Add 2 body- interactions
    for i in range(L - 1):
        H -= nk.operator.LocalOperator(hi, np.kron(pseudo_sigma_z,pseudo_sigma_z), [i, i+1])
        H -= nk.operator.LocalOperator(hi, np.kron(pseudo_sigma_z,pseudo_sigma_z), [i, i+1+L])
    # Add single body term
    for i in range(2*L):
        H -= Gamma * nk.operator.LocalOperator(hi,pseudo_sigma_x,[i])
    return H

def rotated_LONGIsingModel(angle,Gamma,L,hi):
     # Initialize Hamiltonian as a LocalOperator
    pseudo_sigma_x=rotated_sigmax(angle)
    pseudo_sigma_z=rotated_sigmaz(angle)
    H = nk.operator.LocalOperator(hi)

    # Add 2 body- interactions
    for i in range(L - 2):
        H -= nk.operator.LocalOperator(hi, np.kron(pseudo_sigma_z,pseudo_sigma_z), [i, i+1])
        H -= nk.operator.LocalOperator(hi, np.kron(pseudo_sigma_z,pseudo_sigma_z), [i, i+2])
        
    H -= nk.operator.LocalOperator(hi, np.kron(pseudo_sigma_z,pseudo_sigma_z), [L-2,L-1])
    # Add single body term
    for i in range(L):
        H -= Gamma * nk.operator.LocalOperator(hi,pseudo_sigma_x,[i])
    return H


def rotated_XYZModel(angle,Gamma,L,hi):
    
    pseudo_sigma_x=rotated_sigmax(angle)
    pseudo_sigma_z=rotated_sigmaz(angle)
    
    pseudo_sigma_p=(pseudo_sigma_x+isigmay())/2.0
    pseudo_sigma_m=(pseudo_sigma_x-isigmay())/2.0

    H = nk.operator.LocalOperator(hi)
    # Add 2 body- interactions
    for i in range(L - 1):
        H -= 2.0*nk.operator.LocalOperator(hi, np.kron(pseudo_sigma_p,pseudo_sigma_m), [i, i+1])
        H -= 2.0*nk.operator.LocalOperator(hi, np.kron(pseudo_sigma_m,pseudo_sigma_p), [i, i+1])
        H += Gamma*nk.operator.LocalOperator(hi, np.kron(pseudo_sigma_z,pseudo_sigma_z), [i, i+1])
    return H

def rotated_CIMModel(angle,Gamma,L,hi):
    
    pseudo_sigma_x=rotated_sigmax(angle)
    pseudo_sigma_z=rotated_sigmaz(angle)
    pseudo_sigma_p=(pseudo_sigma_x+isigmay())/2.0
    pseudo_sigma_m=(pseudo_sigma_x-isigmay())/2.0
    
    H=nk.operator.LocalOperator(hi)
    
    for i in range(L-2):
        H-=1.0*nk.operator.LocalOperator(hi, np.kron(pseudo_sigma_x,np.kron(pseudo_sigma_z,pseudo_sigma_x)), [i,i+1,i+2])
        H-=(1.0*Gamma)*nk.operator.LocalOperator(hi, np.kron(pseudo_sigma_p-pseudo_sigma_m,pseudo_sigma_p-pseudo_sigma_m), [i,i+1])     

    H-=(1.0*Gamma)*nk.operator.LocalOperator(hi, np.kron(pseudo_sigma_p-pseudo_sigma_m,pseudo_sigma_p-pseudo_sigma_m), [L-2,L-1])     

    return H

def string_order_parameter(angle,L,hi):
    #MATRICES
    pseudo_sigma_x=rotated_sigmax(angle)
    pseudo_sigma_z=rotated_sigmaz(angle)
    #LIST AND INDICES
    ops=[pseudo_sigma_x,isigmay()]+[pseudo_sigma_z for k in range(L-4)]+[isigmay(),pseudo_sigma_x]
    indices=[i for i in range(L)]
    #OPERATOR DEFINITION
    BigO=reduce(np.kron,ops)
    H=nk.operator.LocalOperator(hi,BigO,indices)
    return H




def rotated_m(angle,L,hi):
    pseudo_sigma_z=rotated_sigmaz(angle)
    M = nk.operator.LocalOperator(hi)
    for i in range(L):
        M+=nk.operator.LocalOperator(hi,pseudo_sigma_z, [i])
    return M

def parity_Matrix(angle,L):
    pseudo_sigma_x = rotated_sigmax(angle)
    ops = [pseudo_sigma_x for i in range(L)]
    P_array=reduce(np.kron,ops)
    return P_array

def parity_IsingModel(angle,L,hi):
     # Initialize Hamiltonian as a LocalOperator
    indices=[i for i in range(L)]
    P_array=parity_Matrix(angle,L)
    P = nk.operator.LocalOperator(hi,P_array,indices)
    return P

def Sz0Szj(angle,L,hi,j):
    pseudo_sigma_z=rotated_sigmaz(angle)
     # Initialize Hamiltonian as a LocalOperator
    identities=[pseudo_sigma_z]+[np.eye(2) for i in range(1,j)]+[pseudo_sigma_z]+[np.eye(2) for i in range(j+1,L)]
    identities_list=[i for i in range(0,L)]
    P= nk.operator.LocalOperator(hi,reduce(np.kron,identities),identities_list)
    return P

def Sx0Sxj(angle,L,hi,j):
    pseudo_sigma_x=rotated_sigmax(angle)
     # Initialize Hamiltonian as a LocalOperator
    identities=[pseudo_sigma_x]+[np.eye(2) for i in range(1,j)]+[pseudo_sigma_x]+[np.eye(2) for i in range(j+1,L)]
    identities_list=[i for i in range(0,L)]
    P= nk.operator.LocalOperator(hi,reduce(np.kron,identities),identities_list)
    return P

def to_array(A):
    return A.to_dense()


def parity_X(L,hi):
    aux_0=identity(hi)
    for i in range(L):
        aux_0*=sigmaz(hi,i)
    return aux_0


def parity_Z(L,hi):
    aux_0=identity(hi)
    for i in range(L):
        aux_0*=sigmax(hi,i)
    return aux_0


def IsingModel_X(Gamma,L,hi):
    
    PBC=False
    L_I=L-1
    aux_0=[(-1.0)*sigmax(hi,i)*sigmax(hi,(i+1)%L)-(1.0)*Gamma*sigmaz(hi,i)  for i in range(L_I)]
    H=sum(aux_0)-1.0*Gamma*sigmaz(hi,L-1)       
    return H


def IsingModel_Z(Gamma,L,hi):
    
    PBC=False
    L_I=L-1
    aux_0=[(-1.0)*sigmaz(hi,i)*sigmaz(hi,(i+1)%L)-(1.0)*Gamma*sigmax(hi,i)  for i in range(L_I)]
    H=sum(aux_0)-1.0*Gamma*sigmax(hi,L-1)       
    return H


def CLUSTER_HAM_Z(Gamma,L,hi):

    PBC=False
    if not PBC:
        L_A=L-2
        L_I=L-1
        aux_1=[-1.0*sigmax(hi,i)*sigmaz(hi,(i+1)%L)*sigmax(hi,(i+2)%L) for i in range(L_A)]
        aux_2=[-(1.0*Gamma)*(sigmap(hi,i)-sigmam(hi,i))*(sigmap(hi,(i+1)%L)-sigmam(hi,(i+1)%L)) for i in range(L_I)]
        H=sum(aux_1)+sum(aux_2)
    else:
        L_A=L
        L_I=L
        aux_1=[-1.0*sigmax(hi,i)*sigmaz(hi,(i+1)%L)*sigmax(hi,(i+2)%L) for i in range(L_A)]
        aux_2=[-(1.0*Gamma)*(sigmap(hi,i)-sigmam(hi,i))*(sigmap(hi,(i+1)%L)-sigmam(hi,(i+1)%L)) for i in range(L_I)]
        H=sum(aux_1)+sum(aux_2)
        
    return H

def CLUSTER_HAM_X(Gamma,L,hi):

    PBC=False
    if not PBC:
        L_A=L-2
        L_I=L-1
        aux_1=[-1.0*sigmaz(hi,i)*sigmax(hi,(i+1)%L)*sigmaz(hi,(i+2)%L) for i in range(L_A)]
        aux_2=[-1.0*Gamma*(sigmap(hi,i)-sigmam(hi,i))*(sigmap(hi,(i+1)%L)-sigmam(hi,(i+1)%L)) for i in range(L_I)]
        H=sum(aux_1)+sum(aux_2)
        
    else:
        L_A=L
        L_I=L
        aux_1=[-1.0*sigmaz(hi,i)*sigmax(hi,(i+1)%L)*sigmaz(hi,(i+2)%L) for i in range(L_A)]
        aux_2=[-1.0*Gamma*(sigmap(hi,i)-sigmam(hi,i))*(sigmap(hi,(i+1)%L)-sigmam(hi,(i+1)%L)) for i in range(L_I)]
        H=sum(aux_1)+sum(aux_2)
            
    return H

def CLUSTER_HAM_Y(Gamma,L,hi):

    PBC=False
    if not PBC:
        L_A=L-2
        L_I=L-1
        aux_1=[ complex(1.0)*sigmax(hi,i)*sigmay(hi,(i+1)%L)*sigmax(hi,(i+2)%L) for i in range(L_A)]
        aux_2=[ complex(Gamma)*(sigmaz(hi,i)*sigmaz(hi,(i+1)%L)) for i in range(L_I)]
        H=sum(aux_1)+sum(aux_2)
    else:
        L_A=L
        L_I=L
        aux_1=[ complex(1.0)*sigmax(hi,i)*sigmay(hi,(i+1)%L)*sigmax(hi,(i+2)%L) for i in range(L_A)]
        aux_2=[ complex(Gamma)*(sigmaz(hi,i)*sigmaz(hi,(i+1)%L)) for i in range(L_I)]
        H=sum(aux_1)+sum(aux_2)
        
    return H


def Ham_W(Gamma,V,L,W,hi):
    H=sum([Gamma*sigmax(hi,i%L+int(i/L)*L) for i in range(L*W)])
    H+=sum([V*(sigmaz(hi,i%L+int(i/L)*L)*sigmaz(hi,(i+1)%L+int(i/L)*L)+(W>1)*sigmaz(hi,i%L+int(i/L)*L)*sigmaz(hi,i%L+((int(i/L)+1)%W)*L)) for i in range(L*W)])
    return H

def Ham_PBC(Gamma,V,L,W,hi):
    A=[L,W]
    if W<=1:
        A=[L]
    graph=nk.graph.Grid(extent=A,pbc=True)
    H=sum([Gamma*sigmax(hi,i) for i in range(L*W)])
    H+=sum([V*sigmaz(hi,i)*sigmaz(hi,j) for (i,j) in graph.edges()])
    return H

def Ham_PBC_XYZ(Gamma):
    graph=nk.graph.Grid(extent=[L,W],pbc=True)
    H=sum([Gamma*(sigmaz(hi,i)*sigmaz(hi,j)+sigmax(hi,i)*sigmax(hi,j)+sigmay(hi,i)*sigmay(hi,j)) for (i,j) in graph.edges()])
    return H




def Diag(H,eigv=False):
    sp_h=H.to_sparse()
    eig_vals, eig_vecs = eigsh(sp_h,which="BE")
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
    user_model:{}
    user_H:nk.operator.LocalOperator                                                                                                                                   
    user_state:nk.vqs.MCState
    user_sampler:nk.sampler
    user_optimizer:nk.optimizer.Sgd
    user_driver:nk.driver.VMC
    
    def __init__(self,L,hilbert_space,sampler,preconditioner,model,H,N_samples):

        self.L=L
        self.H=H
        self.N_sample=N_samples
        self.H_space=hilbert_space
        self.user_sampler=sampler
        self.user_model=model
        self.user_state=nk.vqs.MCState(self.user_sampler,model,n_samples=N_samples)        
        self.user_optimizer=nk.optimizer.Momentum(learning_rate=0.05,beta=0.5)
        #self.user_optimizer=nk.optimizer.Sgd(learning_rate=0.05)
        #self.user_optimizer=nk.optimizer.Adam(learning_rate=0.1)
        self.user_driver=nk.driver.VMC(self.H, self.user_optimizer, variational_state=self.user_state,preconditioner=preconditioner)
    def sampling(self):
        return np.array(self.user_state.samples).reshape((self.N_sample,self.L))
    def advance(self,n_run):
        self.user_driver.advance(n_run)
        
    def run(self,obs,n_iter,log=None):
        if log:
            self.user_driver.run(n_iter=n_iter,obs=obs,out=log)
        else:
            self.user_driver.run(n_iter=n_iter,obs=obs)

    def save_params(self,i,log_var):
        log_var(i,self.user_driver.state.variables)

            
    def iteration(self,n_run):
        return self.user_driver.iter(n_run)
    def change_sampler(self,new_sampler):
        self.user_sampler=new_sampler
        self.user_state=nk.vqs.MCState(self.user_sampler,self.user_model,n_samples=self.N_sample)
        self.user_driver=nk.driver.VMC(self.H, self.user_optimizer, variational_state=self.user_state,preconditioner=nk.optimizer.SR(diag_shift=0.01))
        
    def change_H(self,H):
        self.H=H
        self.user_driver._ham=H
    def change_state(self,new_state):
        self.user_state=new_state
        self.user_driver=nk.driver.VMC(self.H, self.user_optimizer, variational_state=self.user_state,preconditioner=nk.optimizer.SR(diag_shift=0.01))
        
    def compute_PCA(self,eps,i=None,log=None,exvar=False,A=None,broken_z2=True):
            if A is None:
                A=self.sampling()
                if broken_z2:
                    size=int(len(A)/2)
                    A[:size]=(-1)*A[:size]
                
               
            S=S_PCA(A,eps,exvar)
            if i is None or log is None:
                return S
            else:
                SPCA={
                "Mean":jnp.array(S_PCA(A,eps,exvar))
                }
                log(i,SPCA)
                return 0;
            
                
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
    def create(self,jj=None):
        
        a=""
        if jj==None:
            self.file=open(self.filename+".txt","w")
        else:
            self.file=open(self.filename+".txt"+str(jj),"w")
            
        for i in range(len(self.variables)):
            a+="d"+self.variables[i]+"\t "+self.variables[i]+"\t "
        self.file.write(a+"\n")

    def name(self):
        return self.filename+".txt"
        
    def write(self,val_variables):
        self.file.write("\t".join(str(val) for val in val_variables) + "\n")
    def close(self):
        self.file.close()



# Define a custom parity constraint
class ParityConstraint(constraint.DiscreteHilbertConstraint):
    """
    Constraint to enforce that the number of -1/2 spins is even (even parity).
    """

    def __call__(self, x):
        # Compute the number of -1/2 spins
        num_down_spins = jnp.sum(x == -1, axis=-1)
        return num_down_spins % 2 == 0  # Keep only even-parity states

    def __hash__(self):
        return hash("ParityConstraint")

    def __eq__(self, other):
        return isinstance(other, ParityConstraint)
