import numpy as np
import matplotlib.pyplot as plt
import class_WF
import var_nk
from scipy.sparse.linalg import eigsh    
import netket as nk
import sys

## CONSTANTS
eps=10**(-8)
dx=0.01
V=-1.0
n_method=3

## PARAMETERS
parameters=sys.argv
n_par=len(parameters)
parameters=[int(parameters[x]) for x in range(1,n_par)]
print(parameters)
L=parameters[0]
W=parameters[1]
Gamma=parameters[2]*(-dx)
GammaF=parameters[3]*(-dx)
n_samples=parameters[4]
n_run=parameters[5]
n_mean=parameters[6]
NG=parameters[7]

#INSERT PUBLISHER DETAILS AND INITIALIZE IT
name_var=["M","L","W","NS","NR","G","GF"]
var=[n_method,L,W,n_samples,n_run,parameters[1],parameters[2]]
variables=["S","E","Id"]
pub=class_WF.publisher(name_var,var,variables)
pub.create()

#INITIALIZE OBJECTS
hi=nk.hilbert.Spin(s=1 / 2,N=L*W)
H=class_WF.Ham_W(Gamma,V,L,W,hi)
eig_vecs=class_WF.Diag(H)
E_WF=class_WF.WF(L*W,var_nk.EWF(eig_vec=tuple(np.abs(eig_vecs[:,0])),L=L*W),H,n_samples)

#ITERATION OVER THE GAMMA VALUES:

for gg in range(NG):
    G=Gamma+(GammaF-Gamma)/NG*gg
    H=class_WF.Ham_W(G,-1.0,L,W,hi)
    eig_vecs=class_WF.Diag(H)
    sampler=nk.sampler.MetropolisLocal(hi)
    aux=np.zeros((n_samples,3))
    E_WF.change_H(H)
    for hh in range(n_samples):
        New_state=nk.vqs.MCState(sampler,var_nk.EWF(eig_vec=tuple(np.abs(eig_vecs[:,0])),L=L*W),n_samples=n_samples)
        E_WF.change_state(New_state)
        A=E_WF.sampling()
        aux[hh,0]=E_WF.compute_PCA(eps,A=A)
        aux[hh,1]=E_WF.compute_E()
        aux[hh,2]=E_WF.compute_ID(eps,A=A)
    
    dS=np.std(aux[:,0])/np.sqrt(n_samples)
    S=np.mean(aux[:,0])
    dE=np.std(aux[:,1])/np.sqrt(n_samples)
    E=np.mean(aux[:,1])
    dID=np.std(aux[:,2])/np.sqrt(n_samples)
    ID=np.mean(aux[:,2])
    pub.write([dS,S,dE,E,dID,ID])

pub.close()





