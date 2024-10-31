import numpy as np
import matplotlib.pyplot as plt
import Methods.class_WF as class_WF
import Methods.var_nk as var_nk
from scipy.sparse.linalg import eigsh    
import netket as nk
import sys

## CONSTANTS
eps=10**(-8)
dx=0.01
V=-1.0
n_between=200
cutoff=2**L
t1=1
t2=5



## ALMOST CONSTANTS
n_neurons=1
n_layers=1


## PARAMETERS
parameters=sys.argv
n_par=len(parameters)
parameters=[int(parameters[x]) for x in range(1,n_par)]
print(parameters)
L=parameters[0]
Gamma=parameters[1]*(-dx)
GammaF=parameters[2]*(-dx)
n_samples=parameters[3]
n_run=parameters[4]
n_mean=parameters[5]
NG=parameters[6]
n_method=parameters[7]

if n_method==2:
    n_neurons=parameters[8]
    n_layers=parameters[9]

methods=[var_nk.MF(),var_nk.JasShort(),var_nk.FFN(alpha=n_neurons,layers=n_layers)]

    
model=methods[n_method]



#INSERT PUBLISHER DETAILS AND INITIALIZE IT

if n_method==2:
    name_var=["M","L","NS","NR","G","GF","t1","t2","NN","NL"]
    var=[n_method,L,n_samples,n_run,parameters[1],parameters[2],t1,t2,n_neurons,n_layers]
else:
    name_var=["M","L","NS","NR","G","GF"]
    var=[n_method,L,n_samples,n_run,parameters[1],parameters[2]]    
    
variables=["S","E","Id"]
pub=class_WF.publisher(name_var,var,variables)
pub.create()

#INITIALIZE OBJECTS
hi=nk.hilbert.Spin(s=1 / 2,N=L)
H=class_WF.Ham(Gamma,V,L,hi)
eig_vecs=class_WF.Diag(H)
E_WF=class_WF.WF(L,model,H,n_samples)

#ITERATION OVER THE GAMMA VALUES:

for gg in range(NG):

    G=Gamma+(GammaF-Gamma)/NG*gg
    H=class_WF.Ham(G,-1.0,L,hi)
    sampler=nk.sampler.MetropolisLocal(hi)
    E_WF.change_H(H)
    aux=np.zeros((n_mean,3))
    
    for hh in range(n_mean):

        E_WF.advance(n_run)
        A=E_WF.sampling()
        if n_method==0 or n_method==2:
            lenght=len(A)
            A[:int(lenght/2),:]=(-1)*A[:int(lenght/2),:]

        aux[hh,0]=E_WF.compute_PCA(eps,A=A)
        aux[hh,1]=E_WF.compute_E()
        aux[hh,2]=E_WF.compute_3ID(t1,t2,cutoff,eps,A=A)

        E_WF.advance(n_between)
        
    dS=np.std(aux[:,0])/np.sqrt(n_mean)
    S=np.mean(aux[:,0])
    dE=np.std(aux[:,1])/np.sqrt(n_mean)
    E=np.mean(aux[:,1])
    dID=np.std(aux[:,2])/np.sqrt(n_mean)
    ID=np.mean(aux[:,2])
    pub.write([dS,S,dE,E,dID,ID])

pub.close()





