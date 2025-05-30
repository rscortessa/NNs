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
## ALMOST CONSTANTS
n_neurons=1
n_layers=1


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
n_method=parameters[8]

if n_method==2:
    n_neurons=parameters[9]
    n_layers=parameters[10]

methods=[var_nk.MF(),var_nk.JasShort(),var_nk.FFN(alpha=n_neurons,layers=n_layers)]

    
model=methods[n_method]



#INSERT PUBLISHER DETAILS AND INITIALIZE IT

if n_method==2:
    name_var=["M","L","W","NS","NR","G","GF","NN","NL"]
    var=[n_method,L,W,n_samples,n_run,parameters[1],parameters[2],n_neurons,n_layers]
else:
    name_var=["M","L","W","NS","NR","G","GF"]
    var=[n_method,L,W,n_samples,n_run,parameters[1],parameters[2]]    
    
variables=["S","E","Id"]
pub=class_WF.publisher(name_var,var,variables)
pub.create()

#INITIALIZE OBJECTS
hi=nk.hilbert.Spin(s=1 / 2,N=L*W)
H=class_WF.Ham_W(Gamma,V,L,W,hi)
eig_vecs=class_WF.Diag(H)
E_WF=class_WF.WF(L*W,model,H,n_samples)

#ITERATION OVER THE GAMMA VALUES:

for gg in range(NG):

    G=Gamma+(GammaF-Gamma)/NG*gg
    H=class_WF.Ham_W(G,-1.0,L,W,hi)
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
        aux[hh,2]=E_WF.compute_ID(eps,A=A)

        E_WF.advance(n_between)
        
    dS=np.std(aux[:,0])/np.sqrt(n_mean)
    S=np.mean(aux[:,0])
    dE=np.std(aux[:,1])/np.sqrt(n_mean)
    E=np.mean(aux[:,1])
    dID=np.std(aux[:,2])/np.sqrt(n_mean)
    ID=np.mean(aux[:,2])
    pub.write([dS,S,dE,E,dID,ID])

pub.close()





