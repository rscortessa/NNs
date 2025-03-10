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
W=parameters[1]
Gamma=parameters[2]*(-dx)
GammaF=parameters[3]*(-dx)
n_samples=parameters[4]
n_run=parameters[5]
n_mean=parameters[6]
NG=parameters[7]
n_method=parameters[8]
try:
    n_neurons=parameters[9]
    n_layers=parameters[10]
except:
    print("no additional parameters")

cutoff=2**L
    
name_var=["M","L","W","NS","NR","G","GF","NN","NL"]
var=[n_method,L,W,n_samples,n_run,parameters[1],parameters[2],n_neurons,n_layers]
print(len(name_var),parameters)
name_var=name_var[:n_par-1]
var=var[:n_par-1]

methods=[var_nk.MF(),var_nk.JasShort(),var_nk.FFN(alpha=n_neurons,layers=n_layers),nk.models.RBM(alpha=n_neurons),var_nk.SymmModel(alpha=n_neurons,layers=n_layers,L=L,W=W)]

#INITIALIZE OBJECTS
hi=nk.hilbert.Spin(s=1 / 2,N=L*W)
H=class_WF.Ham_PBC(Gamma,V,L,W,hi)

#CONDITION IF EXACT
if n_method==5:
    eig_vecs=class_WF.Diag(H)
    methods.append(var_nk.EWF(eig_vec=tuple(np.abs(eig_vecs[:,0])),L=L*W))

model=methods[n_method]
E_WF=class_WF.WF(L*W,model,H,n_samples)

#INSERT PUBLISHER DETAILS AND INITIALIZE IT
    
variables=["S","E"]
pub=class_WF.publisher(name_var,var,variables)
pub.create()


#ITERATION OVER THE GAMMA VALUES:

for gg in range(NG):

    G=Gamma+(GammaF-Gamma)/NG*gg
    #H=class_WF.Ham(G,-1.0,L,hi)
    H=class_WF.Ham_PBC(G,V,L,W,hi)
    sampler=nk.sampler.MetropolisLocal(hi)
    E_WF.change_H(H)
    aux=np.zeros((n_mean,2))
    
    for hh in range(n_mean):        
        if n_method != 5:
            E_WF.advance(n_run)
        A=E_WF.sampling()
        if n_method==0 or n_method==2:
            lenght=len(A)
            A[:int(lenght/2),:]=(-1)*A[:int(lenght/2),:]

        aux[hh,0]=E_WF.compute_PCA(eps,A=A)
        aux[hh,1]=E_WF.compute_E()
        #aux[hh,2]=E_WF.compute_3ID(t1,t2,cutoff,eps,A=A)
        if n_method!=5:
            E_WF.advance(n_between)
        else:
            eig_vecs=class_WF.Diag(H)
            model=var_nk.EWF(eig_vec=tuple(np.abs(eig_vecs[:,0])),L=L*W)
            AA=nk.vqs.MCState(E_WF.user_sampler,model,n_samples=n_samples)
            E_WF.change_state(AA)

    dS=np.std(aux[:,0])/np.sqrt(n_mean)
    S=np.mean(aux[:,0])
    dE=np.std(aux[:,1])/np.sqrt(n_mean)
    E=np.mean(aux[:,1])
    #dID=np.std(aux[:,2])/np.sqrt(n_mean)
    #ID=np.mean(aux[:,2])
    pub.write([dS,S,dE,E])

pub.close()





