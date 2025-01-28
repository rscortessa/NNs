import numpy as np
import matplotlib.pyplot as plt
import Methods.class_WF as class_WF
import Methods.var_nk as var_nk
from scipy.sparse.linalg import eigsh    
import netket as nk
import sys
import os

## CONSTANTS
eps=10**(-8)
dx=-0.01
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
n_par=len(parameters)

print(parameters,n_par)

L=parameters[0]
W=parameters[1]
Gamma=parameters[2]
GammaF=parameters[3] # -
n_samples=parameters[4]
n_run=parameters[5]
n_mean=parameters[6] #-
NG=parameters[7] #-
n_method=parameters[8]

try:
    n_neurons=parameters[9]
    n_layers=parameters[10]
except:
    print("no additional parameters")
print("n_par:",n_par)

name_var=["DATAM","L","W","NS","NR","G","NN","NL"]
var=[n_method,L,W,n_samples,n_run,Gamma,n_neurons,n_layers]

name_var=name_var[:n_par-3]
var=var[:n_par-3]



cutoff=2**L
methods=[var_nk.MF(),var_nk.JasShort(),var_nk.FFN(alpha=n_neurons,layers=n_layers),nk.models.RBM(alpha=n_neurons),var_nk.SymmModel(alpha=n_neurons,layers=n_layers,L=L,W=W)]




#INITIALIZE OBJECTS
hi=nk.hilbert.Spin(s=1 / 2,N=L*W)
H=class_WF.Ham_PBC(Gamma*dx,V,L,W,hi)

#CONDITION IF EXACT
if n_method==5:
    eig_vecs=class_WF.Diag(H)
    methods.append(var_nk.EWF(eig_vec=tuple(np.abs(eig_vecs[:,0])),L=L*W))

model=methods[n_method]
E_WF=class_WF.WF(L*W,model,H,n_samples)




#ITERATION OVER THE GAMMA VALUES:
name_var[0]="M"
pubE=class_WF.publisher(name_var+["GF"],var+[GammaF],["G","E"])
pubE.create()
name_var[0]="DATAM"
E=np.array([[0 for i in range(n_mean)] for gg in range(NG+1)])

for gg in range(NG+1):
    
    G=Gamma+(GammaF-Gamma)/NG*gg
    H=class_WF.Ham_PBC(G*dx,V,L,W,hi)
    sampler=nk.sampler.MetropolisLocal(hi)
    E_WF.change_H(H)
    var[5]=str(round(G))
    
    for hh in range(n_mean):
        
        if n_method != 5:
            E_WF.advance(n_run)
        A=E_WF.sampling()
        
        if n_method==0 or n_method==2:
            lenght=len(A)
            A[:int(lenght/2),:]=(-1)*A[:int(lenght/2),:]

        E[gg][hh]=E_WF.compute_E()
    
        #INSERT PUBLISHER DETAILS AND INITIALIZE IT

        
        pub=class_WF.publisher(name_var,var,[])
        pub.create()
        for x in range(len(A)):
            pub.write(A[x])
        namefile=pub.name()
        pub.close()
        os.rename(namefile,namefile+str(hh))
        
        if n_method!=5:
            E_WF.advance(n_between)
        else:
            eig_vecs=class_WF.Diag(H)
            model=var_nk.EWF(eig_vec=tuple(np.abs(eig_vecs[:,0])),L=L*W)
            AA=nk.vqs.MCState(E_WF.user_sampler,model,n_samples=n_samples)
            E_WF.change_state(AA)

En=np.mean(E,axis=-1)
dEn=np.std(E,axis=-1)
for gg in range(NG+1):
    pubE.write([0,Gamma+(GammaF-Gamma)/NG*gg,dEn[gg],En[gg]])
pubE.close()
