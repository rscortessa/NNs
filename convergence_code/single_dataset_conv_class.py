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
n_samples=parameters[3]
n_between=parameters[4]
n_run=parameters[5] #-
n_mean=parameters[6] #-
n_method=parameters[7]

try:
    n_neurons=parameters[8]
    n_layers=parameters[9]
except:
    print("no additional parameters")

name_var=["DATAM","L","W","NS","NB","G","NN","NL"]
var=[n_method,L,W,n_samples,n_between,Gamma,n_neurons,n_layers]
name_var=name_var[:n_par-2]
var=var[:n_par-2]

    
cutoff=2**L
methods=[var_nk.MF(),var_nk.JasShort(),var_nk.FFN(alpha=n_neurons,layers=n_layers),nk.models.RBM(alpha=n_neurons),var_nk.SymmModel(alpha=n_neurons,layers=n_layers,L=L,W=W)]




#INITIALIZE OBJECTS

hi=nk.hilbert.Spin(s=1 / 2,N=L*W)
H=class_WF.Ham(Gamma*dx,V,L,hi)

#CONDITION IF EXACT

if n_method==5:
    eig_vecs=class_WF.Diag(H)
    methods.append(var_nk.EWF(eig_vec=tuple(np.abs(eig_vecs[:,0])),L=L*W))

model=methods[n_method]
E_WF=class_WF.WF(L*W,model,H,n_samples)

name_var[0]="M"
pubE=class_WF.publisher(name_var+["NR"],var+[n_run],["NS","E"])
pubE.create()
name_var[0]="DATAM"
E=np.array([[0.0 for i in range(n_mean)] for gg in range(n_run)])

#ITERATION OVER THE GAMMA VALUES:

G=Gamma
H=class_WF.Ham(G*dx,V,L,hi)
sampler=nk.sampler.MetropolisLocal(hi)
for hh in range(n_mean):

    E_WF.user_state.init_parameters()
    for steps in range(n_run):

        var[4]=n_between*(1+steps)

        if n_method != 5:
            E_WF.advance(n_between)
            A=E_WF.sampling()    
            if n_method==0 or n_method==2:
                lenght=len(A)
                A[:int(lenght/2),:]=(-1)*A[:int(lenght/2),:]
         
        pub=class_WF.publisher(name_var,var,[])
        pub.create()
        for x in range(len(A)):
            pub.write(A[x])
        namefile=pub.name()
        pub.close()
        os.rename(namefile,namefile+str(hh))

        #THIS PART COMPUTES AND STORES THE HIDDEN VARIABLES

        name_var[0]="VARM"
        pubvar=class_WF.publisher(name_var,var,[])
        pubvar.create()
        if n_method == 3:        
            kernel=E_WF.user_state.variables["params"]["Dense"]["kernel"]
            bias=E_WF.user_state.variables["params"]["Dense"]["bias"]
            visible_bias=E_WF.user_state.variables["params"]["visible_bias"]
            Test=A@kernel
            new_bias=np.array([bias for nns in range(n_samples)])
            Test=1/(1+np.exp(-1.0*(Test+new_bias)))
            for ns in range(n_samples):
                pubvar.write(Test[ns].tolist())
                
        namefile=pubvar.name()    
        pubvar.close()
        os.rename(namefile,namefile+str(hh))
        name_var[0]="DATAM"

        #-------------------------------------------------

        if n_method!=5:
            
        else:
            eig_vecs=class_WF.Diag(H)
            model=var_nk.EWF(eig_vec=tuple(np.abs(eig_vecs[:,0])),L=L*W)
            AA=nk.vqs.MCState(E_WF.user_sampler,model,n_samples=n_samples)
            E_WF.change_state(AA)

        E[steps][hh]=E_WF.compute_E()

        
En=np.mean(E,axis=-1)
dEn=np.std(E,axis=-1)
for gg in range(n_run):
    pubE.write([0,n_between*(1+gg),dEn[gg],En[gg]])
pubE.close()




