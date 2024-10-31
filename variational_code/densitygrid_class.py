import numpy as np
import matplotlib.pyplot as plt
import Methods.TId as TId
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


## PARAMETERS

parameters=sys.argv
n_par=len(parameters)
parameters=[int(parameters[x]) for x in range(1,n_par)]
print(parameters)
L=parameters[0]
Gamma=parameters[1]*(-dx)
tmax=parameters[2]
n_samples=parameters[3]
n_run=parameters[4]
n_mean=parameters[5]
n_method=parameters[6]

if n_method==2:
    n_neurons=parameters[7]
    n_layers=parameters[8]

methods=[var_nk.MF(),var_nk.JasShort(),var_nk.FFN(alpha=n_neurons,layers=n_layers)]

cutoff=2**L

model=methods[n_method]



#INSERT PUBLISHER DETAILS AND INITIALIZE IT

if n_method==2:
    name_var=["M","L","NS","NR","G","tmax","NN","NL"]
    var=[n_method,L,n_samples,n_run,parameters[1],tmax,n_neurons,n_layers]
else:
    name_var=["M","L","NS","NR","G","tmax"]
    var=[n_method,L,n_samples,n_run,parameters[1],tmax]    
    

#INITIALIZE OBJECTS
hi=nk.hilbert.Spin(s=1 / 2,N=L)
H=class_WF.Ham(Gamma,V,L,hi)
eig_vecs=class_WF.Diag(H)
E_WF=class_WF.WF(L,model,H,n_samples)

#ITERATION OVER THE GAMMA VALUES:
E_WF.advance(n_run)
A=E_WF.sampling()


if n_method==0 or n_method==2:
    lenght=len(A)
    A[:int(lenght/2),:]=(-1)*A[:int(lenght/2),:]

B,W=TId.sets(A)
C=TId.neighbors(B)
D=[None for i in range(len(C))]


pub=class_WF.publisher(name_var,var,[])
pub.create()

pub.write([ "S"+str(var_nk.change_to_int(x,L)) for x in B])

for gg in range(tmax):
    D[gg]=TId.n_points(C,W,gg)
    pub.write(D[gg])

pub.close()





