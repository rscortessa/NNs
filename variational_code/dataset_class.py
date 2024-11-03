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
n_neurons=1
n_layers=1


## PARAMETERS

parameters=sys.argv
n_par=len(parameters)
parameters=[int(parameters[x]) for x in range(1,n_par)]
print(parameters)
L=parameters[0]
Gamma=parameters[1]*(-dx)
n_samples=parameters[2]
n_run=parameters[3]
n_method=parameters[4]

if n_method==2:
    n_neurons=parameters[5]
    n_layers=parameters[6]

methods=[var_nk.MF(),var_nk.JasShort(),var_nk.FFN(alpha=n_neurons,layers=n_layers)]

#INITIALIZE OBJECTS
hi=nk.hilbert.Spin(s=1 / 2,N=L)
H=class_WF.Ham(Gamma,V,L,hi)


# CHECK IF IT IS EXACT
if n_method==3:
    eig_vecs=class_WF.Diag(H)
    methods.append(var_nk.EWF(eig_vec=tuple(np.abs(eig_vecs[:,0])),L=L))
    



cutoff=2**L

model=methods[n_method]



#INSERT PUBLISHER DETAILS AND INITIALIZE IT

if n_method==2:
    name_var=["DATAM","L","NS","NR","G","NN","NL"]
    var=[n_method,L,n_samples,n_run,parameters[1],n_neurons,n_layers]
else:
    name_var=["DATAM","L","NS","NR","G"]
    var=[n_method,L,n_samples,n_run,parameters[1]]    
    
#ITERATION OVER THE GAMMA VALUES:

E_WF=class_WF.WF(L,model,H,n_samples)
if n_method != 3:
    E_WF.advance(n_run)

A=E_WF.sampling()

if n_method==0 or n_method==2:
    lenght=len(A)
    A[:int(lenght/2),:]=(-1)*A[:int(lenght/2),:]

B,W=TId.sets(A)


pub=class_WF.publisher(name_var,var,[])
pub.create()
suma=0
for x in range(len(B)):
    suma+=W[x]
    pub.write(np.append(B[x],[W[x],suma]))
    
    
pub.close()





