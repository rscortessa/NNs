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
W=parameters[1]
Gamma=parameters[2]*(-dx)
n_samples=parameters[3]
n_run=parameters[4]
n_method=parameters[5]

try:
    n_neurons=parameters[6]
    n_layers=parameters[7]
except:
    print("no additional parameters")

cutoff=2**L
    
name_var=["DATAM","L","W","NS","NR","G","NN","NL"]
var=[n_method,L,W,n_samples,n_run,parameters[2],n_neurons,n_layers]
print(len(name_var),parameters)
name_var=name_var[:n_par-1]
var=var[:n_par-1]

methods=[var_nk.MF(),var_nk.JasShort(),var_nk.FFN(alpha=n_neurons,layers=n_layers),nk.models.RBM(alpha=n_neurons),var_nk.SymmModel(alpha=n_neurons,layers=n_layers,L=L,W=W)]

#INITIALIZE OBJECTS
hi=nk.hilbert.Spin(s=1 / 2,N=L*W)
H=class_WF.Ham_PBC(Gamma,V,L,W,hi)

# CHECK IF IT IS EXACT
if n_method==5:
    eig_vecs=class_WF.Diag(H)
    methods.append(var_nk.EWF(eig_vec=tuple(np.abs(eig_vecs[:,0])),L=L*W))

# MODEL DETAILS   
cutoff=2**L
model=methods[n_method]
E_WF=class_WF.WF(L,model,H,n_samples)

if n_method != 5:
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





