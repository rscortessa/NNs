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


## PARAMETERS
parameters=sys.argv
n_par=len(parameters)
parameters=[int(parameters[x]) for x in range(1,n_par)]
print(parameters)
L=parameters[0]
W=parameters[1]
Gamma=parameters[2]*(-dx)
GammaF=parameters[3]*(-dx)
NG=parameters[4]

#INSERT PUBLISHER DETAILS AND INITIALIZE IT
name_var=["VECL","W","G","GF"]
name_var2=["ENL","W","G","GF"]
var=[L,W,parameters[2],parameters[3]]
variables=[]

pub=class_WF.publisher(name_var,var,variables)
pub.create()

pub2=class_WF.publisher(name_var2,var,variables)
pub2.create()

#INITIALIZE OBJECTS
hi=nk.hilbert.Spin(s=1 / 2,N=L*W)

#ITERATION OVER THE GAMMA VALUES:
for gg in range(NG):
    G=Gamma+(GammaF-Gamma)/NG*gg
    H=class_WF.Ham_W(G,-1.0,L,W,hi)
    eig_val,eig_vecs=class_WF.Diag(H,True)
    pub2.write(eig_val)
    pub.write(eig_vecs[:,0].tolist())
pub.close()





