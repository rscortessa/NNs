import numpy as np
import matplotlib.pyplot as plt
import Methods.TId as TId
import Methods.class_WF as class_WF
import Methods.var_nk as var_nk
import pandas as pd
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
n_run=1000
n_method=5


#INSERT PUBLISHER DETAILS AND INITIALIZE IT

name_var=["DATAM","L","NS","NR","G"]
var=[n_method,L,n_samples,n_run,parameters[1]]    
name="DATAM"+str(n_method)+"L"+str(L)+"NS"+str(n_samples)+"MPS"+"G"+str(parameters[1])+".txt"
file=pd.read_csv(name,delim_whitespace=True,dtype="a")
file.astype(float)

#ITERATION OVER THE GAMMA VALUES:

A=np.array(file)

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





