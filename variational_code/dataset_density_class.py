import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
rng=np.random.default_rng()
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

#POST CONSTANTS    
cutoff=2**L
tmax=L*W

name_var=["M","L","W","NS","NR","G","NN","NL"]
var=[n_method,L,W,n_samples,n_run,parameters[2],n_neurons,n_layers]
print(len(name_var),parameters)
name_var=name_var[:n_par-1]
var=var[:n_par-1]

#DOCUMENT TO READ
name="DATA"    
for i in range(len(var)):
    name+=name_var[i]+str(var[i])
name+=".txt"
file=pd.read_csv(name,delim_whitespace=True,dtype="a")
file=file.astype(float)
size=len(file)

#PUBLISHER DETAILS AND INTIALIZATION
variables=["D"]
pub=class_WF.publisher(name_var,var,variables)
pub.create()

#ARRAYS INTIALIZATION
LL=L*W
A=np.zeros((size,LL))
W=np.zeros(size)
Wacc=np.zeros(size)
for ii in range(size):
    aux=np.array(file.iloc[ii][0:LL])
#    print(aux.size)
    A[ii]=aux
    W[ii]=file.iloc[ii][LL]
    Wacc[ii]=file.iloc[ii][LL+1]
    
C=TId.neighbors(A)

for jj in range(tmax):
    Aux=TId.n_points(C,W,jj)
    D=np.mean(Aux)
    dD=np.std(Aux)/np.sqrt(size)
    pub.write([dD,D])

pub.close()





