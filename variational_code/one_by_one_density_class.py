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
rng=np.random.default_rng()

## FUNCTIONS

def finder(A,ns):
    lth=len(A)
    i=0
    while np.sum(A[i]==1)!=ns and i<lth:
        i=i+1
    return i if i<lth else lth+1

def finder_string(A,ns):
    lth=len(A)
    return [ i for i in range(lth) if np.sum(A[i]==1)==ns or len(A[i])-np.sum(A[i]==1)==ns]

 



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
n_max=int(parameters[5])
n_method=parameters[6]

if n_method==2:
    n_neurons=parameters[7]
    n_layers=parameters[8]


#INSERT PUBLISHER DETAILS AND INITIALIZE IT

if n_method==2:
    name_var=["M","L","NS","NR","G","NN","NL"]
    var=[n_method,L,n_samples,n_run,parameters[1],n_neurons,n_layers]
else:
    name_var=["M","L","NS","NR","G"]
    var=[n_method,L,n_samples,n_run,parameters[1]]    
    

name="DATA"    
for i in range(len(var)):
    name+=name_var[i]+str(var[i])
name+=".txt"
name_var.append("NMAX")
var.append(n_max)
print(var,name_var)

pub=class_WF.publisher(name_var,var,[])
pub.create()
file=pd.read_csv(name,delim_whitespace=True,dtype="a")
file.astype(float)
size=len(file)

A=np.zeros((size,L))
W=np.zeros(size)
Wacc=np.zeros(size)

for ii in range(size):
    w_min=float(file.iloc[ii][L])
    A[ii]=np.array(file.iloc[ii][0:L])
    W[ii]=w_min
    Wacc[ii]=file.iloc[ii][L+1]

Sts=[]
#for jj in range(n_max):
#    jj_aux=finder(A,jj)
#    if jj_aux!=len(A)+1:
#        Sts.append(jj_aux)
for jj in range(n_max):
    jj_aux=finder_string(A,jj)
    if len(jj_aux)>=1:
        Sts.append(jj_aux)
        
C=TId.neighbors(A)

aux=[None for i in range(len(C))]

pub.write([ "S"+str(x) for x in range(len(Sts))])

for jj in range(tmax):
        aux[jj]=TId.n_points(C,W,jj)[Sts]
        pub.write(aux[jj])

pub.close()
    





