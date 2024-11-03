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
n_rep=parameters[5]
neff_sample=parameters[6]
n_max=parameters[7]
n_method=parameters[8]

if n_method==2:
    n_neurons=parameters[9]
    n_layers=parameters[10]

methods=[var_nk.MF(),var_nk.JasShort(),var_nk.FFN(alpha=n_neurons,layers=n_layers)]

cutoff=2**L

model=methods[n_method]



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
variables=["D"]

pub=class_WF.publisher(name_var,var,variables)
pub.create()






file=pd.read_csv(name,delim_whitespace=True,dtype="a")
file.astype(float)
size=len(file)
A=np.zeros((size,L))
W=np.zeros(size)
Wacc=np.zeros(size)
for ii in range(size):
    A[ii]=np.array(file.iloc[ii][0:L])
    W[ii]=file.iloc[ii][L]
    Wacc[ii]=file.iloc[ii][L+1]

C=TId.neighbors(A)

aux=np.zeros((n_rep,tmax))

for jj in range(tmax):
    for ii in range(n_rep):
        W_aux=np.zeros(size)
        Ns=rng.integers(low=0,high=n_samples,size=neff_sample)
        
        for Number in Ns:
            i=0
            while Wacc[i]<Number:
                i+=1
            W_aux[i]=(W_aux[i]+1)%(n_max+1)
        aux[ii,jj]=np.mean(TId.n_points(C,W_aux,jj))


dD=np.std(aux,axis=0)/np.sqrt(n_rep)
D=np.mean(aux,axis=0)

for i in range(tmax):
    pub.write([dD[i],D[i]])

pub.close()





