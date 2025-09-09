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

if n_method==5:
    jj=parameters[6]
    n_par=n_par-1
else:

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
if n_method==5:
    name+=str(jj)

file=pd.read_csv(name,sep='\s+',header=None,dtype="S")
file=file.astype(float)
size=len(file)

#PUBLISHER DETAILS AND INTIALIZATION
variables=["D"]
pub=class_WF.publisher(name_var,var,variables)

if n_method==5:
    pub.create(jj)
else:
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


print(C.shape,"C")
print(Wacc[-1])
for jj in range(tmax+1):
    aux=TId.n_points(C,W,jj)
    aux_2=W@aux
    D=(aux_2-W@np.ones(size))/2.0
    print(jj,D)
    #+W@(W-1.0)/2.0
    #D=2*D/(Wacc[size-1]*(Wacc[size-1]-1))
    #D=np.mean(Aux)
    #dD=np.std(Aux)/np.sqrt(size)
    pub.write([0,D])
pub.close()





