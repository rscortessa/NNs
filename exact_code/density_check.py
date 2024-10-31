import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Methods.class_WF as class_WF
import Methods.var_nk as var_nk
from scipy.sparse.linalg import eigsh    
import netket as nk
import sys
import time 
import tracemalloc
from memory_profiler import profile
import itertools
import gc

## CONSTANTS
eps=10**(-8)
dx=0.01
V=-1.0

## LOCAL FUNCTIONS

def f(A,i,int):

    l_ones=[A[x][i] for x in range(len(A))]
    s=len(l_ones)
    tt=len(A[0])
    H=[]
    H=[[A[y].copy()+np.array([(j==i+1)*x for j in range(tt)]) for x in range(int,l_ones[y])] for y in range(s)]
    H=np.array(list(itertools.chain.from_iterable(H)))

    return H
def convert(A):
    ss=len(A)
    B=[np.sum(2**(A[i][A[i]!=0]-1)) for i in range(ss)]
    return B

def Basis(L,deep):
    A=np.array([[i*(j==0) for j in range(deep)] for i in range(1,L+1)])
    N=[None for i in range(deep)]

    for j in range(deep-1):
        N[j]=convert(A)
        A=f(A,j,1)

    N[deep-1]=convert(A)
    del A
    gc.collect()
    return N

def density(WaveF,L,NN):
    B=Basis(L,NN)
    rho=[0 for i in range(NN+1)]
    var_rho=[0 for i in range(NN+1)]
    rho[0]=WaveF[0]**2
    var_rho[0]=0
    
    for i in range(1,NN+1):
        kets=B[i-1]
        aux=[]
        for ket in kets:
            aux.append((WaveF[ket])**2)
        aux=np.array(aux)
        rho[i]=np.mean(aux)
        var_rho[i]=np.std(aux)      
    return rho,var_rho
    


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
NN=parameters[5]
name="VECL"+str(L)+"W"+str(W)+"G"+str(parameters[2])+"GF"+str(parameters[3])+".txt"

file=pd.read_csv(name,header=None,delim_whitespace=True,dtype="a")
file=file.astype(float)

name_var=["DENSL","W","NN","G","GF"]
var=[L,W,NN,parameters[2],parameters[3]]
variables=[]
pub=class_WF.publisher(name_var,var,variables)
pub.create()

for nn in range(NG):
    WF=file.iloc[nn].to_numpy()
    D,var=density(WF,L*W,NN)
    #D,var=density(WF,L*W,int(L/2)+int(L/2)%2)
    D.extend(var)
    pub.write(D)
pub.close()









