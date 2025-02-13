import numpy as np
import matplotlib.pyplot as plt
import Methods.TId as TId
import Methods.class_WF as class_WF
import Methods.var_nk as var_nk
import pandas as pd
from scipy.sparse.linalg import eigsh    
import netket as nk
import sys

def name(A,B):
    filename=""
    for i in range(len(A)):
        filename+=A[i]+str(B[i])
    filename+=".txt"
    return filename
    
## CONSTANTS
V=-1.0
eps=10**(-8)
dx=0.01
## ALMOST CONSTANTS
n_neurons=1
n_layers=1


## PARAMETERS
parameters=sys.argv
n_par=len(parameters)
parameters=[int(parameters[x]) for x in range(1,n_par)]
n_par=len(parameters)

print(parameters,n_par)

L=parameters[0]
W=parameters[1]
Gamma=parameters[2]
n_samples=parameters[3]
n_between=parameters[4] 
n_run=parameters[5] #-
n_mean=parameters[6] #-
n_method=parameters[7]

try:
    n_neurons=parameters[8]
    n_layers=parameters[9]
except:
    print("no additional parameters")
print("n_par:",n_par)


tmax=L*W

name_var=["VARM","L","W","NS","NB","G","NN","NL"]
var=[n_method,L,W,n_samples,n_between,Gamma,n_neurons,n_layers]

name_var=name_var[:n_par-2]
var=var[:n_par-2]
G=[x for x in range(n_between,n_run*n_between,n_between)]
D=np.array([[[0.0 for k in range(n_mean)] for i in range(tmax)] for j in range(len(G))])

j=0
for gg in range(len(G)):
    var[4]=round(G[gg])
    for i in range(n_mean):
        filename=name(name_var,var)+str(i)
        file=pd.read_csv(filename,sep="\s+",dtype="a")
        file=file.astype(float)
        A=np.array(file)
        A=2*(A<0.5)-1
        B,W=TId.sets(A)
        C=TId.neighbors(B)
        
        for jj in range(tmax):
            Aux=TId.n_points(C,W,jj)
            D[gg,jj,i]=np.mean(Aux)   

Dn=np.mean(D,axis=-1)
dDn=np.std(D,axis=-1)    
name_var[0]="M"
for gg in range(len(G)):
    var[4]=G[gg]
    pub=class_WF.publisher(name_var,var,["D"])
    pub.create()
    for jj in range(tmax):
        pub.write([dDn[gg,jj],Dn[gg,jj]])
    pub.close()



