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



name_var=["DATAM","L","W","NS","NB","G","NN","NL"]
var=[n_method,L,W,n_samples,n_between,Gamma,n_neurons,n_layers]
name_var=name_var[:n_par-2]
var=var[:n_par-2]
G=range(n_between,(n_run+1)*n_between,n_between)
S=np.array([[0 for i in range(n_mean)] for i in G])


j=0
for gg in G:

    var[4]=round(gg)
    print(gg)
    for i in range(n_mean):
        filename=name(name_var,var)+str(i)
        file=pd.read_csv(filename,sep="\s+",dtype="a")
        file=file.astype(float)
        A=np.array(file)
        lenght=len(A)
        A[:int(lenght/2),:]=(-1)*A[:int(lenght/2),:]
        eps=10**(-10)
        S[j,i]=class_WF.S_PCA(A,eps,False)
    j+=1    

S_PCA=np.mean(S,axis=-1)
DS_PCA=np.std(S,axis=-1)
var[4]=n_between

name_var[0]="M"

name_var+=["NR"]
var+=[n_run]

pub=class_WF.publisher(name_var,var,["NS","S"])
pub.create()
j=0
for x in G:
    pub.write([0,x,DS_PCA[j],S_PCA[j]])
    j+=1
pub.close()





