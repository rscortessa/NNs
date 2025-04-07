import numpy as np
import matplotlib.pyplot as plt
import Methods.TId as TId
import Methods.class_WF as class_WF
import Methods.var_nk as var_nk
import pandas as pd
from scipy.sparse.linalg import eigsh    
import netket as nk
import sys
import os
def name(A,B):
    filename=""
    for i in range(len(A)-1):
        filename+=A[i]+str(B[i])
    filename+="MPS"
    filename+=A[len(A)-1]+str(B[len(A)-1])+".txt"
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
folder_name=None
try:
    folder_name=parameters[8]
except:
    print("No folder name specified")
    
parameters=[int(parameters[x]) for x in range(1,n_par-1)]
n_par=len(parameters)

print(parameters,n_par)

L=parameters[0]
W=parameters[1]
Gamma=parameters[2]
GammaF=parameters[3] # -
n_samples=parameters[4]
n_mean=parameters[5] #-
NG=parameters[6] #-
n_method=5
broken_z2=False


name_var=["DATAM","L","W","NS","G"]
var=[n_method,L,W,n_samples,Gamma]
models_name=["R_QIM","QIM_","CIM_X","XYZ_"]
modelo=0
if folder_name is None:
    folder_name=models_name[modelo]+"L"+str(L)+"W"+str(W)+"NS"+str(n_samples)+"GI"+str(Gamma)+"GF"+str(GammaF)+"NR"+str(n_mean)
    

G=range(Gamma,GammaF+int((GammaF-Gamma)/NG),int((GammaF-Gamma)/NG))
S=np.array([[0.0 for i in range(n_mean)] for i in G])


j=0
for gg in G:

    var[4]=round(gg)
    
    for i in range(n_mean):
        filename=name(name_var,var)+str(i+1)
        file=pd.read_csv(folder_name+"/"+filename,sep="\s+",dtype="a")
        file=file.astype(float)
        A=np.array(file)
        #print(A)
        lenght=len(A)
        #A[:int(lenght/2),:]=(-1)*A[:int(lenght/2),:]
        eps=10**(-10)
        S[j,i]=(class_WF.S_PCA(A,eps,False))*1.0
        #print(S[j,i],"entropy")
    j+=1    
print(S)
S_PCA=np.mean(S,axis=-1)
print(S_PCA)
DS_PCA=np.std(S,axis=-1)
var[4]=Gamma
name_var[0]="M"
name_var+=["GF"]
var+=[GammaF]

pub=class_WF.publisher(name_var,var,["G","S"])
pub.create()
j=0
for x in G:
    pub.write([0,x,DS_PCA[j],S_PCA[j]])
    j+=1
filename=pub.name()
print("everything stored in",folder_name,"/",filename)
pub.close()

os.rename(filename,folder_name+"/"+filename)






