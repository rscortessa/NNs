import netket as nk
import Core_WF
import numpy as np
import matplotlib.pyplot as plt
from netket.operator.spin import sigmax,sigmaz
from scipy.sparse.linalg import eigsh
import jax.numpy as jnp #Search what the difference is
import flax
import jax
import flax.linen as nn # What is this?
from sklearn.decomposition import PCA
import sys
from Core_WF import hi,Ham,Exact_Calculation,MF,JasShort,FFN,v_state_steady

## Define the parameters
parameters=sys.argv
n_par=len(parameters)
print(parameters)
parameters=[int(parameters[x]) for x in range(1,n_par)]
#...
print(parameters)
N=parameters[0]
Gamma=parameters[1]*(-0.01)
GammaF=parameters[2]*(-0.01)
V=-1.0
n_sample=parameters[3]
print(n_sample)
n_run=parameters[4]
n_mean=parameters[5]
each=bool(parameters[6])
NG=parameters[7]
n=parameters[8]
L=N
dh=(GammaF-Gamma)/(1.0*NG)

if each!=False:
    sisj=np.zeros((n_mean,Nh,L-1))

    
model=[MF(),JasShort(),FFN(alpha=1)]
method=["Mean field","Jastrow MF", "Neural Network"]
En=np.zeros((n_mean,NG))
S_ent=np.zeros((n_mean,NG))
m=np.zeros((n_mean,NG))

if each==False:
    for j in range(n_mean):
        m[j,:],En[j,:],S_ent[j,:]=v_state_steady(model[n],n_sample,hi,n_run,L,dh,NG,each) ;
else:
    for j in range(n_mean):
        sisj[j,:],m[j,:],En[j,:],S_ent[j,:]=v_state_steady(model[n],n_sample,hi,n_run,L,dh,NG,each) ;

En_error=np.std(En,axis=0)/np.sqrt(n_mean)        
En=np.mean(En,axis=0)
S_error=np.std(S_ent,axis=0)/np.sqrt(n_mean)
S_ent=np.mean(S_ent,axis=0)


if each!=False:
    s_is_j=np.mean(sisj,axis=0)
m_error=np.std(m,axis=0)/(np.sqrt(n_mean))
m=np.mean(m,axis=0)

name=str(n)+"VAR1"+"G"+str(parameters[1])+"GF"+str(parameters[2])+"L"+str(L)+"N_S"+str(n_sample)+"N_M"+str(n_mean)

file=open(name+".txt","w")
file.write("G"+"\t"+" dE"+"\t"+" E"+"\t"+" dS"+"\t"+" S"+"\t"+" dm"+"\t"+" m"+"\n")
if each!=False:
    file2=open(name+"SiSj.txt","w")

for gg in range(NG):
    Gnew=Gamma+(GammaF-Gamma)/(NG)*gg
    file.write(str(Gnew)+"\t"+str(En_error[gg])+"\t"+str(En[gg])+"\t"+str(S_error[gg])+"\t"+str(S_ent[gg])+"\t"+str(m_error[gg])+"\t"+str(m[gg])+"\n")
    if each!=False:
        for jj in range(L-1):
            file2.write(s_is_j[gg,jj]+"\t")
        file2.write("\n")

        
    


