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
from Core_WF import Ham,Exact_Calculation

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
L=N
name="3VAR1"+"G"+str(parameters[1])+"GF"+str(parameters[2])+"L"+str(L)+"N_S"+str(n_sample)+"N_M"+str(n_mean)
file=open(name+".txt","w")
file.write("G"+"\t"+" E"+"\t"+" S"+"\t"+" m"+"\n")
hi=nk.hilbert.Spin(s=1 / 2,N=L)
for gg in range(NG):
    Gnew=Gamma+(GammaF-Gamma)/(NG)*gg
    H=Ham(Gnew,V)
    sp_h=H.to_sparse()
    eig_vals, eig_vecs = eigsh(sp_h,k=1,which="SA")
    S_exact,m_exact=Exact_Calculation(8192*2,n_run,n_mean,L,eig_vecs[:,0],False)
    file.write(str(Gnew)+"\t"+str(eig_vals[0])+"\t"+str(S_exact)+"\t"+str(m_exact)+"\n")
    
    


