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
from Core_WF import Ham, MF, JasShort, FFN, v_state, Exact_Calculation, min_d


parameters=sys.argv
n_par=len(parameters)
print(parameters)
parameters=[int(parameters[x]) for x in range(1,n_par)]
N=parameters[0]
Gamma=parameters[1]*(-0.01)
V=-1.0
n_sample=parameters[2]
n_run=parameters[3]
n_mean=parameters[4]
each=bool(parameters[5])
k=parameters[6]
L=N
eps=10**(-6)


hi=nk.hilbert.Spin(s=1 / 2,N=L)
H=Ham(Gamma,V,L,hi)
sp_h=H.to_sparse()
eig_vals, eig_vecs = eigsh(sp_h,k=1,which="SA")
print("eig vals: ",eig_vals)


name=str(k)+"con_100VG"+str(parameters[1])+"L"+str(L)+"N_S"+str(n_sample)+"N_M"+str(n_mean)
model=[MF(),JasShort(),FFN(alpha=1)]
method=["Mean field","Jastrow MF", "Neural Network"]
En=np.zeros((n_mean,n_run))
S_ent=np.zeros((n_mean,n_run))
kback=np.zeros((n_mean,n_run))
ex_var=np.zeros((n_mean,n_run,L))

S_error,S_exact,kback_exact,Pv=Exact_Calculation(8192*2,n_run,n_mean,L,np.abs(eig_vecs[:,0]),hi)

print("exact S",S_exact)

for j in range(n_mean):
    kback[j,:],En[j,:],S_ent[j,:],ex_var[j,:]=v_state(model[k],n_sample,hi,n_run,L,V,Gamma,(eig_vecs[:,0])**2,H,each,True) ;

        
dEn=np.std(En,axis=0)/(n_mean)
En=np.mean(En,axis=0)
dS_ent=np.std(S_ent,axis=0)/(n_mean)
S_ent=np.mean(S_ent,axis=0)
dkback=np.std(kback,axis=0)/(n_mean)
kback=np.mean(kback,axis=0)
dex_var=np.std(ex_var,axis=0)/(n_mean)
ex_var=np.mean(ex_var,axis=0)

#rn=ex_var<eps
#ex_var+=rn
#dex_var=dex_var/ex_var
dex_var=np.sum(dex_var,axis=-1)

aux=[0 for i in range(n_run)]
for i in range(n_run):
    for j in range(L):
        if(np.abs(ex_var[i,j])>eps):
            aux[i]+=np.log(ex_var[i,j])*Pv[j]
        
#ex_var=np.log(ex_var[:,0:min_d(ex_var,eps)])*Pv
#ex_var=np.sum(ex_var,axis=-1)
ex_var=np.array(aux)




file=open(name+".txt","w")
file.write("E_exact"+" \t"+" dE"+"\t"+" E"+"\t"+"S_exact"+"\t"+" dS"+"\t"+" S"+"\t"+" dKb"+"\t"+" Kb"+" \t"+"dKbs"+" \t"+"Kbs"+"\n")
for i in range(n_run):
    file.write(str(eig_vals[0])+"\t"+str(dEn[i])+"\t "+str(En[i]-eig_vals[0])+"\t "+str(S_exact)+" \t "+str(dS_ent[i])+"\t "+str(S_ent[i]-S_exact)+"\t "+str(dkback[i])+"\t "+str(kback[i])+"\t "+str(dex_var[i])+"\t"+str(-S_exact-ex_var[i])+" \n")
    
    


