import netket as nk
import Methods.Core_WF as Core_WF
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
from Methods.Core_WF import Ham,Exact_Calculation,MF,JasShort,FFN,v_state_steady,Diag
import os



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
exvar=bool(parameters[9])
L=N
dh=(GammaF-Gamma)/(1.0*NG)
hi=nk.hilbert.Spin(s=1 / 2,N=L)

if each!=False:
    sisj=np.zeros((n_mean,Nh,L-1))

    
model=[MF(),JasShort(),FFN(alpha=1)]
method=["Mean field","Jastrow MF", "Neural Network"]
En=np.zeros((n_mean,NG))
S_ent=np.zeros((n_mean,NG))
I_d=np.zeros((n_mean,NG))
Kback=np.zeros((n_mean,NG,2**L))
m=np.zeros((n_mean,NG))
vars=np.zeros((n_mean,NG,L))

Hts=[Ham(Gamma+dh*i,-1.0,L,hi) for i in range(NG)]
Eigvs=[Diag(Hts[i],False) for i in range(NG)]
Eigvs=np.array([Eigvs[i][:,0] for i in range(NG)])

if exvar==False:
    if each==False:
        for j in range(n_mean):
            m[j,:],I_d[j,:],En[j,:],Kback[j,:],S_ent[j,:]=v_state_steady(model[n],n_sample,hi,n_run,L,Gamma,dh,NG,Hts,each) ;
    else:
        for j in range(n_mean):
            sisj[j,:],I_d[j,:],m[j,:],En[j,:],Kback[j,:],S_ent[j,:]=v_state_steady(model[n],n_sample,hi,n_run,L,Gamma,dh,NG,Hts,each) ;

else:
    if each==False:
        for j in range(n_mean):
            m[j,:],I_d[j,:],En[j,:],Kback[j,:],S_ent[j,:],vars[j,:]=v_state_steady(model[n],n_sample,hi,n_run,L,Gamma,dh,NG,Hts,each) ;
    else:
        for j in range(n_mean):
            sisj[j,:],I_d[j,:],m[j,:],En[j,:],Kback[j,:],S_ent[j,:],vars[j,:]=v_state_steady(model[n],n_sample,hi,n_run,L,Gamma,dh,NG,Hts,each) ;


            
En_error=np.std(En,axis=0)/np.sqrt(n_mean)        
En=np.mean(En,axis=0)
S_error=np.std(S_ent,axis=0)/np.sqrt(n_mean)
S_ent=np.mean(S_ent,axis=0)
I_d_error=np.std(I_d,axis=0)/np.sqrt(n_mean)
I_d=np.mean(I_d,axis=0)

aux=np.mean(Kback,axis=0)
aux[aux==0]=1
Kb_error=np.sum(np.std(Kback,axis=0)/(np.sqrt(n_mean)*aux),axis=1)
Kb=np.mean(Kback,axis=0)

#print(type(Eigvs),type(Kb))
#print(Eigvs.shape,Kb.shape)

eps=10**(-3)
print(Kb)
#Kb[Kb<eps]=1
print(Eigvs**2)
aux=Eigvs
aux_Kb=Kb
aux[Eigvs<eps]=1
aux_Kb[Kb<eps]=1
aux_Kb[Eigvs<eps]=1

Kb=np.matmul(np.log(aux**2)-np.log(Kb),(Eigvs**2).T)


print(Kb.shape)
if exvar==True:
    vars=np.mean(vars,axis=0)



if each!=False:
    s_is_j=np.mean(sisj,axis=0)
m_error=np.std(m,axis=0)/(np.sqrt(n_mean))
m=np.mean(m,axis=0)




name=str(n)+"VAR1"+"G"+str(parameters[1])+"GF"+str(parameters[2])+"L"+str(L)+"N_S"+str(n_sample)+"N_M"+str(n_mean)

file=open(name+".txt","w")
file.write("G"+"\t"+" dE"+"\t"+" E"+"\t"+" dS"+"\t"+" S"+"\t"+" dm"+"\t"+" m"+" \t"+"dKb " +"\t "+ "Kb"+ "\t "+"dI"+" \t"+"I"+"\n")
if each!=False:
    file2=open(name+"SiSj.txt","w")

for gg in range(NG):
    Gnew=Gamma+(GammaF-Gamma)/(NG)*gg
    file.write(str(Gnew)+"\t"+str(En_error[gg])+"\t"+str(En[gg])+"\t"+str(S_error[gg])+"\t"+str(S_ent[gg])+"\t"+str(m_error[gg])+"\t"+str(m[gg])+"\t"+str(Kb_error[gg])+"\t"+str(Kb[gg,gg])+"\t"+str(I_d_error[gg])+"\t"+str(I_d[gg])+"\n")
    if each!=False:
        for jj in range(L-1):
            file2.write(s_is_j[gg,jj]+"\t")
        file2.write("\n")

        
    


