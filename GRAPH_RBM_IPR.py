#!/usr/bin/env python
# coding: utf-8

# In[18]:


import netket as nk
import matplotlib.pyplot as plt
import numpy as np
from netket.operator.spin import sigmax,sigmaz,sigmay,identity,sigmam,sigmap   
import flax
import jax
import jax.numpy as jnp
import flax.linen as nn
from scipy.sparse.linalg import eigsh 
from netket import nn as nknn
from Methods.class_WF import Diag
import Methods.class_WF as class_WF
import Methods.var_nk as var_nk
import os
os.environ["JAX_ENABLE_X64"] = "True"
import matplotlib as mpl
import numpy as np
import sys
import pandas as pd
import equinox as eqx
import os
import json
from matplotlib import cm
import matplotlib as mp
import matplotlib.colors as mcolors
import scipy.special as sc
import scipy.optimize as so
from matplotlib.colors import LinearSegmentedColormap
from functools import reduce


# In[19]:


#DEFINE FUNCTIONS:


# In[21]:


from Methods.class_WF import rotated_sigmax, rotated_sigmaz,isigmay,rotated_IsingModel,rotated_LONGIsingModel,rotated_BROKEN_Z2IsingModel
from Methods.class_WF import rotated_XYZModel, parity_Matrix, parity_IsingModel, Sz0Szj, Sx0Sxj, to_array, rotated_m, rotated_CIMModel


# In[22]:


def GET_PROB_RBM(hi,param_RBM,j):
    
    #DEFINE THE PARAMETERS OF THE RBM
    AA=np.array(param_RBM["params"]["Dense"]["kernel"]["value"]["real"][j])+1j*np.array(param_RBM["params"]["Dense"]["kernel"]["value"]["imag"][j])
    BB=np.array(param_RBM["params"]["Dense"]["bias"]["value"]["real"][j])+1j*np.array(param_RBM["params"]["Dense"]["bias"]["value"]["imag"][j])
    CC=np.array(param_RBM["params"]["visible_bias"]["value"]["real"][j])+1j*np.array(param_RBM["params"]["visible_bias"]["value"]["imag"][j])
    
    #DEFINE THE STATES
    states=hi.all_states()
    
    #AUXILIAR MATRIX
    DD=np.tile(BB,(len(states),1))
    
    #COMPUTE THE PROBABILITIES
    KK=np.exp(states@CC)
    AMP=np.cosh(states@AA+DD)
    ALMOST_PROB=KK*np.prod(AMP,axis=-1)
    
    #NORMALIZE THE PROBABILITY
    NORM=np.sqrt(ALMOST_PROB@np.conjugate(ALMOST_PROB))
    PROB=ALMOST_PROB/NORM
    
    return PROB


# In[23]:


#DEFINE THE VARIABLES:


# In[73]:


L=10
NS=2048
G=10
DG=0.01
NN=1
NL=1
NR=1000
learning_rate=0.05
basis="BROKENZ2_QIM"
modelo="RBM_COMPLEX"
##########################
angle=0
Nangle=12
dangle=np.pi/(2*Nangle)
NSPCA=100
NM=5
##########################



MASTER_DIR="RUN_"+basis+"_"+modelo+"NN"+str(NN)+"L"+str(L)+"G"+str(G)+"NA"+str(Nangle)+"NSPCA"+str(NSPCA)
Nstates=2**L
eps=10**(-10)
angle=[dangle*i for i in range(Nangle+1)]
num_states=[i for i in range(2**L)]

labels=[r"$"+str(i)+r"\times\frac{\pi}{"+str(2*Nangle)+"}"+"$" for i in range(Nangle+1)]
try:
    os.mkdir(MASTER_DIR)
except:
    print("DIRECTORY ALREADY CREATED")
OBS_FILENAME="NANGLE"+str(Nangle)+basis+"M3L"+str(L)+"W1"+"G"+str(G)+"NS"+str(NS)+"NN"+str(NN)+"NL"+str(NL)+"NR"+str(NR)+"OBS"
SPCA_FILENAME="NANGLE"+str(Nangle)+basis+"M3L"+str(L)+"W1"+"G"+str(G)+"NS"+str(NS)+"NN"+str(NN)+"NL"+str(NL)+"NR"+str(NR)+"SPCA"
VAR_FILENAME="NANGLE"+str(Nangle)+basis+"M3L"+str(L)+"W1"+"G"+str(G)+"NS"+str(NS)+"NN"+str(NN)+"NL"+str(NL)+"NR"+str(NR)+"VAR"


# In[74]:


norm = mcolors.Normalize(vmin=np.abs(0), vmax=1)
colors = [(0.0, 'yellow'),(0.25, "green"),(0.5, 'blue'),(0.75,"brown"),(1.0, 'red')]
cmap = LinearSegmentedColormap.from_list('custom_blue_green', colors)


# In[75]:


#EXACT DIAGONALIZATION AND COMPUTING THE ORDER PARAMETER


# In[76]:


#SOLUTION OF THE ISING MODEL AS A FUNCTION OF THETA:


# In[77]:


#SOLUTION OF THE ISING MODEL AND ITS PARAMETERS...


# In[78]:


S_PCA_TEO=[0.0 for i in angle]
PSI_TEO=[None for i in angle]
EXC_PSI_TEO=[None for i in angle]
N_pos=[0.0 for i in range(Nangle+1)]
hi=nk.hilbert.Spin(s=1/2,N=L)
states=hi.all_states()
eps=10**(-10)
for theta in range(len(angle)):
    #PSI AND DIAGONALIZATION
    H=rotated_BROKEN_Z2IsingModel(angle[theta],G*DG,L,hi)
    eig_vals_other,eig_vecs_other=np.linalg.eigh(H.to_dense())
    PSI_TEO[theta]=eig_vecs_other[:,0]
    EXC_PSI_TEO[theta]=eig_vecs_other[:,3]
    if PSI_TEO[theta][0]<0:
        PSI_TEO[theta]=(-1.0)*PSI_TEO[theta]
    
    A=np.random.choice(num_states,size=1000,p=eig_vecs_other[:,0]**2)
    B=np.array([states[a] for a in A])
    S_PCA_TEO[theta]=class_WF.S_PCA(B,10**(-10),exvar=False)
    #SIGN PROBLEM
    H_array=to_array(H)*1.0
    off_diag_indices=np.triu_indices(2**(L),k=1)
    aux_1=H_array[off_diag_indices]
    aux_2=aux_1[np.abs(aux_1)>eps]
    N_pos[theta]=np.sum((aux_2>eps)*1.0)
    
sisj_z=[ PSI_TEO[0].T@Sz0Szj(0.0,L,hi,j).to_dense()@PSI_TEO[0] for j in [1,int(L/2),L-1]]
sisj_x=[ PSI_TEO[0].T@Sx0Sxj(0.0,L,hi,j).to_dense()@PSI_TEO[0] for j in [1,int(L/2),L-1]]



# In[79]:


N_pos


# In[80]:


# INITIALIZE NETKET TOOLS
sites_corr=[1,int(L/2),L-1]
sites_corr=[str(x) for x in sites_corr]
S=[[[0.0 for k in range(NSPCA)] for i in range(len(angle))] for rep in range(NM)]
E=[[[0.0 for k in range(NR)]  for i in range(len(angle))] for rep in range(NM)]
dE=[[[0.0 for k in range(NR)] for i in range(len(angle))] for rep in range(NM)]
P=[[[0.0 for k in range(NR)] for i in range(len(angle))] for rep in range(NM)]
dP=[[[0.0 for k in range(NR)] for i in range(len(angle))] for rep in range(NM)]

indexes=[[[(rep,i,k) for k in range(NR)] for i in range(len(angle))] for rep in range(NM)]
indexes_S=[[[(rep,i,k) for k in range(NSPCA+1)] for i in range(len(angle))] for rep in range(NM)]

CZ=[[[None for corr in sites_corr] for i in range(len(angle))] for rep in range(NM)]
dCZ=[[ [None for corr in sites_corr] for i in range(len(angle))] for rep in range(NM)]

CX=[[ [None for corr in sites_corr] for i in range(len(angle))] for rep in range(NM)]
dCX=[[ [None for corr in sites_corr] for i in range(len(angle))] for rep in range(NM)]

data_RBM=[[[0.0 for k in range(NR)] for i in range(len(angle))] for rep in range(NM)]
data_S_PCA=[[[0.0 for k in range(NSPCA)] for i in range(len(angle))] for rep in range(NM)]
data=[[None for i in range(len(angle))] for rep in range(NM)]

# Load the data from the JSON file
for rep in range(NM):
    for ii in range(len(angle)):
    
        with open(MASTER_DIR+"/"+str(rep)+"NM"+str(ii)+OBS_FILENAME+".json", "r") as f:
            data_RBM[rep][ii] = json.load(f)
        with open(MASTER_DIR+"/"+str(rep)+"NM"+str(ii)+VAR_FILENAME+".json", "r") as f:
            data[rep][ii] = json.load(f)

        with open(MASTER_DIR+"/"+str(rep)+"NM"+str(ii)+SPCA_FILENAME+".json", "r") as f:
            data_S_PCA[rep][ii] = json.load(f)
        
        E[rep][ii]=np.real(data_RBM[rep][ii]["Energy"]["Mean"]["real"])
        dE[rep][ii]=np.real(data_RBM[rep][ii]["Energy"]["Sigma"])
        #P[rep][ii]=np.real(data_RBM[rep][ii]["P"]["Mean"])
        #dP[rep][ii]=np.real(data_RBM[rep][ii]["P"]["Sigma"])
        S[rep][ii]=np.real(data_S_PCA[rep][ii]["Mean"]["value"])
    
        for xi in range(len(sites_corr)):
            CZ[rep][ii][xi]=np.real(data_RBM[rep][ii]["CZ0"+sites_corr[xi]]["Mean"])
            dCZ[rep][ii][xi]=np.real(data_RBM[rep][ii]["CZ0"+sites_corr[xi]]["Sigma"])
            CX[rep][ii][xi]=np.real(data_RBM[rep][ii]["CX0"+sites_corr[xi]]["Mean"])
            dCX[rep][ii][xi]=np.real(data_RBM[rep][ii]["CX0"+sites_corr[xi]]["Sigma"])

#print(np.array(E) == None)

E=np.array(E)
#S=np.array(S)
indexes=np.array(indexes)
indexes_S=np.array(indexes_S)
Nones=[tuple(x) for x in indexes[E==None]]
for Nones_index in Nones:
    E[Nones_index]=0.001
#for None_index in Nones_S:
#    S[Nones_index]=0.001
E_all=np.array(E.copy())
#S=np.mean(np.array(S),axis=0)
#dE=np.mean(np.array(dE),axis=0)
E_all=np.abs((E_all-eig_vals_other[0])/eig_vals_other[0])
E_all[E_all>1.0]=1.0
X=np.array(data_RBM[0][0]["Energy"]["iters"])+1.0


# In[81]:


AA=[10,11,Nangle]
NMM=[[1,2,3,4],[2],[1,4]] #500
#NMM=[[0,2,4,6,7,8],[0,5,6,8,9],[0,2,3,4,5,6,7,8,9]] #170
#NMM=[[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9]]


# In[ ]:





# In[82]:


print(E_all.shape)
for s in range(len(AA)):
    i=AA[s]
    fig,axis=plt.subplots(1,1,figsize=(7,5))
    plt.title(r"$Quantum\:Ising\;Model\;vs\;Iterations$"+"\n"+r"$L="+str(L)+"$ $" r"\lambda="+str(round(G*DG,1))+"$"+" $N_{samples}="+str(NS)+r"\;"+r"\theta=\frac{"+str(i)+"\pi}{"+str(Nangle*2)+"}$"+"\n"+"METROPOLIS HAMILTONIAN SAMPLER",fontsize=15)
    for k in NMM[s]:
        axis.scatter(X,E_all[k][i],marker="*",color=cmap(k/NM*1.0),label=str(k))
    plt.legend(bbox_to_anchor=(1.3,1))
    #plt.xscale("log")    tr
    plt.yscale("log")
    plt.xlabel("$1/iteration$",fontsize=15)
    plt.ylabel(r"$\Delta\;E/E_{teo}$",fontsize=15)
    #axis.legend()
    plt.tight_layout()
    plt.savefig(MASTER_DIR+"/"+"BASIS"+basis+"L"+str(L)+"G"+str(G)+"NS"+str(NS)+"DIF_LEARNING_RATES.png")


# In[83]:


#Naffec=3
Naffec=3
E=[]
dE=[]
NN=np.zeros(Nangle+1)
for i in range(Nangle+1-Naffec):
    E.append(np.mean(E_all[:,i,:],axis=0))
    dE.append(np.var(E_all[:,i,:],axis=0))
    NN[i]=NM
for i in range(Nangle+1-Naffec,Nangle+1):
    E.append(np.mean(E_all[NMM[i-Nangle-1+Naffec],i,:],axis=0))
    dE.append(np.var(E_all[NMM[i-Nangle-1+Naffec],i,:],axis=0))
    NN[i]=len(NMM[i-Nangle-1+Naffec])
E=np.array(E)
dE=np.array(dE)


# In[84]:


# ENERGY
X=np.array(data_RBM[0][0]["Energy"]["iters"])+1.0


# In[ ]:





# In[85]:


fig,axis=plt.subplots(1,1,figsize=(7,5))
plt.title(basis+"$"+r"\;vs\;Iterations$"+"\n"+r"$L="+str(L)+"$ $" r"\lambda="+str(round(G*DG,1))+"$"+" $N_{samples}="+str(NS)+"$"+"\n"+"METROPOLIS HAMILTONIAN SAMPLER",fontsize=15)
AA=[0,5,6,7,8,9,10,Nangle]

for i in AA:
    axis.scatter(X,E[i],marker="*",color=cmap(i/Nangle*1.0))
#plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.xlabel("$1/iteration$",fontsize=15)
plt.ylabel(r"$\Delta\;E/E_{teo}$",fontsize=15)

#axis.legend()
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # ScalarMappable needs an array, even an empty one is fine
cbar = plt.colorbar(sm,ax=axis,label=r'$\theta[\frac{\pi}{2}]$')
cbar.set_label(r'$\theta[\frac{\pi}{2}]$', fontsize=18)
plt.tight_layout()
plt.savefig(MASTER_DIR+"/"+"BASIS"+basis+"L"+str(L)+"G"+str(G)+"NS"+str(NS)+"DIF_LEARNING_RATES.png")


# In[86]:


#RELATIVE ERROR LAST ITERATION


# In[ ]:





# In[87]:


Y=np.array(E)
T_CORR=0
N_SAMPLES=1
X_INDICES=[NR-1-T_CORR*kk for kk in range(N_SAMPLES)]
YY=[]
Y_err=[]
for ii in range(Nangle+1):
    YY.append(np.mean(Y[ii,X_INDICES]))
    Y_err.append(dE[ii,NR-1])
Y_err=np.array(Y_err)


# In[88]:


PSI_TEO=np.array(PSI_TEO)
IPR_TEO=[0.0 for ii in range(Nangle+1)] 
IPR_RBM=[0.0 for ii in range(Nangle+1)]
N_M=[]
for aa in range(Nangle+1-Naffec):
    N_M.append([i for i in range(NM)])
for bb in range(Naffec):
    N_M.append(NMM[bb])
    
for phi in range(Nangle+1): 
    ############
    hi=nk.hilbert.Spin(s=1/2,N=L)
    if phi==Nangle:
        hk=nk.hilbert.Spin(s=1/2,N=L,constraint=class_WF.ParityConstraint())
        if basis=="CIM":
            hk=hi
        
    else:
        hk=hi
    #############    
    WF=PSI_TEO[phi][num_states]
    IPR_TEO[phi]=np.sum((WF**2) * np.conjugate(WF**2))
    aux_param=[]
    for kk in N_M[i]:
        param_RBM=data[kk][phi]
        P=GET_PROB_RBM(hk,param_RBM,NSPCA-1)
        aux_param.append(np.sum(np.absolute(P)**4))
    aux_param=np.array(aux_param)
    
    IPR_RBM[phi]=np.mean(aux_param)
  


# In[89]:


PSI_TEO*PSI_TEO


# In[90]:


#ANGLES
ANGLES=np.array(range(Nangle+1))
Y0=None
for kk in range(Nangle+1):
    if N_pos[kk]<eps:
        continue
    else:
        Y0=kk
        break


# In[91]:


norm_1 = mcolors.Normalize(vmin=np.abs(0), vmax=1)
colors_1 = [(0.0, 'white'),(1.0, "black")]
cmap_1 = LinearSegmentedColormap.from_list('gray_scale', colors)


# In[92]:


N_label_freq=2
fsize=18
xticks=np.arange(0,Nangle+1,N_label_freq)
x_labels=[r"$\frac{"+str(i)+r"\pi}{"+str(2*Nangle)+"}$" for i in xticks]

YYY=YY

fig,axis=plt.subplots(1,2,figsize=(14,5))
axis[0].set_title(r"$L="+str(L)+"$ $" r"\lambda="+str(round(G*DG,2))+"$"+" $N_{samples}="+str(NS)+"$"+" $N_{opt}="+str(NR)+"$"+"\n"+"ENERGY DISCREPANCY",fontsize=fsize)
axis[0].set_xticks(xticks,x_labels,fontsize=fsize)
axis[1].set_xticks(xticks,x_labels,fontsize=fsize)
axis[0].tick_params(axis='y', labelsize=fsize)
axis[1].tick_params(axis='y', labelsize=fsize)
axis[1].set_title(r"$L="+str(L)+"$ $" r"\lambda="+str(round(G*DG,2))+"$"+" $N_{samples}="+str(NS)+"$"+" $N_{opt}="+str(NR)+"$\n"+"INVERSE PARTICIPATION RATIO",fontsize=fsize)

axis[0].errorbar(ANGLES,YYY,yerr=np.sqrt(Y_err)/NN[i],marker="*",linestyle="dashed")
if Y0!=None:
    axis[0].fill_between(ANGLES[Y0:Nangle],np.min(YYY)/1.3,np.max(YYY)*1.3,alpha=.3)
#for i in range(Nangle+1):
#    axis[0].fill_between(ANGLES[i:i+2],np.min(YYY)/1.3,np.max(YYY)*1.3,color=cmap_1(N_pos[i]*1.0/max(N_pos)),alpha=.5)

axis[0].set_ylim([np.min(YYY)/1.1,np.max(YYY)*1.3])
axis[0].set_yscale("log")
axis[0].set_xlabel(r"$\theta$",fontsize=fsize)
axis[0].set_ylabel(r"$\Delta\;E/E_{teo}$",fontsize=fsize)

axis[1].scatter(ANGLES,IPR_RBM,marker="*")
axis[1].plot(ANGLES,IPR_RBM,linestyle="dashed",label="RBM")
axis[1].plot(ANGLES,IPR_TEO,color="black",linestyle="dashed",label="Exact")

axis[1].set_xlabel(r"$\theta$",fontsize=fsize)
axis[1].set_ylabel(r"$IPR$",fontsize=fsize)
if Y0!=None:
    axis[1].fill_between(ANGLES[Y0:Nangle],np.min(IPR_TEO)/20.0,np.max(IPR_TEO)*100.0,alpha=.3)
plt.legend(fontsize=fsize)
#axis[1].set_ylim([np.min(IPR_TEO)/10.0,np.max(IPR_TEO)*2.0])
axis[1].set_yscale("log")

sm = mpl.cm.ScalarMappable(cmap=cmap_1, norm=norm_1)
sm.set_array([])  # ScalarMappable needs an array, even an empty one is fine
#cbar = plt.colorbar(sm,ax=axis,label=r'$N_{+}$')
#cbar.set_label(r'$N_{+}$', fontsize=18)

plt.tight_layout()
plt.savefig("THETA_BASIS"+basis+"L"+str(L)+"G"+str(G)+"NS"+str(NS)+"IPR_ED.jpg")


# In[71]:


N_pos


# In[94]:


phi=0
k_right=[0,2,4]
k_wrong=[1,3,4]

#PARAMETERS OF THE RBM 
r=k_right[0]
w=k_wrong[2]
r_param_RBM=data[r][phi]
w_param_RBM=data[w][phi]
#THEORETICAL RBM
PSI=PSI_TEO[phi]
EXC_PSI=EXC_PSI_TEO[Nangle]
if phi==Nangle:
    hk=nk.hilbert.Spin(s=1/2,N=L,constraint=class_WF.ParityConstraint())
    if basis=="CIM":
        hk=hi
    
else:
    hk=hi
        
PSI=PSI**2
states=hk.all_states()
new_states=hi.states_to_numbers(states)

PSI=PSI[new_states]
perm = np.argsort(-PSI)
#EXP RBM
r_PSI=GET_PROB_RBM(hk,r_param_RBM,NSPCA-1)
w_PSI=GET_PROB_RBM(hk,w_param_RBM,NSPCA-1)
r_P=np.real(r_PSI*np.conjugate(r_PSI))
w_P=np.real(w_PSI*np.conjugate(w_PSI))


# In[112]:


plt.figure()
fig,axis=plt.subplots(1,1,figsize=(7,5))
plt.title(r"$|\psi_{RBM}>-|\psi_{G.S}>$"+r"for different training steps"+ "\n"+ r" $L="+str(L)+r"\;\lambda="+str(round(G*DG,1))+r"\;iteration="+str(kk)+r"["+str(int(NR/NSPCA))+r"\;steps]$"+""+"\n"+r"$\theta="+str(phi)+r"\times\frac{\pi}{"+str(2*Nangle)+r"}$",fontsize=15)
plt.ylabel(r"$Discrepancy \;amplitudes$",fontsize=15)
plt.xlabel(r"$ket\; label$",fontsize=15)
axis.plot(np.abs(r_P),linestyle="dashed",color="blue",label="right convergence")
axis.plot(np.abs(w_P),linestyle="dashed",color="red",label="wrong convergence")
#axis.scatter(np.abs((r_P[perm]-PSI[perm])/PSI[perm]),color="red",s=10)
#axis.plot(num_states,PSI[perm]**2,color="black",label="$ket\;probability$")
axis.legend()
plt.yscale("log")
#plt.xlim([0,50])
plt.savefig(MASTER_DIR+"/DELTAP"+"AG"+str(G)+".png")


# In[113]:


plt.figure()
fig,axis=plt.subplots(1,1,figsize=(7,5))
plt.title(r"$|\psi_{RBM}>-|\psi_{G.S}>$"+r"for different training steps"+ "\n"+ r" $L="+str(L)+r"\;\lambda="+str(round(G*DG,1))+r"\;iteration="+str(kk)+r"["+str(int(NR/NSPCA))+r"\;steps]$"+""+"\n"+r"$\theta="+str(phi)+r"\times\frac{\pi}{"+str(2*Nangle)+r"}$",fontsize=15)
plt.ylabel(r"$Discrepancy \;amplitudes$",fontsize=15)
plt.xlabel(r"$ket\; label$",fontsize=15)
axis.plot(PSI,linestyle="dashed",color="blue",label="right convergence")
#axis.scatter(np.abs((r_P[perm]-PSI[perm])/PSI[perm]),color="red",s=10)
#axis.plot(num_states,PSI[perm]**2,color="black",label="$ket\;probability$")
axis.legend()
plt.yscale("log")
#plt.xlim([0,50])
plt.savefig(MASTER_DIR+"/DELTAP"+"AG"+str(G)+".png")


# In[101]:


plt.figure()
fig,axis=plt.subplots(1,1,figsize=(7,5))
plt.title(r"$|\psi_{RBM}>-|\psi_{G.S}>$"+r"for different training steps"+ "\n"+ r" $L="+str(L)+r"\;\lambda="+str(round(G*DG,1))+r"\;iteration="+str(kk)+r"["+str(int(NR/NSPCA))+r"\;steps]$"+""+"\n"+r"$\theta="+str(phi)+r"\times\frac{\pi}{"+str(2*Nangle)+r"}$",fontsize=15)
plt.ylabel(r"$Discrepancy \;amplitudes$",fontsize=15)
plt.xlabel(r"$ket\; label$",fontsize=15)
axis.plot(np.abs((r_P[perm]-PSI[perm])/PSI[perm]),linestyle="dashed",color="blue",label="right convergence")
axis.plot(np.abs((w_P[perm]-PSI[perm])/PSI[perm]),linestyle="dashed",color="red",label="wrong convergence")
#axis.scatter(np.abs((r_P[perm]-PSI[perm])/PSI[perm]),color="red",s=10)
#axis.plot(num_states,PSI[perm]**2,color="black",label="$ket\;probability$")
axis.legend()
plt.yscale("log")
plt.xlim([0,50])
plt.savefig(MASTER_DIR+"/DELTAP"+"AG"+str(G)+".png")


# In[102]:


plt.figure()
fig,axis=plt.subplots(1,1,figsize=(7,5))
plt.title(r"$|\psi_{RBM}>-|\psi_{G.S}>$"+r"for different training steps"+ "\n"+ r" $L="+str(L)+r"\;\lambda="+str(round(G*DG,1))+r"\;iteration="+str(kk)+r"["+str(int(NR/NSPCA))+r"\;steps]$"+""+"\n"+r"$\theta="+str(12)+r"\times\frac{\pi}{"+str(2*Nangle)+r"}$",fontsize=15)
plt.ylabel(r"$Discrepancy \;amplitudes$",fontsize=15)
plt.xlabel(r"$ket\; label$",fontsize=15)
axis.plot(r_P[perm],linestyle="dashed",color="blue",label="right convergence")
axis.plot(w_P[perm],linestyle="dashed",color="red",label="wrong convergence")
axis.plot(PSI[perm],linestyle="dashed",color="black",label="G.S")

#axis.scatter(np.abs((r_P[perm]-PSI[perm])/PSI[perm]),color="red",s=10)
#axis.plot(num_states,PSI[perm]**2,color="black",label="$ket\;probability$")
axis.legend()
plt.yscale("log")
#plt.xlim([0,50])
plt.savefig(MASTER_DIR+"/DELTAP"+"AG"+str(G)+".png")


# In[103]:


w_perm = np.argsort(-w_P)
r_perm = np.argsort(-r_P)


# In[98]:


hk.numbers_to_states(w_perm[2])


# In[99]:


hk.numbers_to_states(r_perm[4])


# In[100]:


hk.numbers_to_states(perm[0])


# In[ ]:


print(w_P[perm][0],r_P[perm][0],PSI[perm][0])


# In[104]:


labels=[r"$visible\;bias$",r"$dense\;bias$",r"$kernel$"]
labels_name=["visible_bias","dense_bias","kernel"]
label_par=[r"$a_i$",r"$b_i$",r"$k_{ij}$"]
ang=Nangle


# In[105]:


r_Im_Z=[np.array(r_param_RBM["params"]["visible_bias"]["value"]["imag"]),np.array(r_param_RBM["params"]["Dense"]["bias"]["value"]["imag"]),np.array(r_param_RBM["params"]["Dense"]["kernel"]["value"]["imag"])]
#r_Im_Z[2]=Z[2].reshape(-1,Z[2].shape[-1])

r_real_Z=[np.array(r_param_RBM["params"]["visible_bias"]["value"]["real"]),np.array(r_param_RBM["params"]["Dense"]["bias"]["value"]["real"]),np.array(r_param_RBM["params"]["Dense"]["kernel"]["value"]["real"])]
#r_real_Z[2]=Z[2].reshape(-1,Z[2].shape[-1])

w_real_Z=[np.array(w_param_RBM["params"]["visible_bias"]["value"]["real"]),np.array(w_param_RBM["params"]["Dense"]["bias"]["value"]["real"]),np.array(w_param_RBM["params"]["Dense"]["kernel"]["value"]["real"])]
#w_real_Z[2]=Z[2].reshape(-1,Z[2].shape[-1])

w_Im_Z=[np.array(w_param_RBM["params"]["visible_bias"]["value"]["imag"]),np.array(w_param_RBM["params"]["Dense"]["bias"]["value"]["imag"]),np.array(w_param_RBM["params"]["Dense"]["kernel"]["value"]["imag"])]
#w_Im_Z[2]=Z[2].reshape(-1,Z[2].shape[-1])







# In[106]:


# Create the 3D figure
Z=r_real_Z
add="real "
for i in range(len(label_par)-1):

    X = np.arange(0,NSPCA+1, 1)  # X values (rows)
    Y = np.arange(0, len(Z[i][0]), 1)  # Y values (columns)
    print(i,len(Z[i]),len(X))
    X_mesh, Y_mesh = np.meshgrid(Y, X)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X_mesh, Y_mesh,Z[i], cmap='viridis', edgecolor='k')
    # Add a color bar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.05)
    cbar.set_label(label_par[i])
    # Labels and title
    #ax.set_ylim([0,20])
    ax.set_xlabel(r"$site\;i$ ",fontsize=15)
    ax.set_ylabel("(steps["+str(int(NR/NSPCA))+"])",fontsize=15)
    ax.set_zlabel(label_par[i], labelpad=20,fontsize=15) 
    ax.set_title(add+labels[i]+"\n"+"$L="+str(L)+r"\;\lambda="+str(round(G*DG,1))+r"\;\;\theta="+str(angle[phi]/(np.pi/2))+r"\cdot [\pi/2]$",fontsize=15)
    ax.view_init(elev=0, azim=70)  # Adjust angles (elevation=45°, azimuth=120°)
    plt.savefig(MASTER_DIR+"/"+"theta"+str(ang)+labels_name[i]+"L"+str(L)+"G"+str(G)+".png")


# In[107]:


# Create the 3D figure
Z=w_real_Z
add="real "
for i in range(len(label_par)-1):

    X = np.arange(0,NSPCA+1, 1)  # X values (rows)
    Y = np.arange(0, len(Z[i][0]), 1)  # Y values (columns)
    print(i,len(Z[i]),len(X))
    X_mesh, Y_mesh = np.meshgrid(Y, X)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X_mesh, Y_mesh,Z[i], cmap='viridis', edgecolor='k')
    # Add a color bar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.05)
    cbar.set_label(label_par[i])
    # Labels and title
    #ax.set_ylim([0,20])
    ax.set_xlabel(r"$site\;i$ ",fontsize=15)
    ax.set_ylabel("(steps["+str(int(NR/NSPCA))+"])",fontsize=15)
    ax.set_zlabel(label_par[i], labelpad=20,fontsize=15) 
    ax.set_title(add+labels[i]+"\n"+"$L="+str(L)+r"\;\lambda="+str(round(G*DG,1))+r"\;\;\theta="+str(angle[phi]/(np.pi/2))+r"\cdot [\pi/2]$",fontsize=15)
    ax.view_init(elev=0, azim=70)  # Adjust angles (elevation=45°, azimuth=120°)
    plt.savefig(MASTER_DIR+"/"+"theta"+str(ang)+labels_name[i]+"L"+str(L)+"G"+str(G)+".png")


# In[225]:


C=np.array(r_param_RBM["params"]["Dense"]["kernel"]["value"]["real"])
# Plot the heatmap
X = np.arange(0,L, 1)  # X values (rows)
Y = np.arange(0, int(L) , 1)  # Y values (columns)
X_mesh, Y_mesh = np.meshgrid(X, Y)


# In[226]:


for i in range(NSPCA):
    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.imshow(np.abs(C[i].T), cmap='gray', aspect='auto', origin='lower')

# Add a colorbar
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label(r"Parameter Value $k_{ij}$",fontsize=15)

# Labels and title
    ax.set_xlabel(r"$visible\;site \;i$",fontsize=15)
    ax.set_ylabel("$hidden\;site \; j$",fontsize=15)
    ax.set_title(r"Density Plot of Kernel over iter"+"\n"+"$L="+str(L)+r"\;\lambda="+str(round(G*DG,1))+r"\;\;\theta="+str(angle[ang]/(np.pi/2))+r"\cdot [\pi/2]$"+str(i),fontsize=15)

# Set tick labels
    ax.set_xticks(np.arange(0,L,1))
    ax.set_xticklabels(np.arange(0,L,1))
    ax.set_yticks(np.arange(len(Y)))
    ax.set_yticklabels(Y)
    plt.savefig(MASTER_DIR+"/"+"theta"+str(ang)+"L"+str(L)+"G"+str(G)+"NS"+str(NS)+"density_over it"+str(i)+".png")
    plt.show()


# In[227]:


C=np.array(w_param_RBM["params"]["Dense"]["kernel"]["value"]["real"])
# Plot the heatmap
X = np.arange(0,L, 1)  # X values (rows)
Y = np.arange(0, int(L) , 1)  # Y values (columns)
X_mesh, Y_mesh = np.meshgrid(X, Y)


# In[228]:


for i in range(NSPCA):
    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.imshow(np.abs(C[i].T), cmap='gray', aspect='auto', origin='lower')

# Add a colorbar
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label(r"Parameter Value $k_{ij}$",fontsize=15)

# Labels and title
    ax.set_xlabel(r"$visible\;site \;i$",fontsize=15)
    ax.set_ylabel("$hidden\;site \; j$",fontsize=15)
    ax.set_title(r"Density Plot of Kernel over iter"+"\n"+"$L="+str(L)+r"\;\lambda="+str(round(G*DG,1))+r"\;\;\theta="+str(angle[ang]/(np.pi/2))+r"\cdot [\pi/2]$"+str(i),fontsize=15)

# Set tick labels
    ax.set_xticks(np.arange(0,L,1))
    ax.set_xticklabels(np.arange(0,L,1))
    ax.set_yticks(np.arange(len(Y)))
    ax.set_yticklabels(Y)
    plt.savefig(MASTER_DIR+"/"+"theta"+str(ang)+"L"+str(L)+"G"+str(G)+"NS"+str(NS)+"density_over it"+str(i)+".png")
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




