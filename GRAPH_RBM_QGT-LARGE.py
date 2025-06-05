#!/usr/bin/env python
# coding: utf-8

# In[3]:


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
from scipy.sparse import triu
from scipy.linalg import eigvalsh


# In[4]:


#DEFINE FUNCTIONS:


# In[5]:


from Methods.class_WF import rotated_sigmax, rotated_sigmaz,isigmay,rotated_IsingModel,rotated_LONGIsingModel,rotated_BROKEN_Z2IsingModel,rotated_CIMModel_2
from Methods.class_WF import rotated_XYZModel, parity_Matrix, parity_IsingModel, Sz0Szj, Sx0Sxj, to_array, rotated_m, rotated_CIMModel, rotated_CIMModel_Y


# In[6]:


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
    
    logKK=states@CC
    log_AMP=np.log(np.cosh(states@AA+DD))
    log_ALMOST_PROB=np.sum(log_AMP,axis=-1)+logKK
    log_NORM=log_ALMOST_PROB+np.conjugate(log_ALMOST_PROB)
    NORM=np.sqrt(np.sum(np.exp(log_NORM)))
    
    PROB=np.exp(log_ALMOST_PROB)/NORM
    #------------------------------
    #KK=np.exp(states@CC)
    #AMP=np.cosh(states@AA+DD)
    #ALMOST_PROB=KK*np.prod(AMP,axis=-1)
    
    #NORMALIZE THE PROBABILITY
    #NORM=np.sqrt(ALMOST_PROB@np.conjugate(ALMOST_PROB))
    #PROB=ALMOST_PROB/NORM
    
    return PROB

def vstate_par(data_var,it,NS):
    dense_bias_R=np.array(data_var["params"]["Dense"]["bias"]["value"]["real"][it])
    dense_bias_I=np.array(data_var["params"]["Dense"]["bias"]["value"]["imag"][it])
    dense_bias=np.array(dense_bias_R+1j*dense_bias_I,dtype=np.complex128)

    visible_bias_R=np.array(data_var["params"]["visible_bias"]["value"]["real"][it])
    visible_bias_I=np.array(data_var["params"]["visible_bias"]["value"]["imag"][it])
    visible_bias=np.array(visible_bias_R+1j*visible_bias_I,dtype=np.complex128)

    dense_kernel_R=np.array(data_var["params"]["Dense"]["kernel"]["value"]["real"][it])
    dense_kernel_I=np.array(data_var["params"]["Dense"]["kernel"]["value"]["imag"][it])
    dense_kernel=np.array(dense_kernel_R+1j*dense_kernel_I,dtype=np.complex128)

    new_parameters={
        'params': {
            'Dense': {
                'bias': dense_bias,
                'kernel':dense_kernel
            },
            'visible_bias':visible_bias      
        }
    }

    hi=nk.hilbert.Spin(s=1/2,N=L)
    #sampler=nk.sampler.ExactSampler(hi)
    sampler=nk.sampler.MetropolisLocal(hi)
    model=nk.models.RBM(alpha=NN,param_dtype=complex)
    vstate=nk.vqs.MCState(sampler,model,n_samples=NS,variables=new_parameters)

    return vstate





# In[7]:


#DEFINE THE VARIABLES:


# In[8]:


for path in sys.path:
    print(path)


# In[9]:


import os

print("Current working directory:", os.getcwd())


# In[12]:


L=50
NS=2048
G=100
DG=0.01
NN=1
NL=1
NR=1000
learning_rate=0.05
basis="QIM"
modelo="RBM_COMPLEX"
##########################
angle=0
Nangle=12
dangle=np.pi/(2*Nangle)
NSPCA=10
NM=1
##########################
SHIFT=1
SAMPLER="METROPOLIS SAMPLER $off diag="+str(0.1*SHIFT)+"$"
SHIFT="DS"+str(SHIFT)

MASTER_DIR="RUN_"+basis+"_"+modelo+"NN"+str(NN)+"L"+str(L)+"G"+str(G)+"NA"+str(Nangle)+"NSPCA"+str(NSPCA)+SHIFT
Nstates=2**L
eps=10**(-10)
angle=[dangle*i for i in range(Nangle+1)]
labelling=r"$\;L="+str(L)+r"\;h/J="+str(round(G*DG,2))+"$"+"\n"+r"$COMPLEX\;RBM\;\alpha="+str(NN)+r"$"
labels=[r"$"+str(i)+r"\times\frac{\pi}{"+str(2*Nangle)+"}"+"$" for i in range(Nangle+1)]
print(MASTER_DIR)

if os.path.isdir(MASTER_DIR+" "):
    print("DIRECTORY NOT FOUND")
else:
    print("DIRECTORY ALREADY CREATED")
VAR_FILENAME="NANGLE"+str(Nangle)+basis+"M3L"+str(L)+"W1"+"G"+str(G)+"NS"+str(NS)+"NN"+str(NN)+"NL"+str(NL)+"NR"+str(NR)+"VAR"


# In[13]:


# Load the data from the JSON file
data=[[None for i in range(len(angle))] for rep in range(NM)]
for rep in range(NM):
    for ii in range(len(angle)):
            with open(MASTER_DIR+"/"+str(rep)+"NM"+str(ii)+VAR_FILENAME+".json", "r") as f:
                data[rep][ii] = json.load(f)


# In[14]:


AA=[i for i in range(Nangle+1)]
NMM=[[kk for kk in range(NM)] for i in range(Nangle+1)]


# In[ ]:


it=9
eps=10**(-5)
QGT=nk.optimizer.qgt.QGTOnTheFly
av_s=[[None for rep in range(NM)] for ii in angle]
trunked_rank=[[None for rep in range(NM)] for ii in angle]
for ii in range(len(angle)):
    for rep in range(NM):
        data_var=data[rep][ii]
        vstate=vstate_par(data_var,it,NS)
        QGT_rep=QGT(vstate,holomorphic=True)
        s_values=eigvalsh(np.real(QGT_rep.to_dense()))
        av_s[ii][rep]=np.sum(s_values)/len(s_values)
        trunked_rank[ii][rep]=np.sum(np.sum((np.abs(s_values)>eps)*1.0))/len(s_values)
        
av_s=np.array(av_s)
trunked_rank=np.array(trunked_rank)
    


# In[ ]:


N_label_freq=2
fsize=18
xticks=np.arange(0,Nangle+1,N_label_freq)
x_labels=[r"$\frac{"+str(i)+r"\pi}{"+str(2*Nangle)+"}$" for i in xticks]


# In[ ]:


x=np.arange(0,Nangle+1,1)
plt.figure()
plt.title(r"$\frac{1}{N}\sum_{i}\lambda_i$"+labelling+"\n For different iterations"+"\n"+SAMPLER,fontsize=15)
plt.xlabel(r"$Angle\;\theta$",fontsize=15)
plt.ylabel(r"$\frac{1}{N}\sum_{i}\lambda_i$",fontsize=15)
plt.xticks(xticks,x_labels,fontsize=fsize)

for ii in range(NM):
    plt.plot(x,av_s[:,ii])
    plt.scatter(x,av_s[:,ii])
plt.legend()
plt.tight_layout()
plt.savefig("L"+str(L)+"G"+str(G)+"NS"+str(NS)+"sum_QFI"+SHIFT+".png")


# In[7]:


plt.figure()
plt.title(r"$trunked\;rank(QFI)/N\;$"+r"$\epsilon="+str(eps)+"$"+"\n"+labelling+"\n For different iterations"+"\n"+SAMPLER,fontsize=15)
plt.xlabel(r"$Angle\;\theta$",fontsize=15)
plt.ylabel(r"$rank(QFI)/N$",fontsize=15)
plt.xticks(xticks,x_labels,fontsize=fsize)

for ii in range(NM):
    plt.plot(x,trunked_rank[:,ii])
    plt.scatter(x,trunked_rank[:,ii])
plt.legend()
plt.tight_layout()
plt.savefig("L"+str(L)+"G"+str(G)+"NS"+str(NS)+"RANK"+SHIFT+".png")


# In[8]:


# ITERATIONS 


# In[43]:


eps=10**(-5)
chosen=[0,11,12]
QGT=nk.optimizer.qgt.QGTOnTheFly
av_s=[[ [] for rep in range(NM)] for ii in chosen]
trunked_rank=[[ [] for rep in range(NM)] for ii in chosen]
dit=1
for ii in range(len(chosen)):
    for rep in range(NM):
        data_var=data[rep][chosen[ii]]
        for it in range(0,NSPCA,dit):
            print("angle",ii,"rep",rep,"it",it)
            vstate=vstate_par(data_var,it,NS)
            QGT_rep=QGT(vstate,holomorphic=True)
            s_values=eigvalsh(np.real(QGT_rep.to_dense()))
            
            av_s[ii][rep].append(np.sum(s_values)/len(s_values))
            trunked_rank[ii][rep].append(np.sum(np.sum((np.abs(s_values)>eps)*1.0))/len(s_values))
        
av_s=np.array(av_s)
trunked_rank=np.array(trunked_rank)


# In[44]:


xx=np.arange(0,NSPCA,dit)*10
for k in range(len(chosen)):
    plt.figure()
    plt.title(r"$\frac{1}{N}\sum_{i}\lambda_i"+r"\;\theta=\frac{"+str(chosen[k])+r"\pi}{"+str(Nangle*2)+"}$"+labelling+"\n For different iterations"+" "+SAMPLER,fontsize=15)
    plt.xlabel(r"$step$",fontsize=15)
    plt.ylabel(r"$\frac{1}{N}\sum_{i}\lambda_i$",fontsize=15)

    for ii in range(NM):
        plt.plot(xx,av_s[k,ii])
        plt.scatter(xx,av_s[k,ii])
    plt.legend()
    plt.tight_layout()
    plt.savefig("L"+str(L)+"G"+str(G)+"NS"+str(NS)+"k"+str(k)+"CONVERGENCE_sum_QFI"+SHIFT+".png")


# In[45]:


xx=np.arange(0,NSPCA,dit)*10
for k in range(len(chosen)):
    plt.figure()
    plt.title(r"$trunked\;rank(QFI)/N\;"+r"\;\theta=\frac{"+str(chosen[k])+r"\pi}{"+str(Nangle*2)+"}$"+labelling+"\n For different iterations"+" "+SAMPLER,fontsize=15)
    plt.xlabel(r"$step$",fontsize=15)
    plt.ylabel(r"$trunked\;rank(QFI)/N\;$",fontsize=15)

    for ii in range(NM):
        plt.plot(xx,trunked_rank[k,ii])
        plt.scatter(xx,trunked_rank[k,ii])
    plt.legend()
    plt.tight_layout()
    plt.savefig("L"+str(L)+"G"+str(G)+"NS"+str(NS)+"k"+str(k)+"CONVERGENCE_RANK_QFI"+SHIFT+".png")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




