#!/usr/bin/env python
# coding: utf-8

# In[368]:


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


# In[279]:


#DEFINE FUNCTIONS:


# In[280]:


from Methods.class_WF import rotated_sigmax, rotated_sigmaz,isigmay,rotated_IsingModel,rotated_LONGIsingModel,rotated_BROKEN_Z2IsingModel,rotated_CIMModel_2
from Methods.class_WF import rotated_XYZModel, parity_Matrix, parity_IsingModel, Sz0Szj, Sx0Sxj, to_array, rotated_m, rotated_CIMModel, rotated_CIMModel_Y


L=20
G=50
DG=0.01
NR=500
angle=0
Nangle=12
dangle=np.pi/(2*Nangle)
basis="QIM"


MASTER_DIR="DIAG_RUN_"+basis+"_"+"L"+str(L)+"G"+str(G)+"NA"+str(Nangle)
Nstates=2**L
angle=[dangle*i for i in range(Nangle+1)]
num_states=[i for i in range(2**L)]

labels=[r"$"+str(i)+r"\times\frac{\pi}{"+str(2*Nangle)+"}"+"$" for i in range(Nangle+1)]
try:
    os.mkdir(MASTER_DIR)
except:
    print("DIRECTORY ALREADY CREATED")


pub=class_WF.publisher([MASTER_DIR+"/"+"E_L","G"],[L,G],["ANGLE","E","N_pos"])
pub.create()

pub_WF=class_WF.publisher([MASTER_DIR+"/"+"WF_L","G"],[L,G],[])
pub_WF.create()

hi=nk.hilbert.Spin(s=1/2,N=L)
for theta in range(len(angle)):
    #PSI AND DIAGONALIZATION
    H=rotated_IsingModel(angle[theta],G*DG,L,hi)
    #H=rotated_CIMModel_Y(angle[theta],G*DG,L,hi)
    #H=rotated_CIMModel_2(angle[theta],G*DG,L,hi)
    eig_vals_other,eig_vecs_other=eigsh(H.to_sparse())

    pub_WF.write(eig_vecs_other[:,0].tolist())

    #SIGN PROBLEM SPARSE VERSION
    H_triu = triu(H.to_sparse(), k=1)
    # Get the data and row/col indices of non-zero entries
    data = H_triu.data
    # Filter elements with absolute value > eps
    mask = np.abs(data) > eps
    filtered_data = data[mask]
    # Count how many are > eps (i.e., positive and above threshold)
    N_pos=np.sum(filtered_data > eps)
    pub.write([0.0,theta,0.0,eig_vals_other[0],0.0,N_pos]) 
    print("step",theta,"done")

pub.close()
pub_WF.close()





