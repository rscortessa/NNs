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
from matplotlib import cm
import matplotlib as mp
import matplotlib.colors as mcolors
import scipy.special as sc
import scipy.optimize as so
from matplotlib.colors import LinearSegmentedColormap
from functools import reduce

from Methods.class_WF import rotated_sigmax, rotated_sigmaz,isigmay,rotated_IsingModel,rotated_BROKEN_Z2IsingModel
from Methods.class_WF import rotated_XYZModel, parity_Matrix, parity_IsingModel, Sz0Szj, Sx0Sxj, to_array, rotated_m


parameters=sys.argv
n_par=len(parameters)
parameters=[int(parameters[x]) for x in range(1,n_par)]

L=parameters[0]
G=parameters[1]
DG=0.01
NN=parameters[2]
NL=1
NR=parameters[3]
NSPCA=parameters[4]
Nangle=parameters[5]
NMEAN=parameters[6]
learning_rate=0.05
diag_shift=1.0
delta=0.05
basis="QIM"
modelo="RBM_COMPLEX"
broken_z2=False

#Model and optimization details

if modelo=="RBM_COMPLEX":
    model=nk.models.RBM(alpha=NN,param_dtype=complex)
    sr = nk.optimizer.SR(diag_shift=diag_shift*0.01, holomorphic=True)
elif modelo=="RBM_REAL":
    model=nk.models.RBM(alpha=NN)
    sr = nk.optimizer.SR(diag_shift=diag_shift*0.01, holomorphic=False)
    
optimizer=nk.optimizer.Sgd(learning_rate=learning_rate)

#Creation of the folder

MASTER_DIR="FULL_STATE_BROKENZ2_RUN_QIM_"+modelo+"NN"+str(NN)+"L"+str(L)+"G"+str(G)+"NA"+str(Nangle)+"NSPCA"+str(NSPCA)+"SHIFT"+str(diag_shift)

try:
    os.mkdir(MASTER_DIR)
except:
    print("DIRECTORY ALREADY CREATED")
OBS_FILENAME="NANGLE"+str(Nangle)+basis+"M3L"+str(L)+"W1"+"G"+str(G)+"NN"+str(NN)+"NL"+str(NL)+"NR"+str(NR)+"OBS"
VAR_FILENAME="NANGLE"+str(Nangle)+basis+"M3L"+str(L)+"W1"+"G"+str(G)+"NN"+str(NN)+"NL"+str(NL)+"NR"+str(NR)+"VAR"

#Constants initialization

angle=0
dangle=np.pi/(2*Nangle)
Nstates=2**L
eps=10**(-10)
angle=[dangle*i for i in range(Nangle+1)]
num_states=[i for i in range(2**L)]
sites_corr=[1,int(L/2),L-1]
sites_corr=[str(x) for x in sites_corr]

# Hilbert space generation in Netket

hi=nk.hilbert.Spin(s=1/2,N=L)


for tt in range(NMEAN):
    for ii in range(len(angle)):
        
    
        H = rotated_BROKEN_Z2IsingModel(angle[ii],G*DG,L,hi,delta)
        g = nk.graph.Hypercube(length=L, n_dim=1, pbc=False)
        PSI = class_WF.FULL_WF(L,hi,sr,optimizer,model,H)
        
        #THE OUT LOGS ARE CREATED
        log = nk.logging.RuntimeLog()
        log2 = nk.logging.RuntimeLog()
        
        
        #OBSERVABLES INIT.
        obs={}
        obs["P"]=parity_IsingModel(angle[ii],L,hi)
        obs["M"]=rotated_m(angle[ii],L,hi)
    
        for jj in sites_corr:
            obs["CZ0"+jj]=Sz0Szj(angle[ii],L,hi,int(jj))
            obs["CX0"+jj]=Sx0Sxj(angle[ii],L,hi,int(jj))

        NR_eff=int(NR/NSPCA)
        for kk in range(NSPCA):
    
            #FIRST WE COMPUTE THE PARAMETERS
            PSI.save_params(kk,log2)
            #RUNNING FOR NR_EFF ITERATIONS
            PSI.run(obs=obs,n_iter=NR_eff,log=log)
        
        #LAST CALCULATION
        
        PSI.save_params(kk,log2)
        log2.serialize(MASTER_DIR+"/"+str(tt)+"NM"+str(ii)+VAR_FILENAME)
        log.serialize(MASTER_DIR+"/"+str(tt)+"NM"+str(ii)+OBS_FILENAME)
   








