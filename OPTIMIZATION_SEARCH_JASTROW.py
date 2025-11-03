import netket as nk
import matplotlib.pyplot as plt
import numpy as np
from netket.operator.spin import sigmax,sigmaz,sigmay,identity,sigmam,sigmap   
import netket_fidelity as nkf
from netket_fidelity.infidelity import InfidelityOperator
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
from functools import reduce,partial
from flax import struct
import optuna
import sys
import logging
import random
from Methods.class_WF import rotated_sigmax, rotated_sigmaz,isigmay,rotated_IsingModel,rotated_BROKEN_Z2IsingModel,rotated_CIMModel_2
from Methods.class_WF import rotated_XYZModel, parity_Matrix, parity_IsingModel, Sz0Szj, Sx0Sxj, to_array, rotated_m
from Methods.FULL_STATE_OP import objective_I
from Methods.STATES import build_jastrow_wf

def change_to_int(x,L):
    Aux=jnp.array([2**(L-1-i) for i in range(L)])
    Z=jnp.array(jnp.mod(1+x,3)/2,int)
    return np.sum(Aux*Z,axis=-1)

class EWF(nn.Module):                                                                                                                  
    eig_vec:tuple = struct.field(pytree_node=False)                                                                                                                
    L:float
    def setup(self):                                                                                                                   
        self.aux=jnp.array(self.eig_vec)                                                                                               
        self.j1=self.param("j1", nn.initializers.normal(),(1,),float)    
    def __call__(self,x):                                                                                                              
        indices = change_to_int(x,self.L)                                                                                              
        A = [self.aux[idx] for idx in indices]                                                                                         
        return jnp.array(A) 

def save_params(vmc,i,log_var):
        log_var(i,vmc.state.variables)

parameters=sys.argv
n_par=len(parameters)
parameters=[int(parameters[x]) for x in range(1,n_par)]

L=parameters[0]
W=1
DG=0.01
NN=parameters[1]
NL=1
NR=parameters[2]
NSPCA=parameters[3]
NMEAN=parameters[4]
NSAMPLES=parameters[5]
seed=parameters[6]
n_iter=200
#Model Details
basis = "J"+str(seed)
complex_J=True

if complex_J:
    add="COMPLEXJ"
else:
    add=""
    
architecture = "RBM_COMPLEX"
hi=nk.hilbert.Spin(s=1/2,N=L*W,inverted_ordering=True)

#Constants initialization
Nstates=2**(L*W)
eps=10**(-10)
random.seed(a=seed)

#Model and optimization details
architecture_set=["RBM_COMPLEX","RBM_REAL","WSIGNS_RBM_COMPLEX","WSIGNS_RBM_REAL"]
if architecture not in architecture_set:
    print("MODEL IS NOT DEFINED")
    exit()
    


#Creation of the folder
MASTER_DIR="INFIDELITY"
if not os.path.isdir(MASTER_DIR):
    print(os.path.isdir(MASTER_DIR))
    os.mkdir(MASTER_DIR)
    print(f"Directory '{MASTER_DIR}' not found previously but created successfully.")
    
for g in range(NSAMPLES):
    SLAVE_DIR="FULL_STATE_RUN_"+basis+"_"+architecture+"NN"+str(NN)+"L"+str(L)+"G"+str(g)+"NSPCA"+str(NSPCA)+add
    try:
        os.mkdir(MASTER_DIR+"/"+SLAVE_DIR)
    except FileExistsError:
        print(f"Directory "+MASTER_DIR+"/"+SLAVE_DIR+" already exists.")






for g in range(NSAMPLES):
    SLAVE_DIR="FULL_STATE_RUN_"+basis+"_"+architecture+"NN"+str(NN)+"L"+str(L)+"G"+str(g)+"NSPCA"+str(NSPCA)+add
    FILENAME=basis+"M3L"+str(L)+"W1"+"G"+str(g)+"NN"+str(NN)+"NL"+str(NL)+"NR"+str(NR)
    
    log_best_params=nk.logging.RuntimeLog()

    J_couplings=list([random.random() for j in range(int(L*(L+1)/2))])
    
    if complex_J:

        J= np.array(random.sample(J_couplings, k=len(J_couplings)),dtype=complex)
        J_couplings_COMPLEX=list([random.random() for j in range(int(L*(L+1)/2))])
        J_COMPLEX=np.array(random.sample(J_couplings, k=len(J_couplings)),dtype=complex)
        J=J+1j*J_COMPLEX
    else:
        J= np.array(random.sample(J_couplings, k=len(J_couplings)))
        
    rows, cols = np.triu_indices(L)
    idxs=list(zip(rows, cols))
    
    if complex_J:
        J_coeff=np.zeros((L,L),dtype=complex)
    else:
        J_coeff=np.zeros((L,L))

    for idx in range(len(idxs)):
        J_coeff[idxs[idx][1],idxs[idx][0]]=J[idx]
        if complex_J and idxs[idx][0] == idxs[idx][1]:
            J_coeff[idxs[idx][1],idxs[idx][0]]=np.real(J[idx])
        
    GS=build_jastrow_wf(L,J_coeff,hi)
        
    # ARCHITECTURES
    if architecture=="RBM_COMPLEX" or architecture == "WSIGNS_RBM_COMPLEX":
            
        model=nk.models.RBM(alpha=NN,param_dtype=complex)
        holomorphic=True
            
    elif architecture=="RBM_REAL" or architecture == "WSIGNS_RBM_REAL":
        model=nk.models.RBM(alpha=NN)
        holomorphic=False
    if architecture == "WSIGNS_RBM_COMPLEX" or architecture == "WSIGNS_RBM_REAL":
        GS=np.abs(GS)
            
        # EXACT G.S
    GS[np.abs(GS)<10**(-10)]=0.0
    GS=np.log(GS)
    GS[GS!=GS]=-np.inf
    Exact_GS=EWF(L=L*W,eig_vec=tuple(GS))

    phi = nk.vqs.FullSumState(hi, model=model)
    psi = nk.vqs.FullSumState(hi, model=Exact_GS)


        
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(direction="minimize",sampler=optuna.samplers.RandomSampler(),pruner=optuna.pruners.MedianPruner())
    objective_final=partial(objective_I,model=model,psi=psi,phi=phi,L=L*W,hi=hi,n_iter=n_iter,holomorphic=holomorphic)
    study.optimize(objective_final,n_trials=50)    
    best_params=study.best_params
    log_best_params(0,best_params)
        
    log_best_params.serialize(MASTER_DIR+"/"+SLAVE_DIR+"/"+FILENAME+"HYP")
