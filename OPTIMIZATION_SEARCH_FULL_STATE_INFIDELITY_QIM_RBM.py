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

from Methods.class_WF import rotated_sigmax, rotated_sigmaz,isigmay,rotated_IsingModel,rotated_BROKEN_Z2IsingModel,rotated_CIMModel_2,bi_ladder_rotated_IsingModel
from Methods.class_WF import rotated_XYZModel, parity_Matrix, parity_IsingModel, Sz0Szj, Sx0Sxj, to_array, rotated_m
from Methods.FULL_STATE_OP import objective_I

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
Nangle=parameters[4]
NMEAN=parameters[5]
G = [parameters[x] for x in range(6,n_par-1)]


n_iter=200
#Model Details
basis="BI_QIM"
architecture = "RBM_COMPLEX"
pbc=True


add=""
if pbc:
    add+="PBC"

#basis = "BROKENZ2_QIM"

if basis == "QIM":
    add+=""
elif basis == "BI_QIM":
    W=2
elif basis == "BROKENZ2_QIM":
    hpar=0.01
    add+= "HPAR"+str(round(hpar,2))
elif basis == "CIM_2":
    add+=""
else:
    print("MODEL NOT FOUND")
    exit()


    
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
    

for g in G:
    SLAVE_DIR="FULL_STATE_RUN_"+basis+"_"+architecture+"NN"+str(NN)+"L"+str(L)+"G"+str(g)+"NA"+str(Nangle)+"NSPCA"+str(NSPCA)+add
    try:
        os.mkdir(MASTER_DIR+"/"+SLAVE_DIR)
    except FileExistsError:
        print(f"Directory "+MASTER_DIR+"/"+SLAVE_DIR+" already exists.")


#Constants initialization

angle=0
dangle=np.pi/(2*Nangle)
Nstates=2**(L*W)
eps=10**(-10)
angle=[dangle*i for i in range(Nangle+1)]
num_states=[i for i in range(2**(L*W))]
sites_corr=[1,int(L*W/2),L*W-1]
sites_corr=[str(x) for x in sites_corr]

# Hilbert space generation in Netket

hi=nk.hilbert.Spin(s=1/2,N=L*W,inverted_ordering=True)


for g in G:
    SLAVE_DIR="FULL_STATE_RUN_"+basis+"_"+architecture+"NN"+str(NN)+"L"+str(L)+"G"+str(g)+"NA"+str(Nangle)+"NSPCA"+str(NSPCA)+add
    FILENAME="NANGLE"+str(Nangle)+basis+"M3L"+str(L)+"W1"+"G"+str(g)+"NN"+str(NN)+"NL"+str(NL)+"NR"+str(NR)
    log_best_params=nk.logging.RuntimeLog()
    for ii in range(len(angle)):
        #DEFINE THE HAMILTONIAN AND THE EIGENVECTOR
        if basis == "QIM":
            H=rotated_IsingModel(angle[ii],g*DG,L,hi,pbc=pbc)
        if basis == "CIM_2":
            H=rotated_CIMModel_2(angle[ii],g*DG,L,hi,pbc=pbc)
        if basis == "BI_QIM":


            H=bi_ladder_rotated_IsingModel(angle[ii],g*DG,g*DG,1.0,1.0,L,hi,pbc=pbc)
            
        elif basis == "BROKENZ2_QIM":
            H=rotated_BROKEN_Z2IsingModel(angle[ii],g*DG,L,hi,hpar,pbc=pbc)
        eig_vals,eig_vecs=np.linalg.eigh(H.to_dense())
        # DEFINE THE GROUND STATE ENERGY
        E_gs=eig_vals[0]
        # DEFINE THE GROUND STATE EIGEN-VECTOR
        if eig_vecs[:,0][0]<0:
            GS=(-1.0)*eig_vecs[:,0]
        else:
            GS=eig_vecs[:,0]
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
        objective_final=partial(objective_I,model=model,psi=psi,phi=phi,L=L*W,hi=hi,H=H,n_iter=n_iter,holomorphic=holomorphic)
        study.optimize(objective_final,n_trials=50)    
        best_params=study.best_params
        log_best_params(ii,best_params)
        
    log_best_params.serialize(MASTER_DIR+"/"+SLAVE_DIR+"/"+FILENAME+"HYP")







