import netket as nk
import json
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

from Methods.class_WF import rotated_sigmax, rotated_sigmaz,isigmay,rotated_IsingModel,rotated_BROKEN_Z2IsingModel,rotated_CIMModel_2
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
ii_angle=parameters[6]
G = [parameters[x] for x in range(7,n_par-1)]


n_iter=200
#Model Details
basis="QIM"
architecture = "WSIGNS_RBM_COMPLEX"
pbc=True

if basis == "BI_QIM":
    W=2

add=""
if pbc:
    add+="PBC"

#basis = "BROKENZ2_QIM"

if basis == "QIM":
    add+=""
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

hyper_parameters_ready=False

for g in G:
    
    #DEFINE DIRECTORIES
    
    SLAVE_DIR="FULL_STATE_RUN_"+basis+"_"+architecture+"NN"+str(NN)+"L"+str(L)+"G"+str(g)+"NA"+str(Nangle)+"NSPCA"+str(NSPCA)+add
    FILENAME="NANGLE"+str(Nangle)+basis+"M3L"+str(L)+"W1"+"G"+str(g)+"NN"+str(NN)+"NL"+str(NL)+"NR"+str(NR)
    hyper_parameters_ready=False
    try:

        #CHECK IF THE HYPER-PARAMETERS ARE ALREADY THERE

        print("CHECKING PARAMETERS............")
        with open(MASTER_DIR+"/"+SLAVE_DIR+"/"+FILENAME+"HYP"+".json", "r") as f:
            data_best = json.load(f)
    
        hyper_parameters_ready=True

        print("HYPER-PARAMETERS ALREADY FOUND")
        
    except:
        log_best_params=nk.logging.RuntimeLog()


        #CREATE A LOG TO SAVE THE HYPERPARAMETERS
    print("Hyper_parameters exist?",hyper_parameters_ready)

    #DEFINE THE HAMILTONIAN AND THE EIGENVECTOR
    if basis == "QIM":
        H=rotated_IsingModel(angle[ii_angle],g*DG,L,hi,pbc=pbc)
    if basis == "CIM_2":
        H=rotated_CIMModel_2(angle[ii_angle],g*DG,L,hi,pbc=pbc)
    if basis == "BI_QIM":

        H=bi_ladder_rotated_IsingModel(angle[ii_angle],g*DG,g*DG,1.0,1.0,L,hi,pbc=pbc)
            
    elif basis == "BROKENZ2_QIM":
        H=rotated_BROKEN_Z2IsingModel(angle[ii_angle],g*DG,L,hi,hpar,pbc=pbc)

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

    if hyper_parameters_ready:

        best_params={}
        best_params["learning_rate"]=data_best["learning_rate"]["value"][ii_angle]
        best_params["diag_shift"]=data_best["diag_shift"]["value"][ii_angle]
        best_params["cv"]=data_best["cv"]["value"][ii_angle]


    else:
            
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        study = optuna.create_study(direction="minimize",sampler=optuna.samplers.RandomSampler(),pruner=optuna.pruners.MedianPruner())
        objective_final=partial(objective_I,model=model,psi=psi,phi=phi,L=L*W,hi=hi,H=H,n_iter=n_iter,holomorphic=holomorphic)
        study.optimize(objective_final,n_trials=50)    
        best_params=study.best_params
      
    #OPTIMIZER AND PRECONDITIONER INITIALIZATION
        
    optimizer = optimizer=nk.optimizer.Sgd(learning_rate=best_params["learning_rate"])
    preconditioner = nk.optimizer.SR(diag_shift=best_params["diag_shift"], holomorphic=holomorphic)
        
        

    #OBSERVABLES INIT.        

    obs = {}
    obs["Energy"] = H

    if basis == "QIM" or basis == "BROKENZ2_QIM":
        obs["P"]=parity_IsingModel(angle[ii_angle],L*W,hi)
        obs["M"]=rotated_m(angle[ii_angle],L*W,hi)
    
    for jj in sites_corr:
        obs["CZ0"+jj]=Sz0Szj(angle[ii_angle],L*W,hi,int(jj))
        obs["CX0"+jj]=Sx0Sxj(angle[ii_angle],L*W,hi,int(jj))

    #Number of effective steps
    NR_eff=int(NR/NSPCA)

    for tt in range(NMEAN):
            
        #RESTART THE NETWORK
        phi.init_parameters()
        te = nkf.driver.InfidelityOptimizer(psi, optimizer, variational_state=phi, preconditioner=preconditioner, cv_coeff=best_params["cv"])
            
        #THE OUT LOGS ARE CREATED
        log = nk.logging.RuntimeLog()
        log_var = nk.logging.RuntimeLog()
            
        for kk in range(NSPCA):
    
            #FIRST WE COMPUTE THE PARAMETERS
            save_params(te,kk,log_var)
            #RUNNING FOR NR_EFF ITERATIONS
            te.run(obs=obs,n_iter=NR_eff,out=log,show_progress=False)
                
        #LAST CALCULATION
        save_params(te,NSPCA,log_var)
        log_var.serialize(MASTER_DIR+"/"+SLAVE_DIR+"/"+str(tt)+"NM"+str(ii_angle)+FILENAME+"VAR")
        log.serialize(MASTER_DIR+"/"+SLAVE_DIR+"/"+str(tt)+"NM"+str(ii_angle)+FILENAME+"OBS")






