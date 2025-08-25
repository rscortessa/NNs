#!/usr/bin/env python
# coding: utf-8

# In[2]:

import sys
import netket as nk
import numpy as np
import netket_fidelity as nkf
import Methods.var_nk as var_nk
import matplotlib.pyplot as plt
import itertools
from netket.operator.spin import identity
from Methods.class_WF import rotated_IsingModel,rotated_BROKEN_Z2IsingModel
import jax.numpy as jnp
from netket_fidelity.infidelity import InfidelityOperator
import flax.linen as nn
from flax import struct
import os
import json
#FUNCTIONS to implement the exact WF

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
    
# Create the Hilbert space and the variational states |ψ⟩ and |ϕ⟩

MASTER_DIR="FIDELITY"
if os.path.isdir(MASTER_DIR+" "):
    os.mkdir(MASTER_DIR)
    print(f"Directory '{MASTER_DIR}' not found previously but created successfully.")

    
else:
    print("DIRECTORY ALREADY CREATED")

DG = 0.01
eta = 0.001
learning_rate = 500
diag_shift = 10
Nangle = 12
angle = [i for i in range(Nangle)]

parameters=sys.argv
n_par=len(parameters)
parameters = [int(parameters[x]) for x in range(1,n_par)]
L = parameters[0]
NN = parameters[1]
NL = 1
NR = parameters[2]
NSPCA = parameters[3]
Nangle = parameters[4]
NMEAN = parameters[5]

print(n_par,parameters)
G = [parameters[x] for x in range(6,n_par-1)]
pbc=False

basis = "BROKENZ2_QIM"

if basis == "QIM":
    adder =""
elif basis == "BROKENZ2_QIM":
    hpar=0.01
    adder = "HPAR"+str(round(hpar,2))
else:
    print("MODEL NOT FOUND")
    exit()
    

architecture = "MODIFIED_RBM_COMPLEX"

if architecture=="RBM_COMPLEX":
    model=nk.models.RBM(alpha=NN,param_dtype=complex)
    sr = nk.optimizer.SR(diag_shift=diag_shift*eta, holomorphic=True)
elif architecture=="RBM_REAL":
    model=nk.models.RBM(alpha=NN)
    sr = nk.optimizer.SR(diag_shift=diag_shift*eta, holomorphic=False)
elif architecture=="MODIFIED_RBM_REAL" or architecture=="MODIFIED_RBM_COMPLEX":
    print("DEFINED IN ITERATIONS")
else:
    print("MODEL IS NOT DEFINED")
    exit()
    

NREP = [i for i in range(NR)]
hi = nk.hilbert.Spin(0.5, L,inverted_ordering = True)
param_lists = [angle,G,NREP]
flat_dict = {}

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
    
    return PROB

def vstate_par(data_var,it):
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

    hi=nk.hilbert.Spin(s=1/2,N=L,inverted_ordering = True)
    model=nk.models.RBM(alpha=NN,param_dtype=complex)
    vstate = nk.vqs.FullSumState(hi, model=model,variables=new_parameters)
    return vstate


for g in G:
    SLAVE_DIR="FULL_STATE_RUN_"+basis+"_"+architecture+"NN"+str(NN)+"L"+str(L)+"G"+str(g)+"NA"+str(Nangle)+"NSPCA"+str(NSPCA)+"SHIFT"+str(diag_shift)+adder
    try:
        os.mkdir(MASTER_DIR+"/"+SLAVE_DIR)
    except FileExistsError:
        print(f"Directory "+MASTER_DIR+"/"+SLAVE_DIR+" already exists.")
        


for combination in itertools.product(*param_lists):
    
    OBS_FILENAME=str(combination[0])+"NANGLE"+str(Nangle)+basis+"M3L"+str(L)+"W1"+"G"+str(combination[1])+"NN"+str(NN)+"NL"+str(NL)+"NR"+str(NR)
    theta=np.pi/(2.0*Nangle)*combination[0]
    h=combination[1]*DG
    if basis == "QIM":
        H=rotated_IsingModel(theta,h,L,hi,pbc=pbc)
    elif basis == "BROKENZ2_QIM":
        H=rotated_BROKEN_Z2IsingModel(theta,h,L,hi,hpar,pbc=pbc)
    
    eig_vals,eig_vecs=np.linalg.eigh(H.to_dense())
    
    E_gs=eig_vals[0]
    
    if eig_vecs[:,0][0]<0:
        GS=(-1.0)*eig_vecs[:,0]
    else:
        GS=eig_vecs[:,0]
    GS[np.abs(GS)<10**(-10)]=0.0
    GS=np.log(GS)
    GS[GS!=GS]=-np.infty

    # EXACT G.S
    
    Exact_GS=EWF(L=L,eig_vec=tuple(GS))
    SIGNS=(GS<0)*(1j*np.pi)+(GS>=0)*(0.0)
    if architecture=="MODIFIED_RBM_COMPLEX":
        model=var_nk.MODIFIED_RBM(phases=tuple(SIGNS),alpha=NN,param_dtype=complex,L=L,hi=hi)
        sr = nk.optimizer.SR(diag_shift=diag_shift*eta, holomorphic=True)
    elif architecture=="MODIFIED_RBM_REAL":
        model=model=var_nk.MODIFIED_RBM(phases=tuple(SIGNS),alpha=NN,L=L,hi=hi)
        sr = nk.optimizer.SR(diag_shift=diag_shift*eta, holomorphic=False)

    
    phi = nk.vqs.FullSumState(hi, model=model)

    # DEFINE THE OBSERVABLES
    
    obs = {}
    obs["E"] = H
    log = nk.logging.RuntimeLog()
    log_var = nk.logging.RuntimeLog()
    psi = nk.vqs.FullSumState(hi, model=Exact_GS)
    
    op = nk.optimizer.Sgd(learning_rate=learning_rate*eta)
    te = nkf.driver.InfidelityOptimizer(psi, op, variational_state=phi, preconditioner=sr, cv_coeff=0.0)

    # Run the driver
    for steps in range(NSPCA):
        N_run=int(NR/NSPCA)
        te.run(n_iter=N_run,obs=obs,out=log,show_progress=False)
        save_params(te,steps,log_var)
    flat_dict[combination]=log.data
    log.serialize(MASTER_DIR+"/"+SLAVE_DIR+"/"+str(combination[2])+"NM"+OBS_FILENAME+"OBS")
    log_var.serialize(MASTER_DIR+"/"+SLAVE_DIR+"/"+str(combination[2])+"NM"+OBS_FILENAME+"VAR")

