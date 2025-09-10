#!/usr/bin/env python
# coding: utf-8

import netket as nk
import numpy as np
import jax.numpy as jnp
from netket.hilbert import constraint
import equinox as eqx  # Used for struct field handling
from Methods.class_WF import rotated_IsingModel
from Methods.FULL_STATE_OP import objective
import optuna
import logging
import os
import sys
from functools import reduce,partial
import json

# In[57]:


#Define the Hamiltonian and the game rules:
DG=0.01
L=10
G=150
angle=0.0
hi=nk.hilbert.Spin(s=1/2,N=L,inverted_ordering=True)
H=rotated_IsingModel(angle,G*DG,L,hi)
eig_vals,eig_vecs=np.linalg.eigh(H.to_dense())
NMEAN=10
NSPCA=10
NRUN=1000
Nangle=12
NL=1
#Define loop 
NN=np.linspace(0.1,3.0,30)


# In[59]:


#Define the Directories

MASTER_DIR="ENERGY-Z-BASIS"
basis="QIM"
add=""
architecture="RBM_COMPLEX"

if not os.path.isdir(MASTER_DIR):
    print(os.path.isdir(MASTER_DIR))
    os.mkdir(MASTER_DIR)
    print(f"Directory '{MASTER_DIR}' not found previously but created successfully.")

for nn in NN:
    SLAVE_DIR="FULL_STATE_RUN_"+basis+"_"+architecture+"NN"+str(round(nn,1))+"L"+str(L)+"G"+str(G)+"NSPCA"+str(NSPCA)+add
    try:
        os.mkdir(MASTER_DIR+"/"+SLAVE_DIR)
    except FileExistsError:
        print(f"Directory "+MASTER_DIR+"/"+SLAVE_DIR+" already exists.")

best_params=[[] for nn in range(len(NN))]
for nn in range(len(NN)):
    SLAVE_DIR="FULL_STATE_RUN_"+basis+"_"+architecture+"NN"+str(round(NN[nn],1))+"L"+str(L)+"G"+str(G)+"NSPCA"+str(NSPCA)+add
    FILENAME=basis+"M3L"+str(L)+"W1"+"G"+str(G)+"NN"+str(round(nn,1))+"NL"+str(NL)+"NR"+str(NRUN)
    with open(MASTER_DIR+"/"+SLAVE_DIR+"/"+FILENAME+"HYP.json","r") as f:
        best_params[nn]=json.load(f)
    


# In[ ]:


E=np.zeros((30,Nmean))
for nn in range(len(NN)):

    SLAVE_DIR="FULL_STATE_RUN_"+basis+"_"+architecture+"NN"+str(round(nn,1))+"L"+str(L)+"G"+str(G)+"NSPCA"+str(NSPCA)+add
    FILENAME=basis+"M3L"+str(L)+"W1"+"G"+str(G)+"NN"+str(round(nn,1))+"NL"+str(NL)+"NR"+str(NRUN)
    optimizer=optimizer=nk.optimizer.Sgd(learning_rate=best_params[nn]["learning_rate"]["value"][0])
    sr=sr = nk.optimizer.SR(diag_shift=best_params[nn]["diag_shift"]["value"][0], holomorphic=holomorphic)
    
    PSI = class_WF.FULL_WF(L,hi,sr,optimizer,model,H)
    NR_eff=int(NRUN/NSPCA)
    
    for tt in range(NMEAN):

        #RESTART THE NETWORK                                                                                                                                                                                                              

        vstate=PSI.user_state
        vstate.init_parameters()
        PSI.change_state(vstate)

        #THE OUT LOGS ARE CREATED                                                                                                                                                                                                         
        log = nk.logging.RuntimeLog()
        log2 = nk.logging.RuntimeLog()

        for kk in range(NSPCA):

            #FIRST WE COMPUTE THE PARAMETERS                                                                                                                                                                                              
            PSI.save_params(kk,log2)
            #RUNNING FOR NR_EFF ITERATIONS                                                                                                                                                                                                
            PSI.run(obs={},n_iter=NR_eff,log=log)
        
        #LAST CALCULATION                                                                                                                                                                                                                 
        PSI.save_params(kk,log2)
        log2.serialize(MASTER_DIR+"/"+SLAVE_DIR+"/"+str(tt)+"NM"+str(nn)+FILENAME+"VAR")
        log.serialize(MASTER_DIR+"/"+SLAVE_DIR+"/"+str(tt)+"NM"+str(nn)+FILENAME+"OBS")




