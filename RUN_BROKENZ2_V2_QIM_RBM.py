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
NS=parameters[1]
G=parameters[2]
DG=0.01
NN=parameters[3]
NL=1
NR=parameters[4]
NSPCA=parameters[5]
Nangle=parameters[6]
NMEAN=parameters[7]

#BROKEN_Z2
DELTA=1
DELTA_S="D"+str(DELTA)


learning_rate=0.05
diag_shift=1
basis="BROKENZ2_V2_QIM"
modelo="RBM_COMPLEX"
broken_z2=False
compute_obs=False
compute_pca=False
if modelo=="RBM_COMPLEX":
    model=nk.models.RBM(alpha=NN,param_dtype=complex)
    sr = nk.optimizer.SR(diag_shift=diag_shift*0.1, holomorphic=True)
elif modelo=="RBM_REAL":
    model=nk.models.RBM(alpha=NN)
    sr = nk.optimizer.SR(diag_shift=diag_shift*0.1, holomorphic=False)


angle=0
dangle=1.0/(Nangle)
MASTER_DIR="RUN_"+basis+"_"+modelo+"NN"+str(NN)+"L"+str(L)+"G"+str(G)+DELTA_S+"NA"+str(Nangle)+"NSPCA"+str(NSPCA)+"DS"+str(diag_shift)
Nstates=2**L
eps=10**(-10)
angle=[dangle*i for i in range(Nangle+1)]

try:
    os.mkdir(MASTER_DIR)
except:
    print("DIRECTORY ALREADY CREATED")
OBS_FILENAME="NANGLE"+str(Nangle)+basis+"M3L"+str(L)+"W1"+"G"+str(G)+DELTA_S+"NS"+str(NS)+"NN"+str(NN)+"NL"+str(NL)+"NR"+str(NR)+"OBS"
SPCA_FILENAME="NANGLE"+str(Nangle)+basis+"M3L"+str(L)+"W1"+"G"+str(G)+DELTA_S+"NS"+str(NS)+"NN"+str(NN)+"NL"+str(NL)+"NR"+str(NR)+"SPCA"
VAR_FILENAME="NANGLE"+str(Nangle)+basis+"M3L"+str(L)+"W1"+"G"+str(G)+DELTA_S+"NS"+str(NS)+"NN"+str(NN)+"NL"+str(NL)+"NR"+str(NR)+"VAR"

sites_corr=[1,int(L/2),L-1]
sites_corr=[str(x) for x in sites_corr]
hi=nk.hilbert.Spin(s=1/2,N=L)

for tt in range(NMEAN):
    for ii in range(len(angle)):
        
        H=rotated_BROKEN_Z2IsingModel(0.0,G*DG,L,hi,angle[ii]*DG*G)
        alpha=1
        learning_rate=0.05
        g = nk.graph.Hypercube(length=L, n_dim=1, pbc=False)
        #sampler = nk.sampler.MetropolisExchange(hilbert=hi,graph=g)
        sampler=nk.sampler.MetropolisHamiltonian(hi, hamiltonian=H)
        #sampler=nk.sampler.MetropolisLocal(hi,n_chains=128,sweep_size=50)
        vstate=nk.vqs.MCState(sampler,model,n_samples=NS)
        optimizer=nk.optimizer.Sgd(learning_rate=learning_rate)
        PSI=class_WF.WF(L,hi,sampler,sr,model,H,NS)
        #THE OUT LOGS ARE CREATED
        log = nk.logging.RuntimeLog()
        log2 = nk.logging.RuntimeLog()
        log3 = nk.logging.RuntimeLog()
    #OBSERVABLES INIT.
        obs={}
        if compute_obs:
            
            obs["P"]=parity_IsingModel(angle[ii],L,hi)
            obs["M"]=rotated_m(angle[ii],L,hi)
    
            for jj in sites_corr:
                obs["CZ0"+jj]=Sz0Szj(angle[ii],L,hi,int(jj))
                obs["CX0"+jj]=Sx0Sxj(angle[ii],L,hi,int(jj))
        

        NR_eff=int(NR/NSPCA)
        for kk in range(NSPCA):    
        #FIRST WE COMPUTE THE PARAMETERS
            if compute_pca:
                PSI.compute_PCA(10**(-8),i=kk,log=log3,broken_z2=broken_z2)
            PSI.save_params(kk,log2)
            #RUNNING FOR NR_EFF ITERATIONS
            PSI.run(obs=obs,n_iter=NR_eff,log=log)
            print("Nrun",tt,"angle",ii,"iter",kk)

        #LAST CALCULATION:
        if compute_pca:
            PSI.compute_PCA(10**(-8),i=NSPCA-1,log=log3,broken_z2=broken_z2)    
            log3.serialize(MASTER_DIR+"/"+str(tt)+"NM"+str(ii)+SPCA_FILENAME)

        #SAVE LAST PARAMETERS:    
        PSI.save_params(NSPCA-1,log2)
        log2.serialize(MASTER_DIR+"/"+str(tt)+"NM"+str(ii)+VAR_FILENAME)
        log.serialize(MASTER_DIR+"/"+str(tt)+"NM"+str(ii)+OBS_FILENAME)
        








