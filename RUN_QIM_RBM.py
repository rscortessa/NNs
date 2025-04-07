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

from Methods.class_WF import rotated_sigmax, rotated_sigmaz,isigmay,rotated_IsingModel
from Methods.class_WF import rotated_XYZModel, parity_Matrix, parity_IsingModel, Sz0Szj, Sx0Sxj, to_array


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

learning_rate=0.05
basis="QIM"
modelo="RBM_REAL"

if modelo=="RBM_COMPLEX":
    model=nk.models.RBM(alpha=NN,param_dtype=complex)
    sr = nk.optimizer.SR(diag_shift=0.1, holomorphic=True)
elif modelo=="RBM_REAL":
    model=nk.models.RBM(alpha=NN)
    sr = nk.optimizer.SR(diag_shift=0.1, holomorphic=False)

angle=0
Nangle=4
dangle=np.pi/(2*Nangle)
NSPCA=parameters[5]
MASTER_DIR="RUN_QIM_"+modelo+"NN"+str(NN)+"L"+str(L)+"G"+str(G)
Nstates=2**L
eps=10**(-10)
angle=[dangle*i for i in range(Nangle+1)]
num_states=[i for i in range(2**L)]
try:
    os.mkdir(MASTER_DIR)
except:
    print("DIRECTORY ALREADY CREATED")
OBS_FILENAME="NANGLE"+str(Nangle)+basis+"M3L"+str(L)+"W1"+"G"+str(G)+"NS"+str(NS)+"NN"+str(NN)+"NL"+str(NL)+"NR"+str(NR)+"OBS"
SPCA_FILENAME="NANGLE"+str(Nangle)+basis+"M3L"+str(L)+"W1"+"G"+str(G)+"NS"+str(NS)+"NN"+str(NN)+"NL"+str(NL)+"NR"+str(NR)+"SPCA"
VAR_FILENAME="NANGLE"+str(Nangle)+basis+"M3L"+str(L)+"W1"+"G"+str(G)+"NS"+str(NS)+"NN"+str(NN)+"NL"+str(NL)+"NR"+str(NR)+"VAR"

#S_PCA_TEO=[0.0 for i in angle]
#PSI_TEO=[None for i in angle]
#hi=nk.hilbert.Spin(s=1/2,N=L)
#states=hi.all_states()
#for theta in range(len(angle)):
#    H=rotated_IsingModel(angle[theta],G*DG,L,hi)
#    eig_vals_other,eig_vecs_other=np.linalg.eigh(H.to_dense())
#    PSI_TEO[theta]=eig_vecs_other[:,0]
#    A=np.random.choice(num_states,size=1000,p=eig_vecs_other[:,0]**2)
#    B=np.array([states[a] for a in A])
#    S_PCA_TEO[theta]=class_WF.S_PCA(B,10**(-10),exvar=False)
    
#sisj_z=[ eig_vecs_other[:,0].T@Sz0Szj(0.0,L,hi,j).to_dense()@eig_vecs_other[:,0] for j in [1,int(L/2),L-1]]
#sisj_x=[ eig_vecs_other[:,0].T@Sx0Sxj(0.0,L,hi,j).to_dense()@eig_vecs_other[:,0] for j in [1,int(L/2),L-1]]

sites_corr=[1,int(L/2),L-1]
sites_corr=[str(x) for x in sites_corr]

for ii in range(len(angle)):
    if angle[ii]==np.pi/(2.0):
        hi=nk.hilbert.Spin(s=1/2,N=L,constraint=class_WF.ParityConstraint())
    else:
        hi=nk.hilbert.Spin(s=1/2,N=L,constraint=class_WF.ParityConstraint())

    H=rotated_IsingModel(angle[ii],G*DG,L,hi)

    alpha=1
    learning_rate=0.05
    g = nk.graph.Hypercube(length=L, n_dim=1, pbc=False)
    #sampler = nk.sampler.MetropolisExchange(hilbert=hi,graph=g)
    sampler=nk.sampler.MetropolisHamiltonian(hi, hamiltonian=H)
    #sampler=nk.sampler.MetropolisLocal(hi,n_chains=128,sweep_size=50)
    vstate=nk.vqs.MCState(sampler,model,n_samples=NS)
    optimizer=nk.optimizer.Sgd(learning_rate=learning_rate)
    PSI=class_WF.WF(L,hi,sampler,sr,model,H,NS)
    
    log = nk.logging.RuntimeLog()
    log2 = nk.logging.RuntimeLog()
    log3 = nk.logging.RuntimeLog()
    #OBSERVABLES INIT.
    obs={}
    obs["P"]=parity_IsingModel(angle[ii],L,hi)
    for jj in sites_corr:
        obs["CZ0"+jj]=Sz0Szj(angle[ii],L,hi,int(jj))
        obs["CX0"+jj]=Sx0Sxj(angle[ii],L,hi,int(jj))

    NR_eff=int(NR/NSPCA)
    for kk in range(NSPCA):
        PSI.run(obs=obs,n_iter=NR_eff,log=log)
        PSI.save_params(kk,log2)
        if ii!=0:
            broken_z2=False
        else:
            broken_z2=True
            
        PSI.compute_PCA(10**(-8),i=kk,log=log3,broken_z2=broken_z2)
        
    log2.serialize(MASTER_DIR+"/"+str(ii)+VAR_FILENAME)
    log.serialize(MASTER_DIR+"/"+str(ii)+OBS_FILENAME)
    log3.serialize(MASTER_DIR+"/"+str(ii)+SPCA_FILENAME)









