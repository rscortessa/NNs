import json, sys, numpy as np, optuna, netket as nk
from functools import partial
from Methods.var_nk import EWF
from Methods.FULL_STATE_OP import objective_I
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
import json
from Methods.class_WF import rotated_sigmax, rotated_sigmaz,isigmay,rotated_IsingModel,rotated_BROKEN_Z2IsingModel,rotated_CIMModel_2,bi_ladder_rotated_IsingModel
from Methods.class_WF import rotated_XYZModel, parity_Matrix, parity_IsingModel, Sz0Szj, Sx0Sxj, to_array, rotated_m
from Methods.FULL_STATE_OP import objective
from netket.graph import Graph
def save_params(vmc,i,log_var):
        log_var(i,vmc.state.variables)

parameters=sys.argv
n_par=len(parameters)
parameters=[float(parameters[x]) for x in range(1,n_par)]

L = int(parameters[0])
W=1
DG=0.01
NN = parameters[1]
NL = 2
NR = int(parameters[2])
NSPCA = int(parameters[3])
Nangle = int(parameters[4])
NMEAN = int(parameters[5])
angle = int(parameters[6])
g = int(parameters[7])
basis="QIM"
add=""

pbc=False

architecture = "CNN_COMPLEX"

add=""
if pbc:
    add+="PBC"
    
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
    
#Creation of the folder
MASTER_DIR="ENERGY"
SLAVE_DIR="FULL_STATE_RUN_"+basis+"_"+architecture+"NN"+str(NN)+"L"+str(L)+"G"+str(g)+"NA"+str(Nangle)+"NSPCA"+str(NSPCA)+add
study_name = f"MODEL{basis}ARCH{architecture}NN{NN}L{L}G{g}NA{Nangle}ANGLE{angle}BC{add}"
storage = "sqlite:///"+MASTER_DIR+"/"+SLAVE_DIR+"/"+study_name+"optuna_study.db"
FILENAME="NANGLE"+str(Nangle)+basis+"M3L"+str(L)+"W1"+"G"+str(g)+"NN"+str(NN)+"NL"+str(NL)+"NR"+str(NR)



# Hilbert space generation in Netket
hi=nk.hilbert.Spin(s=1/2,N=L*W,inverted_ordering=True)
n_iter=200

if architecture == "RBM_COMPLEX":
    model = nk.models.RBM(alpha=NN, param_dtype=complex)
    holomorphic = True
else:
    model = nk.models.RBM(alpha=NN)
    holomorphic = False

if architecture == "GCNN_COMPLEX":

    #Linear Chain
    edges = [(i,i+1) for i in range(L-1)]
    chain = Graph(edges)
    symmetries = chain.automorphisms()

    zz = int (L*NN)
    if zz == 0:
        print("PROBLEM")
        exit()
    else:
        feature_dims = tuple( [zz for x in range(int(NL))] )
        model = nk.models.GCNN(symmetries=symmetries,layers=NL, features=feature_dims, param_dtype=complex)

if architecture == "CNN_COMPLEX":
    zz= int (L*NN)
    if zz == 0:
        print("PROBLEM")
        exit()
    else:
        feature_dims = tuple( [zz for x in range(int(NL))] )
        model = var_nk.Deep1DCNN(layer_features=feature_dims)

        
phi = nk.vqs.FullSumState(hi, model=model)


if basis == "QIM":
    H=rotated_IsingModel(angle*np.pi/(2*Nangle),g*DG,L,hi,pbc=pbc)
if basis == "CIM_2":
    H=rotated_CIMModel_2(angle*np.pi/(2*Nangle),g*DG,L,hi,pbc=pbc)
if basis == "BROKENZ2_QIM":
    H=rotated_BROKEN_Z2IsingModel(angle*np.pi/(2*Nangle),g*DG,L,hi,hpar,pbc=pbc)
if basis == "BI_QIM":
    H=bi_ladder_rotated_IsingModel(angle*np.pi/(2*Nangle),g*DG,g*DG,1.0,1.0,L,hi,pbc=pbc)


study = optuna.load_study(study_name=study_name, storage=storage)
print(f"🔁 Loaded study '{study_name}' for angle={angle}")

objective_final = partial(
    objective,
    model=model,
    L=L*W,
    hi=hi,
    H=H,
    n_iter=n_iter,
    holomorphic=holomorphic,
)

# -------------------------------
# ---- Run trials ----
# -------------------------------
print(f"🚀 Running Optuna trials for angle={angle}")
study.optimize(objective_final, n_trials=10)
print("✅ Finished trials for this angle.")


