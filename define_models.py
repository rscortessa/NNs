import netket as nk
import argparse
import os
from config import params
from Methods.class_WF import rotated_IsingModel,rotated_BROKEN_Z2IsingModel,rotated_CIMModel_2,bi_ladder_rotated_IsingModel
import Methods.var_nk as var_nk
import numpy as np
import jax.numpy as jnp
from Methods.FULL_STATE_OP import objective,objective_I

hams={}
hams["QIM"]=rotated_IsingModel


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="QIM")
args, unknown = parser.parse_known_args()
params_ = vars(args)

hi=nk.hilbert.Spin(s=1/2,N=params["L"]*params["W"],inverted_ordering=True)
if params_["model"] == "QIM":
    ham=rotated_IsingModel(params["angle"]*np.pi/(2*params["Nangle"]),params["g"]*params["DG"],params["L"],hi,pbc=params["pbc"])
    cost_func=objective
elif params_["model"] == "QUENCH_QIM":
    parser.add_argument("--dt",type=int,default=0)
    parser.add_argument("--nt",type=int,default=1)
    args_2 = parser.parse_known_args()
    params_2 = vars(args_2[0])
    params_["model"]+="dt"+str(params_2["dt"])+"nt"+str(params_2["nt"])
    
    psi_0=np.ones(2**params["L"],dtype=complex)/2**(params["L"]/2.0)
    ham=rotated_IsingModel(params["angle"]*np.pi/(2*params["Nangle"]),params["g"]*params["DG"],params["L"],hi,pbc=params["pbc"])
    E,WFS = np.linalg.eigh(ham.to_dense())
    psi_t = np.sum(WFS*(psi_0@WFS*np.exp(-1j*E*params_2["dt"])),axis=-1)
    psi_t[np.abs(psi_t)<10**(-10)] = 0.0
    psi_t = np.log(psi_t)
    psi_t[psi_t!=psi_t]=-np.infty
    print(psi_t[psi_t!=psi_t])
    Exact_GS=var_nk.EWF(L=params["L"],eig_vec=tuple(psi_t))
    ham = nk.vqs.FullSumState(hi, model=Exact_GS)
    cost_func=objective_I
    
else:
    print("The Hamiltonian does not exist")
    
if params["architecture"] == "CNN_COMPLEX":
    model=var_nk.Deep1DCNN(kernel_size=params["kernel_size"],layer_features=tuple(params["features"]),padding=params["padding"])
elif params["architecture"] == "CNN_REAL":
    model=var_nk.Deep1DCNN(kernel_size=params["kernel_size"],layer_features=tuple(params["features"]),padding=params["padding"],param_dtype=jnp.float64)
elif params["architecture"] == "parity_CNN_COMPLEX":
    model=var_nk.parity_Deep1DCNN(kernel_size=params["kernel_size"],layer_features=tuple(params["features"]),padding=params["padding"])
elif params["architecture"] == "parity_CNN_REAL":
    model=var_nk.parity_Deep1DCNN(kernel_size=params["kernel_size"],layer_features=tuple(params["features"]),padding=params["padding"],param_dtype=jnp.float64)
elif params["architecture"] == "RBM_COMPLEX":
    model=nk.models.RBM(alpha=params["NN"],param_dtype=complex)
elif params["architecture"] == "spin_dependent_T":
    model=var_nk.SpinDependent_TransformerWaveFunction(n_heads=params["n_heads"],head_dim=params["head_dim"],n_patches=params["n_patches"])
elif params["architecture"] == "factored_attention_T":
    model=var_nk.Factored_Attention_TransformerWaveFunction(n_heads=params["n_heads"],head_dim=params["head_dim"],n_patches=params["n_patches"])
    
else:
    print("architecture not found")

models_name=params_["model"]
