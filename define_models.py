import netket as nk
import argparse
import os
from config import params
from Methods.class_WF import rotated_IsingModel,rotated_BROKEN_Z2IsingModel,rotated_CIMModel_2,bi_ladder_rotated_IsingModel
import Methods.var_nk as var_nk
import numpy as np
import jax.numpy as jnp
hams={}
hams["QIM"]=rotated_IsingModel


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="QIM")
args, unknown = parser.parse_known_args()
params_ = vars(args)

hi=nk.hilbert.Spin(s=1/2,N=params["L"]*params["W"],inverted_ordering=True)
if params_["model"] == "QIM":
    ham=rotated_IsingModel(params["angle"]*np.pi/(2*params["Nangle"]),params["g"]*params["DG"],params["L"],hi,pbc=params["pbc"])
else:
    print("The Hamiltonian does not exist")
    
if params["architecture"] == "CNN_COMPLEX":
    model=var_nk.Deep1DCNN(kernel_size=params["kernel_size"],layer_features=tuple(params["features"]),padding=params["padding"])
elif params["architecture"] == "CNN_REAL":
    model=var_nk.Deep1DCNN(kernel_size=params["kernel_size"],layer_features=tuple(params["features"]),padding=params["padding"],param_dtype=jnp.float64)
elif params["architecture"] == "parity_CNN_COMPLEX":
    model=var_nk.parity_Deep1DCNN(kernel_size=params["kernel_size"],layer_features=tuple(params["features"]),padding=params["padding"])
elif params["architecture"] == "RBM_COMPLEX":
    model=nk.models.RBM(alpha=params["NN"],param_dtype=complex)
elif params["architecture"] == "spin_dependent_T":
    model=var_nk.SpinDependent_TransformerWaveFunction(n_heads=params["n_heads"],head_dim=params["head_dim"],n_patches=params["n_patches"])
elif params["architecture"] == "factored_attention_T":
    model=var_nk.Factored_Attention_TransformerWaveFunction(n_heads=params["n_heads"],head_dim=params["head_dim"],n_patches=params["n_patches"])
    
else:
    print("architecture not found")
