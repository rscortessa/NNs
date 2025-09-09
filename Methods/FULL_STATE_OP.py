import netket as nk
import numpy as np
import netket_fidelity as nkf
import Methods.class_WF as class_WF
import Methods.var_nk as var_nk
import matplotlib.pyplot as plt
import itertools
from netket.operator.spin import identity
import jax.numpy as jnp
from netket_fidelity.infidelity import InfidelityOperator
import flax.linen as nn
from flax import struct
import scipy
import optuna
import sys
import logging
from functools import partial

def objective(trial,model,L,hi,H,n_iter,holomorphic):    
    log = nk.logging.RuntimeLog()   
    learning_rate = trial.suggest_float("learning_rate",10**(-5),10**(-1),log=True)
    diag_shift = trial.suggest_float("diag_shift",10**(-5),10**(-1),log=True)
    optimizer = nk.optimizer.Sgd(learning_rate=learning_rate)
    preconditioner = nk.optimizer.SR(diag_shift=diag_shift, holomorphic=holomorphic)
    PSI = class_WF.FULL_WF(L,hi,preconditioner,optimizer,model,H)
    PSI.run(obs={},n_iter=50,log=log)
    for i in range(55,n_iter,10):
        PSI.run(obs={},n_iter=10,log=log)
        intermediate_score=log.data["Energy"]["Mean"][-1]
        trial.report(intermediate_score,i)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()
    score=log.data["Energy"]["Mean"][-1]
    return score

def objective_I(trial,model,psi,phi,L,hi,H,n_iter,holomorphic):    
    log = nk.logging.RuntimeLog()   
    learning_rate = trial.suggest_float("learning_rate",10**(-5),10**(-1),log=True)
    diag_shift = trial.suggest_float("diag_shift",10**(-5),10**(-1),log=True)
    cv = trial.suggest_float("cv",10**(-5),10**(-1),log=True)

    optimizer = nk.optimizer.Sgd(learning_rate=learning_rate)
    preconditioner = nk.optimizer.SR(diag_shift=diag_shift, holomorphic=holomorphic)

    te = nkf.driver.InfidelityOptimizer(psi, optimizer, variational_state=phi, preconditioner=preconditioner, cv_coeff=cv)

    log = nk.logging.RuntimeLog()
    te.run(n_iter=50,out=log,show_progress=False)

    for i in range(60,n_iter,10):
        te.run(n_iter=10,out=log,show_progress=False)
        intermediate_score=log.data["Infidelity"]["Mean"][-1]
        trial.report(intermediate_score,i)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()
    score=log.data["Infidelity"]["Mean"][-1]
    
    return score

