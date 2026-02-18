import netket as nk
import flax
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
from pathlib import Path
def objective(trial,model,L,hi,H,n_iter,holomorphic):    
    log = nk.logging.RuntimeLog()   
    learning_rate = trial.suggest_float("learning_rate",10**(-5),5*10**(-3),log=True)
    diag_shift = trial.suggest_float("diag_shift",10**(-5),10**(-1),log=True)
    optimizer = nk.optimizer.Sgd(learning_rate=learning_rate)
    preconditioner = nk.optimizer.SR(diag_shift=diag_shift, holomorphic=holomorphic)
    PSI = class_WF.FULL_WF(L,hi,preconditioner,optimizer,model,H)
    PSI.run(obs={},n_iter=5,log=log)
    for i in range(5,n_iter):
        PSI.run(obs={},n_iter=1,log=log)
        intermediate_score=log.data["Energy"]["Mean"][-1]
        trial.report(intermediate_score,i)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()
    score=log.data["Energy"]["Mean"][-1]
    return score

def objective_I(trial,model,psi,phi,L,hi,n_iter,holomorphic):    
    log = nk.logging.RuntimeLog()   
    learning_rate = trial.suggest_float("learning_rate",10**(-5),5*10**(-3),log=True)
    diag_shift = trial.suggest_float("diag_shift",10**(-5),10**(-1),log=True)
    cv = trial.suggest_float("cv",10**(-5),10**(-1),log=True)

    optimizer = nk.optimizer.Sgd(learning_rate=learning_rate)
    preconditioner = nk.optimizer.SR(diag_shift=diag_shift, holomorphic=holomorphic)
    phi.init_parameters()
    
    te = nkf.driver.InfidelityOptimizer(psi, optimizer, variational_state=phi, preconditioner=preconditioner, cv_coeff=cv)

    log = nk.logging.RuntimeLog()
    te.run(n_iter=50,out=log,show_progress=True)

    for i in range(60,n_iter,10):
        te.run(n_iter=10,out=log,show_progress=False)
        intermediate_score=log.data["Infidelity"]["Mean"][-1]
        trial.report(intermediate_score,i)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()
    score=log.data["Infidelity"]["Mean"][-1]

    return score

def file_exists(directory, filename):
    """
    Check if a file exists using pathlib.
    """
    return (Path(directory) / filename).exists()


def filenames(directory,FILENAME, tt=0):
    """
    Generate unique filenames for 'VAR' and 'OBS' files in a directory.

    Args:
        directory (str): Directory where files are stored.
        angle (float): The angle (used as part of the filename).
        FILENAME (str): Base filename prefix.
        tt (int, optional): Initial trial number. Defaults to 0.

    Returns:
        tuple[str, str]: filevar, fileobs (unique filenames)
    """
    while True:
        filevar = f"{tt}{FILENAME}VAR"
        fileobs = f"{tt}{FILENAME}OBS"

        # Check if either file already exists
        if not (file_exists(directory, filevar+".json") or file_exists(directory, fileobs+".json")):
            break  # both are free — good to use

        tt+= 1

    with open(directory+"/"+fileobs+".json","x") as f:
            print("File generated:",fileobs)
            
    return filevar, fileobs



def save_history_callback(step, log_data, driver, save_every, prefix):
    if step % save_every == 0:
        filename = f"{prefix}_step_{step}.mpack"
        with open(filename, 'wb') as f:
            f.write(flax.serialization.msgpack_serialize(driver.state.variables))
            print(f"Saved: {filename}") # Optional feedback
    return True
