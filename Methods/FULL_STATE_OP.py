import netket as nk
import flax
import re
import numpy as np
import netket_fidelity as nkf
import Methods.class_WF as class_WF
import Methods.var_nk as var_nk
import matplotlib.pyplot as plt
import itertools
import os
import re
from netket.operator.spin import identity
import jax.numpy as jnp
from netket_fidelity.infidelity import InfidelityOperator
import flax.linen as nn
from flax import struct
import scipy
import optuna
import optax
import sys
import logging
from functools import partial
from pathlib import Path

#def objective(trial,model,L,hi,H,n_iter,holomorphic,clipping=False):    
#    log = nk.logging.RuntimeLog()   
#    learning_rate = trial.suggest_float("learning_rate",10**(-5),5*10**(-3),log=True)
#    diag_shift = trial.suggest_float("diag_shift",10**(-5),10**(-1),log=True)
#    if clipping == True:
#        max_step_size = trial.suggest_float("max_step_size", 1e-3, 0.5, log=True)
#        optimizer = optax.chain(
#        optax.clip_by_global_norm(max_step_size),
#        optax.sgd(learning_rate=learning_rate) 
#        )
#    else:
#        optimizer = nk.optimizer.Sgd(learning_rate=learning_rate)
        
#    preconditioner = nk.optimizer.SR(diag_shift=diag_shift, holomorphic=holomorphic)
#    PSI = class_WF.FULL_WF(L,hi,preconditioner,optimizer,model,H)
#    PSI.run(obs={},n_iter=5,log=log)
#    for i in range(5,n_iter):
#        PSI.run(obs={},n_iter=1,log=log)
#        intermediate_score=log.data["Energy"]["Mean"][-1]
#        trial.report(intermediate_score,i)
#        # Handle pruning based on the intermediate value.
#        if trial.should_prune():
#            raise optuna.TrialPruned()
#    score=log.data["Energy"]["Mean"][-1]
#    return score


def objective(trial, model, L, hi, H, n_iter, holomorphic, clipping=False,num_reports=12):
    log = nk.logging.RuntimeLog()
    
    # 1. Optuna suggests the INITIAL max learning rate
    initial_lr = trial.suggest_float("learning_rate", 10**(-5), 5*10**(-3), log=True)
    initial_ds = trial.suggest_float("diag_shift", 10**(-3), 10**(-1), log=True)
    
    # 2. Create the Optax Learning Rate Schedule
    # alpha=0.01 means the final learning rate will be 1% of the initial_lr.
    # You could also let Optuna optimize `alpha` if you wish!
    lr_schedule = optax.cosine_decay_schedule(
        init_value=initial_lr,
        decay_steps=n_iter,
        alpha=0.01 
    )
    min_ds=1e-5
    
    ds_schedule = optax.cosine_decay_schedule(
        init_value=initial_ds,
        decay_steps=n_iter,
        alpha=min_ds/initial_ds 
    )
    

    # 3. Pass the schedule object instead of a float to the optimizer
    if clipping:
        max_step_size = trial.suggest_float("max_step_size", 1e-3, 0.5, log=True)
        optimizer = optax.chain(
            optax.clip_by_global_norm(max_step_size),
            optax.sgd(learning_rate=lr_schedule) 
        )
    else:
        # We use optax.sgd here directly instead of nk.optimizer.Sgd 
        # because optax seamlessly handles the schedule objects.
        optimizer = optax.sgd(learning_rate=lr_schedule)

    preconditioner = nk.optimizer.SR(diag_shift=ds_schedule, holomorphic=holomorphic)

    # Assuming FULL_WF is your custom wrapper that instantiates a NetKet driver (like VMC)
    PSI = class_WF.FULL_WF(L, hi, preconditioner, optimizer, model, H)
    
    checkpoints = np.unique(np.geomspace(5, n_iter, num=num_reports, dtype=int))    
    # 2. Calculate the number of steps to run per chunk
    # e.g., [5, 4, 6, 11, 20, 36, ...]
    steps_to_run = np.diff(checkpoints, prepend=0)

    # 3. Run the optimization in chunks
    for chunk_size, current_step in zip(steps_to_run, checkpoints):
        
        # Run the chunk. NetKet handles this entirely internally, very fast!
        PSI.run(obs={}, n_iter=int(chunk_size), log=log)
        
        # Extract the intermediate score
        intermediate_score = log.data["Energy"]["Mean"][-1]
        
        # Report the score to Optuna AT the specific step
        trial.report(intermediate_score, int(current_step))
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.TrialPruned()
            
    score = log.data["Energy"]["Mean"][-1]
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


def filenames(directory, NR, FILENAME, tt=0):
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
        filevar = f"{tt}{FILENAME}VAR_step_{NR}.mpack"
        fileobs = f"{tt}{FILENAME}OBS"

        # Check if either file already exists
        if not (file_exists(directory, filevar) or file_exists(directory, fileobs+".json")):
            break  # both are free — good to use

        tt+= 1

    with open(directory+"/"+fileobs+".json","x") as f:
            print("File generated:",fileobs)

    filevar = f"{tt}{FILENAME}VAR"
    return filevar, fileobs


def find_minimum_max_nr_file(directory, fixed_filename):
    """                                                                                                                                                                                                                                                                                                                                                                                            
    Scans a directory for files matching {tt}{FILENAME}VAR_step_{NR}.mpack
    and returns the file with the minimum of the maximum NRs across all tt's 
    where the corresponding log file does not yet exist.                                                                                                                                                                                                                                                                         """
    # 1. Compile the Regular Expression
    pattern_str = r"^(\d+)" + re.escape(fixed_filename) + r"VAR_step_(\d+)\.mpack$"
    pattern = re.compile(pattern_str)

    # Dictionary to store the maximum NR for each tt
    max_nr_per_tt = {}
    # 2. Scan the directory and match files
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            # Extract tt and NR and convert them to integers for proper comparison
            tt = int(match.group(1))
            nr = int(match.group(2))
            # Update the dictionary with the maximum NR found for this tt                                                                                                                                                                                                                                                   
            if tt not in max_nr_per_tt or nr > max_nr_per_tt[tt]:
                max_nr_per_tt[tt] = nr

    # Handle the case where no files match the pattern                                                                                                                                                                                                                                                                                                                                                                                    
    if not max_nr_per_tt:
        print("No matching files found in the directory.")
        return None, None, None

    # 3. Create a list of all possible tt's and their max NR                                                                                                                                                                                                                                                                                                                                                                              
    tt_max_nr_list = list(max_nr_per_tt.items())
    print(f"List of (tt, max_NR) pairs found: {tt_max_nr_list}")

    # 4. Sort the pairs by max_NR in ascending order
    # (x[1] refers to the max_nr in the (tt, max_nr) tuple)
    tt_max_nr_list.sort(key=lambda x: x[1])

    chosen_tt = None
    chosen_nr = None

    # 5. Check if a log file is already created, moving from lowest max_NR to highest
    for tt, nr in tt_max_nr_list:
        log_filename = f"{tt}{fixed_filename}OBSRC_{nr}.log"
        log_filepath = os.path.join(directory, log_filename)
        
        # If the log file is NOT created, we found our target
        if not os.path.exists(log_filepath):
            chosen_tt = tt
            chosen_nr = nr
            break
            
    # Handle the case where ALL files already have a log file created
    if chosen_tt is None:
        print("All matching files already have a corresponding log file created.")
        return None, None, None

    # 6. Reconstruct the exact filename                                                                                                                                                                                                                                                                                                                                                                                                   
    filevar = f"{chosen_tt}{fixed_filename}VAR_step_{chosen_nr}.mpack"

    return filevar, chosen_nr, chosen_tt


def find_minimum_max_nr_file(directory, fixed_filename):
    """                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    Scans a directory for files matching {tt}{FILENAME}VAR_step_{NR}.mpack                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    and returns the log object, the file with the minimum of the maximum NRs, 
    the chosen NR, and the chosen tt where the log file does not yet exist.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    """
    # 1. Compile the Regular Expression                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    pattern_str = r"^(\d+)" + re.escape(fixed_filename) + r"VAR_step_(\d+)\.mpack$"
    pattern = re.compile(pattern_str)

    max_nr_per_tt = {}

    # 2. Scan the directory and match files                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
            tt = int(match.group(1))
            nr = int(match.group(2))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
            if tt not in max_nr_per_tt or nr > max_nr_per_tt[tt]:
                max_nr_per_tt[tt] = nr

    if not max_nr_per_tt:
        print("No matching files found in the directory.")
        return None, None, None, None

    # 3. Create a list of all possible tt's and their max NR                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    tt_max_nr_list = list(max_nr_per_tt.items())

    # 4. Sort the pairs by max_NR in ascending order
    tt_max_nr_list.sort(key=lambda x: x[1])

    chosen_tt = None
    chosen_nr = None

    # 5. Check if a log file is already created, moving from lowest max_NR to highest
    for tt, nr in tt_max_nr_list:
        log_filename = f"{tt}{fixed_filename}OBSRC_{nr}.log"
        log_filepath = os.path.join(directory, log_filename)
        
        if not os.path.exists(log_filepath):
            chosen_tt = tt
            chosen_nr = nr
            break

    # 6. Validate chosen variables
    # This acts as our test to ensure they are valid before proceeding
    if chosen_tt is None or chosen_nr is None:
        print("All matching files already have a corresponding log file created.")
        return None, None, None, None
        
    if not isinstance(chosen_tt, int) or not isinstance(chosen_nr, int):
        raise ValueError(f"Invalid types for tt ({type(chosen_tt)}) or nr ({type(chosen_nr)}). Expected integers.")

    # 7. Reconstruct the exact filename for the .mpack file                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    filevar = f"{chosen_tt}{fixed_filename}VAR_step_{chosen_nr}.mpack"

    # 8. Create the log file with strict creation mode
    # mode="x" guarantees a FileExistsError is raised if the file exists
    log_prefix = os.path.join(directory, f"{chosen_tt}{fixed_filename}OBSRC_{chosen_nr}")
    log = nk.logging.JsonLog(log_prefix, save_params=False, mode="x")

    return log, filevar, chosen_nr, chosen_tt        
        
    


def save_history_callback(step, log_data, driver, save_every, prefix,NR_I):
    if step % save_every == 0:
        filename = f"{prefix}_step_{step+NR_I}.mpack"
        with open(filename, 'wb') as f:
            f.write(flax.serialization.msgpack_serialize(driver.state.variables))
            print(f"Saved: {filename}") # Optional feedback
    return True
