import netket as nk
import argparse
import flax
import optuna
from config import params,ROOT_DIR,working_directory,study_name,storage
from define_models import hi,ham,model
from functools import partial
from Methods.FULL_STATE_OP import objective,filenames,save_history_callback,find_minimum_max_nr_file
import Methods.class_WF as class_WF
import optax
# Hyper parameters....
study = optuna.load_study(study_name=study_name, storage=storage)

parser = argparse.ArgumentParser()
parser.add_argument("--clipping", action="store_true")
parser.add_argument("--resume", action="store_true")

args, unknown = parser.parse_known_args()
params_ = vars(args)

if params_["clipping"] == True:
    optimizer = optax.chain(
    optax.clip_by_global_norm(study.best_params["max_step_size"]),
    optax.sgd(learning_rate=study.best_params["learning_rate"]) 
    )
else:
    optimizer = nk.optimizer.Sgd(learning_rate=study.best_params["learning_rate"])

sr = nk.optimizer.SR(diag_shift=study.best_params["diag_shift"], holomorphic=params["holomorphic"])
PSI = class_WF.FULL_WF(params["L"],hi,sr,optimizer,model,ham)


# OBSERVABLES INIT...
obs={}

# Number of effective steps
NR_eff=int(params["NR"]/params["NSPCA"])

# RESTART THE NETWORK
vstate=PSI.user_state
vstate.init_parameters()
PSI.change_state(vstate)
            
# THE OUT LOGS ARE CREATED
FILENAME = "NM"+working_directory
tt=0

#-----------------------------------------------------------------------------------------------------------------------------------------------------

if params_["resume"] == True:
    log,filevar,nr,tt = find_minimum_max_nr_file(ROOT_DIR+"/"+working_directory,FILENAME)
    fileobs = f"{tt}{FILENAME}OBS"
    filevar = f"{tt}{FILENAME}"
    #log = nk.logging.JsonLog(ROOT_DIR+"/"+working_directory+"/"+fileobs+"RC_"+str(nr), save_params=False)
    n_run = params["NR"]+1
    PSI.load_params(nr,filevar,working_dir = ROOT_DIR+"/"+working_directory)
    filevar +="VAR"
else:
    filevar, fileobs =filenames(ROOT_DIR+"/"+working_directory,params["NR"],FILENAME)
    log = nk.logging.JsonLog(ROOT_DIR+"/"+working_directory+"/"+fileobs, save_params=False)
    n_run = params["NR"]+1
    nr=0
    
my_callback = partial(save_history_callback, save_every=int(params["NR"]/params["NSPCA"]), prefix=ROOT_DIR+"/"+working_directory+"/"+filevar,NR_I=nr)
PSI.run(obs=obs,n_iter=n_run,log=log,callback=my_callback)

#-----------------------------------------------------------------------------------------------------------------------------------------------------

print("🚀 Finished for "+str(params))
print("Results saved ✅ in",filevar,fileobs)

