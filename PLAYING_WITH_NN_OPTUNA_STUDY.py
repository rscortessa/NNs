import netket as nk
import argparse
import flax
import optuna
from config import params,ROOT_DIR
from ground_state_search import working_directory,study_name,storage
from define_models import hi,ham,model,cost_func
from functools import partial

# Hilbert space generation in Netket

parser = argparse.ArgumentParser()
parser.add_argument("--niter", type=int, default=1000)
parser.add_argument("--trials", type=int, default=100)
parser.add_argument("--clipping", action="store_true")
args, unknown = parser.parse_known_args()
params_ = vars(args)

study = optuna.load_study(study_name=study_name, storage=storage)
if len(study.trials) < params_["trials"]:
    
    objective_final = partial(cost_func,model=model,L=params["L"]*params["W"],hi=hi,H=ham,n_iter=params_["niter"],holomorphic=params["holomorphic"],clipping=params_["clipping"])
    print("🚀 Running Optuna trials for "+str(params))
    study.optimize(objective_final, n_trials=params_["trials"]-len(study.trials),n_jobs=1)
    print("🚀 Running Optuna trials for "+str(params))
    print("✅ Finished trials for these parameters.")
else:
    print("number of trials already ",params_["trials"])



