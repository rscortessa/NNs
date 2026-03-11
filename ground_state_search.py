import os
import optuna
from config import params,NN_params,ROOT_DIR
from define_models import hi,ham,model,models_name
import numpy
import Methods.var_nk as var_nk


identifier=params["architecture"]
for name in NN_params:
    identifier+=name+str(params[name])

working_directory = "FULLSUM_"+models_name+identifier+"L"+str(params["L"])+"G"+str(params["g"])+"NA"+str(params["Nangle"])+"NSPCA"+str(params["NSPCA"])+"ANGLE"+str(params["angle"])+params["add"]
                          
try:
    os.mkdir(ROOT_DIR+"/"+working_directory)
except FileExistsError:
    print(f"Root directory exists.")

    
study_name = ROOT_DIR+"/"+working_directory+"/"+working_directory
storage = "sqlite:///"+study_name+".db"
        
try:
    optuna.create_study(study_name=study_name,storage=storage,direction="minimize",sampler=optuna.samplers.RandomSampler(),pruner=optuna.pruners.MedianPruner())
    print(f"✅ Created Optuna study: {study_name}")
except optuna.exceptions.DuplicatedStudyError:
    print(f"ℹ️ Study {study_name} already exists.")

print(f"✅study name {study_name}, storage {storage} ")
