import argparse
import os
import optuna

all_NN_params={}
all_NN_params["CNN_COMPLEX"]=["kernel_size","features","padding"]
all_NN_params["CNN_REAL"]=["kernel_size","features","padding"]
all_NN_params["RBM_COMPLEX"]=["NN"]
all_NN_params["spin_dependent_T"]=["n_heads","head_dim","n_patches"]
all_NN_params["factored_attention_T"]=["n_heads","head_dim","n_patches"]
all_NN_params["parity_CNN_COMPLEX"]=["kernel_size","features","padding"]
all_NN_params["parity_CNN_REAL"]=["kernel_size","features","padding"]
# Initialize the parser
parser = argparse.ArgumentParser()

parser.add_argument("--L", type=int, default=10)
parser.add_argument("--NR", type=int, default=2000)
parser.add_argument("--angle", type=int, default=0)
parser.add_argument("--g", type=int, default=150)
parser.add_argument("--model", type=str, default="QIM")
parser.add_argument("--pbc", action="store_true")
parser.add_argument("--architecture", type=str, default="CNN_COMPLEX")

# This is the magic line
args, unknown = parser.parse_known_args()


if args.architecture == "CNN_COMPLEX":
    parser.add_argument("--kernel_size",type=int,default=2)
    parser.add_argument("--features",type=int,nargs="+",default=[2,2])
    parser.add_argument("--padding",type=str,default="SAME")

if args.architecture == "CNN_REAL":
    parser.add_argument("--kernel_size",type=int,default=2)
    parser.add_argument("--features",type=int,nargs="+",default=[2,2])
    parser.add_argument("--padding",type=str,default="SAME")

if args.architecture == "parity_CNN_COMPLEX":
    parser.add_argument("--kernel_size",type=int,default=2)
    parser.add_argument("--features",type=int,nargs="+",default=[2,2])
    parser.add_argument("--padding",type=str,default="SAME")

if args.architecture == "parity_CNN_REAL":
    parser.add_argument("--kernel_size",type=int,default=2)
    parser.add_argument("--features",type=int,nargs="+",default=[2,2])
    parser.add_argument("--padding",type=str,default="SAME")

    
if args.architecture == "RBM_COMPLEX":
    parser.add_argument("--NN",type=float,default=1.0)
    
if args.architecture == "spin_dependent_T":
    parser.add_argument("--n_heads",type=int,default=1)
    parser.add_argument("--head_dim",type=int,default=10)
    parser.add_argument("--n_patches",type=int,default=1)

if args.architecture == "factored_attention_T":
    parser.add_argument("--n_heads",type=int,default=1)
    parser.add_argument("--head_dim",type=int,default=10)
    parser.add_argument("--n_patches",type=int,default=1)
    
    
    
args = parser.parse_known_args()

params = vars(args[0])

# This are almost unmutable variables

params["NSPCA"]= 10
params["NMEAN"]= 10
params["Nangle"]= 12
params["DG"]= 0.01
params["W"]= 1
params["add"]=""

if params["architecture"] == "CNN_REAL" or params["architecture"] == "spin_dependent_T" or params["architecture"] == "factored_attention_T" or params["architecture"] == "parity_CNN_REAL":
    params["holomorphic"] = False
else:
    params["holomorphic"] = True




if params["pbc"] == True:
    params["add"]="PBC"

ROOT_DIR = "PLAYING_WITH_"+params["architecture"]



try:
    os.mkdir(ROOT_DIR)
except FileExistsError:
    print(f"Root directory exists.")

# NN parameters:
NN_params = all_NN_params[params["architecture"]]

#identifier=params["architecture"]
#for name in NN_params:
#    identifier+=name+str(params[name])

#working_directory = "FULLSUM_"+params["model"]+identifier+"L"+str(params["L"])+"G"+str(params["g"])+"NA"+str(params["Nangle"])+"NSPCA"+str(params["NSPCA"])+"ANGLE"+str(params["angle"])+params["add"]

#try:
#    os.mkdir(ROOT_DIR+"/"+working_directory)
#except FileExistsError:
#    print(f"Root directory exists.")

#study_name = ROOT_DIR+"/"+working_directory+"/"+working_directory
#storage = "sqlite:///"+study_name+".db"
        
#try:
#    optuna.create_study(study_name=study_name,storage=storage,direction="minimize",sampler=optuna.samplers.RandomSampler(),pruner=optuna.pruners.MedianPruner())
#    print(f"✅ Created Optuna study: {study_name}")
#except optuna.exceptions.DuplicatedStudyError:
#    print(f"ℹ️ Study {study_name} already exists.")

#print(f"✅study name {study_name}, storage {storage} ")
