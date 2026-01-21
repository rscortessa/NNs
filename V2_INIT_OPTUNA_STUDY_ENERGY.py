import numpy as np
import sys
import os
import optuna
import logging
import json

parameters=sys.argv
n_par=len(parameters)
parameters=[float(parameters[x]) for x in range(1,n_par)]

L = int(parameters[0])
W=1
DG=0.01
NN = parameters[1]
NL=2
NR = int(parameters[2])
NSPCA = int(parameters[3])
Nangle = int(parameters[4])
NMEAN = int(parameters[5])
g = int(parameters[6])
ii = int(parameters[7])


#Model Details
basis="QIM"
architecture = "CNN_COMPLEX"
pbc=False

add=""
if pbc:
    add+="PBC"

#basis = "BROKENZ2_QIM"

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

#Model and optimization details

architecture_set=["RBM_COMPLEX","RBM_REAL","WSIGNS_RBM_COMPLEX","WSIGNS_RBM_REAL","GCNN_COMPLEX","CNN_COMPLEX"]
if architecture not in architecture_set:
    print("MODEL IS NOT DEFINED")
    exit()
    


#Creation of the folder

MASTER_DIR="ENERGY"

if not os.path.isdir(MASTER_DIR):
    print(os.path.isdir(MASTER_DIR))
    os.mkdir(MASTER_DIR)
    print(f"Directory '{MASTER_DIR}' not found previously but created successfully.")

SLAVE_DIR="FULL_STATE_RUN_"+basis+"_"+architecture+"NN"+str(NN)+"L"+str(L)+"G"+str(g)+"NA"+str(Nangle)+"NSPCA"+str(NSPCA)+add

try:
    os.mkdir(MASTER_DIR+"/"+SLAVE_DIR)
except FileExistsError:
    print(f"Directory "+MASTER_DIR+"/"+SLAVE_DIR+" already exists.")


    
study_name = f"MODEL{basis}ARCH{architecture}NN{NN}L{L}G{g}NA{Nangle}ANGLE{ii}BC{add}"
storage = "sqlite:///"+MASTER_DIR+"/"+SLAVE_DIR+"/"+study_name+"optuna_study.db"
        
try:
    optuna.create_study(study_name=study_name,storage=storage,direction="minimize",sampler=optuna.samplers.RandomSampler(),pruner=optuna.pruners.MedianPruner())
    print(f"✅ Created Optuna study: {study_name}")
except optuna.exceptions.DuplicatedStudyError:
    print(f"ℹ️ Study {study_name} already exists.")

print(f"✅study name {study_name}, storage {storage} ")

