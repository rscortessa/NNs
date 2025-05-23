import numpy as np
import matplotlib.pyplot as plt
import Methods.TId as TId
import Methods.class_WF as class_WF
import Methods.var_nk as var_nk
import pandas as pd
from scipy.sparse.linalg import eigsh    
import netket as nk
import sys

## CONSTANTS

eps=10**(-8)

## PARAMETERS

parameters=sys.argv
n_par=len(parameters)
parameters=[int(parameters[x]) for x in range(1,n_par)]
print(parameters)
L=parameters[0]
W=parameters[1]
Gamma=parameters[2]
n_samples=parameters[3]
n_angles=parameters[4]
jj=parameters[5]
n_run=1000
n_method=5

#INSERT PUBLISHER DETAILS AND INITIALIZE IT
MASTER_FOLDER= "RUN_QIM_DMRG_L" +str(L) + "W"+str(W)+"NS"+str(n_samples)+"G"+str(Gamma)+ "ANG0-90NANG"+str(n_angles)+"NR"+str(jj)


name_var=[MASTER_FOLDER+"/"+"IPR","L","W","NS","NR","G"]
var=[n_method,L,W,n_samples,n_run,parameters[2]]

pub=class_WF.publisher(name_var,var,[])
pub.create(jj)

for theta in range(n_angles+1):

    name=MASTER_FOLDER+"/"+"DATAM5L"+str(L)+"W"+str(W)+"NS"+str(n_samples)+"MPSG"+str(theta)+".txt"+str(jj) 
    file=pd.read_csv(name,sep=r"\s+",dtype="a")
    file=file.astype(float)

    A=np.array(file)
    B,We=TId.sets(A)
    print(We)
    NORM=np.sum(We)
    We=np.array(We)
    IPR=We@(We*1.0)/NORM**2

    pub.write([theta,IPR])
    
pub.close()





