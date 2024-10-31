import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd

## Define the parameters
parameters=sys.argv
n_par=len(parameters)
print(parameters)
parameters=[int(parameters[x]) for x in range(1,n_par)]
#...
print(parameters)
V=-1.0

Gamma=parameters[0]*(-0.01)
GammaF=parameters[1]*(-0.01)
n_sample=parameters[2]
n_mean=parameters[3]
NG=parameters[4]
L=parameters[5]



# Open the files ...
name="VAR1"+"G"+str(parameters[0])+"GF"+str(parameters[1])+"L"+str(L)+"N_S"+str(n_sample)+"N_M"+str(n_mean)


data_1=[0 for i in range(n)]
data_2=[0 for i in range(n)]
for j in range(n):
    data_1[j]=pd.read_csv(str(2)+name_up+str(L[j])+name_down+".txt",delim_whitespace=True)
    data_2[j]=pd.read_csv(str(3)+name_up+str(L[j])+name_down+".txt",delim_whitespace=True)

    
for j in range(3):
    data_1[j]["dE"]=(np.abs(data_1[j]["dE"]+data_2[j]["dE"])/np.abs(data_1[j]["E"]-data_2[j]["E"])+data_2[j]["dE"]/data_2[j]["E"])
    data_1[j]["dS"]=(np.abs(data_1[j]["dS"]+data_2[j]["dS"])/np.abs(data_1[j]["S"]-data_2[j]["S"])+data_2[j]["dS"]/data_2[j]["S"])
    data_1[j]["E"]=np.abs((data_1[j]["E"]-data_2[j]["E"])/data_2[j]["E"])
    data_1[j]["S"]=np.abs(data_1[j]["S"]-data_2[j]["S"])/data_2[j]["S"]
    data_1[j]["dS"]=data_1[j]["dS"]*data_1[j]["S"]
    data_1[j]["dE"]=data_1[j]["dE"]*data_1[j]["E"]

    
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# First plot on the right
axs[0].set_title(r'$\% \;Energy$ '+"\n"+r" $N_{samples}="+str(n_sample)+"$ "+r"$N_{average}="+str(n_mean)+"$")
axs[1].set_title(r'$\%\; S_{PCA} $ '+"\n"+r" $N_{samples}="+str(n_sample)+"$"+r" $N_{average}="+str(n_mean)+"$")
for i in range(n):
    axs[0].errorbar(np.abs(data_1[i]["G"]),data_1[i]["E"],data_1[i]["dE"],marker="o",label=str(L[i]))
axs[0].set_xlabel('$h/J$')
axs[0].set_ylabel(r'$\% E$')
axs[0].legend()
for i in range(n):
    axs[1].set_yscale("log")
    axs[1].errorbar(np.abs(data_1[i]["G"]),np.abs(data_1[i]["S"]),data_1[i]["dS"],marker="o",label=str(L[i]))
axs[1].set_xlabel('$h/J$')
axs[1].set_ylabel(r'$\% S_{PCA}$')
axs[1].legend()
# Adjust layout to prevent overlapping
plt.tight_layout()
# Show the plots
plt.savefig(name_up+name_down+"ESLs.png")
