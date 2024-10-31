import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd

## Define the parameters
parameters=sys.argv
n_par=len(parameters)
dx=-0.01
print(parameters)
parameters=[int(parameters[x]) for x in range(1,n_par)]
#...
print(parameters)
Gamma=parameters[0]*(dx)
GammaF=parameters[1]*(dx)
V=-1.0
n_sample=parameters[2]
n_run=parameters[3]
n_mean=parameters[4]
each=bool(parameters[5])
NG=parameters[6]
L=[12,16,20]
n=len(L)


# Open the files ...
name_up="VAR1"+"G"+str(parameters[0])+"GF"+str(parameters[1])+"L"
name_down="N_S"+str(n_sample)+"N_M"+str(n_mean)
data_1=[0 for i in range(n)]
data_2=[0 for i in range(n)]
for j in range(n):
    data_1[j]=pd.read_csv(str(2)+name_up+str(L[j])+name_down+".txt",delim_whitespace=True)
    data_2[j]=pd.read_csv(str(3)+name_up+str(L[j])+name_down+".txt",delim_whitespace=True)

    
for j in range(n):
    data_1[j]["S"]=data_1[j]["I"]
    data_1[j]["dS"]=data_1[j]["dI"]
    data_1[j]["dI"]=(np.abs(data_1[j]["dI"]+data_2[j]["dI"])/np.abs(data_1[j]["I"]-data_2[j]["I"])+data_2[j]["dI"]/data_2[j]["I"])
    data_1[j]["I"]=np.abs((data_1[j]["I"]-data_2[j]["I"])/data_2[j]["I"])

G=np.linspace(Gamma,GammaF,NG)

    
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# First plot on the right
axs[0].set_title(r'$\;I_{var}$ '+"\n"+r"$N_{sample}"+str(n_sample)+"$"+r"$N_{average}="+str(n_mean)+"$")
axs[1].set_title(r'$\; \% I$'+"\n"+r"$N_{sample}"+str(n_sample)+"$"+r" $N_{average}="+str(n_mean)+"$")
for i in range(n):
    axs[0].errorbar(np.abs(G),data_1[i]["S"],data_1[i]["dS"],marker="o",label=str(L[i]))
axs[0].set_xlabel('$h/J$')
axs[0].set_ylabel(r'$I$')
axs[0].legend()
for i in range(n):
    axs[1].set_yscale("log")
    #axs[1].errorbar(np.abs(G),data_1[i]["S"]-data_2[i]["I"],data_1[i]["dS"]+data_2[i]["dI"],marker="o",label=str(L[i]))
    #axs[1].errorbar(np.abs(G),np.abs(data_1[i]["dI"]),data_1[i]["dI"],marker="o",label=str(L[i]))
    axs[1].plot(np.abs(G),np.abs(data_1[i]["dI"]),marker="o",label=str(L[i]))
    
axs[1].set_xlabel('$h/J$')
axs[1].set_ylabel(r'$\% I$')
axs[1].legend()
# Adjust layout to prevent overlapping
plt.tight_layout()
# Show the plots
plt.savefig(name_up+name_down+"I.png")
