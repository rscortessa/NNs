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
N=parameters[0]
Gamma=parameters[1]*(-0.01)
GammaF=parameters[2]*(-0.01)
V=-1.0
n_sample=parameters[3]
print(n_sample)
n_run=parameters[4]
n_mean=parameters[5]
each=bool(parameters[6])
NG=parameters[7]
n=4
L=N
method=["Mean field","Jastrow MF", "Neural Network","Exact"]


# Open the files ...
name="VAR1"+"G"+str(parameters[1])+"GF"+str(parameters[2])+"L"+str(L)+"N_S"+str(n_sample)+"N_M"+str(n_mean)+".txt"
data_1=[0 for i in range(4)]
data_2=[0 for i in range(4)]
for j in range(4):
    data_1[j]=pd.read_csv(str(j)+name,delim_whitespace=True)
    data_2[j]=pd.read_csv(str(j)+name,delim_whitespace=True)

    
for j in range(3):
    data_2[j]["E"]=data_1[j]["E"]-data_1[3]["E"]
    data_2[j]["S"]=np.abs(data_1[j]["S"]-data_1[3]["S"])/data_1[3]["S"]
    
    

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# First plot on the right
axs[0].set_title('Energy '+"\n"+r"$L="+str(L)+"$"+r" $N_{samples}="+str(n_sample)+"$ "+r"$N_{average}="+str(n_mean)+"$")
axs[1].set_title(r'$S_{PCA} $ '+"\n"+r"$L="+str(L)+"$ "+r" $N_{samples}="+str(n_sample)+"$"+r" $N_{average}="+str(n_mean)+"$")
for i in range(n):
    axs[0].plot(data_1[i]["G"],data_1[i]["E"],label=method[i])
axs[0].set_xlabel('h')
axs[0].set_ylabel('Energy')
axs[0].legend()
for i in range(n):
    axs[1].set_yscale("log")
    axs[1].plot(data_1[i]["G"],np.abs(data_1[i]["S"]),label=method[i])
axs[1].set_xlabel('h')
axs[1].set_ylabel('Entropy')
axs[1].legend()
# Adjust layout to prevent overlapping
plt.tight_layout()
# Show the plots
plt.savefig("Energy_vs_Entropy_h.png")

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# First plot on the right
axs[0].set_title('Energy '+"\n"+r"$L="+str(L)+"$"+r" $N_{samples}="+str(n_sample)+"$ "+r"$N_{average}="+str(n_mean)+"$")
axs[1].set_title(r'$S_{PCA} $ '+"\n"+r"$L="+str(L)+"$ "+r" $N_{samples}="+str(n_sample)+"$"+r" $N_{average}="+str(n_mean)+"$")
for i in range(n):
    axs[0].set_yscale("log")
    axs[0].plot(data_1[i]["G"],data_2[i]["E"],label=method[i])
axs[0].set_xlabel('h')
axs[0].set_ylabel(r'$\% Energy$')
axs[0].legend()
for i in range(n):
    axs[1].set_yscale("log")
    axs[1].plot(data_1[i]["G"],data_2[i]["S"],label=method[i])
axs[1].set_xlabel('h')
axs[1].set_ylabel(r'$\% S_{PCA}$')
axs[1].legend()
# Adjust layout to prevent overlapping
plt.tight_layout()
# Show the plots
plt.savefig("Error_Energy_vs_Entropy_h.png")




# First plot on the right
plt.figure()
plt.title('Magnetization '+"\n"+r"$L="+str(L)+"$"+r" $N_{samples}="+str(n_sample)+"$ "+r"$N_{average}="+str(n_mean)+"$")
for i in range(n):
    plt.plot(data_1[i]["G"],data_1[i]["m"],label=method[i])
plt.xlabel('$h$')
plt.ylabel('$m$')
plt.legend()
# Adjust layout to prevent overlapping
plt.tight_layout()
# Show the plots
plt.savefig("magnetization_vs_correlation_h.png")
