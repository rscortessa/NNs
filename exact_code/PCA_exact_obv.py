import netket as nk
import numpy as np
import matplotlib.pyplot as plt
from netket.operator.spin import sigmax,sigmaz
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import jax.numpy as jnp #Search what the difference is
import flax
import jax
import flax.linen as nn # What is this?
from sklearn.decomposition import PCA
import sys
from Methods.Core_WF import Ham,Exact_Calculation, break_sym,Diag,EWF,FFN

## Define the parameters
parameters=sys.argv
n_par=len(parameters)
print(parameters)
parameters=[int(parameters[x]) for x in range(1,n_par)]
#...
print(parameters)
L=parameters[0]
Gamma=parameters[1]*(-0.01)
n_sample=parameters[2]
n_run=parameters[3]
n_mean=parameters[4]
n=parameters[5]
yes=bool(parameters[6])

V=-1.0
eps=10**(-6)
hi=nk.hilbert.Spin(s=1 / 2,N=L)
H=Ham(Gamma,V,L,hi)
eigv=Diag(H)
sampler = nk.sampler.MetropolisLocal(hi)

aux=tuple(np.abs(eigv[:,0]))

pca=PCA()

fig, axs =plt.subplots(1,2)

plt.suptitle("L="+str(L)+" h/J="+str(Gamma),fontsize=15)
axs[0].set_title("Exact PC",fontsize=15)
axs[0].set_xlabel(r"$1_{PC}$",fontsize=15)
axs[0].set_ylabel("$2_{PC}$",fontsize=15)
axs[1].set_title("NN. Ansatz PC",fontsize=15)
axs[1].set_xlabel(r"$1_{PC}$",fontsize=15)
#axs[1].set_ylabel("$2_{PC}$",fontsize=15)

exvar_E=np.zeros((n_mean,L))
PC1_E=np.zeros((n_mean,L))

for i in range(n_mean):
    n_exact=8192*4
    vstate_exact = nk.vqs.MCState(sampler, EWF(eig_vec=aux,L=L), n_samples=n_exact)
    A_exact=np.array(vstate_exact.samples).reshape((n_exact,L))
    E=pca.fit(A_exact)
    exvar_E[i]=pca.explained_variance_ratio_
    PC1_E[i]=pca.components_[n]
    New_A=pca.transform(A_exact)
    axs[0].scatter(New_A[:,0],New_A[:,1])

dexvar_E=np.std(exvar_E,axis=0)/np.sqrt(n_mean)
exvar_E=np.mean(exvar_E,axis=0)
dPC1_E=np.std(PC1_E,axis=0)/np.sqrt(n_mean)
PC1_E=np.mean(PC1_E,axis=0)

exvar=np.zeros((n_mean,L))
PC1=np.zeros((n_mean,L))
    
for i in range(n_mean):
    vstate= nk.vqs.MCState(sampler,FFN(alpha=1),n_samples=n_sample) ;
    optimizer= nk.optimizer.Sgd(learning_rate=0.05) ;
    log=nk.logging.RuntimeLog() ;    
    gs=nk.driver.VMC(H, optimizer, variational_state=vstate,preconditioner=nk.optimizer.SR(diag_shift=0.1))
    gs.advance(n_run);
    A=np.array(vstate.samples).reshape((n_sample,L))
    if yes==True:
        lenght=len(A)   
        A[:int(lenght/2),:]=(-1)*A[:int(lenght/2),:]        
    E=pca.fit(A)
    exvar[i]=pca.explained_variance_ratio_
    PC1[i]=pca.components_[2]
    New_A=pca.transform(A)
    axs[1].scatter(New_A[:,0],New_A[:,1])

plt.savefig("PCAG"+str(parameters[1])+"NS"+str(parameters[2])+"NR"+str(parameters[5])+"NM"+str(parameters[4])+"Y"+str(parameters[6])+".png")


error_exvar=np.std(exvar,axis=0)/np.sqrt(n_mean)
error_PC1=np.std(PC1,axis=0)/np.sqrt(n_mean)

exvar=np.mean(exvar,axis=0)
PC1=np.mean(PC1,axis=0)
X=np.arange(0,L,1)

file=open("PC"+str(n)+str(parameters[1])+"NS"+str(parameters[2])+"NR"+str(parameters[5])+"NM"+str(parameters[4])+"Y"+str(parameters[6])+".txt","w")
file.write(" ".join(map(str,PC1)))
file.write(" ".join(map(str,PC1_E/n_mean)))

file.close()

fig, axs =plt.subplots(1,2)
plt.suptitle("L="+str(L)+" h/J="+str(Gamma),fontsize=15)
axs[0].set_title("Exact PC"+str(n),fontsize=15)
axs[0].set_xlabel(r"$Sites$",fontsize=15)
axs[0].set_ylabel("$PC"+str(n)+"$",fontsize=15)
axs[1].set_title("NN. Ansatz PC"+str(n),fontsize=15)
axs[1].set_xlabel(r"$Sites$",fontsize=15)
axs[0].errorbar(X,PC1_E,yerr=dPC1_E,fmt="o",linestyle="None")
axs[1].errorbar(X,PC1,yerr=error_PC1,fmt="o",linestyle="None")
plt.savefig("PC"+str(n)+"G"+str(parameters[1])+"NS"+str(parameters[2])+"NR"+str(parameters[5])+"NM"+str(parameters[4])+"Y"+str(parameters[6])+".png")

fig, axs =plt.subplots(1,2)
plt.suptitle("L="+str(L)+" h/J="+str(Gamma),fontsize=15)
axs[0].set_title("Exact EVR",fontsize=15)
axs[0].set_xlabel(r"$Sites$",fontsize=15)
axs[0].set_ylabel("$EVR$",fontsize=15)
axs[1].set_title("NN. Ansatz EVR",fontsize=15)
axs[1].set_xlabel(r"$Sites$",fontsize=15)
axs[0].errorbar(X,exvar_E,yerr=dexvar_E,fmt="o",linestyle="None")
axs[1].errorbar(X,exvar,yerr=error_exvar,fmt="o",linestyle="None")
axs[0].set_yscale("log")
axs[1].set_yscale("log")
plt.savefig("EVR"+str(parameters[1])+"NS"+str(parameters[2])+"NR"+str(parameters[5])+"NM"+str(parameters[4])+"Y"+str(parameters[6])+".png")

file=open("EVR"+str(parameters[1])+"NS"+str(parameters[2])+"NR"+str(parameters[5])+"NM"+str(parameters[4])+"Y"+str(parameters[6])+".txt","w")
file.write(" ".join(map(str,exvar_E/n_mean-exvar)))
file.close()
plt.figure()
plt.title("L="+str(L)+" h/J="+str(Gamma)+"\n"+"$\Delta EVR$",fontsize=15)
plt.errorbar(X,exvar_E-exvar,yerr=dexvar_E+error_exvar,fmt="o",linestyle="None",label=r"$EVR_{exact}-EVR$")
plt.xlabel("Component",fontsize=15)
plt.ylabel(r"$\Delta EVR$",fontsize=15)
plt.legend()
plt.savefig("Delta_EVR"+str(parameters[1])+"NS"+str(parameters[2])+"NR"+str(parameters[5])+"NM"+str(parameters[4])+"Y"+str(parameters[6])+".png")

