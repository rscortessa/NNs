import netket as nk
import Core_WF
import numpy as np
import matplotlib.pyplot as plt
from netket.operator.spin import sigmax,sigmaz
from scipy.sparse.linalg import eigsh
import jax.numpy as jnp #Search what the difference is
import flax
import jax
import flax.linen as nn # What is this?
from sklearn.decomposition import PCA

from Core_WF import Ham, MF, JasShort, FFN, v_state_steady, Exact_Calculation_steady
from Core_WF import hi, H, n_sample, n_run, n_mean, each, k, L, Gamma, V,eig_vals, parameters

Nh=parameters[8]
dh=parameters[9]*(-0.01)

n=3
model=[MF(),JasShort(),FFN(alpha=1)]
method=["Mean field","Jastrow MF", "Neural Network"]
En=np.zeros((n_mean,n,Nh))
S_ent=np.zeros((n_mean,n,Nh))
m=np.zeros((n_mean,n,Nh))
if each==False:
    sisj=np.zeros((n_mean,n,L-1))
else:
    sisj=np.zeros((n_mean,n,Nh,L-1))

for j in range(n_mean):
    for i in range(n):
        sisj[j,i,:],m[j,i,:],En[j,i,:],S_ent[j,i,:]=v_state_steady(model[i],n_sample,hi,n_run,L,dh,Nh,each) ;
        
En=np.sum(En,axis=0)/n_mean
S_ent=np.sum(S_ent,axis=0)/n_mean
s_is_j=np.sum(sisj,axis=0)/n_mean
m=np.sum(m,axis=0)/n_mean

S_exact,m_exact,var_sisj_exact,s_is_j_exact=Exact_Calculation_steady(8192*2,n_run,n_mean,L,Nh,dh)


fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# First plot on the right
axs[0].set_title('Energy '+"\n"+r"$L="+str(L)+"$"+r" $N_{samples}="+str(n_sample)+"$ "+r"$N_{average}="+str(n_mean)+"$")
axs[1].set_title(r'$S_{PCA} $ '+"\n"+r"$L="+str(L)+"$ "+r" $N_{samples}="+str(n_sample)+"$"+r" $N_{average}="+str(n_mean)+"$")
for i in range(n):
    axs[0].set_yscale("log")
    axs[0].plot(np.abs((En[i]-eig_vals[0])/eig_vals[0]),label=method[i])
axs[0].set_xlabel('h')
axs[0].set_ylabel('Energy')
axs[0].legend()
for i in range(n):
    axs[1].set_yscale("log")
    axs[1].plot(np.abs(S_ent[i]-S_exact)/S_exact,label=method[i])
axs[1].set_xlabel('h')
axs[1].set_ylabel('Entropy')
axs[1].legend()
# Adjust layout to prevent overlapping
plt.tight_layout()
# Show the plots
plt.savefig("Energy_vs_Entropy_h.png")

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# First plot on the right
axs[0].set_title('Magnetization '+"\n"+r"$L="+str(L)+"$"+r" $N_{samples}="+str(n_sample)+"$ "+r"$N_{average}="+str(n_mean)+"$")
axs[1].set_title(r'$\langle S_0 \cdot S_{i} \rangle $ '+"\n"+r"$L="+str(L)+"$ "+r" $N_{samples}="+str(n_sample)+" $"+r" $N_{average}="+str(n_mean)+"$")
for i in range(n):
    axs[0].plot(m[i],label=method[i])
axs[0].plot(m_exact,label="exact",markersize=12,linestyle="dashed")
axs[0].set_xlabel('h')
axs[0].set_ylabel('m')
axs[0].legend()
if each==False:
    corr=np.array([[np.abs(s_is_j[i,j]-s_is_j_exact[j])/s_is_j_exact[j] for j in range(L-1)] for i in range(n)])
    for i in range(n):
        axs[1].set_yscale("log")
        axs[1].plot(corr[i,:],label=method[i])
    axs[1].set_xlabel('site i')
    axs[1].set_ylabel(r"$ |\langle S_0 \cdot S_{i} \rangle_{exact} -\langle S_0 \cdot S_{"+str(k)+r"} \rangle|/ \langle S_0 \cdot S_{"+str(k)+r"} \rangle_{exact} $",fontsize=15)
if each==True:
    axs[1].set_yscale("log")
    for i in range(n):
        axs[1].plot(s_is_j[i,:,k],label=method[i])
    axs[1].plot(s_is_j_exact[j],label="exact",linestyle="dashed")
    axs[1].set_xlabel('h')
    axs[1].set_ylabel(r"$ |\langle S_0 \cdot S_{i} \rangle_{exact} -\langle S_0 \cdot S_{"+str(k)+r"} \rangle|/ \langle S_0 \cdot S_{"+str(k)+r"} \rangle_{exact} $",fontsize=15)
    axs[1].legend()
# Adjust layout to prevent overlapping
plt.tight_layout()
# Show the plots
plt.savefig("magnetization_vs_correlation_h.png")
