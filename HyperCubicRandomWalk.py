#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib as mpl
import numpy as np
import random as rn
import Methods.TId as TId
import matplotlib.pyplot as plt
import scipy.optimize as so
import scipy.special as sc
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import random
import sys
aa=10
random.seed(aa)


# In[2]:


def Hamming(A,B,L):
    C= [A[i]!=B[i] for i in range(L)]
    return np.sum(C)

def prob(i,d):
    i_new=i+1
    x=rn.random()
    p_av=(d-i)/d  
    if x>p_av:
        i_new=i-1 
    return i_new

def step(i,d):
    return prob(i,d)

def step_diff(A,L,p):
    B=A.copy()
    x=np.random.randint(0,L)
    yy=random.random()
    if B[x]==-1: B[x]=1
    else: B[x]+=-2*(yy<=p/(1-p))
        
    
    return B

def walk(A,L,N,p):
    for i in range(N):
        A=step_diff(A,L,p)
    return A

def bounds(x,L):
    
    eps=10**(-6)
    x=np.array(x[0:int(L/2)])
    max=x.argmax()
    return max


def initial_guess(x,L):
    eps=10**(-6)
    maxi=bounds(x,L)
    
    delta_left=maxi

    while delta_left>0 and x[delta_left]>eps:
        delta_left-=1
    delta_left+=1
    delta_right=maxi
    while delta_right<int(L/2) and x[delta_right]>eps:
        delta_right+=1
    delta_right-=1
    
    return [delta_left,maxi,delta_right]
    


# In[3]:


#Decide the system size, number of samples:
parameters=sys.argv
n_par=len(parameters)
print(parameters)
parameters=[int(parameters[x]) for x in range(1,n_par)]
dp=0.01
L=parameters[0]
W=1
n_method=0
n_mean=parameters[1]
snap_time=parameters[2]
snaps=parameters[3]
start=1
delta=10
p=parameters[4]


#Define the data-structure

dataset=[np.zeros((n_mean,L)) for i in range(snaps)]
sets=[None for i in range(snaps)]
Weights= [None for i in range(snaps)]
neighbors=[None for i in range(snaps)]
density=[None for i in range(snaps)]
PDF_density=[[0 for i in range(0,L+1,2)] for i in range(snaps)]
#initial state:
dd=[-1,1]
a=[ dd[np.random.randint(0,2)] for i in range(L)]

#Perform a loop over the number of samples:
for j in range(n_mean):
    x=a
    for snap in range(snaps):
        x=walk(x,L,snap_time,p*dp)
        dataset[snap][j]=x



        


# In[4]:


#GENERATE THE FILES
aux=[i for i in range(0,L+1,2)]
size=len(aux)
for snap in range(snaps):
    sets[snap],Weights[snap]=TId.sets(dataset[snap])
    neighbors[snap]=TId.neighbors(sets[snap])
    density[snap]=np.array([np.mean(TId.n_points(neighbors[snap],Weights[snap],i)) for i in range(L+1)])
    
    PDF_density[snap][0]=density[snap][0]/n_mean
    PDF_density[snap][1:size]=(density[snap][aux[1:]]-density[snap][aux[0:size-1]])/n_mean
    


# In[5]:


def exp_Ansatz(x, k, d, d0):
    # Ensure x is a numpy array
    x = np.array(x, dtype=float)
    d_aux =d * x + d0
    prob = np.ones_like(x)  # Initialize an array for probabilities
    
    # Handle each element in x independently due to range(1, x_i + 1)
    for i, x_i in enumerate(x):
        for j in range(1, int(x_i) + 1):  # Loop from 1 to x_i (inclusive)
            prob[i] *= (d_aux[i] + 1 - j) / (j * 2 ** (d_aux[i] / x_i))
            
    return prob*k


# In[6]:


#CREATE THE FIT 
start=1
delta=10

AA=[i for i in range(start,snaps-5)]
init_A=[None for i in AA]
ID0=[]
DID0=[]

for i in range(len(AA)):
    par=initial_guess(PDF_density[AA[i]],L*W)
    init_A[i]=par
    #print(par) 
    x=np.arange(par[0]*2,par[1]*2,2)
    #print(PDF_density[AA[i]][par[0]:par[2]])
    A=so.curve_fit(exp_Ansatz,x,PDF_density[AA[i]][par[0]:par[1]],p0=(1.0,1.0,2*par[1]))
    ID0.append(A[0])
    DID0.append(np.sqrt(np.diag(A[1])))
    
ID0=np.array(ID0)
DID0=np.array(DID0)


# In[7]:


norm = mcolors.Normalize(vmin=np.abs(start), vmax=np.abs(snaps))
colors = [(0.0, 'yellow'),(0.25, "green"),(0.5, 'blue'),(0.75,"brown"),(1.0, 'red')]
# Create the custom colormap
cmap = LinearSegmentedColormap.from_list('custom_blue_green', colors)


name_figs="RW"
fig,axis=plt.subplots(1,1,figsize=(7,5))
plt.title("Model evaluation for different values of $t$",fontsize=15)
plt.xlabel("Hamming Distance $r$",fontsize=15)
plt.ylabel("P(r)",fontsize=15)
plt.yscale("log")
#plt.ylim([1e-6,1e-1])
for i in range(0,len(AA)):
    x=np.arange(2*init_A[i][0],2*init_A[i][2],2)
    y=exp_Ansatz(x,ID0[i,0],ID0[i,1],ID0[i,2])
    axis.plot(x,y,color="black",label=str(i+start))
    axis.scatter(x,PDF_density[AA[i]][init_A[i][0]:init_A[i][2]],color=cmap(norm(i+start)))

#plt.legend(bbox_to_anchor=[1.0, 1.0])
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # ScalarMappable needs an array, even an empty one is fine
# Create the colorbar
cbar = plt.colorbar(sm,ax=axis,label='$i$')
cbar.set_label('$t$', fontsize=14)
plt.savefig("validation"+name_figs)
plt.show()

xx=[a*snap_time for a in AA]
plt.figure()
plt.title("BID for different values of $t$",fontsize=15)
plt.xlabel(" $t$",fontsize=15)
plt.ylabel("BID(t)",fontsize=15)
plt.errorbar(xx,ID0[:,2],yerr=DID0[:,2],linestyle="dashed")
plt.scatter(xx,ID0[:,2],color="black")
plt.legend()
plt.savefig("BID"+name_figs)

plt.show()

plt.figure()
plt.title("$d_1$ for different values of $t$",fontsize=15)
plt.xlabel(" $t$",fontsize=15)
plt.ylabel("$d_1$(t)",fontsize=15)
plt.errorbar(xx,ID0[:,1]/(L*W),yerr=DID0[:,1]/(L*W),color="green",linestyle="dashed")
plt.scatter(xx,ID0[:,1]/(L*W),color="black")
plt.legend()
plt.savefig("D1"+name_figs)

plt.show()

plt.figure()
plt.title("$N$ (Normalization) for different values of $t$",fontsize=15)
plt.xlabel(" $t$",fontsize=15)
plt.ylabel("$N$(t)",fontsize=15)
plt.errorbar(xx,ID0[:,0],yerr=DID0[:,0],color="red",linestyle="dashed")
plt.scatter(xx,ID0[:,0],color="black")
plt.legend()
plt.savefig("N"+name_figs)

plt.show()
        


# In[19]:


I_D=np.mean(ID0[:,2])
DI_D=np.std(ID0[:,2])/np.sqrt(len(ID0[:,2]))
print(I_D,DI_D)


# In[8]:


I_D=np.mean(ID0[:,1])
DI_D=np.std(ID0[:,1])/np.sqrt(len(ID0[:,1]))
print(I_D,DI_D)


# In[ ]:





# In[ ]:





# In[ ]:




