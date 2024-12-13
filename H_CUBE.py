import matplotlib as mpl
import numpy as np
import random as rn
import Methods.TId as TId
import Methods.class_WF as class_WF
import scipy.optimize as so
import scipy.special as sc
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import random
import sys

## DEFINITION OF MAIN FUNCTIONS:

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

aux=[i for i in range(0,L+1,2)]
size=len(aux)
for snap in range(snaps):
    sets[snap],Weights[snap]=TId.sets(dataset[snap])
    neighbors[snap]=TId.neighbors(sets[snap])
    density[snap]=np.array([np.mean(TId.n_points(neighbors[snap],Weights[snap],i)) for i in range(L+1)])
    
    PDF_density[snap][0]=density[snap][0]/n_mean
    PDF_density[snap][1:size]=(density[snap][aux[1:]]-density[snap][aux[0:size-1]])/n_mean
    

AA=[i for i in range(start,snaps-1)]
init_A=[None for i in AA]
ID0=[]
DID0=[]

for i in range(len(AA)):
    par=initial_guess(PDF_density[AA[i]],L*W)
    init_A[i]=par
    
    x=np.arange(par[0]*2,par[1]*2,2)
    
    A=so.curve_fit(exp_Ansatz,x,PDF_density[AA[i]][par[0]:par[1]],p0=(1.0,1.0,2*par[1]))
    ID0.append(A[0])
    DID0.append(np.sqrt(np.diag(A[1])))

ID0=np.array(ID0)
IDP=np.mean(ID0[:,2])
DIDP=np.std(ID0[:,2])/np.sqrt(snaps)
ID1P=np.mean(ID0[:,1])
DID1P=np.std(ID0[:,1])/np.sqrt(snaps)

#INSERT PUBLISHER DETAILS AND INITIALIZE IT

name_var=["RWALK","L","NS","NR","P"]
var=[n_method,L,n_mean,snap_time,p]    

pub=class_WF.publisher(name_var,var,["ID","ID1"])
pub.create()
pub.write([DIDP,IDP,DID1P,ID1P])    
pub.close()
