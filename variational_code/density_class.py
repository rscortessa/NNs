import numpy as np
import matplotlib.pyplot as plt
import Methods.TId as TId
import Methods.class_WF as class_WF
import Methods.var_nk as var_nk

from scipy.sparse.linalg import eigsh    
import netket as nk
import sys
import time
## CONSTANTS
eps=10**(-8)
dx=0.01
V=-1.0
n_between=200

t1=1
t2=5



## ALMOST CONSTANTS
n_neurons=1
n_layers=1


## PARAMETERS
parameters=sys.argv
n_par=len(parameters)
parameters=[int(parameters[x]) for x in range(1,n_par)]
print(parameters)
L=parameters[0]
W=parameters[1]
Gamma=parameters[2]*(-dx)
tmax=parameters[3]
n_samples=parameters[4]
n_run=parameters[5]
n_mean=parameters[6]
n_method=parameters[7]

try:
    n_neurons=parameters[9]
    n_layers=parameters[10]
except:
    print("no additional parameters")

cutoff=2**L
    
name_var=["M","L","W","NS","NR","G","GF","NN","NL"]
var=[n_method,L,W,n_samples,n_run,parameters[1],parameters[2],n_neurons,n_layers]
print(len(name_var),parameters)
name_var=name_var[:n_par-1]
var=var[:n_par-1]    
methods=[var_nk.MF(),var_nk.JasShort(),var_nk.FFN(alpha=n_neurons,layers=n_layers),nk.models.RBM(alpha=n_neurons),var_nk.SymmModel(alpha=n_neurons,layers=n_layers,L=L,W=W)]

#INITIALIZE OBJECTS
hi=nk.hilbert.Spin(s=1 / 2,N=L*W)
H=class_WF.Ham_PBC(Gamma,V,L,W,hi)

if n_method==5:
    eig_vecs=class_WF.Diag(H)
    methods.append(var_nk.EWF(eig_vec=tuple(np.abs(eig_vecs[:,0])),L=L))


cutoff=2**L

model=methods[n_method]



#INSERT PUBLISHER DETAILS AND INITIALIZE IT

variables=["D"]
pub=class_WF.publisher(name_var,var,variables)
pub.create()

E_WF=class_WF.WF(L,model,H,n_samples)

aux=np.zeros((n_mean,tmax))

#ITERATION OVER THE GAMMA VALUES:
if n_method!=5:
    zz=time.time()
    E_WF.advance(n_run)
    yy=time.time()
for hh in range(n_mean):
    
    for gg in range(tmax):
        aa=time.time()
        A=E_WF.sampling()
        bb=time.time()
        if n_method==0 or n_method==2:
            lenght=len(A)
            A[:int(lenght/2),:]=(-1)*A[:int(lenght/2),:]
        cc=time.time()
        B,W=TId.sets(A)
        dd=time.time()
        C=TId.neighbors(B)
        ee=time.time()
        aux[hh,gg]=np.mean(TId.n_points(C,W,gg))
        ff=time.time()
    if n_method !=5:
        gg=time.time()
        E_WF.advance(n_between)
        hh=time.time()
    
    else:
        E_state=nk.vqs.MCState(E_WF.user_sampler,model,n_samples=n_samples)
        E_WF.change_state(E_state)
    print(hh-gg,gg-ff,ff-ee,ee-dd,dd-cc,cc-bb,bb-aa,yy-zz)
SD=[]
dSD=[]
for i in range(tmax):
    pub.write([np.std(aux[:,i])/np.sqrt(n_mean),np.mean(aux[:,i])])

pub.close()





