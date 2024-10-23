import numpy as np
import matplotlib.pyplot as plt
import class_WF
import var_nk
from scipy.sparse.linalg import eigsh    
import netket as nk
import sys
import TId
import time
## CONSTANTS
eps=10**(-8)
dx=0.01
V=-1.0
n_between=200

t1=4
t2=5
nt=20


## ALMOST CONSTANTS
n_neurons=1
n_layers=1


## PARAMETERS
parameters=sys.argv
n_par=len(parameters)
parameters=[int(parameters[x]) for x in range(1,n_par)]
print(parameters)
L=parameters[0]
Gamma=parameters[1]*(-dx)
n_samples=parameters[2]
n_run=parameters[3]
n_mean=parameters[4]
NG=parameters[5]
n_method=parameters[6]
cutoff=2**L
if n_method==2:
    n_neurons=parameters[7]
    n_layers=parameters[8]

methods=[var_nk.MF(),var_nk.JasShort(),var_nk.FFN(alpha=n_neurons,layers=n_layers)]

    
model=methods[n_method]



#INSERT PUBLISHER DETAILS AND INITIALIZE IT

if n_method==2:
    name_var=["M","L","NS","NR","G","t1","NN","NL"]
    var=[n_method,L,n_samples,n_run,parameters[1],t1,n_neurons,n_layers]
else:
    name_var=["M","L","NS","NR","G","t1"]
    var=[n_method,L,n_samples,n_run,parameters[1],t1]    
    
variables=["Id"]
pub=class_WF.publisher(name_var,var,variables)
pub.create()

#INITIALIZE OBJECTS
hi=nk.hilbert.Spin(s=1 / 2,N=L)
H=class_WF.Ham(Gamma,V,L,hi)
eig_vecs=class_WF.Diag(H)
E_WF=class_WF.WF(L,model,H,n_samples)
E_WF.advance(n_run)
#ITERATION OVER THE GAMMA VALUES:


aux=np.zeros((n_mean,nt))
    
for hh in range(n_mean):

    A=E_WF.sampling()
    
    if n_method==0 or n_method==2:
        lenght=len(A)
        A[:int(lenght/2),:]=(-1)*A[:int(lenght/2),:]

    start=time.time()
    states,weights=TId.sets(A)
    middle=time.time()
    neigh=TId.neighbors(states)
    end=time.time()
    print(hh,end-start,middle-start)
    for gg in range(t1+1,nt+t1+1):
        start=time.time()
        aux[hh,gg-t1-1]=E_WF.compute_3ID(t1,gg,cutoff,eps,A=A,neigh=neigh,weights=weights,states=states)
        end=time.time()
        #print(hh,gg,end-start)    
    E_WF.advance(n_between)

        

    
for gg in range(nt):
    
    dID=np.std(aux[:,gg])/np.sqrt(n_mean)
    ID=np.mean(aux[:,gg])
    pub.write([t1+1+gg,dID,ID])

pub.close()





