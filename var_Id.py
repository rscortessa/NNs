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



## ALMOST CONSTANTS


## PARAMETERS
parameters=sys.argv
n_par=len(parameters)
parameters=[int(parameters[x]) for x in range(1,n_par)]
print(parameters)
L=parameters[0]
Gamma=parameters[1]*(-dx)
t1=parameters[2]
nt=t1
n_samples=parameters[3]
n_run=parameters[4]
n_mean=parameters[5]
n_method=parameters[6]
cutoff=2**L


if n_method==2:
    n_neurons=parameters[7]
    n_layers=parameters[8]

methods=[var_nk.MF(),var_nk.JasShort(),var_nk.FFN(alpha=n_neurons,layers=n_layers)]

    
model=methods[n_method]




    


#INSERT PUBLISHER DETAILS AND INITIALIZE IT


if n_method==2:
    name_var=["M","L","NS","NR","G","t","NN","NL"]
    var=[n_method,L,n_samples,n_run,parameters[1],t1,n_neurons,n_layers]    
else:
    name_var=["M","L","NS","NR","G","t"]
    var=[n_method,L,n_samples,n_run,parameters[1],t1]   
    
    
variables=["Id","Ymin","Vratio"]
pub=class_WF.publisher(name_var,var,variables)
pub.create()

#INITIALIZE OBJECTS
hi=nk.hilbert.Spin(s=1 / 2,N=L)
H=class_WF.Ham(Gamma,V,L,hi)
eig_vecs=class_WF.Diag(H)
E_WF=class_WF.WF(L,model,H,n_samples)



#E_WF.advance(n_run)
#ITERATION OVER THE GAMMA VALUES:


aux=np.zeros((n_mean,L-t1-1))
aux_1=np.zeros((n_mean,L-t1-1))    
aux_2=np.zeros((n_mean,L-t1-1))

for hh in range(n_mean):
    
    E_WF.advance(n_run)
    
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
    for gg in range(t1+1,L):
        start=time.time()
        aux[hh,gg-t1-1],aux_1[hh,gg-t1-1],aux_2[hh,gg-t1-1]=E_WF.compute_3ID(t1,gg,cutoff,eps,A=A,neigh=neigh,weights=weights,states=states)
        end=time.time()

    E_WF.advance(n_between)

    #print(hh,gg,end-start)    

        

    
for gg in range(L-t1-1):
    ymin=np.mean(aux_1[:,gg])
    dymin=np.std(aux_1[:,gg])/np.sqrt(n_mean)
    
    dID=np.std(aux[:,gg])/np.sqrt(n_mean)
    ID=np.mean(aux[:,gg])
    DV=np.std(aux_2[:,gg])/np.sqrt(n_mean)
    V_ratio=np.mean(aux_2[:,gg])
    
    pub.write([t1+1+gg,dID,ID,dymin,ymin,DV,V_ratio])

pub.close()





