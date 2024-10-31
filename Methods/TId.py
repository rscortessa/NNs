import numpy as np
import scipy.special as sc
import itertools

## Body of the functions:

def fc(x):
    return sc.gamma(x+1)

def fac(x2,x1):
    suma=1
    for i in range(int(x2-x1-1)):
        suma*=(x2-i)
    return suma
    
def sets(A):
    Sa=len(A)
    nw=[]
    ns=[]
    
    while Sa>=1:
        Sa=len(A)
        if Sa<1:
            break
        a=A[0]
        B=np.array(np.sum(a!=A,axis=1),dtype=bool)
        A=A[B]
        Sb=len(A)
        nw.append(a)
        ns.append(Sa-Sb)    
    return np.array(nw),ns
    
def neighbors(A):
    Sa=len(A)
    B=np.zeros((Sa,Sa))

    ## Fast implementation
    Aux=[ np.add.reduce( ((A[i]+A[i:Sa])/2.0+1.0)%2,axis=1 ) for i in range(Sa) ]
    Aux=list(itertools.chain.from_iterable(Aux))
    B[np.triu_indices(Sa)]=Aux
    
    #print(B[0],"fast")

    ## SLOW implementation
    #B=[[np.sum(((A[i]+A[j])/2.0 +1.0)%2.0)*(j>=i+1)+0.0 for j in range(0,Sa)] for i in range(Sa)] 
    #print(B[0],"slow")
    #C=np.triu(B,0)+np.triu(B,1).T
    
    C=B+B.T
    
    return C

def n_points(C,W,t1):
    N_s=len(C)
    
    #slow implementation:
    #Np=np.array([ np.sum(np.array([(C[i,j]<=t1)*W[j] for j in range(N_s)]))  for i in range(N_s)])

    #Fast implementation:
    Aux=C<=t1
    Np=Aux@W

    #Medium  implementation:
    #Np=np.zeros(N_s)
    #for i in range(N_s):
    #    for j in range(i,N_s):
    #        if(C[i,j]<=t1):
    #            if i!=j:
    #                Np[i]+=W[j]
    #                Np[j]+=W[i]
    #            else:
    #                Np[i]+=W[i]

    return Np

def Volume_ratio(C,W,t1,t2):
    Va=np.mean(n_points(C,W,t1))
    Vb=np.mean(n_points(C,W,t2))
    return Va/Vb

## Other Functions
def n_exact(t,x):
    return sc.hyp2f1(-x,-t,-x-t,-1)

def func(t1,t2,x):
    return n_exact(t1,x)/n_exact(t2,x)*(fac(t2,t1)/fac(x+t2,x+t1))


def roots(ns,cutoff,N,eps,func,*args):

    x=np.linspace(0,cutoff,100000)
    y=np.array([ np.abs(func(i,*args)-ns) for i in x])
    
    min=x[np.min(np.abs(y))==y]

    dx=cutoff/100    
    n=0
    while np.abs(func(min,*args)-ns)>eps and n<N:
        x=np.linspace((min-dx)*(min-dx>0),min+dx,1000)
        y=np.array([ np.abs(func(i,*args)-ns) for i in x])
        min=x[np.min(np.abs(y))==y]
        min=min[0]
        dx*=2/100.
        n+=1
        
    return min
