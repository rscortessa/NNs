import numpy as np
import scipy.special as sc


## Body of the functions:

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
    B=[[np.sum(((A[i]+A[j])/2.0 +1.0)%2.0)*(j>=i+1)+0.0 for j in range(0,Sa)] for i in range(Sa)]
    C=np.triu(B,0)+np.triu(B,1).T
    return C

def n_points(C,W,t1):
    N_s=len(C)
    Np=np.zeros(N_s)   
    for i in range(N_s):
        for j in range(i,N_s):
            if(C[i,j]<=t1):
                if i!=j:
                    Np[i]+=W[j]
                    Np[j]+=W[i]
                else:
                    Np[i]+=W[i]
    return Np

def Volume_ratio(C,W,t1,t2):
    Va=np.mean(n_points(C,W,t1))
    Vb=np.mean(n_points(C,W,t2))
    return Vb/Va

## Other Functions

def func(t1,t2,x):
    return sc.hyp2f1(-x,-t1,-x-t1,-1)/sc.hyp2f1(-x,-t2,-x-t2,-1)


def roots(ns,cutoff,N,eps,func,*args):
    eps=10**(-8)
    x=np.linspace(0,cutoff,100000)
    y=np.abs(func(x,*args)-ns)
    
    min=x[np.min(np.abs(y))==y]
    dx=cutoff/100
    
    n=0
    while func(min,*args)>eps or n>N:
        print(n)
        x=np.linspace(min-dx,min+dx,100)
        y=np.abs(func(x,*args)-ns)
        min=x[np.min(np.abs(y))==y]
        dx*=2/100.
        
    return min

