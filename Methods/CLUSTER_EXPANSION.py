import itertools
import netket as nk
import numpy as np

def set_fixed_node_spin_indices(L,hi,idxs=None,value=1):
    
    states = hi.all_states()    
    
    if idxs==None or idxs == ():
        mask=np.array([True for state in states])
    else:
        node_values = np.prod(states[:,idxs],axis=-1)
        mask = node_values == value
        
    return mask,states[mask]
    

def log_p_contribution_node(L,hi,prob,idxs=None,value=1,log=True):

    eps=10**(-14)
    set_states, states = set_fixed_node_spin_indices(L,hi,idxs,value)
    cardinal_set,L=states.shape
    if log:
        aux = np.log(prob[set_states]+eps)
    else:
        aux = prob[set_states]
        
    node_coeff = np.sum(aux)/cardinal_set
    
    return node_coeff


def get_coeff_WF(n_order,L,hi,psi_GS,log=True):
    coeff = np.zeros((2**L),dtype=complex)
    subset_masks = range(1,2**L)
    for mask in subset_masks:
        idxs = [ i for i in range(L) if (mask >> i) &1 ]
        cont_1 = log_p_contribution_node(L,hi,psi_GS,idxs=idxs,value=1,log=log) # cont/(2) +a, 
        cont_2 = log_p_contribution_node(L,hi,psi_GS,idxs=idxs,value=-1,log=log) # cont/(2) -a
        coeff[mask]=(cont_1-cont_2)/2.0
        
    return coeff


def Fast_Hadamard(a,hi=None):
    
    if hi is not None:
        ordering = hi._inverted_ordering*(-2.0)+1.0
    else:
        ordering = 1.0
        
    a = a.copy()
    N = len(a)
    h = 1
    while h < N:
        # reshape into blocks of size 2h
        a2 = a.reshape(-1, 2*h)

        # slices
        x = a2[:, :h]
        y = a2[:, h:]

        # butterfly
        t = x.copy()
        a2[:, :h] = 0.5 * (t + y)
        a2[:, h:] = 0.5 * (t-  y) * ordering

        h *= 2

    return a

def truncated_vec(sorting,array,n_coeff):

    aux = array.copy()
    aux[sorting[n_coeff:]] = 0.0
    
    return aux
