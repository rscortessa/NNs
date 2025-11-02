import numpy as np
import netket as nk
from functools import reduce
from netket.operator.spin import sigmax,sigmaz,sigmay,identity,sigmam,sigmap


def build_jastrow_wf(L,J,hi):
    
    states=hi.all_states()
    democratic_state=np.ones((2**L),dtype=complex)/np.sqrt(2**L)
    
    for l in range(len(states)):

        two_op_list = [[np.cosh(J[i,j])+states[l,i]*states[l,j]*np.sinh(J[i,j]) for j in range(i+1,L)] for i in range(L)]
        two_op_list = np.array([x for sublist in two_op_list for x in sublist])
        one_op_list = [np.cosh(J[i,i])+np.sinh(J[i,i])*states[l,i] for i in range(L)]
        two_op = reduce(lambda x, y: x * y, two_op_list)
        one_op = reduce(lambda x, y: x * y, one_op_list)
        JF = one_op*two_op

        democratic_state[l]*=JF
        
    return democratic_state/np.linalg.norm(democratic_state)
