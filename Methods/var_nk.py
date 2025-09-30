
import netket as nk
import jax.numpy as jnp
import flax
import jax
import flax.linen as nn
import numpy as np
import netket.nn as nknn
from typing import Any

import numpy as np
import netket
from flax import struct
from jax.nn.initializers import normal
from netket.utils import HashableArray
from netket.utils.types import NNInitFunc
from netket.utils.group import PermutationGroup

default_kernel_init = normal(stddev=0.01)

def change_to_int(x,L):
    Aux=jnp.array([2**(L-1-i) for i in range(L)])
    Z=jnp.array(jnp.mod(1+x,3)/2,int)
    return np.sum(Aux*Z,axis=-1)


class MF(nn.Module):                                                                                                                   
    @nn.compact # What is this ?                                                                                                       
    def __call__(self,x): #x.shape(Nsamples,L)                                                                                         hi
        lam= self.param("lambda", nn.initializers.normal(),(1,),float)                                                                 
        p= nn.log_sigmoid(lam*x) ## How does the initializers work?                                                                    
        return 0.5*jnp.sum(p,axis=-1)                                                                                                  
                                                                                                                                       
class EWF(nn.Module):                                                                                                                  
    eig_vec:tuple = struct.field(pytree_node=False)                                                                                                                
    L:float
    def setup(self):                                                                                                                   
        self.aux=jnp.array(self.eig_vec)                                                                                               
        self.j1=self.param("j1", nn.initializers.normal(),(1,),float)    
    def __call__(self,x):                                                                                                              
        indices = change_to_int(x,self.L)                                                                                              
        A = [self.aux[idx] for idx in indices]                                                                                         
        return jnp.array(A) 

    
class JasShort(nn.Module):                                                                                                             
    @nn.compact                                                                                                                        
    def __call__(self,x):                                                                                                              
        j1=self.param("j1", nn.initializers.normal(),(1,),float)                                                                       
        j2=self.param("j2", nn.initializers.normal(),(1,),float)                                                                       
        ## Nearest neighbor correlations                                                                                               
        corr1=x*jnp.roll(x,-1,axis=-1)                                                                                                 
        corr2=x*jnp.roll(x,-2,axis=-1)                                                                                                 
        return jnp.sum(j1*corr1+j2*corr2,axis=-1)

class FFN(nn.Module):
    alpha : int = 1
    layers : int = 1
    @nn.compact
    def __call__(self, x):
        aux=x
        for i in range(self.layers):
            dense = nn.Dense(features=self.alpha * aux.shape[-1])
            aux_1=dense(aux)
            aux = nn.relu(aux_1)
        y=aux
        return jnp.sum(y, axis=-1)

class SymmModel(nn.Module):
    alpha : int = 1
    layers : int = 1
    L : int = 1
    W : int = 1
    @nn.compact
    
    def __call__(self, x):
        aux=x.reshape(-1,1,x.shape[-1])
        
        for i in range(self.layers):
            if self.W <= 1:
                A=[self.L]
            else:
                A=[self.L,self.W]
            graph=nk.graph.Grid(extent=A,pbc=True)
            aux_1 = nknn.DenseSymm(symmetries=graph.translation_group(),features=self.alpha,kernel_init=nn.initializers.normal(stddev=0.01))(aux)
            aux = nn.relu(aux_1)
        y=aux
        
        return jnp.sum(y, axis=(-1,-2))
    

class MODIFIED_RBM(nn.Module):
    r"""A restricted boltzman Machine, equivalent to a 2-layer FFNN with a
    nonlinear activation function in between.
    """
    phases:tuple
    hi: nk.hilbert.Spin
    param_dtype: Any = np.float64
    """The dtype of the weights."""
    activation: Any = nknn.log_cosh
    """The nonlinear activation function."""
    alpha: float | int = 1
    """feature density. Number of features equal to alpha * input.shape[-1]"""
    use_hidden_bias: bool = True
    """if True uses a bias in the dense layer (hidden layer bias)."""
    use_visible_bias: bool = True
    """if True adds a bias to the input not passed through the nonlinear layer."""
    precision: Any = None
    """numerical precision of the computation see :class:`jax.lax.Precision` for details."""
    L : float | int = 10
    inverse_ordering: Any|bool = False
    
    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the Dense layer matrix."""
    hidden_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the hidden bias."""
    visible_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the visible bias."""
    
    
    
    @nn.compact
    def __call__(self, input):
        x = nn.Dense(
            name="Dense",
            features=int(self.alpha * input.shape[-1]),
            param_dtype=self.param_dtype,
            precision=self.precision,
            use_bias=self.use_hidden_bias,
            kernel_init=self.kernel_init,
            bias_init=self.hidden_bias_init,
        )(input)
        x = self.activation(x)
        x = jnp.sum(x, axis=-1)
        indices=self.hi.states_to_numbers(input)
        angles=jnp.array(self.phases)[indices]
        x+=angles
        if self.use_visible_bias:
            v_bias = self.param(
                "visible_bias",
                self.visible_bias_init,
                (input.shape[-1],),
                self.param_dtype,
            )
            out_bias = jnp.dot(input, v_bias)
            return x + out_bias
        else:
            return x
