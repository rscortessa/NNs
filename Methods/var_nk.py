import netket as nk
import jax.numpy as jnp
import flax
import jax
import flax.linen as nn
import numpy as np
import netket.nn as nknn

def change_to_int(x,L):
    Aux=jnp.array([2**(L-1-i) for i in range(L)])
    Z=jnp.array(jnp.mod(1+x,3)/2,int)
    return np.sum(Aux*Z,axis=-1)


class MF(nn.Module):                                                                                                                   
    @nn.compact # What is this ?                                                                                                       
    def __call__(self,x): #x.shape(Nsamples,L)                                                                                         
        lam= self.param("lambda", nn.initializers.normal(),(1,),float)                                                                 
        p= nn.log_sigmoid(lam*x) ## How does the initializers work?                                                                    
        return 0.5*jnp.sum(p,axis=-1)                                                                                                  
                                                                                                                                       
class EWF(nn.Module):                                                                                                                  
    eig_vec:tuple                                                                                                                      
    L:float                                                                                                                            
    def setup(self):                                                                                                                   
        self.aux=jnp.array(self.eig_vec)                                                                                               
        self.j1=self.param("j1", nn.initializers.normal(),(1,),float)                                                                  
                                                                                                                                       
    def __call__(self,x):                                                                                                              
        indices = change_to_int(x,self.L)                                                                                              
        A = [self.aux[idx] for idx in indices]                                                                                         
        return jnp.log(jnp.array(A))                                                                                                   
                                                                                                                                       
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
    
    
