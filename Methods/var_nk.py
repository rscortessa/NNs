import jax
import flax
import numpy as np
import netket as nk
from typing import Any
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
import netket.nn as nknn
from jax.nn.initializers import normal,zeros
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



class Deep1DCNN(nn.Module):
    # You can pass a list of features, e.g., (8, 16, 8)
    layer_features: tuple = (8, 8) 
    kernel_size: int = 2
    param_dtype: any = jnp.complex64
    padding: str = "SAME"
    @nn.compact
    def __call__(self, x):
        # 1. Reshape Input
        x = x.reshape(x.shape[0], -1, 1)

        # 2. Dynamic Layer Loop
        # We loop through all feature sizes provided in the list
        for feat in self.layer_features:
            x = nn.Conv(features=feat, 
                        kernel_size=(self.kernel_size,), 
                        padding=self.padding, # Keeps size constant
                        param_dtype=self.param_dtype)(x)
            
            # Non-linearity between layers
            x = nk.nn.log_cosh(x)

        return jnp.sum(x, axis=(-1, -2))
    


class SpinDependent_TransformerWaveFunction(nn.Module):
    """
    Transformer Wave Function with Input-Dependent (Dynamic) Attention.
    
    This architecture computes attention weights alpha_ij based on the 
    spin configuration itself (Self-Attention), allowing the model to 
    dynamically decide which patches are correlated.
    """
    n_heads: int            # Number of attention heads
    head_dim: int           # Dimension of each head
    n_patches: int          # Number of patches
    
    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: Input spins with shape (batch, n_patches, patch_size)
        
        Returns:
            log_psi: Complex scalar wave function amplitude (batch,)
        """
        batch_size = x.shape[0]

        try:
            x = x.reshape((x.shape[0],self.n_patches,x.shape[1]//self.n_patches))
        except:
            print("ERROR, number of patches does not agree with total dimension")
            
        # Total feature dimension = (number of heads) * (dimension per head)
        total_dim = self.n_heads * self.head_dim
        
        # Define initializer: Small complex random numbers to break symmetry
        init_fn = normal(stddev=0.01, dtype=jnp.complex128)
        
        # ---------------------------------------------------------
        # 1. Q, K, V Projections (The Core of Self-Attention)
        # ---------------------------------------------------------
        # We project the input 'x' into three distinct complex vectors per patch.
        
        # Queries (Q): What patch 'i' is looking for.
       
        q = nn.Dense(features=total_dim, use_bias=False, param_dtype=jnp.complex128, kernel_init=init_fn, name="Q")(x)
       
        # Keys (K): What patch 'j' announces about its content.

        k = nn.Dense(features=total_dim, use_bias=False, param_dtype=jnp.complex128, kernel_init=init_fn, name="K")(x)
        
        # Values (V): The actual content/features to be transported.

        v = nn.Dense(features=total_dim, use_bias=False, param_dtype=jnp.complex128, kernel_init=init_fn, name="V")(x)

        # ---------------------------------------------------------
        # 2. Multi-Head Splitting
        # ---------------------------------------------------------
        # Reshape (batch, patches, total_dim) -> (batch, patches, heads, head_dim)
        # This isolates the heads so they can operate independently.
        
        q = q.reshape(batch_size, self.n_patches, self.n_heads, self.head_dim)
        k = k.reshape(batch_size, self.n_patches, self.n_heads, self.head_dim)
        v = v.reshape(batch_size, self.n_patches, self.n_heads, self.head_dim)

        # ---------------------------------------------------------
        # 3. Calculate Attention Scores (Scaled Dot Product)
        # ---------------------------------------------------------
        # Calculate the raw similarity between Queries and Keys.
        # We perform a complex dot product: Q_i * conjugate(K_j).
        #
        # Einsum indices:
        # b: batch
        # i: target patch index (Query)
        # j: source patch index (Key)
        # h: head index
        # d: feature dimension (summed over)
        # Output shape: (batch, heads, i, j)
        
        # Note: We conjugate 'k' here. (Q . K*) is the standard inner product.
        score_complex = jnp.einsum('bihd, bjhd -> bhij', q, jnp.conj(k))
        
        # CRITICAL STEP: 
        # 1. Take the REAL part (Probabilities must be real).
        # 2. Scale by 1/sqrt(d) (Standard Transformer scaling for stability).
        score_real = jnp.real(score_complex) / jnp.sqrt(self.head_dim)
        
        # Apply Softmax to get probabilities that sum to 1 over the 'j' dimension.
        # alpha shape: (batch, heads, n_patches, n_patches)
        alpha = nn.softmax(score_real, axis=-1)

        # ---------------------------------------------------------
        # 4. Apply Attention (Weighted Sum)
        # ---------------------------------------------------------
        # Create the new patch representation by summing Values weighted by Alpha.
        # A_i = sum_j (alpha_ij * v_j)
        #
        # indices:
        # b: batch, h: head, i: target, j: source, d: dim
        # alpha: (b, h, i, j)
        # v:     (b, j, h, d)
        # output: (b, i, h, d)
        attn_out = jnp.einsum('bhij, bjhd -> bihd', alpha, v)

        # ---------------------------------------------------------
        # 5. Concatenation & Linear Mixing
        # ---------------------------------------------------------
        # Flatten the heads back into a single vector (Concatenation)
        # Shape: (batch, n_patches, total_dim)
        attn_flat = attn_out.reshape(batch_size, self.n_patches, total_dim)

        
        
        # Apply Output Projection (W_O) to mix info between heads.
        # We add the bias here.
        mixed_out = nn.Dense(
            features=total_dim,
            use_bias=True,
            param_dtype=jnp.complex128,
            kernel_init=init_fn,
            name="OutputProjection"
        )(attn_flat)

        # ---------------------------------------------------------
        # 6. Nonlinearity & Final Sum
        # ---------------------------------------------------------
        # Apply ln(cosh(z)) element-wise
        activation = nk.nn.log_cosh(mixed_out)
        
        # Sum over all patches and feature dimensions to get the log amplitude
        log_psi = jnp.sum(activation, axis=(-2, -1))
        
        return log_psi

class parity_Deep1DCNN(nn.Module):
    layer_features: tuple = (8, 8)
    kernel_size: int = 2
    param_dtype: any = jnp.complex64
    padding: str = "SAME"

    @nn.compact
    def __call__(self, x):
        # 1. Parity Check (Using Modulo and jnp.where)
        # Assuming we want to penalize states with an ODD number of 1s
        num_ones = jnp.sum(x == 1, axis=-1)
        is_odd = (num_ones % 2) == 1
        
        # Safely assign -inf to odd parity, 0.0 to even parity
        parity_penalty = jnp.where(is_odd, -jnp.inf, 0.0)

        # 2. Reshape Input
        x_reshaped = x.reshape(x.shape[0], -1, 1)

        # 3. Dynamic Layer Loop                                                                                                                                                                                                                                                         
        for feat in self.layer_features:
            x_reshaped = nn.Conv(
                features=feat,
                kernel_size=(self.kernel_size,),
                padding=self.padding,                                                                                                                                                                           
                param_dtype=self.param_dtype
            )(x_reshaped)

            x_reshaped = nk.nn.log_cosh(x_reshaped)

        # Add penalty at the end
        return jnp.sum(x_reshaped, axis=(-1, -2)) + parity_penalty

class Factored_Attention_TransformerWaveFunction(nn.Module):
    """
    Transformer-based Variational Wave Function.
    
    Architecture:
    Input -> V (Dense) -> Split Heads -> Attention (alpha) -> 
    Concat Heads -> Linear Mixing (W_O) -> LogCosh -> Sum
    """
    n_heads: int            # Number of attention heads
    head_dim: int           # Dimension of each head
    n_patches: int          # Number of patches in the lattice
    
    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: Input spins with shape (batch, n_patches, patch_size)
               (Ensure your graph partitioning logic produces this shape)
        
        Returns:
            log_psi: Complex scalar wave function amplitude (batch,)
        """
        batch_size = x.shape[0]

        try:
            x = x.reshape((x.shape[0],self.n_patches,x.shape[1]//self.n_patches))
        except:
            print("ERROR, number of patches does not agree with total dimension")
        
        # ---------------------------------------------------------
        # 1. Feature Extraction (The V Matrix)
        # ---------------------------------------------------------
        # We project the raw spin patch into a complex feature vector.
        # Total dimension = n_heads * head_dim
        # Output shape: (batch, n_patches, n_heads * head_dim)
        total_dim = self.n_heads * self.head_dim
        
        v_out = nn.Dense(
            features=total_dim,
            use_bias=False,             # Bias usually handled later
            param_dtype=jnp.complex128, # Must be complex for quantum phases
            kernel_init=default_kernel_init,
            name="V"
        )(x)
        
        # ---------------------------------------------------------
        # 2. Multi-Head Splitting
        # ---------------------------------------------------------
        # Reshape to separate the heads.
        # Shape: (batch, n_patches, n_heads, head_dim)
        v_reshaped = v_out.reshape(batch_size, self.n_patches, self.n_heads, self.head_dim)

        # ---------------------------------------------------------
        # 3. Spatial Correlations (The Alpha Matrix)
        # ---------------------------------------------------------
        # Learnable parameters for attention weights.
        # Shape: (n_heads, n_patches, n_patches)
        # We use a real-valued matrix for the interaction strengths.
        attn_logits = self.param(
            'alpha_logits',
            zeros,
            (self.n_heads, self.n_patches, self.n_patches),
            jnp.complex128
        )+0j
        
        # Apply Softmax along the last dimension (j) so sum_j(alpha_ij) = 1
        alpha = nn.softmax(jnp.real(attn_logits), axis=-1)

        # ---------------------------------------------------------
        # 4. Attention Application
        # ---------------------------------------------------------
        # Calculate A_i^mu = sum_j alpha_{ij}^mu * v_j^mu
        # h: heads, i: target patch, j: source patch, b: batch, d: head_dim
        #
        # alpha shape: (h, i, j)
        # v shape:     (b, j, h, d)
        # output:      (b, i, h, d)
        attn_out = jnp.einsum('hij, bjhd -> bihd', alpha, v_reshaped)

        # ---------------------------------------------------------
        # 5. Concatenation & Linear Mixing
        # ---------------------------------------------------------
        # Flatten the heads back together (Concatenation)
        # Shape: (batch, n_patches, n_heads * head_dim)
        attn_flat = attn_out.reshape(batch_size, self.n_patches, total_dim)
        
        

        # Apply the Output Projection (W_O) to mix the heads
        # This is the "Linear Combination" step before nonlinearity.
        mixed_out = nn.Dense(
            features=total_dim,         # Maintain the feature size
            use_bias=True,              # Add bias here (complex)
            param_dtype=jnp.complex128,
            kernel_init=normal(stddev=0.01),
            name="OutputProjection"
        )(attn_flat)

        # ---------------------------------------------------------
        # 6. Nonlinearity & Output
        # ---------------------------------------------------------
        # Apply ln(cosh(z)). NetKet's implementation is numerically stable.
        activation = nk.nn.log_cosh(mixed_out)
        
        # Sum over all patches and feature dimensions to get log_psi
        # Shape: (batch,)
        log_psi = jnp.sum(activation, axis=(-2, -1))
        
        return log_psi
