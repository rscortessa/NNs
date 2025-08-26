import netket as nk
import numpy as np
import os
import json
import jax.numpy as jnp
def GET_PROB_RBM(hi,param_RBM,j):
    
    #DEFINE THE PARAMETERS OF THE RBM
    AA=np.array(param_RBM["params"]["Dense"]["kernel"]["value"]["real"][j])+1j*np.array(param_RBM["params"]["Dense"]["kernel"]["value"]["imag"][j])
    BB=np.array(param_RBM["params"]["Dense"]["bias"]["value"]["real"][j])+1j*np.array(param_RBM["params"]["Dense"]["bias"]["value"]["imag"][j])
    CC=np.array(param_RBM["params"]["visible_bias"]["value"]["real"][j])+1j*np.array(param_RBM["params"]["visible_bias"]["value"]["imag"][j])
    
    #DEFINE THE STATES
    states=hi.all_states()
    
    #AUXILIAR MATRIX
    DD=np.tile(BB,(len(states),1))
    
    #COMPUTE THE PROBABILITIES
    
    logKK=states@CC
    log_AMP=np.log(np.cosh(states@AA+DD))
    log_ALMOST_PROB=np.sum(log_AMP,axis=-1)+logKK
    log_NORM=log_ALMOST_PROB+np.conjugate(log_ALMOST_PROB)
    NORM=np.sqrt(np.sum(np.exp(log_NORM)))
    
    PROB=np.exp(log_ALMOST_PROB)/NORM
    
    return PROB

def name_files(basis,modelo,L,G,NN,NL,NR,Nangle,NSPCA,NM,DS):
    adder="SHIFT"+str(DS)
    MASTER_DIR="FULL_STATE_RUN_"+basis+"_"+modelo+"NN"+str(NN)+"L"+str(L)+"G"+str(G)+"NA"+str(Nangle)+"NSPCA"+str(NSPCA)+adder
    if os.path.isdir(MASTER_DIR+" "):
        print("DIRECTORY NOT FOUND")
    else:
        print("DIRECTORY ALREADY CREATED")
    OBS_FILENAME="NANGLE"+str(Nangle)+basis+"M3L"+str(L)+"W1"+"G"+str(G)+"NN"+str(NN)+"NL"+str(NL)+"NR"+str(NR)+"OBS"
    SPCA_FILENAME="NANGLE"+str(Nangle)+basis+"M3L"+str(L)+"W1"+"G"+str(G)+"NN"+str(NN)+"NL"+str(NL)+"NR"+str(NR)+"SPCA"
    VAR_FILENAME="NANGLE"+str(Nangle)+basis+"M3L"+str(L)+"W1"+"G"+str(G)+"NN"+str(NN)+"NL"+str(NL)+"NR"+str(NR)+"VAR"
    return MASTER_DIR,OBS_FILENAME,SPCA_FILENAME,VAR_FILENAME

def import_data_obs(MASTER_DIR,OBS_FILENAME,obs,NR,NM,Nangle):
    #How much data I want for the average
    T_CORR=1
    N_SAMPLES=1
    
    E=[[[1.0 for k in range(NR)]  for i in range(Nangle+1)] for rep in range(NM)]
    dE=[[[0.0 for k in range(NR)] for i in range(Nangle+1)] for rep in range(NM)]
    data_RBM=[[[1.0 for k in range(NR)] for i in range(Nangle+1)] for rep in range(NM)]
    AA=[i for i in range(Nangle+1)]
    NMM=[[kk for kk in range(NM)] for i in range(Nangle+1)]

    for rep in range(NM):
        for ii in range(Nangle+1):
            with open(MASTER_DIR+"/"+str(rep)+"NM"+str(ii)+OBS_FILENAME+".json", "r") as f:
                data_RBM[rep][ii] = json.load(f)
            E[rep][ii]=np.real(data_RBM[rep][ii][obs]["Mean"]["real"])
            dE[rep][ii]=np.real(data_RBM[rep][ii][obs]["Variance"])
    
    E=np.array(E)
    
    dE=np.array(dE)
    
    E_mean_RBM=[]
    dE=[]
    NN=np.zeros(Nangle+1)

    for i in range(Nangle+1):
        E_mean_RBM.append(np.mean(E[NMM[i],i,:],axis=0))
        dE.append(np.var(E[NMM[i],i,:],axis=0))
        NN[i]=len(NMM[i])
    E_mean_RBM=np.array(E_mean_RBM)
    dE=np.array(dE)

    A=E.shape
    NRR=A[-1]
    X_INDICES=[NRR-1-T_CORR*kk for kk in range(N_SAMPLES)]
    EY=[]
    Y_err=[]
    
    for ii in range(Nangle+1):
        EY.append(np.mean(E_mean_RBM[ii,X_INDICES]))
        Y_err.append(dE[ii,NR-1])
    EY=np.array(EY)
    Y_err=np.array(Y_err)
    #print(EY,Y_err)
    return EY, Y_err

def import_data_obs_min(MASTER_DIR,OBS_FILENAME,obs,NR,NM,Nangle):
    #How much data I want for the average
   
    E=[[[1.0 for k in range(NR)]  for i in range(Nangle+1)] for rep in range(NM)]
    dE=[[[0.0 for k in range(NR)] for i in range(Nangle+1)] for rep in range(NM)]
    data_RBM=[[[1.0 for k in range(NR)] for i in range(Nangle+1)] for rep in range(NM)]
    AA=[i for i in range(Nangle+1)]
    NMM=[[kk for kk in range(NM)] for i in range(Nangle+1)]

    for rep in range(NM):
        for ii in range(Nangle+1):
            with open(MASTER_DIR+"/"+str(rep)+"NM"+str(ii)+OBS_FILENAME+".json", "r") as f:
                data_RBM[rep][ii] = json.load(f)
            E[rep][ii]=np.real(data_RBM[rep][ii][obs]["Mean"]["real"])
            dE[rep][ii]=np.real(data_RBM[rep][ii][obs]["Variance"])
    
    E=np.array(E)    
    dE=np.array(dE)
    A=E.shape
    NRR=A[-1]
    minis=[0 for i in range(Nangle+1)]
    E_RBM=np.array([[0.0 for i in range(NRR)] for i in range(Nangle+1)])
    dE_RBM=np.array([[0.0 for i in range(NRR)] for i in range(Nangle+1)])
    for ii in range(Nangle+1):
        mini=0
        for rep in range(NM):
            if E[rep,ii,NRR-1]<=E[mini,ii,NRR-1]:
                mini=rep
                E_RBM[ii,:]=E[mini,ii,:]
                dE_RBM[ii,:]=dE[mini,ii,:]
                minis[ii]=mini
                
    return E_RBM[:,NRR-1],dE_RBM[:,NRR-1],minis


def import_data_WF(MASTER_DIR,VAR_FILENAME,NR,NM,Nangle,L,NSPCA,hi):
    data=[[None for i in range(Nangle+1)] for rep in range(NM)]
    PSI_RBM=np.zeros((2**L,Nangle+1,NM),dtype=complex)
    for rep in range(NM):
        for ii in range(Nangle+1):
            with open(MASTER_DIR+"/"+str(rep)+"NM"+str(ii)+VAR_FILENAME+".json", "r") as f:
                data[rep][ii] = json.load(f)
            PSI_RBM[:,ii,rep]=GET_PROB_RBM(hi,data[rep][ii],NSPCA-1)
    return PSI_RBM

def import_net_parameters(MASTER_DIR,VAR_FILENAME,rep,ii,L,hi):
    with open(MASTER_DIR+"/"+str(rep)+"NM"+str(ii)+VAR_FILENAME+".json", "r") as f:
                data  = json.load(f)
    return data

def vstate_par(data_var,it,inverted_ordering=False):
    
    dense_bias_R=np.array(data_var["params"]["Dense"]["bias"]["value"]["real"][it])
    dense_bias_I=np.array(data_var["params"]["Dense"]["bias"]["value"]["imag"][it])
    dense_bias=np.array(dense_bias_R+1j*dense_bias_I,dtype=np.complex128)

    visible_bias_R=np.array(data_var["params"]["visible_bias"]["value"]["real"][it])
    visible_bias_I=np.array(data_var["params"]["visible_bias"]["value"]["imag"][it])
    visible_bias=np.array(visible_bias_R+1j*visible_bias_I,dtype=np.complex128)

    dense_kernel_R=np.array(data_var["params"]["Dense"]["kernel"]["value"]["real"][it])
    dense_kernel_I=np.array(data_var["params"]["Dense"]["kernel"]["value"]["imag"][it])
    dense_kernel=np.array(dense_kernel_R+1j*dense_kernel_I,dtype=np.complex128)

    new_parameters={
        'params': {
            'Dense': {
                'bias': dense_bias,
                'kernel':dense_kernel
            },
            'visible_bias':visible_bias      
        }
    }

    #hi=nk.hilbert.Spin(s=1/2,N=L,inverted_ordering = inverted_ordering)
    #model=nk.models.RBM(alpha=NN,param_dtype=complex)
    #vstate = nk.vqs.FullSumState(hi, model=model,variables=new_parameters)
    return new_parameters

def to_probabilities(hi,model, parameters):
    # begin by generating all configurations in the hilbert space.
    all_configurations = hi.all_states()

    # now evaluate the model, and convert to a normalised wavefunction.
    logpsi = model.apply(parameters, all_configurations)
    psi = jnp.exp(logpsi)
    psi = psi / jnp.linalg.norm(psi)
    return psi
