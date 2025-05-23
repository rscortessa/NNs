import numpy as np
import matplotlib.pyplot as plt
import Methods.class_WF as class_WF
import Methods.var_nk as var_nk
from scipy.sparse.linalg import eigsh    
import netket as nk
import sys
import os
import time
## CONSTANTS

eps=10**(-8)
dx=0.01
V=-1.0
t1=1
t2=5



## ALMOST CONSTANTS

n_neurons=1
n_layers=1


## PARAMETERS

parameters=sys.argv
n_par=len(parameters)
parameters=[int(parameters[x]) for x in range(1,n_par)]
n_par=len(parameters)
print(parameters,n_par)

L=parameters[0]
W=parameters[1]
Gamma=parameters[2]
n_samples=parameters[3]
n_between=parameters[4]
n_run=parameters[5] #-
n_mean=parameters[6] #-
n_method=parameters[7]
save_hiddens=False
modelo=parameters[8]
try:
    n_neurons=parameters[9]
    n_layers=parameters[10]
except:
    print("no additional parameters")

    


    
cutoff=2**L

methods=[var_nk.MF(),var_nk.JasShort(),var_nk.FFN(alpha=n_neurons,layers=n_layers),nk.models.RBM(alpha=n_neurons,param_dtype=complex),nk.models.RBM(alpha=n_neurons),var_nk.SymmModel(alpha=n_neurons,layers=n_layers,L=L,W=W)]
method_name=["MF_","JS_","FFN_","RBM_COMPLEX","RBM_REAL","SYMFFN_","EX"]


hi=nk.hilbert.Spin(s=1/2,N=L*W,constraint=class_WF.ParityConstraint())


models=[class_WF.IsingModel_Z(Gamma*dx,L,hi),class_WF.IsingModel_X(Gamma*dx,L,hi),class_WF.CLUSTER_HAM_X(Gamma*dx,L,hi),class_WF.CLUSTER_HAM_Z(Gamma*dx,L,hi),class_WF.CLUSTER_HAM_Y(Gamma*dx,L,hi)]
models_name=[r"QIM_Z",r"QIM_X",r"CIM_X",r"CIM_Z",r"CIM_Y",r"XYZ_"]
folder_name="P"+models_name[modelo]+method_name[n_method]+"NN"+str(n_neurons)+"NL"+str(n_layers)+"L"+str(L)+"W"+str(W)+"G"+str(Gamma)+"NS"+str(n_samples)+"NB"+str(n_between)

try:
    os.mkdir(folder_name)
except:
    print("The folder",folder_name,"already exists. All the files will be stored there")

name_var=["PDATAM","L","W","NS","NB","G","NN","NL"]
var=[n_method,L,W,n_samples,n_between,Gamma,n_neurons,n_layers]
name_var=name_var[:n_par-2]
var=var[:n_par-2]

    
filename_CONTROL="L"+str(L)+"W"+str(W)+"G"+str(Gamma)+"NS"+str(n_samples)+models_name[modelo]+method_name[n_method]+".txt"
file_CONTROL=open(folder_name+"/"+filename_CONTROL,"w")
file_CONTROL.write("RUNNING WITH"+"L="+str(L)+"W="+str(W)+"GAMMA="+str(Gamma)+"N samples="+str(n_samples)+"time for each snapshot(n_between)"+str(n_between)+"N_snapshots"+str(n_run)+"N_method="+str(method_name[n_method])+"model="+str(models_name[modelo]))
file_CONTROL.close()


#INITIALIZE OBJECTS
H=models[modelo]
#CONDITION IF EXACT
if n_method==5:
    eig_vals,eig_vecs=class_WF.Diag(H,True)
    methods.append(var_nk.EWF(eig_vec=tuple(np.abs(eig_vecs[:,0])),L=L*W))
    print(eig_vals)
    quit()
model=methods[n_method]
E_WF=class_WF.WF(L*W,model,H,n_samples,constraint=class_WF.ParityConstraint())

name_var[0]=folder_name+"/"+"M"
pubE=class_WF.publisher(name_var+["NR"],var+[n_run],["NS","E"])
pubE.create()
name_var[0]=folder_name+"/"+"DATAM"
E=np.array([[0.0 for i in range(n_mean)] for gg in range(n_run)],dtype=float)

#ITERATION OVER THE GAMMA VALUES:

G=Gamma
#H=class_WF.Ham(G*dx,V,L,hi)




    

for hh in range(n_mean):
    print(hh)
    if hh>0:
        file_CONTROL=open(folder_name+"/"+filename_CONTROL,"a")
        #file_CONTROL.write("iter_num="+str(hh)+" time="+str(bb-aa)+"\n")
        file_CONTROL.write("E"+"\t"+"STEPS"+"\n")
        file_CONTROL.close()
    aa=time.time()
    E_WF.user_state.init_parameters()
    for steps in range(n_run):

        var[4]=n_between*(1+steps)

        if n_method != 5:
            E_WF.advance(n_between)
            A=E_WF.sampling()    
            #if n_method==0 or n_method==2:
            #    lenght=len(A)
            #    A[:int(lenght/2),:]=(-1)*A[:int(lenght/2),:]
        file_CONTROL=open(folder_name+"/"+filename_CONTROL,"a")
        file_CONTROL.write(str(E_WF.compute_E())+"\t"+str(steps)+"\n")
        file_CONTROL.close()
        pub=class_WF.publisher(name_var,var,[])
        pub.create()
        A=E_WF.sampling()
        for x in range(len(A)):
            pub.write(A[x])
        #namefile=pub.name()
        pub.close()
        #os.rename(namefile,folder_name+"/"+namefile+str(hh))

        #THIS PART COMPUTES AND STORES THE HIDDEN VARIABLES

        name_var[0]=folder_name+"/"+"VARM"
        pubvar=class_WF.publisher(name_var,var,[])
        pubvar.create()
        if n_method == 3 and save_hiddens:        
            kernel=E_WF.user_state.variables["params"]["Dense"]["kernel"]
            bias=E_WF.user_state.variables["params"]["Dense"]["bias"]
            visible_bias=E_WF.user_state.variables["params"]["visible_bias"]
            Test=A@kernel
            new_bias=np.array([bias for nns in range(n_samples)])
            Test=1/(1+np.exp(-1.0*(Test+new_bias)))
            for ns in range(n_samples):
                pubvar.write(Test[ns].tolist())
                
        #namefile=pubvar.name()    
        pubvar.close()
        #os.rename(namefile,folder_name+"/"+namefile+str(hh))
        name_var[0]=folder_name+"/"+"DATAM"

        #-------------------------------------------------

        if n_method==5:
            eig_vecs=class_WF.Diag(H)
            model=var_nk.EWF(eig_vec=tuple(np.abs(eig_vecs[:,0])),L=L*W)
            AA=nk.vqs.MCState(E_WF.user_sampler,model,n_samples=n_samples)
            E_WF.change_state(AA)

        E[steps][hh]=E_WF.compute_E()

        bb=time.time()
        
En=np.mean(E,axis=-1)
dEn=np.std(E,axis=-1)

for gg in range(n_run):
    pubE.write([0,n_between*(1+gg),dEn[gg],En[gg]])
#filename=pubE.name()
pubE.close()
#os.rename(filename,folder_name+"/"+filename)

file_CONTROL=open(folder_name+"/"+filename_CONTROL,"a")
#file_CONTROL.write("FINISHED")
file_CONTROL.close()




