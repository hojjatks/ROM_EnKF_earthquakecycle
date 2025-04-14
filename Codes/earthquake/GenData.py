#%%
import numpy as np
import cte_eq
import sys
sys.path.append('./..')
import cte
import matplotlib.pyplot as plt
from ProcessFunctions import Ploteigs,ApplyPODStateSpace2D,GenRandom_ai,FindInitFromAi
from Forward2d import forwardmodel
import sys
index = int(sys.argv[1])
np.random.seed(index)  # seed is set using the index
import time
start_time = time.time()
#%%
# load the long simulation.
drs=0.012
T_final_load=10500
ApplyPOD=False # 1 if you want to apply POD, 0 if you want to load the data
Nxout=cte_eq.Nxout
Ntout=cte_eq.Ntout
T_filter=500 

#%%
if ApplyPOD:
    direct='/central/groups/astuart/hkaveh/Data/LearnROM/transfer/2DSim_MainSimulation_Tf'+str(T_final_load)+"Nx="+str(Nxout)+"Nt="+str(Ntout)+'drs'+str(drs)+".npz"
    data=np.load(direct)
    v=data['v']
    theta=data['theta']
    dc=data['dc']
    t=data['t']
    Nx=dc.shape[0]
    U,S,VT,q_bar=ApplyPODStateSpace2D(v,theta,t,T_filter,Nx)
    # saving U,S,VT,q_bar to later use in if ApplyPOD==0
    np.savez('/central/groups/astuart/hkaveh/Data/LearnROM/transfer/ApplyPOD_V_theta_together'+str(T_final_load)+"Nx="+str(Nxout)+"Nt="+str(Ntout)+'drs'+str(drs),U=U,S=S,VT=VT,q_bar=q_bar)
else:
    # load U,S,VT,q_bar
    data=np.load('/central/groups/astuart/hkaveh/Data/LearnROM/transfer/ApplyPOD_V_theta_together'+str(T_final_load)+"Nx="+str(Nxout)+"Nt="+str(Ntout)+'drs'+str(drs)+'.npz')
    U=data['U']
    S=data['S']
    VT=data['VT']
    q_bar=data['q_bar']
#%%
Nt2=VT.shape[0] # number of snapshots in the dataset after filtering first T_filter years
Nx=U.shape[0]//2
#%%
N_m=30
coeff=1
Specifyinit=True
T_final_run=350 # The interevent time is 34 years, how many events you want to include? I think 10 events is good
ai=GenRandom_ai(U,S,N_m,Nt2,coeff)
print(ai)
u_init=FindInitFromAi(ai,U,N_m,q_bar)
print(u_init.shape)
v0=10**(u_init[:Nx])
v0max=np.max(v0)
while v0max>2e1: # making sure the initial condition is somewhere that has less than 1e3 maximum slip rate
    ai=GenRandom_ai(U,S,N_m,Nt2,coeff)
    u_init=FindInitFromAi(ai,U,N_m,q_bar)
    print(u_init.shape)
    v0=10**(u_init[:Nx])
    v0max=np.max(v0)

#%%
p=forwardmodel(T_final_run,cte_eq.Ntout,cte_eq.Nxout,Specifyinit,u_init,drs)
#%%
direct='/central/groups/astuart/hkaveh/Data/LearnROM/transfer/SampleSimulation_Tf_2D'+str(T_final_run)+"Nt="+str(cte_eq.Ntout)+"N_m"+str(N_m)+"coeff"+str(coeff)+"number"+str(index)
# You only need time, V, theta:
x_ox=p.ox["x"].unique()
Nt=len(p.ox["v"])//(len(x_ox)) # Number of Snapshots
t_ox=p.ox["t"].values.reshape((Nt,len(x_ox)))
V_ox=p.ox["v"].values.reshape((Nt,len(x_ox)))
theta_ox=p.ox["theta"].values.reshape((Nt,len(x_ox)))
np.savez_compressed(direct+'.npz', array1=V_ox, array2=theta_ox, array3=t_ox)

end_time = time.time()
print(f"Total runtime: {(end_time - start_time)/60:.2f} minutes")