#%%
# This code is used to generate raw data for ROM.
#%%
import numpy as np
import cte
import sys
import matplotlib.pyplot as plt
sys.path.append('/central/groups/astuart/hkaveh/QDYN/qdyn-read-only/src')  # For pyqdyn
from pyqdyn import qdyn
from ProcessFunctions import ApplyPODStateSpace,GrabData,PltPODModesVorthetav2,SaveAsPickle,GenRandom_ai,FindInitFromAi,ApplyPODV,PltPODmodesV,ApplyPODtheta,Grab_SaveData
import pickle

#%%
def forwardmodelnorthCascadia(a_VW,b_VW,dc,W_asp=25000):
    L_asp=cte.L_asp_north
    L=cte.L_asp_north+cte.L_buffer    
    stringadditional="north"
    p=forwardmodel(a_VW,b_VW,dc,W_asp,L,L_asp,stringadditional)


def forwardmodelsmallsystem(a_VW,b_VW,dc,W_asp=25000):
    L_asp=cte.L_asp
    L=cte.L_asp_north+cte.L_buffer    
    stringadditional="smallsystem"
    p=forwardmodel(a_VW,b_VW,dc,W_asp,L,L_asp,stringadditional)



    
def forwardmodel(a_VW,b_VW,dc,W_asp,L,L_asp,stringadditional,T_final,u_init,Specifyinit):
        #% Specifying parameters that are always constant
    p = qdyn()
    set_dict=p.set_dict
    set_dict["MESHDIM"] = 2     # Simulation dimensionality (2D fault in 3D medium)
    set_dict["FAULT_TYPE"]= 2   # Thrust fault
    set_dict["TMAX"]= T_final*cte.t_yr # Maximum simulation time
    set_dict["NTOUT"]= cte.Ntout     # Save outputs every N steps
    set_dict["NXOUT"]=cte.Nxout         # Snapshot resolution along-strike
    set_dict["NWOUT"]=cte.Nwout         # Snapshot resolution along-dip
    set_dict["V_PL"]=cte.V_PL     # Plate velocity
    set_dict["MU"]=cte.G         # Shear modulus
    set_dict["SIGMA"]=cte.sigma       # Effective Normal stress
    set_dict["ACC"]=1e-7        # Solver Accuracy
    set_dict["SOLVER"]=2        # Solver type (Runge-Kutta)
    set_dict["Z_CORNER"] = -cte.W    # Base of the fault (depth taken <0); NOTE: Z_CORNER must be < -W !
    set_dict["DIP_W"]=cte.dipangle

    # Setting some (default) RSF parameter values
    set_dict["SET_DICT_RSF"]["A"] = cte.a_VS    # Direct effect (will NOT be overwritten later)
    set_dict["SET_DICT_RSF"]["B"] = cte.b_VS    # Evolution effect (will be overwritten later)
    set_dict["SET_DICT_RSF"]["V_SS"] = 1e-6   # Reference velocity [m/s]*
    set_dict["SET_DICT_RSF"]["V_0"] = set_dict["V_PL"]     # Initial velocity [m/s]
    set_dict["SET_DICT_RSF"]["TH_0"] = 0.99 * set_dict["SET_DICT_RSF"]["DC"] / set_dict["V_PL"]    # Initial (steady-)state [s]
    set_dict["SET_DICT_RSF"]["DC"] = dc     # Characteristic slip distance
    Lb = set_dict["MU"] * set_dict["SET_DICT_RSF"]["DC"] / (b_VW* set_dict["SIGMA"])
        # Nucleation length [m]
    Lc = set_dict["MU"] * set_dict["SET_DICT_RSF"]["DC"] / ((b_VW - a_VW) * set_dict["SIGMA"])
    pi1=(b_VW-a_VW)/b_VW
    pi2=dc/W_asp
    pi3=(b_VW-a_VW)*cte.sigma/cte.G
        # These parameters are not in QDYN I just need to record them for post process
    set_dict["a_VW"]=a_VW
    set_dict["b_VW"]=b_VW
    set_dict["pi1"]=pi1
    set_dict["pi2"]=pi2
    set_dict["pi3"]=pi3

        # print("pi1="+str(pi1))
        # print("pi2="+str(pi2))
        # print("pi3="+str(pi3))
        # print("h_rr/W="+str(np.pi/4*pi2/pi3))
        # print(f"Process zone size: {Lb} m \t Nucleation length: {Lc} m")
        # print("counter is "+str(counter))
    Nx = int(np.power(2, np.ceil(np.log2(cte.resolution * L / Lb))))
    Nw = int(np.power(2, np.ceil(np.log2(cte.resolution * cte.W / Lb))))


    x = np.linspace(-L/2, L/2, Nx, dtype=float)
    z = np.linspace(-cte.W/2, cte.W/2, Nw, dtype=float)
    X, Z = np.meshgrid(x, z)

        # Set mesh size and fault length
    set_dict["NX"] = Nx
    set_dict["NW"] = Nw
    set_dict["L"] = L
    set_dict["W"] = cte.W 
    set_dict["DW"] = cte.W / Nw
# Set time series output node to the middle of the fault
    set_dict["IC"] = Nx * (Nw // 2) + Nx // 2


    """ Step 2: Set (default) parameter values and generate mesh """
    p.settings(set_dict)
    p.render_mesh()
    nx=int((L_asp/L)*Nx) # Number of elements along-strike for VW region
    nw=int((W_asp/cte.W)*Nw) # Number of elements along the dip for VS region


    x_VW=np.linspace(-L_asp/2, L_asp/2, nx, dtype=float)
    z_VW=np.linspace(-W_asp/2, W_asp/2, nw, dtype=float)
    X_VW, Z_VW = np.meshgrid(x_VW, z_VW)


    B=np.ones((Nw,Nx))*cte.b_VS  
    A=np.ones((Nw,Nx))*cte.a_VS
    A[Nw//2-nw//2:Nw//2+nw//2,Nx//2-nx//2:Nx//2+nx//2]=a_VW  # a=0.004 in all of the pickle files whose a's are not specified.
    B[Nw//2-nw//2:Nw//2+nw//2,Nx//2-nx//2:Nx//2+nx//2]=b_VW  # a=0.004 in all of the pickle files whose a's are not specified.
        

    
    p.mesh_dict["A"] = A.ravel()
    p.mesh_dict["B"] = B.ravel()
    if Specifyinit==True:
        N=Nx*Nw*2
        plt.figure()
        plt.imshow(u_init[:N//2].reshape(32,256))
        V_0=10**(u_init[:N//2])
        TH_0=10**(u_init[N//2:])
        p.mesh_dict["TH_0"]=TH_0.ravel()
        p.mesh_dict["V_0"]=V_0.ravel()
        plt.show()
    else:
        TH_0=np.ones((Nw,Nx))*0.99 * set_dict["SET_DICT_RSF"]["DC"] / set_dict["V_PL"]    # Initial (steady-)state [s]
        TH_0[Nw//2-nw//2:Nw//2+nw//2,Nx//2-nx//8:Nx//2]=0.8 * set_dict["SET_DICT_RSF"]["DC"] / set_dict["V_PL"]    # Initial (steady-)state [s]
        p.mesh_dict["TH_0"]=TH_0.ravel()
    p.write_input()
    p.run()
        #%
    p.read_output(read_ot=True, read_ox=True)    
    string=f"Optimizing_SSE_pi1={pi1:.8f}_pi2={pi2:.8f}_pi3={pi3:.8f}"+stringadditional
    #% Plotting maximum slip rate
    plt.figure(figsize=(4, 4))

    fig = plt.figure(figsize=(7.4, 3))
    font_size=7
    plt.rc('font',family='Serif',size=font_size)
    plt.rcParams.update({'font.family':'Serif', 'font.size': font_size})
# Slip rate

    time = p.ot_vmax["t"]  # Vector of simulation time steps
    Vmax = p.ot_vmax["v"]  # Vector of maximum slip velocity
    
    ax1=fig.add_subplot(1,1,1)
    ax1.plot(time/ cte.t_yr, Vmax)
    ax1.set_xlabel("t [years]")
    ax1.set_ylabel("Vmax [m/s]")
    ax1.set_yscale("log")
    string2=f"pi/4*PI2/PI3={pi2/pi3*3.14/4:.3f}"+f" and pi1 is {pi1:.8f}, pi2 is {pi2:.8f}, and pi3 is {pi3:.8f}"
    ax1.set_title(string+string2)
# Shear stress
    ax1.axhline(y=cte.V_thresh, color='black', linestyle='dashed')
        #% Saving
    directory="./../Data/"+string+".pickle"
    # SaveAsPickle(p,directory)

    return p


#%%
sample=False # If true, it generates random initial conditions and solve the equations for those initial conditions
CalPOD=False # If true, it calculates the POD modes
a_VW,b_VW,dc=[0.004,0.014,0.045]
W_asp=25000
POD_on_V=0
L_asp=cte.L_asp
L=cte.L_asp+cte.L_buffer    
stringadditional="smallsystem"
T_final=600
Specifyinit=False
u_init=0
p=forwardmodel(a_VW,b_VW,dc,W_asp,L,L_asp,stringadditional,T_final,u_init,Specifyinit)
direct='/central/groups/astuart/hkaveh/Data/LearnROM/transfer/MainSimulation_Tf'+str(T_final)+"Nt="+str(cte.Ntout)
#SaveAsPickle(p,direct)
Grab_SaveData(p,direct)

#%% Here I save the data:
# 
savegeom=0
if savegeom==1:
    Nx=p.set_dict["NX"]
    Nw=p.set_dict["NW"]
    np.savez("meshdict.npz",a=p.mesh_dict["A"].reshape(Nw,Nx),b=p.mesh_dict["B"].reshape(Nw,Nx),drs=p.mesh_dict["DC"].reshape(Nw,Nx),f0=p.set_dict["SET_DICT_RSF"]["MU_SS"],V0=p.set_dict["SET_DICT_RSF"]["V_SS"],sigma=p.set_dict["SIGMA"])
# finding stress from slip rate and theta and compare the time sreis with the qdyn stress
# reshape a,b:
    a_array=p.mesh_dict["A"].reshape(Nw,Nx)
    b_array=p.mesh_dict["B"].reshape(Nw,Nx)
    drs=p.mesh_dict["DC"].reshape(Nw,Nx)
    f0=p.set_dict["SET_DICT_RSF"]["MU_SS"]
    v0=p.set_dict["SET_DICT_RSF"]["V_SS"]
    sigma=p.set_dict["SIGMA"]
    Nt = len(p.ox["v"]) // (Nx* Nw)
    vel=p.ox["v"].values.reshape((Nt, Nw, Nx))
    theta=p.ox["theta"].values.reshape((Nt, Nw, Nx))
    
    stress_qdyn=p.ox["tau"].values.reshape((Nt, Nw, Nx))
    stress=sigma*(f0+a_array*np.log(vel/v0)+b_array*np.log(v0*theta/drs))
    # difference:
    error=stress-stress_qdyn

#%% Plotting the vmax vs dt
    
W=p.set_dict["W"]
t_ox,x_ox,z_ox,V_ox,vmin,vmax,ax1_xlim,ax1_ylim,theta_ox=GrabData(p,L,W,cte.t_yr)
Time=t_ox[:,0,0]
dTime=Time[1:]-Time[:-1]
V_ox_max=np.max(V_ox,axis=(1,2))
# plt.scatter(np.log10(V_ox_max[100:-1]),np.log10(dTime[100:])) 
# plt.xlabel(r'$log_{10} v_{max}$')
# plt.ylabel(r'$log_{10}dt$')  
# plt.savefig('./Figs/Fordebugging', dpi=600)
#%% Plotting the POD modes
T_filter=100
V_thresh=10
if CalPOD==True:
    if POD_on_V==0:
        U,S,VT,q_bar,Nz,Nx,x_ox,z_ox,vmin,vmax,V_ox_filtered,Nt2,t_ox_filtered,Theta_filtered,Nz=ApplyPODStateSpace(p,T_filter,V_thresh)
        direct='MainSimulation_Tf'+str(T_final)+"Nt="+str(cte.Ntout)
        #np.savez_compressed(direct+'.npz', U=U, S=S, VT=VT,q_bar=q_bar)

    else:
        U,S,VT,P_bar,Nz,Nx,x_ox,z_ox,vmin,vmax,V_ox_filtered,Nt2,t_ox_filtered,Nz=ApplyPODV(p,T_filter)
        direct='MainSimulation_Tf'+str(T_final)+"Nt="+str(cte.Ntout)+'PODonlyonV'
        # np.savez_compressed(direct+'.npz', U=U, S=S, VT=VT,q_bar=P_bar)
        
        U,S,VT,P_bar,Nz,Nx,x_ox,z_ox,vmin,vmax,Theta_ox_filtered,Nt2,t_ox_filtered,Nz=ApplyPODtheta(p,T_filter)
        direct='MainSimulation_Tf'+str(T_final)+"Nt="+str(cte.Ntout)+'PODonlyontheta'
        # np.savez_compressed(direct+'.npz', U=U, S=S, VT=VT,q_bar=P_bar)    

    # ATTTENTION, For future, try to save number of time snapshots as well, because to find the variance you need to divide by Nt 
    # Actually you dont really need it, the number of snapshots are the number of rows(or columns) of VT

#%%
    if POD_on_V==0:
        NMods=8
        V_or_Theta="V"
        PltPODModesVorthetav2(U,q_bar,Nz,Nx,NMods,V_or_Theta,V_thresh,x_ox,z_ox)
    else:
        NMods=8

        PltPODmodesV(U,P_bar,NMods,Nz,Nx,x_ox,z_ox)
            

#%%
Num_samples=220
N_m=30
coeff=2
Specifyinit=True
T_final_samples=250 #300
if sample==True:
    for index in range(130):
        print('Simulating random initial condition number : ' +str(index))
        ai=GenRandom_ai(U,S,N_m,Nt2,coeff)
        u_init=FindInitFromAi(ai,U,N_m,q_bar)
        p=forwardmodel(a_VW,b_VW,dc,W_asp,L,L_asp,stringadditional,T_final_samples,u_init,Specifyinit)
        direct='./Data/SampleSimulation_Tf'+str(T_final_samples)+"Nt="+str(cte.Ntout)+"N_m"+str(N_m)+"coeff"+str(coeff)+"number"+str(index)
        # You only need time, V, theta, so only saving those:
        x_ox=p.ox["x"].unique()
        z_ox=p.ox["z"].unique()
        Nt=len(p.ox["v"])//(len(x_ox)*len(z_ox)) # Number of Snapshots
        t_ox=p.ox["t"].values.reshape((Nt,len(z_ox),len(x_ox)))
        V_ox=p.ox["v"].values.reshape((Nt,len(z_ox),len(x_ox)))
        theta_ox=p.ox["theta"].values.reshape((Nt,len(z_ox),len(x_ox)))
        np.savez_compressed(direct+'.npz', array1=V_ox, array2=theta_ox, array3=t_ox)










