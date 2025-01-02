#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:08:28 2024

@author: hojjat
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

#%%
import matplotlib
import sys
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as mcolors

#%%
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pickle
# import ProcessFunctions

import sys
from numpy import random
import seaborn as sns

from scipy.optimize import fsolve
from scipy.special import psi
import ProcessFunctions
from ProcessFunctions import GenRandom_ai,FindInitFromAi
#%%
# Add QDYN directory to PATH

sys.path.append('/central/groups/astuart/hkaveh/QDYN/qdyn-read-only/src')  # For pyqdyn

from pyqdyn import qdyn
#%%
# Instantiate the QDYN class object


T_final=35#2500
Ntout=300
Nxout=2
t_yr = 3600 * 24 * 365.25   # seconds per 
Long_sim=1 # if one it will run the simulation and save it, if 0 it will load the simulation
Specifyinit=0 if Long_sim==1 else 1
CalculatePOD=0 # if 1 it will calculate the POD, if 0 it will load the POD
OnlyPostProcess=0
POD_seperate=1
downsampleratio=1
#%%
def forwardmodel(T_final,Ntout,Nxout,Specifyinit,u_init,Drs):
    p = qdyn()
    L =240e3
    resolution = 7              # Mesh resolution / process zone width
    set_dict = p.set_dict
    set_dict["MESHDIM"] = 1        # Simulation dimensionality (1D fault in 2D medium)
     #   Boundary conditions when MESHDIM=1:
     # 0 = Periodic fault: the fault is infinitely long, but slip is spatially periodic with period L, loaded by steady displacement at distance W from the fault.
    set_dict["FINITE"] = 1         # Periodic fault
     # set_dict["TMAX"] = 2500*t_yr     # Maximum simulation time [s]
    set_dict["TMAX"] = T_final*t_yr     # Maximum simulation time [s]
    set_dict["NTOUT"] = Ntout        # Save output every N steps
    set_dict["NXOUT"] = Nxout          # Snapshot resolution (every N elements)
    set_dict["L"] = L
    set_dict["V_PL"] = 50e-3/t_yr  # Plate velocity
    set_dict["VS"] = 3.3*1000          # Shear wave speed
    set_dict["MU"] = 3e10          # Shear modulus
    set_dict["SIGMA"] = 50e6       # Effective normal stress [Pa]    
     # Setting some (default) RSF parameter values
    set_dict["SET_DICT_RSF"]["A"] = 1e-2       # Direct effect (will not be overwritten later)
    set_dict["SET_DICT_RSF"]["B"] = 0.015      # Evolution effect (will be overwritten later)
    set_dict["SET_DICT_RSF"]["DC"] = Drs      # Characteristic slip distance
    set_dict["SET_DICT_RSF"]["V_SS"] = 1e-6    # Reference velocity [m/s]
    set_dict["SET_DICT_RSF"]["MU_SS"] = 0.6    # Reference friction coeff [m/s]
    
    set_dict["SET_DICT_RSF"]["TH_0"] = set_dict["SET_DICT_RSF"]["DC"] / set_dict["V_PL"]    # Initial state [s] (will be overwritten later)
    set_dict["SET_DICT_RSF"]["V_0"] = 1e-6 
    
     # Compute relevant length scales:
     # Process zone width [m]
    Lb = set_dict["MU"] * set_dict["SET_DICT_RSF"]["DC"] / (set_dict["SET_DICT_RSF"]["B"] * set_dict["SIGMA"])
    set_dict["ACC"] = 1e-7         # Solver accuracy
    set_dict["SOLVER"] = 2         # Solver type (Runge-Kutta)
    
    
    
     #set_dict["W"] = 50e3           # Loading distance [m]
    
    
     # Nucleation length [m]
     # Lc = Lb / cab_ratio
     # Length of asperity [m]
     # Lasp *= Lc
     # Fault length [m]
    N = int(np.power(2, np.ceil(np.log2(resolution * L / Lb))))
    
     # Spatial coordinate for mesh
    x = np.linspace(-L/2, L/2, N, dtype=float)
    
     # Set mesh size and fault length
    set_dict["N"] = N
    set_dict["L"] = L
     # Set time series output node to the middle of the fault
    set_dict["IC"] = N // 2
    
    
    
     ## Specifying init stress
     # Very important: in the simulation of the QDYN itself they do not specify V0 and mu0. Here since I want to specify Tau0, I first specify initial velocity to be V_ss and then find theta0
     # Theta0=(Dc/V0)*exp(1/b*(tau/sigma_n-f0))
     #set_dict["V_0"] = 1e-6          # Initial velocity =V_ss 
    
    
    """ Step 2: Set (default) parameter values and generate mesh """
    p.settings(set_dict)
    p.render_mesh()
    
    
    """ Step 3: override default mesh values """
     # Distribute direct effect a over mesh according to some arbitrary function
    b_list=[]
    
    for i in range(x.size):
        if x[i]<40e3-L/2:
            b_list.append(-.01) # VS 1
        elif x[i]<(40+72.5)*1000-L/2:
            b_list.append(0.015) # VW
        elif x[i]<(40+72.5+15)*1000-L/2:
            b_list.append(0.008) # VS 2
        elif x[i]<(40+72.5+15+72.5)*1000-L/2:
            b_list.append(0.015) # VW
        else:
            b_list.append(-.01) # VS 1
                
    p.mesh_dict["B"] = b_list
    if Specifyinit==True:
        v_init=u_init[0]
        theta_init=u_init[1]
        p.mesh_dict["V_0"] = 10**(v_init)
        p.mesh_dict["TH_0"] = 10**(theta_init)
    else:

    
        V0=set_dict["SET_DICT_RSF"]["V_0"] 
        Dc=set_dict["SET_DICT_RSF"]["DC"]
        sigma_n=set_dict["SIGMA"]
        f0=set_dict["SET_DICT_RSF"]["MU_SS"]
        b1=-.01
        b2=0.015
        b3=0.008
        b4=0.015
        b5=-.01
        tau1=26.1e6
        tau2=28.5e6
        tau3=28.2e6
        tau4=28.8e6
        tau5=26.1e6
        
        theta_list=[]
        Tau_list=[]
        for i in range(x.size):
            if x[i]<40e3-L/2:
                theta_i=(Dc/V0)*np.exp(1/b1*(tau1/sigma_n-f0))
                Tau_i=sigma_n*(f0+b1*np.log(V0*theta_i/Dc))
                theta_list.append(theta_i) 
                Tau_list.append(Tau_i)
            elif x[i]<(40+72.5)*1000-L/2:
                theta_i=(Dc/V0)*np.exp(1/b2*(tau2/sigma_n-f0))
                Tau_i=sigma_n*(f0+b2*np.log(V0*theta_i/Dc))
        
                theta_list.append(theta_i) # VW
                Tau_list.append(Tau_i) 
        
            elif x[i]<(40+72.5+15)*1000-L/2:
        
        
                theta_i=(Dc/V0)*np.exp(1/b3*(tau3/sigma_n-f0))
                Tau_i=sigma_n*(f0+b3*np.log(V0*theta_i/Dc))
                theta_list.append(theta_i) # VS 2
                Tau_list.append(Tau_i)
            elif x[i]<(40+72.5+15+72.5)*1000-L/2:
                
                theta_i=1*(Dc/V0)*np.exp(1/b4*(tau4/sigma_n-f0))
                Tau_i=sigma_n*(f0+b4*np.log(V0*theta_i/Dc))
        
                theta_list.append(theta_i) # VW
                Tau_list.append(Tau_i)
            else:
                theta_i=(Dc/V0)*np.exp(1/b5*(tau5/sigma_n-f0))
                Tau_i=sigma_n*(f0+b5*np.log(V0*theta_i/Dc))
        
                theta_list.append(theta_i) # VS 1
                Tau_list.append(Tau_i)
        plt.figure()
        plt.plot(x/1000,theta_list)
        p.mesh_dict["TH_0"] = theta_list
        
        
    p.write_input()
    
    # print(N)
    # plt.figure()
    # plt.clf()
    # plt.plot(x, p.mesh_dict["A"] - p.mesh_dict["B"])
    # plt.axhline(0, ls=":", c="k")
    # plt.xlabel("position [m]")
    # plt.ylabel("(a-b) [-]")
    # plt.tight_layout()
    # plt.savefig("asperity_a-b.png")
    # plt.show()
    
    p.run()
    p.read_output()
    return p

#%%
drs=np.array([6,9,13,16])*0.001
b=0.015
a=0.01
nu=0.25
G=3e10/(1-nu) # plane strain
sigma=50e6
L=240e3
h_ra=2*G*drs*b/(np.pi*sigma*(b-a)**2)
Ins_ratio=L/h_ra
cmap = cm.viridis  # Choose a colormap
norm = mcolors.Normalize(vmin=min(Ins_ratio), vmax=max(Ins_ratio))  # Normalize color range




if OnlyPostProcess==0:
    for i in range(np.size(drs)):
        u_init=0
        p=forwardmodel(T_final,Ntout,Nxout,Specifyinit,u_init,drs[i])
        direct='/central/groups/astuart/hkaveh/Data/transfer/2DSim_MainSimulation_Tf'+str(T_final)+"Nt="+str(Ntout)+'drs'+str(drs[i])+".pickle"
        ProcessFunctions.SaveAsPickle(p,direct)
else:
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 8  # You can adjust this value as needed
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    serif_font = fm.FontProperties(family="serif", size=8)
    fig = plt.figure(figsize=(7.4, 6))

# Use serif font and set font size for all text in the figure
    plt.rc('font', family='serif')
    plt.rcParams.update({'font.family': 'serif'})
    plt.rcParams.update({'font.size': 8})
    gs = gridspec.GridSpec(nrows=3, ncols=2 ,height_ratios=[1, 1,.1])
    axes0 = fig.add_subplot(gs[0, 0])
    axes1 = fig.add_subplot(gs[0, 1])
    axes2 = fig.add_subplot(gs[1, 0])
    axes3 = fig.add_subplot(gs[1, 1])
    axs=[axes0,axes1,axes2,axes3]
    cax1 = fig.add_subplot(gs[2,:])  # Colorbar subplot spanning all rows

    for i in range(np.size(drs)):
        direct='./Data/2DSim_MainSimulation_Tf'+str(T_final)+"Nt="+str(Ntout)+'drs'+str(drs[i])+".pickle"
        p=ProcessFunctions.ReadData(direct)

# #%%
        #qdyn_plot.slip_profile(p.ox, warm_up=1000*t_yr)
        T_filter=400 # years, remove everything before this year.
        N_snapshots=8000
        if CalculatePOD==1:
            if POD_seperate==1:
                v_or_theta="v"
                U,S,VT,P_bar,Nx,x_ox,V_ox_filtered,theta_ox_filtered,Nt2,t_ox_filtered=ProcessFunctions.ApplyPODV_2D(p,T_filter,v_or_theta,downsampleratio,N_snapshots)
                direct='./Data/MainSimulation2D_Tf'+str(T_final)+"Nt="+str(Ntout)+'PODonlyonV'+'drs'+str(drs[i])
                np.savez_compressed(direct+'.npz', U=U, S=S, VT=VT,q_bar=P_bar)
                v_or_theta="theta"
                U,S,VT,P_bar,Nx,x_ox,V_ox_filtered,theta_ox_filtered,Nt2,t_ox_filtered=ProcessFunctions.ApplyPODV_2D(p,T_filter,v_or_theta,downsampleratio,N_snapshots)
                direct='./Data/MainSimulation2D_Tf'+str(T_final)+"Nt="+str(Ntout)+'PODonlyontheta'+'drs'+str(drs[i])
                np.savez_compressed(direct+'.npz', U=U, S=S, VT=VT,q_bar=P_bar)    
        else:
            direct1='./Data/MainSimulation2D_Tf'+str(T_final)+"Nt="+str(Ntout)+'PODonlyonV'+'drs'+str(drs[i])
            direct2='./Data/MainSimulation2D_Tf'+str(T_final)+"Nt="+str(Ntout)+'PODonlyontheta'+ 'drs'+str(drs[i])
            Data_V=np.load(direct1+'.npz')
            Data_theta=np.load(direct2+'.npz')

# #%% 
            # # Loading the POD






#            fig,axs=plt.subplots(1,2,figsize=(15,5))
            color = cmap(norm(Ins_ratio[i]))  # Map value to a color
            axs[0].plot(np.diag(Data_V["S"])**2/N_snapshots,color=color,label=r'$v$')
            axs[1].plot(np.diag(Data_theta["S"])**2/N_snapshots,color=color,label=r'$\theta$')
            axs[0].set_yscale('log')
            axs[1].set_yscale('log')

#            axs[0].set_title('Singular values')
            axs[2].plot(np.cumsum(np.diag(Data_V["S"]**2))/np.sum(np.diag(Data_V["S"]**2)),color=color,label=r'$v$')
            axs[3].plot(np.cumsum(np.diag(Data_theta["S"]**2))/np.sum(np.diag(Data_theta["S"]**2)),color=color,label=r'$\theta$')



    axs[0].set_xlim([0,3000])
    axs[1].set_xlim([0,3000])
    axs[0].set_ylim([1e-14,1e4])
    axs[1].set_ylim([1e-14,1e4])
    axs[2].set_ylim(bottom=.4,top=1)
    axs[3].set_ylim(bottom=.4,top=1)
    axs[0].set_xlabel(r'$i$')
    axs[1].set_xlabel(r'$i$')
    axs[2].set_xlabel(r'$i$')
    axs[3].set_xlabel(r'$i$')
    axs[0].set_ylabel(r'$\lambda_i^{v}$')
    axs[1].set_ylabel(r'$\lambda_i^{\theta}$')
    axs[2].set_ylabel(r'$\sum_{j=1}^i\lambda_i^{v}/\sum_{j=1}^m\lambda_i^{v}$')
    axs[3].set_ylabel(r'$\sum_{j=1}^i\lambda_i^{\theta}/\sum_{j=1}^m\lambda_i^{\theta}$')




    axs[2].set_xlim([0,40])
    axs[3].set_xlim([0,40])

    # # Enable major and minor ticks
    for j in range(4):
        axs[j].minorticks_on()
        axs[j].grid(which='major', linestyle='-', linewidth='0.5', color='black')
        axs[j].grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    # Add a colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(Ins_ratio)  # Set the array for the colorbar
    cbar = plt.colorbar(sm, cax=cax1, orientation='horizontal')
    cbar.set_label('Instability ratio')  # Label for the colorbar
    plt.tight_layout()
    plt.savefig('./Figs/POD_SingularValues2.png',dpi=300)
    plt.show()

# #%%
# # Plotting the POD components for slip rate and state variable
# # making everything serif font
# # Set global font family to 'serif' and font size to 14
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.size'] = 8  # You can adjust this value as needed
# plt.rcParams['mathtext.fontset'] = 'dejavuserif'
# fig,axs=plt.subplots(3,1,figsize=(5,8))

# L=p.set_dict["L"]
# Nx=Data_V["U"].shape[0]
# x_grid=np.linspace(-L/2,L/2,Nx)/1e3
# N_plot=5

# # Plotting on axs[0] two lines with shared x axis but different y axis
# axs[0].plot(x_grid,Data_V["q_bar"],color='black',label=r'$v$')
# # make another plot with different y-axis using twinx:
# ax2=axs[0].twinx()
# ax2.plot(x_grid,Data_theta["q_bar"],color='red',linestyle='--',label=r'$\theta$')
# ax2.tick_params('y', colors='r')
# ax2.spines['right'].set_color('r')

# for i in range(3):
#     axs[i].set_xlabel(r'$\mathrm{x (km)}$')
#     axs[i].set_xlim(-L/2/1e3,L/2/1e3)
# axs[0].set_ylabel(r'$\overline{\mathrm{log}_{10}v}$')
# ax2.set_ylabel(r'$\overline{\mathrm{log}_{10}\theta}$')
# axs[1].set_ylabel(r'$\phi_i^v$')
# axs[2].set_ylabel(r'$\phi_i^\theta$')

# for i in range(N_plot):


#     axs[1].plot(x_grid,Data_V["U"][:,i],label="$i={}$".format(i+1))
#     axs[2].plot(x_grid,Data_theta["U"][:,i],label="$i={}$".format(i+1))

# axs[1].legend(ncol=2)
# axs[2].legend(ncol=2)
# plt.tight_layout()
# plt.savefig('./Figs/Eq_PODcomponents.png',dpi=300)


# #%%
# Num_samples=20 # 220
# N_m=70
# coeff=1
# Specifyinit=True
# T_final_samples=800 #300
# Nt2=Data_V["VT"].shape[0]
# # What to do:
# # 1- generate random ai for v and theta
# # 2- find the initial condition from ai
# # 3- give the right initial conditioon to the forward model (double check)
# if GenDataforML==1:
#     for index in range(Num_samples):
#         max_v=3
#         while max_v>2: # ensures that the initial condition is reasonable and not too large
#             print('Simulating random initial condition number : ' +str(index))
#             ai_v=GenRandom_ai(Data_V["U"],Data_V["S"],N_m,Nt2,coeff)
#             v_init=FindInitFromAi(ai_v,Data_V["U"],N_m,Data_V["q_bar"])
#             ai_theta=GenRandom_ai(Data_theta["U"],Data_theta["S"],N_m,Nt2,coeff)
#             theta_init=FindInitFromAi(ai_theta,Data_theta["U"],N_m,Data_theta["q_bar"])

#             N_coarse=Data_V["U"].shape[0]
#             N_fine=N_coarse*Nxout
#             x_coarse=np.linspace(-L/2,L/2,N_coarse)
#             x_fine=np.linspace(-L/2,L/2,N_fine)
#             # interpolate v_init and theta_init to the fine grid
#             v_init_fine=np.interp(x_fine,x_coarse,v_init.ravel())
#             theta_init_fine=np.interp(x_fine,x_coarse,theta_init.ravel())
#             u_init=[v_init_fine,theta_init_fine]
#             max_v=np.max(v_init_fine)
#         p=forwardmodel(T_final_samples,Ntout,Nxout,Specifyinit,u_init)

#         # qdyn_plot.timeseries(p.ot[0], p.ot_vmax)

#         direct='./Data/SampleSimulation2D_Tf'+str(T_final_samples)+"Nt="+str(Ntout)+"Nx="+str(Nxout)+"N_m"+str(N_m)+"coeff"+str(coeff)+"number"+str(index)
#         # You only need time, V, theta, so only saving those:
#         x_ox=p.ox["x"].unique()
#         Nt=len(p.ox["v"])//(len(x_ox)) # Number of Snapshots
#         t_ox=p.ox["t"].values.reshape((Nt,len(x_ox)))
#         V_ox=p.ox["v"].values.reshape((Nt,len(x_ox)))
#         theta_ox=p.ox["theta"].values.reshape((Nt,len(x_ox)))
#         np.savez_compressed(direct+'.npz', array1=V_ox, array2=theta_ox, array3=t_ox)






# %%
