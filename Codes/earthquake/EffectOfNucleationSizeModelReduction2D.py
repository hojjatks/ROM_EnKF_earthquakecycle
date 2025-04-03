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
import cte_eq
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
from forward2d import forwardmodel
#%%
# Instantiate the QDYN class object


T_final=3500#2500
Long_sim=1 # if one it will run the simulation and save it, if 0 it will load the simulation
Specifyinit=0 if Long_sim==1 else 1
CalculatePOD=0 # if 1 it will calculate the POD, if 0 it will load the POD
OnlyPostProcess=1
POD_seperate=1
downsampleratio=1
Ploteigenfunctions=1
#%%
drs=np.array([6,9,12,15])*0.001
b=0.015
a=0.01
G=3e10 
sigma=50e6
L=cte_eq.L
h_ra=2*G*drs*b/(np.pi*sigma*(b-a)**2)
Ins_ratio=L/h_ra
cmap = cm.viridis  # Choose a colormap
norm = mcolors.Normalize(vmin=min(Ins_ratio), vmax=max(Ins_ratio))  # Normalize color range

Ntout=cte_eq.Ntout
Nxout=cte_eq.Nxout



if OnlyPostProcess==0:
    for i in range(np.size(drs)):
        u_init=0
        p=forwardmodel(T_final,Ntout,Nxout,Specifyinit,u_init,drs[i])
        direct='/central/groups/astuart/hkaveh/Data/LearnROM/transfer/2DSim_MainSimulation_Tf'+str(T_final)+"Nx="+str(Nxout)+"Nt="+str(Ntout)+'drs'+str(drs[i])+".npz"
        v=p.ox["v"]
        theta=p.ox["theta"]
        tau=p.ox["tau"]
        slip=p.ox["slip"]
        t=p.ox["t"]
        a=p.mesh_dict["A"]
        b=p.mesh_dict["B"]
        dc=p.mesh_dict["DC"]
        sigma=p.mesh_dict["SIGMA"]
        # saving the output of the simulation in a numpy file in the directory
        np.savez(direct,v=v,theta=theta,tau=tau,slip=slip,t=t,a=a,b=b,dc=dc,sigma=sigma)
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
        direct='/central/groups/astuart/hkaveh/Data/LearnROM/transfer/2DSim_MainSimulation_Tf'+str(T_final)+"Nt="+str(Ntout)+'drs'+str(drs[i])+".npz"
        data=np.load(direct)
        v=data["v"]
        theta=data["theta"]
        #tau=p.ox["tau"]
        slip=data["slip"]
        t=data["t"]
        a=data["a"]
        Nx=int(a.shape[0]//2)
# #%%
        #qdyn_plot.slip_profile(p.ox, warm_up=1000*t_yr)
        T_filter=400 # years, remove everything before this year.
        N_snapshots=70000
        if CalculatePOD==1:
            if POD_seperate==1:
                v_or_theta="v"
                U,S,VT,P_bar,Nx,V_ox_filtered,theta_ox_filtered,Nt2,t_ox_filtered=ProcessFunctions.ApplyPODV_2D(v,theta,t,Nx,T_filter,v_or_theta,downsampleratio,N_snapshots)
                direct='/central/groups/astuart/hkaveh/Data/LearnROM/transfer/MainSimulation2D_Tf'+str(T_final)+"Nt="+str(Ntout)+'PODonlyonV'+'drs'+str(drs[i])
                np.savez_compressed(direct+'.npz', U=U, S=S, VT=VT,q_bar=P_bar)
                v_or_theta="theta"
                U,S,VT,P_bar,Nx,V_ox_filtered,theta_ox_filtered,Nt2,t_ox_filtered=ProcessFunctions.ApplyPODV_2D(v,theta,t,Nx,T_filter,v_or_theta,downsampleratio,N_snapshots)
                direct='/central/groups/astuart/hkaveh/Data/LearnROM/transfer/MainSimulation2D_Tf'+str(T_final)+"Nt="+str(Ntout)+'PODonlyontheta'+'drs'+str(drs[i])
                np.savez_compressed(direct+'.npz', U=U, S=S, VT=VT,q_bar=P_bar)    
        else:
            direct1='/central/groups/astuart/hkaveh/Data/LearnROM/transfer/MainSimulation2D_Tf'+str(T_final)+"Nt="+str(Ntout)+'PODonlyonV'+'drs'+str(drs[i])
            direct2='/central/groups/astuart/hkaveh/Data/LearnROM/transfer/MainSimulation2D_Tf'+str(T_final)+"Nt="+str(Ntout)+'PODonlyontheta'+'drs'+str(drs[i])
            Data_V=np.load(direct1+'.npz')
            Data_theta=np.load(direct2+'.npz')

# #%% 
            # # Loading the POD






#            fig,axs=plt.subplots(1,2,figsize=(15,5))
            color = cmap(norm(Ins_ratio[i]))  # Map value to a color
            Is    = range(1,len(np.diag(Data_V["S"])**2)+1)

            axs[0].plot(Is,np.diag(Data_V["S"])**2/N_snapshots,color=color,label=r'$v$')
            axs[1].plot(Is,np.diag(Data_theta["S"])**2/N_snapshots,color=color,label=r'$\theta$')
            axs[0].set_yscale('log')
            axs[1].set_yscale('log')

#            axs[0].set_title('Singular values')
            axs[2].plot(Is,np.cumsum(np.diag(Data_V["S"]**2))/np.sum(np.diag(Data_V["S"]**2)),color=color,label=r'$v$')
            axs[3].plot(Is,np.cumsum(np.diag(Data_theta["S"]**2))/np.sum(np.diag(Data_theta["S"]**2)),color=color,label=r'$\theta$')



    axs[0].set_xlim([1,3000])
    axs[1].set_xlim([1,3000])
    axs[0].set_ylim([1e-14,1e4])
    axs[1].set_ylim([1e-14,1e4])
    axs[2].set_ylim(bottom=.4,top=1)
    axs[3].set_ylim(bottom=.4,top=1)
    axs[0].text(100,2e4,'(a)')
    axs[1].text(100,2e4,'(b)')
    axs[2].text(2,1.01,'(c)')
    axs[3].text(2,1.01,'(d)')
    axs[0].set_xlabel(r'$i$')
    axs[1].set_xlabel(r'$i$')
    axs[2].set_xlabel(r'$i$')
    axs[3].set_xlabel(r'$i$')
    axs[0].set_ylabel(r'$\lambda_i^{v}$')
    axs[1].set_ylabel(r'$\lambda_i^{\theta}$')
    axs[2].set_ylabel(r'$\sum_{j=1}^i\lambda_i^{v}/\sum_{j=1}^m\lambda_i^{v}$')
    axs[3].set_ylabel(r'$\sum_{j=1}^i\lambda_i^{\theta}/\sum_{j=1}^m\lambda_i^{\theta}$')




    axs[2].set_xlim([1,40])
    axs[3].set_xlim([1,40])

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
    plt.savefig('/central/groups/astuart/hkaveh/Figs/ROM/POD_SingularValues2.png',dpi=300)
    plt.show()

# #%%

if Ploteigenfunctions==1:
    i=1 # plot eigenfunctinos for i=1
    direct='/central/groups/astuart/hkaveh/Data/LearnROM/transfer/2DSim_MainSimulation_Tf'+str(T_final)+"Nt="+str(Ntout)+'drs'+str(drs[i])+".npz"
    data=np.load(direct)
    v=data["v"]
    theta=data["theta"]
    #tau=p.ox["tau"]
    slip=data["slip"]
    t=data["t"]
    a=data["a"]
    Nx=int(a.shape[0]//2)
    direct1='/central/groups/astuart/hkaveh/Data/LearnROM/transfer/MainSimulation2D_Tf'+str(T_final)+"Nt="+str(Ntout)+'PODonlyonV'+'drs'+str(drs[i])
    direct2='/central/groups/astuart/hkaveh/Data/LearnROM/transfer/MainSimulation2D_Tf'+str(T_final)+"Nt="+str(Ntout)+'PODonlyontheta'+'drs'+str(drs[i])
    Data_V=np.load(direct1+'.npz')
    Data_theta=np.load(direct2+'.npz')
    # Plotting the POD components for slip rate and state variable
    # making everything serif font
    # Set global font family to 'serif' and font size to 14
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 8  # You can adjust this value as needed
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    fig,axs=plt.subplots(3,1,figsize=(3.7,8))

    Nx=Data_V["U"].shape[0]
    x_grid=np.linspace(-L/2,L/2,Nx)/1e3
    N_plot=4

# # Plotting on axs[0] two lines with shared x axis but different y axis
    axs[0].plot(x_grid,Data_V["q_bar"],color='black',label=r'$v$')
# # make another plot with different y-axis using twinx:
    ax2=axs[0].twinx()
    ax2.plot(x_grid,Data_theta["q_bar"],color='red',linestyle='--',label=r'$\theta$')
    ax2.tick_params('y', colors='r')
    ax2.spines['right'].set_color('r')

    for i in range(3):
        axs[i].set_xlabel('Distance along strike (km)')
        axs[i].set_xlim(-L/2/1e3,L/2/1e3)
    axs[0].set_ylabel(r'$\phi_0^v$')
    ax2.set_ylabel(r'$\phi_0^\theta$',color='red')
    axs[1].set_ylabel(r'$\phi_i^v$')
    axs[2].set_ylabel(r'$\phi_i^\theta$')

    for i in range(N_plot):


        axs[1].plot(x_grid,Data_V["U"][:,i],label="$i={}$".format(i+1))
        axs[2].plot(x_grid,Data_theta["U"][:,i],label="$i={}$".format(i+1))
    axs[1].set_ylim(top=0.04)
    axs[2].set_ylim(top=0.04)
    axs[1].legend(ncol=4,frameon=False,fontsize=6)
    axs[2].legend(ncol=4,frameon=False,fontsize=6)
    axs[0].text(-100,-5.3,'(a)')
    axs[1].text(-100,0.042,'(b)')
    axs[2].text(-100,0.042,'(c)')
    plt.tight_layout()
    
    plt.savefig('/central/groups/astuart/hkaveh/Figs/ROM/Eq_PODcomponents.png',dpi=300)


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
