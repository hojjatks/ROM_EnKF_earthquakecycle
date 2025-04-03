import sys
import numpy as np
import pickle
from numpy import random
import seaborn as sns
import cte_eq
from scipy.optimize import fsolve
from scipy.special import psi

sys.path.append('/central/groups/astuart/hkaveh/QDYN/qdyn-read-only/src')  # For pyqdyn

from pyqdyn import qdyn
def forwardmodel(T_final,Ntout,Nxout,Specifyinit,u_init,Drs):
    p = qdyn()
    L=cte_eq.L
    resolution = 7              # Mesh resolution / process zone width
    set_dict = p.set_dict
    set_dict["MESHDIM"] = 1        # Simulation dimensionality (1D fault in 2D medium)
     #   Boundary conditions when MESHDIM=1:
     # 0 = Periodic fault: the fault is infinitely long, but slip is spatially periodic with period L, loaded by steady displacement at distance W from the fault.
    set_dict["FINITE"] = 1         # Periodic fault
     # set_dict["TMAX"] = 2500*t_yr     # Maximum simulation time [s]
    set_dict["TMAX"] = T_final*cte_eq.t_yr     # Maximum simulation time [s]
    set_dict["NTOUT"] = Ntout        # Save output every N steps
    set_dict["NXOUT"] = Nxout          # Snapshot resolution (every N elements)
    set_dict["L"] = L
    set_dict["V_PL"] = 50e-3/cte_eq.t_yr  # Plate velocity
    set_dict["VS"] = 3.3*1000          # Shear wave speed
    set_dict["MU"] = 3e10          # Shear modulus, IMPORTANT: 1-nu has to be here because we want to study the in plane case
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
        p.mesh_dict["TH_0"] = theta_list
        
        
    p.write_input()
    p.run()
    p.read_output()
    return p