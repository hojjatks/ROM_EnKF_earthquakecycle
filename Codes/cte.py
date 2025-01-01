#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:30:28 2024

@author: hkaveh
"""
import numpy as np
#%% Specifying the parameters that I dont want to change
t_yr=3600*24*365.24      # Seconds per year
G=30e9# shear stress
sigma=10e6# normal stress
#L= 320e3               # Length of fault along strike
W=50e3                 # Length of fault along dip
L_asp_north=680000      # from google earth (above latitude )
L_asp_south=770000
L_buffer=20000
resolution=5            # Mesh resolution/process zone width
V_PL=40e-3/t_yr         # Plate velocity
L_asp=300e3 # Length of Asperity along-strike
W_asp=25e3  # Length of Asperity along dip
Poissonratio=0.25

a_VS=0.019
b_VS=0.014
V_thresh=10*V_PL # to define events
V_thresh_max=1e-3 # Any simulation with sliprate bigger than is earthquake
Ntout=5
Nxout=1
Nwout=1
dipangle=17.5
slope_constraint=1.6
constraint_lower = np.array([2/12, -np.inf,0.1]) # defining the orange  acceptable region. First constraint is a vertical line, second one is the slope, and the third one is the horizental line
constraint_upper = np.array([6/12, 0,0.9])
# T_filter=5 # Remove the first 100 years from the data

L_thresh=35e3 # Almost half of the length of the smallest segment of the fault

# observational data
mu_IT= 0.545565 #(yr)
sigma_IT=0.354709  #(yr)
mu_ED=0.126426
sigma_ED=0.0346
mu_Mw=6.288342
sigma_Mw=0.436717 

obsv1=np.array([mu_IT,sigma_IT,mu_ED,sigma_ED,mu_Mw,sigma_Mw]).reshape(6,1)
obsv2=np.array([mu_IT,sigma_IT,mu_Mw]).reshape(3,1)
obsv3=np.array([mu_IT,mu_Mw]).reshape(2,1)
obsv4=np.array([mu_IT,mu_ED,mu_Mw]).reshape(3,1)
# I simulated a bunch of faults (the one to plot the behaviour of the fault in the pi1, pi2/pi3 plane), here is the range of values of the cost:
# array([[282.31572044]]), array([[84.10963196]]), array([[382.18543548]]), array([[1906.55319985]]), array([[1705.45528323]]),array([[2262.11002572]]), array([[8246.79086723]]),array([[3112.51604163]]), array([[2934.96879664]]), array([[3445.11929637]]), array([[7642.8973813]]), array([[10852.00931244]]), array([[14575.32184417]]), array([[5380.08394645]]), array([[6445.60956332]]),array([[7651.90612958]]),array([[17147.02766678]]),array([[22236.04997299]]),array([[26958.54336203]]),array([[12186.13236975]]),array([[12786.4935196]]),array([[13396.02540959]]),array([[17233.41997847]]),array([[19604.44808856]]), array([[62133.07686862]])]
# maximum value is 62000
# So I if we choose when we have creep, we the cost will be -1e6
# what about earthquakes:
    # I think earthquakes are less dangerous than creep, so I will just keep the samme order of mags?
    # I suggest we do not add any additional constraint for when we have earthquakes.
cost_creep=-1e6
cost_constraint=-1e6
cost_creep_secondopt=100

constraint_lower = np.array([0.25, -np.inf])
constraint_upper = np.array([0.5, 0])
minimum_window=np.array([0.01,1e-7,1e-7])
