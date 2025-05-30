# -*- coding: utf-8 -*-
#[your name]

# The purpose of this problem is to apply Blade Element Momentum theory to
# the analysis of an entire wind turbine, using the tools developed in the
# previous problems.

# You will need the following helper functions to complete this
# problem.
# -get_foil_data
# -get_design_alpha
# -interp_foil
# -tiploss
# -BEM_CL
# -CL_intersect
# -calculate_parameters

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.io
from helperfunctions_template import get_foil_data, interp_foil, tiploss, BEM_CL, CL_intersect, calculate_parameters, get_design_alpha

plt.close('all')

#%% Setup
def BEM_analysis(U_design, R, eta_0, lambda_design, B, r_hub, N, performance_file, Re_design, options, BEM_color, rho, nu):
    #%% Part A: Blade Shape

    # Load airfoil performance data
    foil = get_foil_data(performance_file)

    # Get optimal angle of attack at design Re
    alpha_design = get_design_alpha(foil, Re_design)

    # Get coefficients of lift and drag at optimal angle of attack
    CL_design, CD_design = interp_foil(foil, Re_design, alpha_design)

    # Define edges of the blade elements
    r_edge = np.linspace(0,R,N+1) #dividing up the blade from the hub to the rotor tip

    #%%

    r = np.empty(N)

    # Define mid-point of each segment
    for i in range(N):
        r[i] = (r_edge[i] + r_edge[i+1]) / 2

    # Calculate blade geometric parameters and optimal flow conditions

    # Local speed ratio and ideal angle of relative wind
    lambda_r = lambda_design * r / R  # Local speed ratio (all sections)

    phi_design = (2/3) * np.arctan(1/lambda_r)  # Design angle of relative wind [radians] (all sections)

    # Blade chord length
    c = (8*np.pi*r*(1-np.cos(phi_design)))/(B*CL_design)  # Blade chord length [m] (all sections)

    # Pitch and twist angles
    theta_P = phi_design - alpha_design  # Blade pitch angle [radians] (all sections)
    theta_P0 = theta_P[-1]  # Blade pitch angle [radians] (tip section)
    theta_T = theta_P - theta_P0 # Blade twist angle [radians] (all sections)

    #%%
    # Visualization of blade geometry
    # Visualize blade geometry
    fig, axes = plt.subplots(1, 3, figsize=(14.08, 5.18))  # Set figure size

    # Chord length plot
    axes[0].plot(r / R, c, '--o', label='Chord Length')
    axes[0].axvspan(0, r_hub / R, color=[153 / 255, 0, 0], alpha=0.3)  # Hub region
    axes[0].set_xlabel('r/R', fontweight='bold')
    axes[0].set_ylabel('c (m)', fontweight='bold')
    axes[0].grid(True)
    axes[0].legend()

    # Pitch angle plot
    axes[1].plot(r / R, theta_P * 180 / np.pi, '--o', label='Pitch Angle')
    axes[1].axvspan(0, r_hub / R, color=[153 / 255, 0, 0], alpha=0.3)  # Hub region
    axes[1].set_xlabel('r/R', fontweight='bold')
    axes[1].set_ylabel(r'$\theta_P$ (degrees)', fontweight='bold')
    axes[1].grid(True)
    axes[1].legend()

    # Twist angle plot
    axes[2].plot(r / R, theta_T * 180 / np.pi, '--o', label='Twist Angle')
    axes[2].axvspan(0, r_hub / R, color=[153 / 255, 0, 0], alpha=0.3)  # Hub region
    axes[2].set_xlabel('r/R', fontweight='bold')
    axes[2].set_ylabel(r'$\theta_T$ (degrees)', fontweight='bold')
    axes[2].grid(True)
    axes[2].legend()

    # Display the plots
    plt.tight_layout()
    plt.show()


    # %% Part B: Blade Element Performance
    # --- Part B: Blade Element Performance ---
    # Rotor solidity (all sections)
    sigma = B*c/(2*np.pi*r) 

    # Initialize angle of attack to design condition as a starting guess
    alpha = np.full_like(r, alpha_design)

    #%%
    #calculate intersection of airfoil performance and turbine performance (all sections)
    #'minimize' tries to minimize the function output (the residual between coefficients
    #of lift based on BEM and foil data given an initial
    #guess for the return variable ('alpha'), solver 'options', and multiple constant values
    #This problem needs bounds to ensure the minimization doesn't solve to a non-physical alpha

    alphaBound = [(0,10)]*len(alpha) #make initial array of alpha guesses

    res = minimize(CL_intersect, x0=alpha, bounds = alphaBound, args=(foil, theta_P0, theta_T, sigma, lambda_r, B, R, r, Re_design) ,method = 'L-BFGS-B')
    alpha = res.x

    # Warning if negative angles of attack
    if np.any(alpha <= 0):
        print('Warning: angle of attack should not be negative!')

    #calculate actual lift and drag coefficients using airfoil data (note: CL could also be found
    #equivalently from BEM relation)
    CL, CD = interp_foil(foil, Re_design, alpha)

    #calculate induction factors, angle of relative wind, and tip loss
    #correction (all sections)
    a_axial, a_angular, phi, F = calculate_parameters(CL, alpha, theta_P0, theta_T, sigma, B, R, r)

    # --- Plot rotor performance as a function of blade span position ---
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    r_norm = r / R

    # Lift coefficient
    #axs[0, 0].plot(..., '-k', linewidth=2)
    axs[0, 0].plot(r_norm, CL, 'o-.', color=BEM_color)
    axs[0, 0].axvspan(0, r_hub / R, color=[153 / 255, 0, 0], alpha=0.3)  # Hub region
    axs[0, 0].set_xlabel('r/R')
    axs[0, 0].set_ylabel('$C_L$')
    axs[0, 0].grid(True)

    # Drag coefficient
    #axs[0, 1].plot(..., '-k', linewidth=2)
    axs[0, 1].plot(r_norm, CD, 'o-.', color=BEM_color)
    axs[0, 1].axvspan(0, r_hub / R, color=[153 / 255, 0, 0], alpha=0.3)  # Hub region
    axs[0, 1].set_xlabel('r/R')
    axs[0, 1].set_ylabel('$C_D$')
    axs[0, 1].grid(True)

    # Angle of attack
    #axs[0, 2].plot(..., '-k', linewidth=2)
    axs[0, 2].plot(r_norm, alpha * 180 / np.pi, 'o-.', color=BEM_color)
    axs[0, 2].axvspan(0, r_hub / R, color=[153 / 255, 0, 0], alpha=0.3)  # Hub region
    axs[0, 2].set_xlabel('r/R')
    axs[0, 2].set_ylabel(r'$\alpha$ [degrees]')
    axs[0, 2].grid(True)

    # Axial induction factor
    #axs[1, 0].plot(..., '-k', linewidth=2)
    axs[1, 0].plot(r_norm, a_axial, 'o-.', color=BEM_color)
    axs[1, 0].axvspan(0, r_hub / R, color=[153 / 255, 0, 0], alpha=0.3)  # Hub region
    axs[1, 0].set_xlabel('r/R')
    axs[1, 0].set_ylabel('a')
    axs[1, 0].grid(True)

    # Angular induction factor
    axs[1, 1].plot(r_norm, a_angular, 'o-.', color=BEM_color)
    axs[1, 1].axvspan(0, r_hub / R, color=[153 / 255, 0, 0], alpha=0.3)  # Hub region
    axs[1, 1].set_xlabel('r/R')
    axs[1, 1].set_ylabel("a'")
    axs[1, 1].grid(True)

    # Tip loss correction
    axs[1, 2].plot(r_norm, F, 'o-.', color=BEM_color)
    axs[1, 2].axvspan(0, r_hub / R, color=[153 / 255, 0, 0], alpha=0.3)  # Hub region
    axs[1, 2].set_xlabel('r/R')
    axs[1, 2].set_ylabel('F')
    axs[1, 2].grid(True)

    plt.tight_layout()
    plt.show()

    #%%
    # --- Part C: Turbine Performance ---

    # Calculate U_rel for all blade points
    U_rel = np.sqrt((U_design*(1-a_axial))**2 + (lambda_r*U_design*(1+a_angular))**2)

    #%%

    # calculate actual Reynolds number for all points along the blade
    Re =  U_rel*c/nu

    # calculate CP for each annular blade element
    CP = (8/(lambda_design*N))*F*(lambda_r**3)*a_angular*(1-a_axial)*(1-(CD/(CL*np.tan(phi))))

    # calculate CT for each annular blade element
    CT = (8/(R**2))*a_axial*(1-a_axial)*r*F

    #%%
    # Find first element without hub overlap
    first_element = np.argmax((r > r_hub) & (r <= R))

    #%%
    # Calculate average Reynolds number for non-hub elements
    Re_blade = np.average(Re[first_element:N])

    # Calculate CP and CT for the entire rotor, excluding hub elements
    CP_turbine = sum(CP[first_element:N])
    CT_turbine = sum(CT[first_element:N])
    print('CP:',CP_turbine)
    print('CT:',CT_turbine)
    # --- Plotting ---
    fig, axs = plt.subplots(1, 3, figsize=(12, 5))

    # Performance coefficient (CP)
    axs[0].plot(r / R, CP, 'o-.', color=BEM_color, label='CP')
    axs[0].axvspan(0, r_hub / R, color=[153 / 255, 0, 0], alpha=0.3)  # Hub region
    axs[0].set_xlabel('r/R')
    axs[0].set_ylabel('$C_{P,i}$')
    axs[0].grid()

    # Thrust coefficient (CT)
    axs[1].plot(r / R, CT, 'o-.', color=BEM_color, label='CT')
    axs[1].axvspan(0, r_hub / R, color=[153 / 255, 0, 0], alpha=0.3)  # Hub region
    axs[1].set_xlabel('r/R')
    axs[1].set_ylabel('$C_{T,i}$')
    axs[1].grid()

    # Reynolds number (Re)
    axs[2].plot(r / R, Re, 'o-.', color=BEM_color, label='Element Re')
    axs[2].plot(r / R, np.full_like(r, Re_blade), '--k', label='Blade Average')
    axs[2].plot(r / R, np.full_like(r, Re_design), '-k', linewidth=2, label='Design Re')
    axs[2].axvspan(0, r_hub / R, color=[153 / 255, 0, 0], alpha=0.3,label = '$r_{hub}$')  # Hub region
    axs[2].set_xlabel('r/R')
    axs[2].set_ylabel('Re')
    axs[2].grid()
    axs[2].legend()

    plt.tight_layout()
    plt.show()

    return CP_turbine, CT_turbine, Re_blade


    # Calculate electrical power output [kW]
    #P_e = ...
    #print(f"Electrical Power Output: {P_e:.2f} kW")