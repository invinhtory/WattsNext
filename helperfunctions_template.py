# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
def get_foil_data(foil_file):
    """
    Description: load airfoil data from data file

    return a foil structure with four fields (Re, angle of attack, coef.
    of lift, and coef. of drag, based on performance data
    
    Load performance file data from Excel spreadsheet
    Output:
    - foil: A dictionary containing Reynolds number, angle of attack (in radians),
      coefficient of lift (CL), and coefficient of drag (CD)
    """
    # Load performance file data from the Excel spreadsheet
    # 'output 1: numeric data, output 2: text data, output 3: raw data'
    data = pd.read_excel(foil_file)  # Reads the spreadsheet
    
    # Extract columns from the spreadsheet
    # num:
    # - col 1: Reynolds number (chord-based)
    # - col 2: angle of attack [degrees]
    # - col 3: coefficient of lift
    # - col 4: coefficient of drag
    # - col 5: pitching moment coefficient (not used)
    foil = {
        'Re': np.array(data.iloc[1:-1, 0].to_numpy(),dtype = np.float64),                  # Reynolds number
        'alpha': np.radians(np.array(data.iloc[1:-1, 1].to_numpy(),dtype = np.float64)),   # Angle of attack [radians]
        'CL': np.array(data.iloc[1:-1, 2].to_numpy(),dtype = np.float64),                 # Coefficient of lift
        'CD': np.array(data.iloc[1:-1, 3].to_numpy(),dtype = np.float64)                  # Coefficient of drag
    }
    
    return foil

def interp_foil(foil, Re, alpha):
    """
    %Airfoil data interpolation

    Changelog 
    2020-23-04, Greg Talpey
       edited to use griddata function for interpolation of 2D scattered
       data to eliminate (i) need for looping through mulitple alphas and (ii) 
       need for two step interpolation.

    Given specific Reynolds number and angle of attack [degrees], determine
    the lift and drag coefficients by using interpolation of scattered data
    by MATLAB's griddata function
    Airfoil data interpolation using 2D scattered data.
    
    Inputs:
    - foil: A dictionary containing CL, CD as functions of alpha and Re
    - Re: Chord-based Reynolds number (scalar)
    - alpha: Array of angles of attack [radians]
    
    Outputs:
    - CL: Coefficient of lift for each angle of attack
    - CD: Coefficient of drag for each angle of attack
    """
    # Extend Reynolds number to vector the same length as alpha
    Re = np.full(np.shape(alpha), Re)
    # Check if Reynolds number is outside tabulated range
    if np.min(Re) < foil['Re'].min():
        print('Warning: Re below tabulated range of data')
        CL = np.full(np.shape(alpha), np.nan)
        CD = np.full(np.shape(alpha), np.nan)
    elif np.max(Re) > foil['Re'].max():
        print('Warning: Re above tabulated range of data')
        CL = np.full(np.shape(alpha), np.nan)
        CD = np.full(np.shape(alpha), np.nan)
    else:
        CL = griddata(np.column_stack((foil['Re'], foil['alpha'])), foil['CL'], (Re, alpha), method='linear',rescale = True)
        CD = griddata(np.column_stack((foil['Re'], foil['alpha'])), foil['CD'], (Re, alpha), method='linear',rescale = True)
    return CL, CD

def tiploss(B, R, r, phi):
    """
    Helper function for Prandtl tip-loss correction.
    
    Inputs:
    - B: Number of blades
    - R: Turbine radius [m]
    - r: Position along radius [m] (can be scalar or array)
    - phi: Relative wind angle [radians] (can be scalar or array)
    
    Outputs:
    - F: Tip-loss correction factor (scalar or array depending on input)
    
    Note: If `r` and `phi` are arrays, F will be a vector of tip-loss corrections along the blade span.
    """
    # Equation for Prandtl tip-loss correction
    F = (2/np.pi)*np.acos(np.exp(-(B/2)*(R-r)/(r*np.sin(phi))))
    
    return F

def BEM_CL(alpha, theta_P0, theta_T, sigma, lambda_r, B, R, r):
    """
    Coefficient of lift based on BEM relations.
    
    Inputs:
    - alpha: Angle of attack [radians]
    - theta_P0: Pitch angle at blade tip [radians]
    - theta_T: Twist angle of blade element [radians]
    - sigma: Solidity [~]
    - lambda_r: Local speed ratio [~]
    - B: Number of blades
    - R: Turbine radius [m]
    - r: Position along radius [m]
    
    Output:
    - CL_BEM: Coefficient of lift from BEM relation
    
    Note: alpha, theta_T, sigma, lambda_r, and r can be a single element or a vector of elements. If a
    vector of elements, CL_EM will be a vector of lift coefficients along the blade span.
    """
    # Calculate angle of relative wind
    phi = alpha + theta_P0 + theta_T
    
    # Calculate tip loss correction
    F = tiploss(B, R, r, phi)  
    
    # Calculate coefficient of lift based on BEM relations for thrust and torque
    CL_BEM = (4*F*np.sin(phi)/sigma)*((np.cos(phi)-lambda_r*np.sin(phi))/(np.sin(phi)+lambda_r*np.cos(phi)))
    
    return CL_BEM

def CL_intersect(guess, foil, theta_P0, theta_T, sigma, lambda_r, B, R, r, Re):
    """
    Helper routine to calculate alpha for a blade segment that satisfies both
    aerodynamic lift coefficient data and BEM lift coefficient relations.
    
    Inputs:
    - guess: Current value for angle of attack (iterated by optimization until residual is small)
    - foil: Structure containing lift and drag coefficients as functions of alpha and Reynolds number
    - theta_P0: Pitch angle at the tip of the blade [radians]
    - theta_T: Twist angle [radians]
    - sigma: Solidity [~]
    - lambda_r: Local speed ratio [~]
    - B: Number of blades
    - R: Turbine radius [m]
    - r: Element position along radius [m]
    
    Outputs:
    - residual: Residual value between aerodynamic lift coefficient and BEM lift coefficient for the current guess for alpha. 
    Used by minimize to inform a guess for a new value for alpha that might reduce the residual.
    
    When the residual satisfies the criteria specified in "options",
    'fminsearch' will return the current value of "guess" (i.e., alpha).
    """
    # Copy over "guess" to a variable with physical meaning
    alpha = guess  # Angles of attack (either single angle or all blade elements)
    
    # Calculate coefficient of lift from BEM relations
    CL_BEM = BEM_CL(alpha, theta_P0, theta_T, sigma, lambda_r, B, R, r) #Note: You will want to call the BEM_CL function that you completed
    #in Problem 2 to calculate this
    
    # Calculate coefficient of lift based on airfoil data, angle of attack, and Reynolds number
    CL_foil = CL_foil = interp_foil(foil, Re, alpha)[0] #You will want to call the interp_foil function, as you did in problem 2
    
    # Calculate residual (minimization target) - difference between BEM and aerodynamic lift coefficient
    # Note: summation is to evaluate over all foil elements
    residual = np.sum((CL_BEM - CL_foil) ** 2)
    
    return residual

def calculate_parameters(CL, alpha, theta_P0, theta_T, sigma, B, R, r):
    """
    Function to calculate various parameters for an element once actual angle of 
    attack, as well as lift and drag coefficients, are known.
    
    Inputs:
    - CL: Coefficient of lift [~]
    - alpha: Angle of attack [radians]
    - theta_P0: Pitch angle at blade tip [radians]
    - theta_T: Twist angle of blade element [radians]
    - sigma: Solidity [~]
    - B: Number of blades
    - R: Turbine radius [m]
    - r: Position along radius [m]

    Outputs:
    - a_axial: Axial induction factor
    - a_angular: Angular induction factor
    - phi: Angle of relative wind [radians]
    - F: Tip-loss correction factor
    """
    # Calculate angle of relative wind
    phi = alpha + theta_P0 + theta_T

    # Calculate tip loss
    F = tiploss(B, R, r, phi)
    
    # Calculate axial induction factor - This calculation should neglect C_D/C_L terms
    a_axial = (1 + ((4*F*(np.sin(phi)**2))/(sigma*CL* np.cos(phi))))**(-1)

    # Calculate angular induction factor - This calculation should neglect C_D/C_L terms
    a_angular = (((4*F*np.cos(phi))/(sigma*CL)) -1)**(-1)

    return a_axial, a_angular, phi, F

def get_design_alpha(foil, Re):
    """
    Get the angle of attack [radians] with maximum CL/CD at design Reynolds number.

    Inputs:
    - foil: Dictionary containing Reynolds number, angle of attack, coefficient of lift, and coefficient of drag
    - Re: Design Reynolds number

    Outputs:
    - alpha: Angle of attack [radians] with maximum CL/CD at design Reynolds number
    """

    print(f"foil: {foil}")
    print(f"Re: {Re}")

    # Create a common range of angles of attack for interpolation of foil data
    alphas = np.arange(min(foil["alpha"]), max(foil["alpha"]), 0.0001)
    # Initialize return value
    CL_CD = np.zeros_like(alphas)

    # Calculate lift and drag coefficient at each angle of attack
    CL, CD = interp_foil(foil, Re, alphas)

    # Calculate ratio of lift to drag
    CL_CD = CL / CD

    print(f"CL_CD: {CL_CD}")

    # plt.figure(1)
    # plt.plot(alphas, CL_CD, '--o')
    # plt.grid(True)
    # Identify maximum ratio of lift to drag coefficient
    alpha = alphas[np.nanargmax(CL_CD)]

    return alpha