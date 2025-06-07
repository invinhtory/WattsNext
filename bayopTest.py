import numpy as np
from bayes_opt import BayesianOptimization, acquisition
from BEM import BEM_analysis, BEM_analysis_looped
from neuralfoil import get_aero_from_kulfan_parameters
import traceback
import pandas as pd
import os
import json
from datetime import datetime
import scipy as sci


def bo_to_kulfan(bo):
    """Convert optimization parameters to Kulfan format"""
    if isinstance(bo, dict):
        bo = [bo[f'p{i}'] for i in range(18)]
    return {
        "leading_edge_weight": bo[0],
        "TE_thickness": bo[1],
        "upper_weights": list(bo[2:10]),
        "lower_weights": list(bo[10:18])
    }


def get_CL_CD_perf(kf_foil, alpha_range):
    """Calculates lift and drag coffecients of kf_foil over a range of angle of attack (alpha) using NeuralFoil"""
    # Define range of alphas and Reynolds numbers
    alpha_start, alpha_end = alpha_range
    alpha = np.linspace(alpha_start, alpha_end, 200)
    Re_list = np.linspace(1e6, 1e7, 10)

    Re_all, alpha_all, CL_all, CD_all = [], [], [], []
    # For each Reynold number, get the aerodynamic performance of the foil from NeuralFoil and add to list
    for Re in Re_list:
        aero_ = get_aero_from_kulfan_parameters(
            kulfan_parameters=kf_foil,
            alpha=alpha,
            Re=Re,
            model_size="xxsmall"
        )
        Re_all.extend([Re] * len(alpha))
        alpha_all.extend(alpha)
        CL_all.extend(aero_["CL"])
        CD_all.extend(aero_["CD"])

    # Return all Reynolds numbers, alphas, and resulting lift and drag coefficients
    return {
        "Re": np.array(Re_all, dtype=np.float64),
        "alpha": np.radians(np.array(alpha_all, dtype=np.float64)),
        "CL": np.array(CL_all, dtype=np.float64),
        "CD": np.array(CD_all, dtype=np.float64)
    }


def compute_shape_penalty(upper, lower):
    """Sets penalties to the coefficient of performance if the airfoil shape has very sharp or jagged edges"""
    roughness_penalty = np.sum(np.diff(upper)**2 + np.diff(lower)**2)
    curvature_penalty = np.sum(np.diff(upper, 2)**2 + np.diff(lower, 2)**2)
    return roughness_penalty, curvature_penalty


def bayop_test(U_design, R, eta_0, lambda_design, B, R_hub, N,
               initial_kulfan, Re_design, rho, nu, alpha_range):
    """runs the Bayesian optimization process to maximize coefficient of performance by changing airfoil shape"""

    # creates a directory to store all iteration data as a .json file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_dir = os.path.join("optimization_history", f"run_{timestamp}")
    os.makedirs(history_dir, exist_ok=True)

    optimization_history = []
    iteration_counter = [0]

    # the objective function for optimizing Cp
    def cst_objective(**params):
        try:
            # creates a list of Bayesian optimization parameters
            bo_params = [params[f'p{i}'] for i in range(18)]

            if np.any(np.isnan(bo_params)) or np.any(np.isinf(bo_params)):
                print("Invalid params: NaN or Inf detected")
                return 0.0

            # converts the list of BO parameters into Kulfan, CST-parameter friendly dictionary
            kulfan = bo_to_kulfan(bo_params)

            # looks at specifically the upper and lower weights of the Kulfan parameters
            upper = np.array(kulfan["upper_weights"])
            lower = np.array(kulfan["lower_weights"])

            if np.any(lower >= upper):
                print("Constraint violated: lower weights cross or equal upper weights")
                return 0.0

            # Calculates lift and drag coefficients of the kulfan Airfoil
            try:
                aero = get_CL_CD_perf(kulfan, alpha_range)
            except Exception as e:
                print(f"Aero performance failed: {e}")
                return 0.0

            for key in ['CL', 'CD', 'alpha', 'Re']:
                if key not in aero or np.any(np.isnan(aero[key])) or np.any(np.isinf(aero[key])):
                    print(f"Bad aero data: {key}")
                    return 0.0

            # saves the Aerodynamic performance of the kulfan Airfoil into a dataframe, which is then saved into an excel file
            df = pd.DataFrame({
                'Re': aero['Re'],
                'alpha': np.degrees(aero['alpha']),
                'CL': aero['CL'],
                'CD': aero['CD']
            })

            if df.isnull().any().any():
                print("DataFrame contains NaN values")
                return 0.0

            performance_file = os.path.join(history_dir, "temp_airfoil_data.xlsx")
            with pd.ExcelWriter(performance_file) as writer:
                df.to_excel(writer, index=False)

            # Uses the saved aerodynamic performance to calculate a coefficient of performance
            try:
                CP, P_e, Re_blade = BEM_analysis_looped(
                    U_design, R, eta_0, lambda_design, B, R_hub, N, performance_file, Re_design, rho, nu
                )

                # saves this iteration (airfoil data + Cp) into the .json file for later use in visualization
                iteration_data = {
                    'iteration': iteration_counter[0],
                    'parameters': kulfan,
                    'CP': float(CP) if CP else 0.0,
                    'P_e': float(P_e) if P_e else 0.0,
                    'Re_blade': float(Re_blade) if Re_blade else 0.0
                }

                optimization_history.append(iteration_data)

                with open(os.path.join(history_dir, f"iteration_{iteration_counter[0]}.json"), 'w') as f:
                    json.dump(iteration_data, f, indent=2)

                iteration_counter[0] += 1

                if not CP or not np.isfinite(CP):
                    print("Invalid CP returned")
                    return 0.0

                upper = np.array(kulfan["upper_weights"])
                lower = np.array(kulfan["lower_weights"])
                
                # applies additional penalties to Cp for weird shapes
                roughness_penalty, curvature_penalty = compute_shape_penalty(upper, lower)

                return float(CP) - 0.1 * roughness_penalty - 0.5 * curvature_penalty

            finally:
                if os.path.exists(performance_file):
                    try:
                        os.remove(performance_file)
                    except Exception as e:
                        print(f"Failed to remove temp file: {e}")

        except Exception as e:
            print(f"Objective function error: {e}")
            traceback.print_exc()
            return 0.0

    # A constraint function that stops the Bayesian Optimization from looking at infeasible airfoils
    def constraint_function(**params):
        try:
            #gets the airfoil parameters from Bayesian Optimization and converts into airfoil
            bo_params = [params[f'p{i}'] for i in range(18)]

            if np.any(np.isnan(bo_params)) or np.any(np.isinf(bo_params)):
                print("Invalid params: NaN or Inf detected")
                return 0.0

            kulfan = bo_to_kulfan(bo_params)

            upper = np.array(kulfan["upper_weights"])
            lower = np.array(kulfan["lower_weights"])

            # if any of the upper weights is < the lower weight in a specific region, 
            # the airfoil is invalid. If invalid, return -1, else return 1.
            for i in np.arange(len(upper)):
                if (lower[i] >= upper[i]):
                    return -1
            return 1
            
        except Exception as e:
            print(f"Objective function error: {e}")
            traceback.print_exc()
            return 0.0

    # Set the bounds of the leading edge and trailing edge thickness
    pbounds = {
        'p0': (0.01, 0.3),    # leading_edge_weight
        'p1': (0.01, 0.05),   # TE_thickness
    }
    # Upper weights p2 to p9 should be >= 0 (e.g. 0 to 0.1)
    for i in range(2, 10):
        pbounds[f'p{i}'] = (0, 1.2)

    # Lower weights p10 to p17 should be <= 0 (e.g. -0.1 to 0)
    for i in range(10, 18):
        pbounds[f'p{i}'] = (-0.2, 0.2)

    # Set different acquisition functions to try to change the exploration vs. exploitation balance
    acquisition_function = acquisition.ExpectedImprovement(xi=0.03)
    acquisition_function2 = acquisition.ProbabilityOfImprovement(xi=1E-3)
    constraint = sci.optimize.NonlinearConstraint(constraint_function, 0, np.inf)

    # Create the Bayesian Optimization optimizer with appropriate objective, constraint, & acqisition functions.  
    optimizer = BayesianOptimization(
        f=cst_objective,
        acquisition_function=acquisition_function,
        constraint=constraint,
        pbounds=pbounds,
        verbose=2,
        random_state=42,
        allow_duplicate_points=True
    )

    # Set Gaussian Process parameters, defaults to a Matern 2.5 Kernel.
    optimizer.set_gp_params(alpha=1e-3, n_restarts_optimizer=10)

    # Probes the initial Kulfan airfoil set in the function. This is a starting point value.
    if initial_kulfan:
        init_point = {
            'p0': initial_kulfan['leading_edge_weight'],
            'p1': initial_kulfan['TE_thickness']
        }
        for i, w in enumerate(initial_kulfan['upper_weights']):
            init_point[f'p{i+2}'] = w
        for i, w in enumerate(initial_kulfan['lower_weights']):
            init_point[f'p{i+10}'] = w
        optimizer.probe(init_point)

    # Runs the optimization process with 20 random points and 100 following iterations.
    try:
        optimizer.maximize(init_points=20, n_iter=100)
    except Exception as e:
        print(f"Bayesian optimization failed: {e}")
        traceback.print_exc()

    # saves the entire history into a json file
    with open(os.path.join(history_dir, "optimization_history.json"), 'w') as f:
        json.dump(optimization_history, f, indent=2)

    best_params = optimizer.max['params']
    kulfan_best = bo_to_kulfan(best_params)

    # returns the best airfoil, highest Cp achieved, and the directory the json file is saved
    return kulfan_best, optimizer.max['target'], history_dir
