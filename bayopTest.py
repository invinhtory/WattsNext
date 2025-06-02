import numpy as np
from bayes_opt import BayesianOptimization
from BEM import BEM_analysis
from neuralfoil import get_aero_from_kulfan_parameters
import traceback
import pandas as pd
import os
import json
from datetime import datetime


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
    alpha_start, alpha_end = alpha_range
    alpha = np.linspace(alpha_start, alpha_end, 200)
    Re_list = np.linspace(1e6, 1e7, 10)

    Re_all, alpha_all, CL_all, CD_all = [], [], [], []
    for Re in Re_list:
        aero_ = get_aero_from_kulfan_parameters(
            kulfan_parameters=kf_foil,
            alpha=alpha,
            Re=Re,
            model_size="xxlarge"
        )
        Re_all.extend([Re] * len(alpha))
        alpha_all.extend(alpha)
        CL_all.extend(aero_["CL"])
        CD_all.extend(aero_["CD"])

    return {
        "Re": np.array(Re_all, dtype=np.float64),
        "alpha": np.radians(np.array(alpha_all, dtype=np.float64)),
        "CL": np.array(CL_all, dtype=np.float64),
        "CD": np.array(CD_all, dtype=np.float64)
    }


def compute_shape_penalty(upper, lower):
    roughness_penalty = np.sum(np.diff(upper)**2 + np.diff(lower)**2)
    curvature_penalty = np.sum(np.diff(upper, 2)**2 + np.diff(lower, 2)**2)
    return roughness_penalty, curvature_penalty


def bayop_test(U_design, R, eta_0, lambda_design, B, R_hub, N,
               initial_kulfan, Re_design, rho, nu, alpha_range):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_dir = os.path.join("optimization_history", f"run_{timestamp}")
    os.makedirs(history_dir, exist_ok=True)

    optimization_history = []
    iteration_counter = [0]

    def cst_objective(**params):
        try:
            bo_params = [params[f'p{i}'] for i in range(18)]

            if np.any(np.isnan(bo_params)) or np.any(np.isinf(bo_params)):
                print("Invalid params: NaN or Inf detected")
                return 0.0

            kulfan = bo_to_kulfan(bo_params)

            upper = np.array(kulfan["upper_weights"])
            lower = np.array(kulfan["lower_weights"])

            if np.any(lower >= upper):
                print("Constraint violated: lower weights cross or equal upper weights")
                return 0.0

            try:
                aero = get_CL_CD_perf(kulfan, alpha_range)
            except Exception as e:
                print(f"Aero performance failed: {e}")
                return 0.0

            for key in ['CL', 'CD', 'alpha', 'Re']:
                if key not in aero or np.any(np.isnan(aero[key])) or np.any(np.isinf(aero[key])):
                    print(f"Bad aero data: {key}")
                    return 0.0

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

            try:
                CP, P_e, Re_blade = BEM_analysis(
                    U_design, R, eta_0, lambda_design, B, R_hub, N,
                    performance_file, Re_design, rho=rho, nu=nu
                )

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

    pbounds = {
        'p0': (0.01, 0.3),    # leading_edge_weight
        'p1': (0.01, 0.05),   # TE_thickness
    }
    # Upper weights p2 to p9 should be >= 0 (e.g. 0 to 0.1)
    for i in range(2, 10):
        pbounds[f'p{i}'] = (0.0, 0.15)

    # Lower weights p10 to p17 should be <= 0 (e.g. -0.1 to 0)
    for i in range(10, 18):
        pbounds[f'p{i}'] = (-0.15, 0.0)

    optimizer = BayesianOptimization(
        f=cst_objective,
        pbounds=pbounds,
        verbose=2,
        random_state=42,
        allow_duplicate_points=True
    )

    optimizer.set_gp_params(alpha=1e-3, n_restarts_optimizer=10)

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

    try:
        optimizer.maximize(init_points=20, n_iter=80)
    except Exception as e:
        print(f"Bayesian optimization failed: {e}")
        traceback.print_exc()

    with open(os.path.join(history_dir, "optimization_history.json"), 'w') as f:
        json.dump(optimization_history, f, indent=2)

    best_params = optimizer.max['params']
    kulfan_best = bo_to_kulfan(best_params)

    return kulfan_best, optimizer.max['target']
