# --- Part 5.4: Bayesian Optimization Loop ---
import numpy as np
from scipy.stats import norm
from scipy.stats.qmc import Sobol
import aerosandbox as asb
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor

# kernel
kernel = RBF()

# gp regressor
gp = GaussianProcessRegressor(kernel=kernel)

# ---------------- Helper Functions ----------------
def bo_to_kulfan(bo_vec):
    """Convert 18‑element design vector -> Kulfan dictionary."""
    assert len(bo_vec) == 18, "Design vector must have 18 parameters"
    return {
        "leading_edge_weight": bo_vec[0],
        "TE_thickness":       bo_vec[1],
        "upper_weights":      list(bo_vec[2:10]),
        "lower_weights":      list(bo_vec[10:18]),
    }

def make_foil_kf(kf_dict, name="candidate"):
    """Build an AeroSandbox KulfanAirfoil from a Kulfan dictionary."""
    return asb.KulfanAirfoil(
        name=name,
        lower_weights=kf_dict["lower_weights"],
        upper_weights=kf_dict["upper_weights"],
        leading_edge_weight=kf_dict["leading_edge_weight"],
        TE_thickness=kf_dict["TE_thickness"],
    )

# Objective: *negative* figure‑of‑merit so we can minimise (GP/EI code = minimisation)
def objective_function(x_vec, re, alpha_range):
    """Returns a scalar to *minimise* (−C_p surrogate)."""
    try:
        kf = bo_to_kulfan(x_vec)
        foil = make_foil_kf(kf, "opt")

        # Use NeuralFoil surrogate for aero data (fast, differentiable)
        polar = foil.get_aero_from_neuralfoil(alpha_range, re)
        CL = np.array(polar['CL'])
        CD = np.array(polar['CD'])
        ld_ratio = CL / CD
        # Simple proxy for C_p: maximise max(L/D)
        cp_proxy = np.max(ld_ratio)
        return -float(cp_proxy)  # negative because we minimise
    except Exception as err:
        # Penalise failed evaluations so they are not selected again
        return 1e3

# Expected Improvement acquisition (minimisation form)
def expected_improvement(X, X_sample, y_sample, model, xi=0.01):
    mu, sigma = model.predict(X, return_std=True)
    mu_sample_opt = np.min(y_sample)
    with np.errstate(divide='warn'):
        imp = mu_sample_opt - mu - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    return ei

def bo_loop(re, alpha_range):
    dim = 18
    bounds = np.array([[-1.0, 1.0]] * dim)

    # Initial design (Sobol sequence)
    sob_sampler = Sobol(d=dim, scramble=True)
    n_initial = 20
    X_sample = sob_sampler.random_base2(m=int(np.ceil(np.log2(n_initial))))
    X_sample = bounds[:,0] + X_sample * (bounds[:,1] - bounds[:,0])

    y_sample = np.array([objective_function(x, re, alpha_range) for x in X_sample])

    # Initialise GP (re‑use kernel defined earlier)
    gp.fit(X_sample, y_sample)

    # History trackers
    best_idx = np.argmin(y_sample)
    print(f'Initial best (proxy C_p) = {-y_sample[best_idx]:.4f}')

    n_iter = 30  # optimisation budget
    for it in range(n_iter):
        # Random candidate set for acquisition (quasi‑Monte Carlo)
        X_cand = bounds[:,0] + np.random.rand(1024, dim) * (bounds[:,1] - bounds[:,0])
        ei = expected_improvement(X_cand, X_sample, y_sample, gp)
        x_next = X_cand[np.argmax(ei)]
        y_next = objective_function(x_next, re, alpha_range)

        # Update dataset
        X_sample = np.vstack((X_sample, x_next))
        y_sample = np.append(y_sample, y_next)

        # Re‑fit GP
        gp.fit(X_sample, y_sample)

        # Progress report
        if y_next < y_sample[best_idx]:
            best_idx = len(y_sample) - 1
        print(f"Iter {it+1:02d}: proxy C_p = {-y_sample[best_idx]:.4f}")

    # Store best design
    best_design = X_sample[best_idx]
    best_kf = bo_to_kulfan(best_design)
    best_foil = make_foil_kf(best_kf, name='best_foil')
    print('\nBest design Kulfan vector:', best_design)

    # Quick look at geometry
    fig, ax = plt.subplots(figsize=(6, 2))
    best_foil.draw()

    return best_design, best_kf, best_foil

