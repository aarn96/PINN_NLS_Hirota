# spearman.py
# ============================================================
# Computation functions for the Spearman correlation analysis
# between the PINN error and physical/numerical proxies.
#
# Operationalized hypotheses:
#   rho1 : error vs |q_exact|        (H1: amplitude)
#   rho2 : error vs |d_x q_exact|    (H1: gradient)
#   rho3 : error vs dist_NN          (H2: sampling)
#   rho4 : error vs F_mag            (internal consistency)
#   rho4a: error vs u_pinn           (consistency, real component)
#   rho4b: error vs v_pinn           (consistency, imaginary comp.)
#
# Note on large samples (n ~ 20000):
#   With large n, small values of rho are statistically
#   significant even if physically negligible. Always report
#   magnitude + p-value. Suggested threshold of practical
#   relevance: |rho| > 0.3 (Cohen 1988).
#
# Usage in the notebooks:
#   from spearman import (spearman_safe, compute_gradient_magnitude,
#                         compute_dist_nn)
# ============================================================

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats   import spearmanr


def spearman_safe(a, b):
    """
    Spearman correlation between two arrays.

    Ignores NaN and Inf (griddata artifacts at the borders).
    Returns (nan, nan) if there are fewer than 10 valid points.

    Parameters
    ----------
    a, b : arrays (flattened internally)

    Returns
    -------
    rho : float  Spearman coefficient
    p   : float  p-value under H0: rho = 0
    """
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 10:
        return np.nan, np.nan
    return spearmanr(a[mask], b[mask])


def compute_gradient_magnitude(H_exact, x_vals):
    """
    Magnitude of the spatial gradient of |q|:
        |d_x q_exact|(x, t)

    Proxy for local complexity (hypothesis H1: gradient).

    Parameters
    ----------
    H_exact : (Nt, Nx)
    x_vals  : (Nx,)

    Returns
    -------
    grad_mag : (Nt, Nx)
    """
    dx = x_vals[1] - x_vals[0]
    return np.abs(np.gradient(H_exact, dx, axis=1))


def compute_dist_nn(X_f_train, x_vals, t_vals):
    """
    Distance from each grid point to the nearest collocation
    point (nearest-neighbor distance).

    Proxy for the effect of sampling (hypothesis H2: numerical).
    rho3 ≈ 0 rejects the claim that the error pattern is
    controlled by the distribution of collocation points.

    Parameters
    ----------
    X_f_train : (Nf, 2)  collocation points [x, t]
    x_vals    : (Nx,)
    t_vals    : (Nt,)

    Returns
    -------
    dist : (Nt, Nx)
    """
    X_grid, T_grid = np.meshgrid(x_vals, t_vals)
    query_pts = np.column_stack([X_grid.ravel(), T_grid.ravel()])
    tree = cKDTree(X_f_train)
    dist, _ = tree.query(query_pts)
    return dist.reshape(X_grid.shape)
