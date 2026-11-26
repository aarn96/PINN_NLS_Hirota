# conservation.py
# ============================================================
# Computation functions for the conservation diagnostic of
# the quantity N(t) = integral |psi|^2 dx in NLS and Hirota.
#
# Formal derivation:
#   Multiplying the PDE by psi* and taking the imaginary part
#   yields the local continuity equation:
#       partial_t |psi|^2 + partial_x J = 0
#   Global conservation dN/dt = 0 requires [J]_boundary = 0,
#   guaranteed analytically by periodic BC, or verified
#   numerically by evaluating J(±L, t) directly on the
#   reference solution from the .mat file.
#
# Usage in the notebooks:
#   from conservation import (compute_norm, compute_J_hirota,
#                              compute_J_nls, check_boundary_flux)
# ============================================================

import numpy as np


def compute_norm(U, V, x_vals):
    """
    Conserved quantity: N(t) = integral |psi(x,t)|^2 dx
                             = integral (u^2 + v^2) dx

    |psi|^2 = u^2 + v^2 is invariant under permutation of u and v,
    so possible sign inversions in the training-code convention
    do not affect this computation.

    Trapezoidal integration: valid for uniform and non-uniform
    grids.

    Parameters
    ----------
    U, V   : (Nt, Nx)  Re(psi), Im(psi)
    x_vals : (Nx,)     spatial axis

    Returns
    -------
    N : (Nt,)  value of the conserved quantity at each t
    """
    # NumPy >= 2.0 removes np.trapz; keep a fallback for older versions.
    _trapezoid = getattr(np, "trapezoid", np.trapz)
    return _trapezoid(U**2 + V**2, x=x_vals, axis=1)


def compute_J_hirota(U, V, x, D2=1.0, D3=0.01, R3=0.01):
    """
    Local flux J_Hir(x,t) of the continuity equation for N(t)
    for the Hirota equation:
        partial_t |psi|^2 + partial_x J_Hir = 0

        J_Hir = J1 + J2 + J3
          J1 = D2 * (u*v_x - v*u_x)              [NLS-like]
          J2 = D3 * [2*(u*u_xx + v*v_xx)          [third-order dispersion]
                      - (u_x^2 + v_x^2)]
          J3 = 3*R3 * (u^2 + v^2)^2              [Raman / self-steepening]

    Note: R2 does not appear -- it cancels algebraically in the
    derivation (it generates no transport).

    Required fields: U and V -- H = |psi| is not enough for J1 and J2.

    Parameters
    ----------
    U, V     : (Nt, Nx)
    x        : (Nx,)     spatial grid (assumed uniform)
    D2,D3,R3 : Hirota parameters

    Returns
    -------
    J : (Nt, Nx)
    """
    dx  = x[1] - x[0]
    Ux  = np.gradient(U,  dx, axis=1)
    Vx  = np.gradient(V,  dx, axis=1)
    Uxx = np.gradient(Ux, dx, axis=1)
    Vxx = np.gradient(Vx, dx, axis=1)

    J1 = D2  * (U * Vx - V * Ux)
    J2 = D3  * (2*(U*Uxx + V*Vxx) - (Ux**2 + Vx**2))
    J3 = 3.0 * R3 * (U**2 + V**2)**2

    return J1 + J2 + J3

def compute_J_Yan(U, V, x, alpha=1.0, beta=0.01):
    """
    Local flux J_Yan(x,t) of the continuity equation for N(t)
    for the Hirota equation in the Zhou & Yan parametrization:
        partial_t |psi|^2 + partial_x J_Yan = 0

        J_Yan = J1 + J2 + J3
          J1 = 2*alpha * (u*v_x - v*u_x)              [NLS-like]
          J2 = beta * [2*(u*u_xx + v*v_xx)            [third-order dispersion]
                      - (u_x^2 + v_x^2)]
          J3 = 3*beta * (u^2 + v^2)^2                 [Raman / self-steepening]

    Note: unlike compute_J_hirota, here alpha plays a double role:
    it contributes to the flux through the dispersion term (J1, as D2=2*alpha) and is
    also the coefficient of the cubic term that cancels algebraically
    in the derivation (equivalent to R2 in the general Hirota equation).

    Required fields: U and V -- H = |psi| is not enough for J1 and J2.

    Parameters
    ----------
    U, V     : (Nt, Nx)
    x        : (Nx,)     spatial grid (assumed uniform)
    alpha, beta : Yan parameters

    Returns
    -------
    J : (Nt, Nx)
    """
    dx  = x[1] - x[0]
    Ux  = np.gradient(U,  dx, axis=1)
    Vx  = np.gradient(V,  dx, axis=1)
    Uxx = np.gradient(Ux, dx, axis=1)
    Vxx = np.gradient(Vx, dx, axis=1)

    J1 = 2*alpha  * (U * Vx - V * Ux)
    J2 = beta  * (2*(U*Uxx + V*Vxx) - (Ux**2 + Vx**2))
    J3 = 3.0 * beta * (U**2 + V**2)**2

    return J1 + J2 + J3

def compute_J_nls(U, V, x):
    """
    Local flux J_NLS(x,t) of the continuity equation for N(t)
    for NLS:
        J_NLS = u*v_x - v*u_x = Im(psi* psi_x)

    Limiting case of Hirota with D2=1, D3=0, R3=0.

    Parameters
    ----------
    U, V : (Nt, Nx)
    x    : (Nx,)

    Returns
    -------
    J : (Nt, Nx)
    """
    dx = x[1] - x[0]
    Ux = np.gradient(U, dx, axis=1)
    Vx = np.gradient(V, dx, axis=1)
    return U * Vx - V * Ux


def check_boundary_flux(U, V, x, equation='hirota',
                        D2=1.0, D3=0.01, R3=0.01):
    """
    Evaluates the residual boundary flux [J(+L,t) - J(-L,t)].

    Main use: case with no imposed BC (central experiment).
    If max|J(+L) - J(-L)| << 1 for all t, the domain is
    wide enough and dN/dt ≈ 0 is numerically valid.

    Parameters
    ----------
    U, V     : (Nt, Nx)  reference solution (U_star, V_star)
    x        : (Nx,)
    equation : 'hirota', 'yan' or 'nls'
    D2,D3,R3 : Hirota parameters (ignored for 'yan' and 'nls')

    Returns
    -------
    flux_residual : (Nt,)  J(+L,t) - J(-L,t)
    J_left        : (Nt,)  J(-L,t)
    J_right       : (Nt,)  J(+L,t)
    """

    if equation == 'hirota':
        J = compute_J_hirota(U,V,x, D2, D3, R3)
    elif equation == 'yan':
        J = compute_J_Yan(U,V, x)
    elif equation == 'nls':
        J = compute_J_nls(U, V, x)
    else:
        raise ValueError(f"Unknown equation: {equation}")

    J_left        = J[:, 0]
    J_right       = J[:, -1]
    flux_residual = J_right - J_left

    print(f"  max|J(+L) - J(-L)| = {np.max(np.abs(flux_residual)):.2e}")
    print(f"  max|J(-L,t)|       = {np.max(np.abs(J_left)):.2e}")
    print(f"  max|J(+L,t)|       = {np.max(np.abs(J_right)):.2e}")

    return flux_residual, J_left, J_right
