"""
sampling_strategies.py
======================
Module of collocation-point sampling strategies for PINNs.
Designed to be integrated with notebooks/PINN_Torch_NLS_Hirota.ipynb.

THEORETICAL BACKGROUND
----------------------
In a PINN, the PDE residual loss is defined as:

    L_res = (1/Nf) * Σ_i |f(x_i, t_i)|²

where f(x,t) is the PDE residual evaluated at the collocation points
{(x_i, t_i)}. This sum is an approximation of the integral:

    I = ∫∫_Ω |f(x,t)|² dx dt

The quality of that approximation depends directly on how the
collocation points are distributed. This is the same problem as in
numerical integration: different sampling strategies approximate the
integral with different accuracy.

The metric that measures how uniformly a set of points covers the
domain is called DISCREPANCY. The lower the discrepancy, the better
the coverage of the domain, and hence the better the approximation of
the residual integral. The Koksma-Hlawka theorem formalizes this relation:

    |I - approximation| ≤ D*(P) × V(f)

where D*(P) is the discrepancy of the point set P and V(f) is the total
variation of f over the domain.

IMPLEMENTED STRATEGIES
----------------------
1. LHS       — Latin Hypercube Sampling (current baseline)
2. Grid      — Coherent uniform grid, finite-difference style
3. Random    — Uniform random sampling (pseudo-random, Monte Carlo)
4. Sobol     — Base-2 low-discrepancy quasi-random sequence

KEY DIFFERENCE BETWEEN GRID AND THE REST
----------------------------------------
Grid builds an explicit rows × columns structure over the domain.
It therefore needs to define how many points go in each dimension
(Nx_g × Nt_g = Nf), which requires knowing the aspect ratio of the
domain (Lx / Lt) in order to keep the cells approximately square.

Random, LHS and Sobol sample the continuous domain [lb, ub]²
directly, with no grid structure. They only need the total number
of points Nf. They are agnostic to the shape of the domain.

DEPENDENCIES
------------
- numpy
- scipy.stats.qmc  (available in scipy >= 1.7, no extra installation on Colab)

COMPATIBILITY
-------------
Designed for Google Colab with PyTorch. It does not require pyDOE or any
external libraries beyond numpy and scipy.

BASIC USAGE
-----------
    from sampling_strategies import get_sampling

    points = get_sampling(
        strategy = 'Grid',
        lb       = np.array([-5.0, 0.0]),
        ub       = np.array([ 5.0, np.pi/2]),
        Nf       = 10000,
        N0       = 50,
        Nb       = 50,
        Exact_u  = Exact_u,
        Exact_v  = Exact_v,
        x_all    = x_all,
        t_all    = t_all,
        seed     = 1234,
    )

    x0        = points['x0']
    u0_train  = points['u0_train']
    v0_train  = points['v0_train']
    tb_train  = points['tb_train']
    X_f_train = points['X_f_train']

REFERENCES
----------
- Wu, C., Zhu, M., Tan, Q., Kartha, Y., & Lu, L. (2023).
  A comprehensive study of non-adaptive and residual-based adaptive
  sampling for physics-informed neural networks.
  Computer Methods in Applied Mechanics and Engineering, 403, 115671.
  https://doi.org/10.1016/j.cma.2022.115671

- Sobol, I. M. (1967). On the distribution of points in a cube and the
  approximate evaluation of integrals. USSR Computational Mathematics
  and Mathematical Physics, 7(4), 86-112.

- McKay, M. D., Beckman, R. J., & Conover, W. J. (1979). A comparison
  of three methods for selecting values of input variables in the
  analysis of output from a computer code. Technometrics, 21(2), 239-245.

- Niederreiter, H. (1992). Random Number Generation and Quasi-Monte Carlo
  Methods. SIAM.
"""

import numpy as np
from scipy.stats import qmc


# ==============================================================================
# AVAILABLE STRATEGIES
# ==============================================================================

AVAILABLE_STRATEGIES = ['LHS', 'Grid', 'Random', 'Sobol']


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _nearest_exact_ic(x_pts, x_all, Exact_u, Exact_v):
    """
    Retrieves the exact values of the initial condition (t=0) at the
    points x_pts by nearest-neighbor search in x_all.

    This approach is necessary because the analytical solution is only
    available on the discrete grid of the .mat file. It is used only
    in sample_grid, where the IC points come from the interior grid
    and do not coincide exactly with the nodes of x_all.

    LHS, Random and Sobol do not use this function: their IC points are
    obtained directly from x_all via np.random.choice, guaranteeing
    exact coincidence with the grid without any interpolation.

    Parameters
    ----------
    x_pts   : np.ndarray, shape (N, 1)
        x coordinates where the initial condition is needed.
    x_all   : np.ndarray, shape (Nx, 1)
        Spatial grid of the .mat file.
    Exact_u : np.ndarray, shape (Nx, Nt)
        Real part of the exact solution.
    Exact_v : np.ndarray, shape (Nx, Nt)
        Imaginary part of the exact solution.

    Returns
    -------
    u0 : np.ndarray, shape (N, 1)
        Real part of the exact solution at t=0 for each point x.
    v0 : np.ndarray, shape (N, 1)
        Imaginary part of the exact solution at t=0 for each point x.

    Notes
    -----
    Column 0 of Exact_u and Exact_v is always indexed, which corresponds
    to t=0 under the convention of the .mat file (columns = time instants).
    """
    x_flat = x_all.flatten()                                      # (Nx,)
    idx    = np.array([np.argmin(np.abs(x_flat - xi))
                       for xi in x_pts.flatten()])                # (N,)
    u0 = Exact_u[idx, 0:1]    # column 0 -> t = 0
    v0 = Exact_v[idx, 0:1]
    return u0, v0


def _compute_grid_dims(Nf, lb, ub):
    """
    Computes the dimensions of the interior grid (Nx_g × Nt_g ≈ Nf)
    keeping the cells approximately square in physical units.

    The aspect ratio of the domain guides the distribution of points:
        ratio = Lx / Lt

    The system of equations to solve is:
        Nx_g × Nt_g = Nf        (total number of points)
        Nx_g / Nt_g = ratio     (square cells in physical units)

    Substituting: Nt_g = sqrt(Nf / ratio), Nx_g = Nf / Nt_g

    Parameters
    ----------
    Nf  : int
        Desired number of interior points.
    lb  : np.ndarray, shape (2,)
        Lower bound of the domain [x_min, t_min].
    ub  : np.ndarray, shape (2,)
        Upper bound of the domain [x_max, t_max].

    Returns
    -------
    Nx_g : int
        Number of points in the spatial direction x (excluding boundaries).
    Nt_g : int
        Number of points in the temporal direction t (excluding IC).

    Notes
    -----
    Nx_g × Nt_g may differ slightly from Nf because of rounding to integers.
    The difference is always below 5%.
    """
    Lx    = ub[0] - lb[0]
    Lt    = ub[1] - lb[1]
    ratio = Lx / Lt

    Nt_g = max(2, int(np.round(np.sqrt(Nf / ratio))))
    Nx_g = max(2, int(np.round(Nf / Nt_g)))

    return Nx_g, Nt_g


# ==============================================================================
# SAMPLING STRATEGIES
# ==============================================================================

def sample_lhs(lb, ub, Nf, N0, Nb, x_all, t_all, Exact_u, Exact_v, seed=1234):
    """
    LHS — Latin Hypercube Sampling.

    IMPLEMENTATION
    --------------
    Uses scipy.stats.qmc.LatinHypercube, which implements LHS with an
    option to optimize the distribution (centered, maximin, etc.).

    IC and BC: sampled randomly from the discrete grid of the .mat file,
    replicating the behavior of the original notebook. The legacy generator
    np.random.seed + np.random.choice is used to guarantee reproducibility
    identical to the training notebook PINN_Torch_NLS_Hirota.ipynb.
    The interior sampler (qmc.LatinHypercube) uses its own internal seed,
    independent of the state of the legacy generator.

    Parameters
    ----------
    lb, ub   : np.ndarray, shape (2,)
        Domain bounds [x_min, t_min] and [x_max, t_max].
    Nf       : int
        Number of interior collocation points.
    N0       : int
        Number of initial-condition points (t=0).
    Nb       : int
        Number of boundary-condition points per boundary.
    x_all    : np.ndarray, shape (Nx, 1)
        Spatial grid of the .mat file.
    t_all    : np.ndarray, shape (Nt, 1)
        Temporal grid of the .mat file.
    Exact_u  : np.ndarray, shape (Nx, Nt)
        Real part of the exact solution.
    Exact_v  : np.ndarray, shape (Nx, Nt)
        Imaginary part of the exact solution.
    seed     : int, optional
        Seed for reproducibility. Default 1234.

    Returns
    -------
    x0       : np.ndarray, shape (N0, 1)  — x coordinates of the IC
    u0_train : np.ndarray, shape (N0, 1)  — real part at the IC
    v0_train : np.ndarray, shape (N0, 1)  — imaginary part at the IC
    tb_train : np.ndarray, shape (Nb, 1)  — t instants of the BC
    X_f_train: np.ndarray, shape (Nf, 2)  — interior points [x, t]
    """
    # IC: N0 random indices of x_all (discrete grid of the .mat file)
    # The legacy generator is used to replicate the original notebook exactly.
    np.random.seed(seed)
    idx_x    = np.random.choice(x_all.shape[0], N0, replace=False)
    x0       = x_all[idx_x, :]
    u0_train = Exact_u[idx_x, 0:1]
    v0_train = Exact_v[idx_x, 0:1]

    # BC: Nb random instants of t_all (discrete grid of the .mat file)
    # Generator state continues from the previous choice (same as the original).
    idx_t    = np.random.choice(t_all.shape[0], Nb, replace=False)
    tb_train = t_all[idx_t, :]

    # Interior: continuous LHS over the whole domain.
    # qmc.LatinHypercube uses its own internal seed, independent of the state
    # of the legacy generator. No interference between the two.
    sampler   = qmc.LatinHypercube(d=2, seed=seed)
    X_f_train = lb + (ub - lb) * sampler.random(n=Nf)   # scale to [lb, ub]

    return x0, u0_train, v0_train, tb_train, X_f_train


def sample_grid(lb, ub, Nf, N0, Nb, x_all, t_all, Exact_u, Exact_v):
    """
    Grid — Coherent uniform grid, finite-difference style.

    GRID STRUCTURE
    --------------
    A coherent grid is built in which IC and BC are natural subsets
    of the global grid:

        Total nodes:   (Nx_g + 2) × (Nt_g + 1)
        Interior:       Nx_g      × Nt_g         ← X_f_train (≈ Nf points)
        IC (t=t_min):  (Nx_g + 2) × 1            ← x0 (N0 = Nx_g + 2 points)
        BC (x=lb,ub):   2         × Nt_g         ← tb (Nb = Nt_g points)

    The IC and BC nodes are subsets of the global grid, so
    there is no duplication between regions.

    Parameters
    ----------
    lb, ub   : np.ndarray, shape (2,)
        Domain bounds [x_min, t_min] and [x_max, t_max].
    Nf       : int
        Desired number of interior collocation points.
        The effective number may differ slightly because of rounding
        in the computation of Nx_g and Nt_g.
    N0       : int
        Not used directly. In Grid, N0 = Nx_g + 2 (the whole t=0 row).
        Included for compatibility with the get_sampling() interface.
    Nb       : int
        Not used directly. In Grid, Nb = Nt_g (boundary column).
        Included for compatibility with the get_sampling() interface.
    x_all    : np.ndarray, shape (Nx, 1)
        Spatial grid of the .mat file. Used to interpolate the IC.
    t_all    : np.ndarray, shape (Nt, 1)
        Not used directly in Grid (the grid is independent of the .mat file).
    Exact_u  : np.ndarray, shape (Nx, Nt)
        Real part of the exact solution. Used to evaluate the IC.
    Exact_v  : np.ndarray, shape (Nx, Nt)
        Imaginary part of the exact solution. Used to evaluate the IC.

    Returns
    -------
    x0       : np.ndarray, shape (Nx_g+2, 1) — x coordinates of the IC
    u0_train : np.ndarray, shape (Nx_g+2, 1) — real part at the IC
    v0_train : np.ndarray, shape (Nx_g+2, 1) — imaginary part at the IC
    tb_train : np.ndarray, shape (Nt_g,   1) — t instants of the BC
    X_f_train: np.ndarray, shape (Nx_g*Nt_g, 2) — interior points [x, t]

    Notes
    -----
    The effective N0 and Nb may differ from the requested values because
    in Grid they are determined by the grid dimensions (Nx_g, Nt_g).
    This is reported in the diagnostic print.
    """
    x_min, t_min = lb
    x_max, t_max = ub

    # Compute grid dimensions from the aspect ratio
    Nx_g, Nt_g = _compute_grid_dims(Nf, lb, ub)

    # Full grid nodes (including boundaries)
    x_nodes = np.linspace(x_min, x_max, Nx_g + 2)   # (Nx_g+2,) includes lb, ub
    t_nodes = np.linspace(t_min, t_max, Nt_g + 1)   # (Nt_g+1,) includes t_min

    # Interior: excludes the boundaries in x and the IC in t
    x_int = x_nodes[1:-1]    # (Nx_g,) — without x_min and x_max
    t_int = t_nodes[1:]      # (Nt_g,) — without t_min (IC)

    XX, TT    = np.meshgrid(x_int, t_int)             # (Nt_g, Nx_g)
    X_f_train = np.hstack([XX.flatten()[:, None],
                            TT.flatten()[:, None]])    # (Nx_g*Nt_g, 2)

    # IC: whole row t=t_min, including the corners (x_min and x_max)
    x0       = x_nodes[:, None]                        # (Nx_g+2, 1)
    u0_train, v0_train = _nearest_exact_ic(x0, x_all, Exact_u, Exact_v)

    # BC: t instants of the interior grid (without t_min to avoid duplicating the IC)
    tb_train = t_int[:, None]                          # (Nt_g, 1)

    print(f"   [Grid] Aspect ratio: Lx/Lt = "
          f"{(x_max-x_min):.2f}/{(t_max-t_min):.4f} = "
          f"{(x_max-x_min)/(t_max-t_min):.2f}")
    print(f"   [Grid] Grid: {Nx_g} × {Nt_g} = {Nx_g*Nt_g} interior pts "
          f"(requested: {Nf})")
    print(f"   [Grid] Effective N0: {x0.shape[0]} | "
          f"Effective Nb: {tb_train.shape[0]}")

    return x0, u0_train, v0_train, tb_train, X_f_train


def sample_random(lb, ub, Nf, N0, Nb, x_all, t_all, Exact_u, Exact_v,
                  seed=1234):
    """
    Random — Continuous uniform random sampling (pure Monte Carlo).

    IMPLEMENTATION
    --------------
    IC and BC are sampled from the discrete grid of the .mat file via
    np.random.choice (legacy generator), the same as LHS and the original
    notebook. This guarantees that the only variable that changes between
    strategies is the distribution of the interior points X_f_train.
    The interior is generated with default_rng for continuous uniform sampling.

    Parameters
    ----------
    lb, ub   : np.ndarray, shape (2,)
        Domain bounds [x_min, t_min] and [x_max, t_max].
    Nf       : int
        Number of interior collocation points.
    N0       : int
        Number of initial-condition points (t=0).
    Nb       : int
        Number of boundary-condition points per boundary.
    x_all    : np.ndarray, shape (Nx, 1)
        Spatial grid of the .mat file. Used to interpolate the IC.
    t_all    : np.ndarray, shape (Nt, 1)
        Not used directly (continuous sampling).
    Exact_u  : np.ndarray, shape (Nx, Nt)
        Real part of the exact solution.
    Exact_v  : np.ndarray, shape (Nx, Nt)
        Imaginary part of the exact solution.
    seed     : int, optional
        Seed for reproducibility. Default 1234.

    Returns
    -------
    x0       : np.ndarray, shape (N0, 1)  — x coordinates of the IC
    u0_train : np.ndarray, shape (N0, 1)  — real part at the IC
    v0_train : np.ndarray, shape (N0, 1)  — imaginary part at the IC
    tb_train : np.ndarray, shape (Nb, 1)  — t instants of the BC
    X_f_train: np.ndarray, shape (Nf, 2)  — interior points [x, t]
    """
    x_min, t_min = lb
    x_max, t_max = ub

    # IC and BC: from the discrete grid of the .mat file with the legacy generator,
    # the same as LHS and the original notebook. This guarantees comparability
    # between strategies: the only variable that changes is X_f_train.
    np.random.seed(seed)
    idx_x    = np.random.choice(x_all.shape[0], N0, replace=False)
    x0       = x_all[idx_x, :]
    u0_train = Exact_u[idx_x, 0:1]
    v0_train = Exact_v[idx_x, 0:1]

    idx_t    = np.random.choice(t_all.shape[0], Nb, replace=False)
    tb_train = t_all[idx_t, :]

    # Interior: continuous uniform over the whole domain (pure Monte Carlo).
    # default_rng is used for the interior, with a seed independent of the legacy one,
    # so as not to interfere with the state of the IC/BC generator.
    rng = np.random.default_rng(seed)
    X_f_train = np.hstack([
        rng.uniform(x_min, x_max, (Nf, 1)),
        rng.uniform(t_min, t_max, (Nf, 1)),
    ])

    return x0, u0_train, v0_train, tb_train, X_f_train


def sample_sobol(lb, ub, Nf, N0, Nb, x_all, t_all, Exact_u, Exact_v,
                 seed=1234):
    """
    Sobol — Base-2 low-discrepancy quasi-random sequence.

    IMPLEMENTATION
    --------------
    IC and BC are sampled from the discrete grid of the .mat file via
    np.random.choice (legacy generator), the same as LHS and the original
    notebook. This guarantees that the only variable that changes between
    strategies is the distribution of the interior points X_f_train.
    The interior is generated with qmc.Sobol (scramble=True, seed=seed).

    Technical note: Sobol is more efficient when N is a power of 2
    (N = 2^m). For other values of N, scipy generates the first N
    points of the sequence of length 2^ceil(log2(N)).

    Parameters
    ----------
    lb, ub   : np.ndarray, shape (2,)
        Domain bounds [x_min, t_min] and [x_max, t_max].
    Nf       : int
        Number of interior collocation points.
    N0       : int
        Number of initial-condition points (t=0).
    Nb       : int
        Number of boundary-condition points per boundary.
    x_all    : np.ndarray, shape (Nx, 1)
        Spatial grid of the .mat file. Used to interpolate the IC.
    t_all    : np.ndarray, shape (Nt, 1)
        Not used directly (continuous sampling).
    Exact_u  : np.ndarray, shape (Nx, Nt)
        Real part of the exact solution.
    Exact_v  : np.ndarray, shape (Nx, Nt)
        Imaginary part of the exact solution.
    seed     : int, optional
        Seed for the Owen scrambling randomization.
        Default 1234.

    Returns
    -------
    x0       : np.ndarray, shape (N0, 1)  — x coordinates of the IC
    u0_train : np.ndarray, shape (N0, 1)  — real part at the IC
    v0_train : np.ndarray, shape (N0, 1)  — imaginary part at the IC
    tb_train : np.ndarray, shape (Nb, 1)  — t instants of the BC
    X_f_train: np.ndarray, shape (Nf, 2)  — interior points [x, t]

    Notes
    -----
    IC and BC use the legacy generator to be comparable with LHS and the
    original notebook. The interior uses qmc.Sobol, which has its own
    internal seed independent of the state of the legacy generator.
    """
    x_min, t_min = lb
    x_max, t_max = ub

    # IC and BC: from the discrete grid of the .mat file with the legacy generator.
    # Identical to LHS and the original notebook to guarantee comparability.
    np.random.seed(seed)
    idx_x    = np.random.choice(x_all.shape[0], N0, replace=False)
    x0       = x_all[idx_x, :]
    u0_train = Exact_u[idx_x, 0:1]
    v0_train = Exact_v[idx_x, 0:1]

    idx_t    = np.random.choice(t_all.shape[0], Nb, replace=False)
    tb_train = t_all[idx_t, :]

    # Interior: 2D Sobol scaled to the domain.
    # qmc.Sobol uses its own internal seed, without interference with the legacy one.
    sampler_f = qmc.Sobol(d=2, scramble=True, seed=seed)
    pts_f     = sampler_f.random(n=Nf)
    X_f_train = lb + (ub - lb) * pts_f                 # scale to [lb, ub]

    return x0, u0_train, v0_train, tb_train, X_f_train


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def get_sampling(strategy, lb, ub, Nf, N0, Nb,
                 Exact_u, Exact_v, x_all, t_all,
                 seed=1234):
    """
    Generates training points for a PINN according to the given strategy.

    Unified interface for all sampling strategies. All of them
    produce the same output structure, directly compatible with
    the PhysicsInformedNN class of the notebook PINN_Torch_NLS_Hirota.

    Parameters
    ----------
    strategy : str
        Sampling strategy. Options: 'LHS', 'Grid', 'Random', 'Sobol'.
    lb       : np.ndarray, shape (2,)
        Lower bound of the domain: [x_min, t_min].
    ub       : np.ndarray, shape (2,)
        Upper bound of the domain: [x_max, t_max].
    Nf       : int
        Number of interior collocation points (PDE residual).
    N0       : int
        Number of initial-condition points (t=t_min).
        In Grid, the effective value is Nx_g+2 (determined by the grid).
    Nb       : int
        Number of boundary-condition points per boundary (x=lb, x=ub).
        In Grid, the effective value is Nt_g (determined by the grid).
    Exact_u  : np.ndarray, shape (Nx, Nt)
        Real part of the exact solution from the .mat file.
    Exact_v  : np.ndarray, shape (Nx, Nt)
        Imaginary part of the exact solution from the .mat file.
    x_all    : np.ndarray, shape (Nx, 1)
        Spatial grid of the .mat file.
    t_all    : np.ndarray, shape (Nt, 1)
        Temporal grid of the .mat file.
    seed     : int, optional
        Randomness seed for reproducibility. Default 1234.
        Does not apply to Grid (deterministic).

    Returns
    -------
    dict with the following keys:
        'x0'        : np.ndarray, shape (N0_eff, 1)
            x coordinates of the initial-condition points.
        'u0_train'  : np.ndarray, shape (N0_eff, 1)
            Real part of the exact solution at t=0 (IC supervision values).
        'v0_train'  : np.ndarray, shape (N0_eff, 1)
            Imaginary part of the exact solution at t=0.
        'tb_train'  : np.ndarray, shape (Nb_eff, 1)
            t instants for the periodic boundary condition.
        'X_f_train' : np.ndarray, shape (Nf_eff, 2)
            Interior collocation points, columns [x, t].
        'strategy'  : str
            Name of the strategy used.
        'N0_eff'    : int
            Effective number of IC points generated.
        'Nb_eff'    : int
            Effective number of BC points generated.
        'Nf_eff'    : int
            Effective number of interior points generated.

    Example
    -------
        points    = get_sampling('Sobol', lb, ub, 10000, 50, 50,
                                 Exact_u, Exact_v, x_all, t_all, seed=1234)
        x0        = points['x0']
        u0_train  = points['u0_train']
        v0_train  = points['v0_train']
        tb_train  = points['tb_train']
        X_f_train = points['X_f_train']
    """
    strategy = strategy.strip()
    if strategy not in AVAILABLE_STRATEGIES:
        raise ValueError(
            f"Strategy '{strategy}' not recognized.\n"
            f"Available options: {AVAILABLE_STRATEGIES}"
        )

    print(f"\n   {'='*55}")
    print(f"   Sampling strategy : {strategy}")
    print(f"   Nf={Nf}, N0={N0}, Nb={Nb}, seed={seed}")
    print(f"   {'='*55}")

    if strategy == 'LHS':
        x0, u0, v0, tb, Xf = sample_lhs(
            lb, ub, Nf, N0, Nb, x_all, t_all, Exact_u, Exact_v, seed)

    elif strategy == 'Grid':
        x0, u0, v0, tb, Xf = sample_grid(
            lb, ub, Nf, N0, Nb, x_all, t_all, Exact_u, Exact_v)

    elif strategy == 'Random':
        x0, u0, v0, tb, Xf = sample_random(
            lb, ub, Nf, N0, Nb, x_all, t_all, Exact_u, Exact_v, seed)

    elif strategy == 'Sobol':
        x0, u0, v0, tb, Xf = sample_sobol(
            lb, ub, Nf, N0, Nb, x_all, t_all, Exact_u, Exact_v, seed)

    print(f"   -> IC : {x0.shape[0]:>6} points at t={lb[1]}")
    print(f"   -> BC : {tb.shape[0]:>6} points per boundary")
    print(f"   -> Int: {Xf.shape[0]:>6} interior points")

    return {
        'x0':        x0,
        'u0_train':  u0,
        'v0_train':  v0,
        'tb_train':  tb,
        'X_f_train': Xf,
        'strategy':  strategy,
        'N0_eff':    x0.shape[0],
        'Nb_eff':    tb.shape[0],
        'Nf_eff':    Xf.shape[0],
    }
