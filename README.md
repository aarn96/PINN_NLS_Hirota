# PINN_NLS_Hirota

**Physics-Informed Neural Networks for soliton-type solutions of the Nonlinear Schrödinger (NLS) and Hirota equations.**

Code and reproducibility artifacts accompanying the M.Sc. thesis:

> Contreras Roque, A. E. (2026). *Aplicación de redes neuronales informadas por la física a ecuaciones no lineales dispersivas relevantes en ciencias de la atmósfera y el océano* [M.Sc. thesis, Universidad Nacional Autónoma de México]. TESIUNAM. https://hdl.handle.net/20.500.14330/TES01000894006

*English translation of the title: Application of physics-informed neural networks to nonlinear dispersive equations relevant to atmospheric and ocean sciences.*

The thesis PDF is in Spanish (274 pages total); this repository is in English.

---

## 1. Scope

This work characterizes the **viability** of Physics-Informed Neural Networks (PINNs) as forward solvers for two dispersive nonlinear PDEs with geophysical relevance:

- **Nonlinear Schrödinger equation (NLS):** the canonical model for modulational instability, rogue waves, and envelope dynamics in deep water and internal waves.
- **Hirota equation:** a generalization of NLS including third-order dispersion that unifies elements of the KdV and NLS hierarchies, relevant to shorter-scale dispersive effects.

The main contribution is a **combined diagnostic protocol** $(\varepsilon_{L^2},\, \mathcal{R}(x,t),\, \mathcal{N}(t),\, \Delta J_{\mathrm{ref}},\, \rho_1\!-\!\rho_4)$ and a set of design heuristics (boundary conditions, sampling, architecture) tested across 8 experimental phases. The work is methodological — it does **not** claim PINNs replace classical numerical methods (SSF, symplectic, spectral schemes remain 3–5 orders of magnitude more accurate for the forward problem). The differential advantage of PINNs is expected in inverse problems, equations with variable coefficients, and unstructured data — directions outlined for the planned doctoral project on NLS with variable coefficients (bathymetry / currents) and rogue-wave identification from sparse instrumental data.

### Geophysical motivation

Solitons, breathers, rogue waves, internal waves, Rossby waves, and atmospheric blocking share the mathematical structure addressed here. The study is a controlled benchmark before extending the method to equations of direct interest to geophysical fluid dynamics and numerical weather prediction.

---

## 2. Equations

### NLS
$$ i\psi_t - \tfrac{1}{2}\psi_{xx} - |\psi|^2\psi = 0 $$

Tested configurations: $N=1$ fundamental soliton, $N=2$ Satsuma–Yajima bound state, $N=2$ two-soliton collision.

### Hirota (Zhou & Yan parametrization)
$$ i\psi_t + \alpha\left(\psi_{xx} + 2|\psi|^2\psi\right) + i\beta\left(\psi_{xxx} + 6|\psi|^2\psi_x\right) = 0,\quad \alpha=1,\ \beta=0.01 $$

Tested configuration: $N=1$ fundamental Hirota soliton, reproducing Zhou & Yan (2021).

### Hirota (general, Serkin parametrization)
$$ i\psi_t + \tfrac{1}{2}D_2\psi_{xx} + R_2|\psi|^2\psi + iD_3\psi_{xxx} + 6iR_3|\psi|^2\psi_x = 0 $$

with $(D_2, D_3, R_2, R_3) = (1.0,\, 0.01,\, 1.0,\, 0.01)$. Tested configurations: $N=2$ **bound state** (breather) and $N=2$ **two-soliton collision**. No closed-form analytical expression is reproduced in the thesis for these two cases; reference solutions were provided by Dr. L. Serkin as `.mat` files (see §7, Credits and data provenance).

---

## 3. Main results

### 3.1 Central finding — Boundary conditions incompatible with internal flux

The headline result concerns the Hirota $N=2$ bound state (breather) with architecture $[4\times 80]$ and seed $1234$. Two learning rates were used in mirror batteries (Bat. A, $\eta=5\times 10^{-4}$; Bat. B, $\eta=10^{-3}$).

**Table 4.9, $\eta = 5\times 10^{-4}$ (thesis, §4.2.3):**

| Boundary condition         | $\varepsilon_{L^2}(\|\psi\|)$ |
|----------------------------|-------------------------------|
| **No BC**                  | **$9.91\times 10^{-3}$**      |
| Val Only                   | $1.50\times 10^{-2}$          |
| Der1 Only                  | $7.84\times 10^{-2}$          |
| Val + Der1                 | $6.57\times 10^{-2}$          |
| Val + Der1 + Der2          | $6.55\times 10^{-2}$          |

**Table 4.11, Bat. B, $\eta = 10^{-3}$:** No BC $= 5.488\times 10^{-3}$, Val+Der1 $= 6.620\times 10^{-2}$ (degradation factor $12.06$).

Degradation factor from imposing periodic BC: **6–13×** depending on $\eta$. The effect is robust across architectures: **19 / 20 architectures without BC stay below the ceiling reached with BC**.

**Physical explanation.** From the continuity equation $\partial_t|\psi|^2 + \partial_x J = 0$, global norm balance reads
$$\frac{d\mathcal{N}}{dt} = -\bigl[J(+L,t) - J(-L,t)\bigr].$$
On the reference breather solution, the two boundaries oscillate **in antiphase** following the breathing period: imposing periodic BC forces $J(+L,t)=J(-L,t)$, which is structurally incompatible with the actual flux of the solution, and the training landscape is pulled toward an unphysical minimum.

> *"An apparently more physical constraint can make the solution less physical when it is incompatible with transport through the boundaries of the domain."*

### 3.2 Diagnostic scenarios E1–E4

Each configuration is classified by jointly inspecting the global error $\varepsilon_{L^2}$ and the residual proxy $L_{\mathrm{res}}$ at the end of training:

| Scenario | $\varepsilon_{L^2}$ | $L_{\mathrm{res}}$ | Interpretation                                                   |
|----------|---------------------|---------------------|------------------------------------------------------------------|
| E1       | Low (< $2\times 10^{-2}$) | Low (< $10^{-5}$)   | Accurate and physically consistent solution                       |
| E2       | High                | High ($\geq 10^{-4}$) or intermediate + L-BFGS exhausted | Network did not converge                                          |
| E3       | Low                 | High                | Not observed empirically                                         |
| E4       | High                | Low                 | Incompatible BC — spurious local consistency                      |

Scenario **E4** is the diagnostic signature of the BC-incompatibility case above.

### 3.3 Spearman correlations $\rho_1$–$\rho_4$ (error vs physical/numerical proxies)

Operationalization of four hypotheses tested across all configurations:

- $\rho_1$: error vs $|\psi_{\mathrm{exact}}|$ (amplitude) — **confirmed with nuances**;
- $\rho_2$: error vs $|\partial_x\psi_{\mathrm{exact}}|$ (gradient) — **empirically refuted**;
- $\rho_3$: error vs nearest-collocation-point distance — **$\approx 0$, confirmed** (point density does not control the error pattern);
- $\rho_4$: error vs local residual magnitude — **nuanced, not refuted**.

Threshold of practical relevance (Cohen 1988): $|\rho| > 0.3$.

---

## 4. Repository layout

```
PINN_NLS_Hirota/
├── README.md                              ← this file
├── LICENSE                                ← CC BY-NC 4.0 (code only; see §9)
├── requirements.txt                       ← pip dependencies
├── environment.yml                        ← conda environment (optional)
├── .gitignore
├── CITATION.cff                           ← citation metadata (APA 7)
│
├── src/                                   ← reusable Python modules
│   ├── sampling_strategies.py             ← LHS / Grid / Random / Sobol
│   ├── conservation.py                    ← N(t), J_NLS, J_Yan, J_Hirota, boundary-flux check
│   └── spearman.py                        ← ρ₁–ρ₄ with NaN/Inf-safe wrappers
│
├── notebooks/
│   ├── PINN_Torch_NLS_Hirota.ipynb        ← training notebook (PyTorch + Colab GPU)
│   └── Postprocessing_PINN_NLS_Hirota.ipynb  ← postprocessing, diagnostics, figures
│
├── data/                                  ← reference solutions (see §7)
│   ├── README.md                          ← request procedure for the three missing files
│   ├── NLS_N1.mat                         ← included
│   ├── NLS_N2.mat                         ← included
│   └── Bright_soliton.mat                 ← included
│
└── results/                               ← .npz / .json outputs per experiment (gitignored by default)
```

The `results/` directory mirrors the folder structure of the Google Drive working copy used during the thesis.

---

## 5. Installation

### 5.1 Local (Python 3.10+)

```bash
git clone https://github.com/aarn96/PINN_NLS_Hirota.git
cd PINN_NLS_Hirota
python -m venv .venv
source .venv/bin/activate              # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 5.2 Conda

```bash
conda env create -f environment.yml
conda activate pinn-nls-hirota
```

### 5.3 Google Colab

The notebooks were developed on Google Colab with a Tesla T4 GPU. Copy or clone the repository into Google Drive (default location: `MyDrive/PINN_NLS_Hirota`; otherwise edit `REPO_PATH` in the setup cell), then open the notebook from Drive. The setup cell mounts Drive and adds `src/` to `sys.path`; PyTorch, NumPy, SciPy and Matplotlib are preinstalled on Colab, and the postprocessing notebook installs `cmocean` if needed.

**Reference environment:** Tesla T4, `torch 2.11.0+cu130`, `numpy 2.1.3`, Python 3.13.

---

## 6. Usage

### 6.1 Training a configuration

**Quick start.** `notebooks/PINN_Torch_NLS_Hirota.ipynb` runs as is with the data shipped in `data/`. Its default case is the Hirota $N=1$ fundamental soliton (Zhou & Yan, 2021; `Bright_soliton.mat`) on $x\in[-10,10]$, $t\in[0,5]$, with the Hirota baseline below and one run per sampling strategy (LHS, Sobol, Grid, Random). Then run `notebooks/Postprocessing_PINN_NLS_Hirota.ipynb` with the same configuration cell to obtain the figures and tables of that batch. On a CPU the full budget is slow; reduce `n_adam` and `n_lbfgs` for a quick check.

Both notebooks start with the same configuration cell:

- `equation` and `case` select the residual, the reference `.mat` file and the domain (Table 3.2 of the thesis):

  | `equation` | `case`        | Reference file                 | $x$          | $t$           | Availability |
  |------------|---------------|--------------------------------|--------------|---------------|--------------|
  | `'yan'`    | `fundamental` | `Bright_soliton.mat`           | $[-10, 10]$  | $[0, 5]$      | Included (default) |
  | `'nls'`    | `fundamental` | `NLS_N1.mat`                   | $[-5, 5]$    | $[0, \pi/2]$  | Included |
  | `'nls'`    | `bound_state` | `NLS_N2.mat`                   | $[-5, 5]$    | $[0, \pi/2]$  | Included |
  | `'nls'`    | `collision`   | `NLS_2soliton_interaction.mat` | $[0, 20]$    | $[0, 5]$      | On request |
  | `'hirota'` | `bound_state` | `hirota_N_2.mat`               | $[-5, 5]$    | $[0, 2\pi]$   | On request |
  | `'hirota'` | `collision`   | `N_2_hirota_inter.mat`         | $[-10, 10]$  | $[0, 4]$      | On request |

  `'yan'` is the Hirota equation in the Zhou & Yan parametrization and `'hirota'` the Serkin parametrization (§2). The central result of §3.1 uses the `'hirota'` / `bound_state` case; its reference file is available on request (§7, `data/README.md`). If a selected file is missing, the notebook stops with a message that points to the request procedure.
- `architecture`: `[2, n, ..., n, 2]` (input, hidden layers, output), e.g. `[2, 80, 80, 80, 80, 2]` for $[4\times 80]$
- `sampling_strategies`: any subset of `LHS`, `Sobol`, `Grid`, `Random` (one experiment each)
- `bc_mode`: integer code for the boundary-condition variant:
    - `0`, `1` → **no BC** (unpenalized boundaries)
    - `2` → **Val Only** (periodic in $\psi$)
    - `3` → **Der1 Only** (periodic in $\partial_x\psi$)
    - `4` → **Val + Der1**
    - `5` → **Val + Der1 + Der2** (adds periodicity in $\partial^2_x\psi$)
- `n_adam`, `n_lbfgs`, `lr`: Adam epochs, maximum L-BFGS iterations and Adam learning rate (see §6.2)
- `seed`: default $1234$

Outputs go to `outputs/<BATCH_ID>/` (training) and `results/<BATCH_ID>/` (postprocessing), where `BATCH_ID` is built from the case, architecture and BC mode (e.g. `Yan_N1_4x80_NoBC_Sampling`).

### 6.2 Baseline configurations

Two baselines cover most experiments in the thesis. All runs use seed $1234$.

**NLS baseline**

- Architecture: $[3 \times 40]$
- $(N_0, N_b, N_f) = (50, 50, 10{,}000)$
- Optimizer: Adam $10{,}000$ it. + L-BFGS $10{,}000$ it. (max)
- Learning rate: $\eta = 10^{-3}$

**Hirota baseline**

- Architecture: $[4 \times 80]$
- $(N_0, N_b, N_f) = (50, 50, 10{,}000)$
- Optimizer: Adam $5{,}000$ it. + L-BFGS $10{,}000$ it. (max)
- Learning rate: $\eta = 10^{-3}$, **except** for the Hirota $N=2$ bound state with periodic BC, which uses $\eta = 5 \times 10^{-4}$. Removing the BC recovers $\eta = 10^{-3}$.

Spatio-temporal domains and the configurations of the experiments outside these baselines (architecture sweeps, data-regime sensitivity, optimizer-budget sweep, BC × domain crops, sampling-strategy comparison, 20-architecture sweep) are documented in the thesis, **Chapter 3 — Methodology, §3.2 (domains) and §3.3 (experimental design)**.

### 6.3 Postprocessing and diagnostics

Open `notebooks/Postprocessing_PINN_NLS_Hirota.ipynb`. The notebook loads `.npz` + `.json` outputs and computes:

- Global error $\varepsilon_{L^2}$ per component ($u$, $v$, $|\psi|$)
- Pointwise error maps and residual maps $\mathcal{R}(x,t)$
- $\mathcal{N}(t)$ and $\Delta\mathcal{N}_{\mathrm{rel}}(t)$ for conservation
- Boundary-flux residual $\Delta J_{\mathrm{ref}}(t) = J(+L,t) - J(-L,t)$ on the reference solution
- Spearman $\rho_1$–$\rho_4$ with NaN/Inf-safe implementations
- Scenario E1–E4 assignment

### 6.4 Using the `src` modules directly

```python
from src.sampling_strategies import get_sampling
from src.conservation import compute_norm, compute_J_hirota, check_boundary_flux
from src.spearman import spearman_safe, compute_gradient_magnitude, compute_dist_nn

points = get_sampling(
    strategy="LHS",
    lb=np.array([-5.0, 0.0]),
    ub=np.array([ 5.0, 2*np.pi]),
    Nf=10_000, N0=50, Nb=50,
    Exact_u=U_ref, Exact_v=V_ref,
    x_all=x, t_all=t, seed=1234,
)
```

`get_sampling` returns a dict with `x0`, `u0_train`, `v0_train`, `tb_train`, `X_f_train`, and the effective point counts (`N0_eff`, `Nb_eff`, `Nf_eff`), ready to feed the training loop.

---

## 7. Credits and data provenance

- **Advisor:** Dr. Ignacio Arturo Quintanar Isaías (UNAM, ICAyCC).
- **Committee member with ML background:** Dr. Carlos Ignacio Hernández Castellanos — first flagged the single-seed limitation of the thesis experiments.
- **Dr. Leonid Serkin** provided the `.mat` reference solutions for the Hirota $N=2$ bound state, the Hirota $N=2$ two-soliton collision, and the NLS $N=2$ two-soliton collision, as well as the base MATLAB code underlying the Hirota two-soliton solutions used throughout Chapters 4 and 5.

### Reference `.mat` files — inclusion vs request by email

| File                               | Equation and configuration                        | Status in this repo |
|------------------------------------|---------------------------------------------------|---------------------|
| `NLS_N1.mat`                       | NLS, $N=1$ fundamental soliton                    | **Included**        |
| `NLS_N2.mat`                       | NLS, $N=2$ Satsuma–Yajima bound state             | **Included**        |
| `Bright_soliton.mat`               | Hirota (Zhou & Yan), $N=1$ bright soliton         | **Included**        |
| `NLS_2soliton_interaction.mat`     | NLS, $N=2$ two-soliton collision                  | **Request by email**|
| `hirota_N_2.mat`                   | Hirota (Serkin), $N=2$ bound state (breather)     | **Request by email**|
| `N_2_hirota_inter.mat`             | Hirota (Serkin), $N=2$ two-soliton collision      | **Request by email**|

To request the three missing files, write to:

> **Aaron Ernesto Contreras Roque** — `aron.contreras@atmosfera.unam.mx`

Please include your name and affiliation, the specific files you need, and the intended academic use. Access to the Hirota two-soliton solutions requires permission from Dr. L. Serkin; the request will be forwarded.

Other references:

- Classical reproduction of the Hirota fundamental soliton follows **Zhou & Yan (2021)** (cited in the thesis `.bib`).
- Low-discrepancy sampling draws on McKay, Beckman & Conover (1979), Sobol (1967), Niederreiter (1992), and the comparative study by Wu et al. (2023, CMAME).

---

## 8. Citation

### APA 7

Contreras Roque, A. E. (2026). *Aplicación de redes neuronales informadas por la física a ecuaciones no lineales dispersivas relevantes en ciencias de la atmósfera y el océano* [M.Sc. thesis, Universidad Nacional Autónoma de México]. TESIUNAM. https://hdl.handle.net/20.500.14330/TES01000894006

### BibTeX

```bibtex
@mastersthesis{ContrerasRoque2026_PINN_NLS_Hirota,
  author  = {Contreras Roque, Aaron Ernesto},
  title   = {Aplicaci{\'o}n de redes neuronales informadas por la f{\'\i}sica
             a ecuaciones no lineales dispersivas relevantes en ciencias de
             la atm{\'o}sfera y el oc{\'e}ano},
  school  = {Universidad Nacional Aut{\'o}noma de M{\'e}xico (UNAM)},
  year    = {2026},
  type    = {M.Sc. thesis in Earth Sciences (Atmospheric Sciences)},
  address = {Mexico City, Mexico},
  pages   = {274},
  url     = {https://hdl.handle.net/20.500.14330/TES01000894006},
  language = {spanish}
}
```

A machine-readable `CITATION.cff` is included at the root of the repository.

---

## 9. License

- **Code** (the Python modules in `src/`, the notebooks in `notebooks/`, and the repository-level configuration files) is released under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license — https://creativecommons.org/licenses/by-nc/4.0 — which requires attribution (citation of the thesis, §8) and limits reuse to non-commercial purposes. See `LICENSE`.
- **The thesis document** (available at the TESIUNAM handle above) is licensed under **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)** — https://creativecommons.org/licenses/by-nc-nd/4.0 — as registered by UNAM.
- **Reference `.mat` files** that are included in `data/` share the code license (CC BY-NC 4.0). The three files available on request (see §7) are not covered by either license and their reuse conditions are set by Dr. L. Serkin on a case-by-case basis.

---

## 10. Limitations and planned work

- All experiments in the thesis use a single random seed ($1234$); a seed-variability analysis is left for future work.
- Single-domain, periodic-BC setting; no multi-domain / adaptive decomposition.
- Residual-based adaptive sampling (RAR, RAD, R3) is out of scope for this thesis.
- Analytical period of the Hirota breather was inferred empirically ($T=2\pi$); analytical verification is pending.
- Scenario E3 (low $L_{\mathrm{res}}$, high $\varepsilon_{L^2}$) was not observed — this is an empirical result, not a taxonomy limitation.

Planned extension (doctoral project): NLS with variable coefficients tied to bathymetry and currents, and identification of rogue waves from sparse instrumental data.

---

## 11. Contact

Aaron Ernesto Contreras Roque — Instituto de Ciencias de la Atmósfera y Cambio Climático, UNAM.
Email: `aron.contreras@atmosfera.unam.mx`
Repository issues: https://github.com/aarn96/PINN_NLS_Hirota/issues
