#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# 
# Author: Oumaima Badraoui
# 📧 o.badraoui@studenti.unipi.it
# 
# Edited by: Felipe Rincón
# 📧 felipe.rincon@phd.unipi.it
# =============================================================================

import os
import sys
import re
import time
import datetime
import tracemalloc
import psutil
import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse import linalg, csr_matrix, eye

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Setting up
# =============================================================================

TEST_ID = "001"  # <-- set per run: "001", "002", "003", ...

OUT_DIR = os.path.abspath(f"INR_results_{TEST_ID}")
os.makedirs(OUT_DIR, exist_ok=True)

# ----- log to file + console (tee) -----
log_file_path = os.path.join(OUT_DIR, "run.log")
_log_fh = open(log_file_path, "w", buffering=1)

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

sys.stdout = Tee(sys.stdout, _log_fh)
sys.stderr = Tee(sys.stderr, _log_fh)

print("=" * 80)
print("RUN START")
print("Timestamp:", datetime.datetime.now().isoformat())
print("OUT_DIR   :", OUT_DIR)
print("=" * 80)

# plt.rcParams["font.family"] = "arial"
plt.rcParams.update({"font.size": 12})
FIG_COUNTER = 0

def _slugify(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^\w\s\-]+", "", s)
    s = re.sub(r"[\s\-]+", "_", s)
    return s[:120] if len(s) > 120 else s

def save_current_fig(name: str, dpi: int = 500) -> str:
    """Save current matplotlib figure as PDF to OUT_DIR and close it."""
    global FIG_COUNTER
    FIG_COUNTER += 1
    safe = _slugify(name) or "figure"
    filename = f"{FIG_COUNTER:03d}_{safe}.pdf"
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, format="pdf", dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[saved figure] {path}")
    return path

def save_npz(name: str, **arrays) -> str:
    """Save compressed NPZ to OUT_DIR."""
    safe = _slugify(name) or "arrays"
    path = os.path.join(OUT_DIR, f"{safe}.npz")
    np.savez_compressed(path, **arrays)
    print(f"[saved npz] {path}")
    return path


# =============================================================================
# for tracking
# =============================================================================

class PerfTracker:
    """Tracks wall-clock time and peak CPU RAM for a code block (context manager).
    Also tracks GPU peak mem if torch+CUDA are available.
    """
    def __init__(self, label):
        self.label   = label
        self.elapsed = None
        self.peak_mb = None
        self.gpu_peak_mb = 0.0

    def __enter__(self):
        tracemalloc.start()
        self._proc = psutil.Process(os.getpid())
        self._t0   = time.perf_counter()

        # GPU tracking (optional)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
        except Exception:
            pass

        return self

    def __exit__(self, *args):
        # GPU tracking (optional)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                self.gpu_peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        except Exception:
            self.gpu_peak_mb = 0.0

        self.elapsed = time.perf_counter() - self._t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_mb = peak / 1024**2

    def report(self):
        print(f"  [{self.label}] time = {self.elapsed:.3f} s | CPU peak mem = {self.peak_mb:.1f} MB")
        print(f"  [{self.label}] GPU peak mem = {self.gpu_peak_mb:.1f} MB")


def collect_perf_summary(trackers: dict) -> None:
    print("\n" + "=" * 84)
    print("PERFORMANCE SUMMARY")
    print(f"  {'Method':<22s}  {'Time (s)':>10s}  {'CPU peak (MB)':>14s}  {'GPU peak (MB)':>14s}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*14}  {'-'*14}")
    for label, pt in trackers.items():
        if pt is not None and pt.elapsed is not None:
            gpu_mb = getattr(pt, "gpu_peak_mb", 0.0)
            print(f"  {label:<22s}  {pt.elapsed:>10.3f}  {pt.peak_mb:>14.1f}  {gpu_mb:>14.1f}")
    print("=" * 84 + "\n")


# =============================================================================
# GRID CLASS
# =============================================================================

class grid:
    """Grid class for defining the numerical grid"""

    def __init__(self, dimension, origin, spacing, npoints):
        self.dimension = dimension
        self.origin = np.array(origin)
        self.spacing = np.array(spacing)
        self.npoints = np.array(npoints, dtype=int)
        self.extent = self.origin + self.spacing * self.npoints
        self.x = np.arange(self.npoints[0]) * self.spacing[0] + self.origin[0]
        self.y = np.arange(self.npoints[1]) * self.spacing[1] + self.origin[1]

    def get_gaussian_prior(self, correlation_length):
        n_total = int(np.prod(self.npoints))
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        coords = np.column_stack([X.ravel(), Y.ravel()])
        rows, cols, data = [], [], []
        cutoff = 3.0 * correlation_length
        for i in range(n_total):
            row_sum = 0.0
            temp_cols, temp_data = [], []
            for j in range(n_total):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist <= cutoff:
                    value = np.exp(-(dist / correlation_length) ** 2)
                    temp_cols.append(j)
                    temp_data.append(value)
                    row_sum += value
            for j, val in zip(temp_cols, temp_data):
                rows.append(i)
                cols.append(j)
                data.append(val / row_sum)
        return csr_matrix((data, (rows, cols)), shape=(n_total, n_total))


# =============================================================================
# RAY TRACING FUNCTIONS
# =============================================================================

def get_all_to_all_locations(src_locations, rec_locations):
    n_src = src_locations.shape[1]
    n_rec = rec_locations.shape[1]
    sources = np.zeros((2, n_src * n_rec))
    receivers = np.zeros((2, n_src * n_rec))
    idx = 0
    for i in range(n_src):
        for j in range(n_rec):
            sources[:, idx] = src_locations[:, i]
            receivers[:, idx] = rec_locations[:, j]
            idx += 1
    return sources, receivers


def create_forward_operator(sources, receivers, g):
    n_rays = sources.shape[1]
    n_cells = int(np.prod(g.npoints))
    rows, cols, data = [], [], []
    for ray_idx in range(n_rays):
        source = sources[:, ray_idx]
        receiver = receivers[:, ray_idx]
        x0, y0 = source
        x1, y1 = receiver
        dx_ray = x1 - x0
        dy_ray = y1 - y0
        ray_length = np.sqrt(dx_ray**2 + dy_ray**2)
        if ray_length < 1e-10:
            continue
        n_samples = max(100, int(ray_length / min(g.spacing) * 5))
        t_vals = np.linspace(0, 1, n_samples)
        cell_lengths = {}
        for k in range(len(t_vals) - 1):
            t = t_vals[k]
            x = x0 + t * dx_ray
            y = y0 + t * dy_ray
            i = int((x - g.origin[0]) / g.spacing[0])
            j = int((y - g.origin[1]) / g.spacing[1])
            if 0 <= i < g.npoints[0] and 0 <= j < g.npoints[1]:
                cell_idx = i * g.npoints[1] + j
                segment_length = ray_length / (n_samples - 1)
                cell_lengths[cell_idx] = cell_lengths.get(cell_idx, 0.0) + segment_length
        for cell_idx, length in cell_lengths.items():
            rows.append(ray_idx)
            cols.append(cell_idx)
            data.append(length)
    G = csr_matrix((data, (rows, cols)), shape=(n_rays, n_cells))
    return G


def build_gradient_operators(Nx, Ny):
    n = Nx * Ny
    rows_x, cols_x, vals_x = [], [], []
    for i in range(Nx - 1):
        for j in range(Ny):
            row = i * Ny + j
            rows_x += [row, row]
            cols_x += [i * Ny + j, (i + 1) * Ny + j]
            vals_x += [-1.0, 1.0]
    Dx = csr_matrix((vals_x, (rows_x, cols_x)), shape=((Nx - 1) * Ny, n))

    rows_y, cols_y, vals_y = [], [], []
    row_ctr = 0
    for i in range(Nx):
        for j in range(Ny - 1):
            rows_y += [row_ctr, row_ctr]
            cols_y += [i * Ny + j, i * Ny + (j + 1)]
            vals_y += [-1.0, 1.0]
            row_ctr += 1
    Dy = csr_matrix((vals_y, (rows_y, cols_y)), shape=(Nx * (Ny - 1), n))
    return Dx, Dy


# =============================================================================
# for plotting
# =============================================================================

def plot_rays(sources, receivers, g):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(g.origin[0], g.extent[0])
    ax.set_ylim(g.origin[1], g.extent[1])
    n_rays = sources.shape[1]
    for i in range(n_rays):
        ax.plot([sources[0, i], receivers[0, i]],
                [sources[1, i], receivers[1, i]],
                color='gray', alpha=0.3, linewidth=0.5)
    ax.plot(sources[0, :], sources[1, :], 'r^', markersize=10,
            markeredgecolor='darkred', markeredgewidth=1, label='Sources')
    ax.plot(receivers[0, :], receivers[1, :], 'bv', markersize=10,
            markeredgecolor='darkblue', markeredgewidth=1, label='Receivers')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Ray coverage')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    save_current_fig("ray_coverage")


def plot_ray_density(G, g):
    ray_density = np.array(G.sum(axis=0)).flatten()
    ray_density = ray_density.reshape((g.npoints[0], g.npoints[1]))
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(ray_density.T, origin='lower', cmap='viridis',
                   extent=[g.origin[0], g.extent[0], g.origin[1], g.extent[1]])
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Ray density')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    save_current_fig("ray_density")


def plot_model(m, g, title, caxis=None):
    model_2d = (1.0 / m).reshape((g.npoints[0], g.npoints[1]))
    fig, ax = plt.subplots(figsize=(8, 8))
    if caxis is not None:
        im = ax.imshow(model_2d.T, origin='lower', cmap='RdBu_r',
                       extent=[g.origin[0], g.extent[0], g.origin[1], g.extent[1]],
                       vmin=caxis[0], vmax=caxis[1])
    else:
        im = ax.imshow(model_2d.T, origin='lower', cmap='RdBu_r',
                       extent=[g.origin[0], g.extent[0], g.origin[1], g.extent[1]])
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Velocity [m/s]')
    plt.tight_layout()
    save_current_fig(title)


def plot_traveltime_comparison(d_obs, d_pred, sigma_d, method_label, ylim_residual=None):
    N = len(d_obs)
    ray_idx = np.arange(N)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                             gridspec_kw={'height_ratios': [2, 1]})

    ax = axes[0]
    ax.plot(ray_idx, 1000.0 * d_obs,  'k.',  markersize=3,  label='Observed',  zorder=3)
    ax.plot(ray_idx, 1000.0 * d_pred, 'r-',  linewidth=1.2, label='Predicted', zorder=4)
    ax.set_xlim([0, 220])
    ax.set_ylabel('Travel time [ms]')
    ax.set_title(f'Traveltime fit — {method_label}', fontweight='bold', pad=10)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    residuals = 1000.0 * (d_obs - d_pred)
    ax2.plot(ray_idx, residuals, 'kx', markersize=3, label='Residual')
    ax2.axhline( 1000.0 * sigma_d, color='r', linestyle='--', linewidth=1.2,
                 label=r'$\pm\sigma_d$')
    ax2.axhline(-1000.0 * sigma_d, color='r', linestyle='--', linewidth=1.2)
    ax2.axhline(0.0, color='gray', linestyle='-', linewidth=0.7)
    ax2.set_xlim([0, 220])
    if ylim_residual is not None:
        ax2.set_ylim(ylim_residual)
    ax2.set_xlabel('Ray path index')
    ax2.set_ylabel('Residual [ms]')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_current_fig(f"traveltime_comparison_{method_label}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("STRAIGHT RAY TOMOGRAPHY")
    print("=" * 70)

    # Define the numerical grid
    print("\n1. Setting up grid...")
    dimension = 2
    x_min = 0.0
    y_min = 0.0
    dx = 2.5
    dy = 2.5
    Nx = 20
    Ny = 20
    g = grid(dimension, [x_min, y_min], [dx, dy], np.array([Nx, Ny]))

    # Sources and receivers – left-wall sources, right-wall receivers (crosshole)
    print("2. Setting up sources and receivers (crosshole geometry)...")
    src_locations = np.array([0.0 * np.ones((11,)), np.linspace(0, 50, 11)])
    rec_locations = np.array([50.0 * np.ones((21,)), np.linspace(0, 50, 21)])

    sources, receivers = get_all_to_all_locations(src_locations, rec_locations)
    plot_rays(sources, receivers, g)

    # Compute G
    print("3. Computing forward operator G...")
    G = create_forward_operator(sources, receivers, g)

    print("\nMatrix statistics:")
    print(f"  Matrix shape:            {G.shape}")
    print(f"  Data points:             {G.shape[0]}")
    print(f"  Unknowns in model space: {G.shape[1]}")
    print(f"  Non-zero entries:        {G.count_nonzero()}")
    print(f"  Ratio of non-zeros:      {100 * G.count_nonzero() / (G.shape[0] * G.shape[1]):.4f} %")

    # Plot ray density
    print("4. Plotting ray density...")
    plot_ray_density(G, g)

    # Plot sparsity pattern
    print("5. Plotting sparsity pattern...")
    plt.figure(figsize=(10, 12))
    plt.spy(G, markersize=2, color='k')
    plt.gca().xaxis.tick_bottom()
    plt.xlabel('model space index')
    plt.ylabel('data space index')
    plt.title(r'non-zero entries of $\mathbf{G}$')
    plt.tight_layout()
    save_current_fig("G_sparsity")

    # Input model setup (checkerboard)
    print("6. Creating true model (checkerboard)...")
    dvp = 100.0  # velocity variations in m/s
    dd = 4       # width of checkerboard cells

    vp = 3000.0 * np.ones((g.npoints[0], g.npoints[1]))
    s = 1.0
    for i in range(0, g.npoints[0], dd):
        s_row = s
        for j in range(0, g.npoints[1], dd):
            end_i = min(g.npoints[0], i + dd)
            end_j = min(g.npoints[1], j + dd)
            vp[i:end_i, j:end_j] += s_row * dvp
            s_row *= -1
        s *= -1

    m_true = (1 / vp).ravel()

    clim = [2900.0, 3100.0]   # velocity [m/s]
    plot_model(m_true, g, 'True model [m/s]', caxis=clim)

    # Create observed data
    print("7. Creating observed data...")
    d_true = G * m_true

    sigma_d = 0.2e-4
    np.random.seed(42)
    d_obs = d_true + sigma_d * np.random.randn(len(d_true))

    Cd = sigma_d ** 2 * eye(len(d_obs))
    Cd_inv = 1 / sigma_d ** 2 * eye(len(d_obs))

    # Prior model
    print("9. Setting up prior model...")
    m_prior = np.ones(m_true.shape) / 3000.0

    # Prior covariance
    print("10. Computing prior covariance...")
    correlation_length = 3.0
    regularization_weight = 2.5e-5

    Cm = g.get_gaussian_prior(correlation_length)

    smoothed = Cm * m_true
    plot_model(smoothed, g, 'Smoothed true model [m/s]', caxis=clim)

    Cm *= regularization_weight ** 2
    print("11. Inverting prior covariance...")
    Cm_inv = linalg.inv(Cm.tocsc())

    # Hessian
    print("12. Computing Hessian...")
    H = G.T * Cd_inv * G + Cm_inv

    # Starting model (shared)
    print("12b. Plotting starting model (used for all inversions)...")
    plot_model(m_prior, g, 'Starting model — velocity [m/s]', caxis=clim)

    # -------------------------------------------------------------------------
    # L2 inversion
    # -------------------------------------------------------------------------
    print("13. Running L2 (Bayesian) inversion...")
    pt_l2 = PerfTracker("L2 inversion")

    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpsp
        import cupyx.scipy.sparse.linalg as cplinalg
        _use_gpu_l2 = True
    except ImportError:
        _use_gpu_l2 = False

    with pt_l2:
        if _use_gpu_l2:
            G_gpu       = cpsp.csr_matrix(G)
            Cd_inv_gpu  = cpsp.eye(len(d_obs), dtype=cp.float32) * (1.0 / sigma_d**2)
            Cm_inv_gpu  = cpsp.csr_matrix(Cm_inv)
            H_gpu       = G_gpu.T @ Cd_inv_gpu @ G_gpu + Cm_inv_gpu
            Cm_post_gpu = cplinalg.inv(H_gpu)
            rhs_gpu     = G_gpu.T @ Cd_inv_gpu @ cp.array(d_obs) + Cm_inv_gpu @ cp.array(m_prior)
            m_est       = (Cm_post_gpu @ rhs_gpu).get()
        else:
            Cm_post = scipy.sparse.linalg.inv(H.tocsc())
            m_est   = Cm_post * (G.T * Cd_inv * d_obs + Cm_inv * m_prior)

    d_est   = G * m_est
    d_prior = G * m_prior
    pt_l2.report()

    print("15. Plotting reconstructed model...")
    plot_model(m_est, g, 'L2 reconstructed velocity [m/s]', caxis=clim)

    # -------------------------------------------------------------------------
    # Conjugate Gradient inversion
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CONJUGATE GRADIENT INVERSION")
    print("=" * 70)

    print("16. Setting up CG system (same normal equations as Bayesian)...")
    A_cg = H
    b_cg = G.T * Cd_inv * d_obs + Cm_inv * m_prior

    cg_residuals = []

    def cg_callback(xk):
        r = b_cg - A_cg * xk
        cg_residuals.append(np.linalg.norm(r))

    print("17. Running Conjugate Gradient solver...")
    pt_cg = PerfTracker("CG inversion")

    with pt_cg:
        try:
            import cupy as cp
            import cupyx.scipy.sparse as cpsp
            import cupyx.scipy.sparse.linalg as cplinalg
            A_cg_gpu  = cpsp.csr_matrix(A_cg)
            b_cg_gpu  = cp.array(b_cg)
            m_cg_gpu, info = cplinalg.cg(A_cg_gpu, b_cg_gpu,
                                         x0=cp.array(m_prior), maxiter=1000)
            m_cg = m_cg_gpu.get()
        except ImportError:
            m_cg, info = sp.linalg.cg(
                A_cg,
                b_cg,
                x0=m_prior.copy(),
                maxiter=1000,
                atol=1e-8,
                callback=cg_callback,
            )
    pt_cg.report()

    if info == 0:
        print(f"    CG converged in {len(cg_residuals)} iterations.")
    else:
        print(f"    CG did not fully converge (info={info}) after {len(cg_residuals)} iterations.")

    print("18. Plotting CG reconstructed model...")
    plot_model(m_cg, g, 'CG reconstructed velocity [m/s]', caxis=clim)

    # -------------------------------------------------------------------------
    # Total Variation inversion
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TOTAL VARIATION INVERSION")
    print("=" * 70)

    print("19. Building gradient operators...")
    Dx, Dy = build_gradient_operators(Nx, Ny)

    lambda_tv = 1e4
    n_irls    = 20
    cg_tol    = 1e-10

    Cm_inv_diag = (1.0 / regularization_weight**2) * eye(int(np.prod(g.npoints)))
    A_data_tv = G.T * Cd_inv * G
    b_data_tv = G.T * Cd_inv * d_obs + Cm_inv_diag * m_prior
    A_base_tv = A_data_tv + Cm_inv_diag

    m_tv = m_prior.copy()

    print(f"20. Running IRLS ({n_irls} iterations), lambda_tv={lambda_tv:.0e}...")
    pt_tv = PerfTracker("TV inversion")

    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpsp
        import cupyx.scipy.sparse.linalg as cplinalg
        _use_gpu_tv = True
    except ImportError:
        _use_gpu_tv = False

    with pt_tv:
        for irls_iter in range(n_irls):
            gx = Dx @ m_tv
            gy = Dy @ m_tv

            grad_std = np.std(np.concatenate([gx, gy]))
            eps_tv   = max(0.01 * grad_std, 1e-8)

            wx = 1.0 / np.sqrt(gx**2 + eps_tv**2)
            wy = 1.0 / np.sqrt(gy**2 + eps_tv**2)
            L_tv = Dx.T @ sp.diags(wx) @ Dx + Dy.T @ sp.diags(wy) @ Dy

            A_tv = A_base_tv + lambda_tv * L_tv
            if _use_gpu_tv:
                A_tv_gpu    = cpsp.csr_matrix(A_tv)
                b_data_gpu  = cp.array(b_data_tv)
                m_tv_gpu, _ = cplinalg.cg(A_tv_gpu, b_data_gpu,
                                         x0=cp.array(m_tv), maxiter=1000,
                                         tol=cg_tol)
                m_tv = m_tv_gpu.get()
            else:
                m_tv_new, cg_info = sp.linalg.cg(A_tv, b_data_tv, x0=m_tv.copy(),
                                                 maxiter=1000, atol=cg_tol)
                m_tv = m_tv_new
    pt_tv.report()

    print("21. Plotting TV reconstructed model...")
    plot_model(m_tv, g, 'TV reconstructed velocity [m/s]', caxis=clim)

    # -------------------------------------------------------------------------
    # INR inversion
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("INR INVERSION")
    print("=" * 70)

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import time as _time
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  PyTorch device: {_device}")
        torch.manual_seed(42)
        np.random.seed(42)
    except ImportError:
        raise SystemExit("PyTorch not installed — run: pip install torch")

    class _FourierEncoding(nn.Module):
        def __init__(self, in_features=2, n_frequencies=16, sigma=6.0):
            super().__init__()
            B = torch.randn(in_features, n_frequencies) * sigma
            self.register_buffer('B', B)

        @property
        def out_features(self):
            return self.B.shape[1] * 2

        def forward(self, x):
            proj = x @ self.B
            return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    class _SineLayer(nn.Module):
        def __init__(self, in_features, out_features, omega_0=6.0, is_first=False):
            super().__init__()
            self.omega_0  = omega_0
            self.is_first = is_first
            self.linear   = nn.Linear(in_features, out_features)
            self._init_weights()

        def _init_weights(self):
            with torch.no_grad():
                n = self.linear.in_features
                if self.is_first:
                    self.linear.weight.uniform_(-1.0 / n, 1.0 / n)
                else:
                    bound = np.sqrt(6.0 / n) / self.omega_0
                    self.linear.weight.uniform_(-bound, bound)
                self.linear.bias.zero_()

        def forward(self, x):
            return torch.sin(self.omega_0 * self.linear(x))

    class _SIREN(nn.Module):
        def __init__(self, in_features=2, out_features=1,
                     hidden_features=128, hidden_layers=5, omega_0=6.0,
                     n_fourier=16, fourier_sigma=6.0):
            super().__init__()
            self.encoding = _FourierEncoding(in_features, n_fourier, fourier_sigma)
            first_in      = self.encoding.out_features

            layers = [_SineLayer(first_in, hidden_features, omega_0=omega_0, is_first=True)]
            for _ in range(hidden_layers):
                layers.append(_SineLayer(hidden_features, hidden_features, omega_0=omega_0))

            out = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                bound = np.sqrt(6.0 / hidden_features) / omega_0
                out.weight.uniform_(-bound, bound)
                out.bias.zero_()
            layers.append(out)
            self.net = nn.Sequential(*layers)

        def forward(self, coords):
            coords = self.encoding(coords)
            return self.net(coords)

    vp_bg    = 3000.0
    dvp_val  = 100.0
    s0_inr   = 1.0 / vp_bg
    ds_scale = abs(1.0 / (vp_bg - dvp_val) - s0_inr)
    x_max    = float(g.extent[0])
    y_max    = float(g.extent[1])

    def _raw_to_slowness(raw, s0, ds):
        return s0 + ds * torch.tanh(raw)

    def _raw_to_slowness_np(raw, s0, ds):
        return s0 + ds * np.tanh(raw)

    def _cell_centre_coords(nx, ny, xmax, ymax):
        dx  = xmax / nx;  dy  = ymax / ny
        xc  = (np.arange(nx) + 0.5) * dx
        yc  = (np.arange(ny) + 0.5) * dy
        xc_n = xc / xmax;  yc_n = yc / ymax
        return np.meshgrid(xc_n, yc_n, indexing='ij')

    _Xc, _Yc     = _cell_centre_coords(Nx, Ny, x_max, y_max)
    _coords_np   = np.stack([_Xc.ravel(), _Yc.ravel()], axis=1).astype(np.float32)
    _coords_grid = torch.tensor(_coords_np, device=_device)

    # NOTE: this is dense — keep as-is from your code
    _G_torch = torch.tensor(G.toarray().astype(np.float32), device=_device)
    _d_obs_t = torch.tensor(d_obs.astype(np.float32), device=_device)

    siren_net = _SIREN(in_features=2, out_features=1,
                       hidden_features=256, hidden_layers=7, omega_0=3.0,
                       n_fourier=32, fourier_sigma=4.0).to(_device)
    print(f"  Parameters: {sum(p.numel() for p in siren_net.parameters()):,}")

    lr_inr        = 3e-4
    n_iter_inr    = 20000
    warmup_iters  = 800
    lambda_tv_inr = 8e-5
    lambda_lap    = 5e-6
    eps_tv        = 1e-4
    print_every   = 500

    optimizer_inr = optim.Adam(siren_net.parameters(), lr=lr_inr)

    def _lr_lambda(it):
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        p = (it - warmup_iters) / max(1, n_iter_inr - warmup_iters)
        return max(5e-7 / lr_inr, 0.5 * (1.0 + np.cos(np.pi * p)))

    scheduler_inr = optim.lr_scheduler.LambdaLR(optimizer_inr, _lr_lambda)

    best_ssi         = -np.inf
    best_state_dict  = None
    best_iter        = 0
    inr_ssi_history  = []
    log_iters        = []

    t0_inr = _time.time()
    print(f"  Training ({n_iter_inr} iters, warmup={warmup_iters})...")

    pt_inr = PerfTracker("INR (SIREN)")
    pt_inr.__enter__()

    for it in range(n_iter_inr):
        siren_net.train()
        optimizer_inr.zero_grad()

        raw_train  = siren_net(_coords_grid).squeeze()
        m_train    = _raw_to_slowness(raw_train, s0_inr, ds_scale)
        d_pred     = _G_torch @ m_train
        loss_data  = torch.mean((d_pred - _d_obs_t) ** 2)

        S = m_train.reshape(Nx, Ny)

        gx_tv   = S[1:, :] - S[:-1, :]
        gy_tv   = S[:, 1:] - S[:, :-1]
        gx_tv_p = torch.nn.functional.pad(gx_tv, (0, 0, 0, 1))
        gy_tv_p = torch.nn.functional.pad(gy_tv, (0, 1, 0, 0))
        tv_norm = torch.sqrt(gx_tv_p**2 + gy_tv_p**2 + eps_tv**2)
        loss_tv = torch.mean(tv_norm)

        lap_x    = S[2:, :] - 2*S[1:-1, :] + S[:-2, :]
        lap_y    = S[:, 2:] - 2*S[:, 1:-1] + S[:, :-2]
        loss_lap = torch.mean(lap_x**2) + torch.mean(lap_y**2)

        loss = loss_data + lambda_tv_inr * loss_tv + lambda_lap * loss_lap

        loss.backward()
        torch.nn.utils.clip_grad_norm_(siren_net.parameters(), max_norm=1.0)
        optimizer_inr.step()
        scheduler_inr.step()

        if (it + 1) % print_every == 0:
            siren_net.eval()
            with torch.no_grad():
                raw_mon = siren_net(_coords_grid).cpu().numpy().squeeze()
            m_mon = _raw_to_slowness_np(raw_mon, s0_inr, ds_scale).ravel()

            d_pred_mon = G @ m_mon
            rpe_mon    = 100.0 * np.mean(np.abs(d_pred_mon - d_obs) /
                                         (np.abs(d_obs) + 1e-30))

            mu_p   = np.mean(m_mon);  mu_t   = np.mean(m_true)
            sig_p  = np.std(m_mon);   sig_t  = np.std(m_true)
            sig_pt = np.mean((m_mon - mu_p) * (m_true - mu_t))
            c1 = (0.01 * (m_true.max() - m_true.min())) ** 2
            c2 = (0.03 * (m_true.max() - m_true.min())) ** 2
            ssi_mon = ((2*mu_p*mu_t + c1) * (2*sig_pt + c2)) / \
                      ((mu_p**2 + mu_t**2 + c1) * (sig_p**2 + sig_t**2 + c2))

            if ssi_mon > best_ssi:
                best_ssi        = ssi_mon
                best_iter       = it + 1
                best_state_dict = {k: v.clone() for k, v in siren_net.state_dict().items()}

            inr_ssi_history.append(ssi_mon)
            log_iters.append(it + 1)

            elapsed   = _time.time() - t0_inr
            remaining = elapsed / (it + 1) * (n_iter_inr - it - 1)
            p_bar     = (it + 1) / n_iter_inr
            bar       = '█' * int(30 * p_bar) + '░' * (30 - int(30 * p_bar))
            print(f"  [{bar}] {100*p_bar:5.1f}% | iter {it+1:5d} | "
                  f"RPE {rpe_mon:.4f}% | SSI {ssi_mon:.5f} | "
                  f"best SSI {best_ssi:.5f}@{best_iter} | ETA {remaining/60:.1f} min")

    pt_inr.__exit__(None, None, None)
    pt_inr.report()

    siren_net.load_state_dict(best_state_dict)
    siren_net.eval()
    with torch.no_grad():
        raw_best = siren_net(_coords_grid).cpu().numpy().squeeze()
    m_inr = _raw_to_slowness_np(raw_best, s0_inr, ds_scale).ravel()

    print("22. Plotting INR reconstructed model...")
    plot_model(m_inr, g, 'INR reconstructed velocity [m/s]', caxis=clim)

    print("22b. Traveltime comparison plots (L2 + INR) with shared residual y-axis...")
    d_inr_pred = G @ m_inr

    _res_l2  = 1000.0 * (d_obs - d_est)
    _res_inr = 1000.0 * (d_obs - d_inr_pred)
    _all_res = np.concatenate([_res_l2, _res_inr])
    _pad     = 0.1 * (np.max(np.abs(_all_res)) or 1.0)
    _ylim    = (-np.max(np.abs(_all_res)) - _pad, np.max(np.abs(_all_res)) + _pad)

    plot_traveltime_comparison(d_obs, d_est, sigma_d, method_label='L2 inversion', ylim_residual=_ylim)
    plot_traveltime_comparison(d_obs, d_inr_pred, sigma_d, method_label='INR', ylim_residual=_ylim)

    extent = [g.origin[0], g.extent[0], g.origin[1], g.extent[1]]

    print("23a. Classical methods comparison (L2 | CG | TV)...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    classical = [
        ('L2 inversion',  m_est),
        ('CG inversion',  m_cg),
        ('TV inversion',  m_tv),
    ]
    for ax, (title, model) in zip(axes, classical):
        im = ax.imshow((1.0 / model.reshape(Nx, Ny).T),
                       extent=extent, origin='lower',
                       cmap='RdBu_r', vmin=clim[0], vmax=clim[1], aspect='equal')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        plt.colorbar(im, ax=ax, label='Velocity [m/s]')
    fig.suptitle('Classical inversion methods — velocity [m/s]',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    save_current_fig("classical_methods_L2_CG_TV")

    # True model | L2 | INR
    print("23c. True model vs L2 vs INR comparison...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    final_comp = [
        ('True model',   m_true),
        ('L2 inversion', m_est),
        ('INR',          m_inr),
    ]
    for ax, (title, model) in zip(axes, final_comp):
        im = ax.imshow((1.0 / model.reshape(Nx, Ny).T),
                       extent=extent, origin='lower',
                       cmap='RdBu_r', vmin=clim[0], vmax=clim[1], aspect='equal')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        plt.colorbar(im, ax=ax, label='Velocity [m/s]')
    fig.suptitle('True model vs L2 vs INR — velocity [m/s]',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    save_current_fig("true_vs_L2_vs_INR")

    # Quantitative metrics
    def _compute_metrics(m, m_true, d_obs, G):
        mu_p   = np.mean(m);     mu_t   = np.mean(m_true)
        sig_p  = np.std(m);      sig_t  = np.std(m_true)
        sig_pt = np.mean((m - mu_p) * (m_true - mu_t))
        c1     = (0.01 * (m_true.max() - m_true.min())) ** 2
        c2     = (0.03 * (m_true.max() - m_true.min())) ** 2
        ssi    = ((2*mu_p*mu_t + c1) * (2*sig_pt + c2)) / \
                 ((mu_p**2 + mu_t**2 + c1) * (sig_p**2 + sig_t**2 + c2))
        d_pred = G @ m
        rpe    = 100.0 * np.mean(np.abs(d_pred - d_obs) / (np.abs(d_obs) + 1e-30))
        return float(ssi), float(rpe)

    ssi_l2,  rpe_l2  = _compute_metrics(m_est, m_true, d_obs, G)
    ssi_inr, rpe_inr = _compute_metrics(m_inr, m_true, d_obs, G)

    print("\n" + "-" * 60)
    print("Model SSI & data RPE comparison:")
    print(f"  {'Method':<22s}  {'SSI (model)':>14s}  {'RPE data [%]':>14s}")
    print(f"  {'-'*22}  {'-'*14}  {'-'*14}")
    print(f"  {'L2 inversion':<22s}  {ssi_l2:>14.5f}  {rpe_l2:>14.4f}")
    print(f"  {'INR':<22s}  {ssi_inr:>14.5f}  {rpe_inr:>14.4f}")
    print("-" * 60)

    # Save arrays / results
    save_npz(
        "results_models_and_data",
        m_true=m_true,
        m_prior=m_prior,
        m_l2=m_est,
        m_cg=m_cg,
        m_tv=m_tv,
        m_inr=m_inr,
        d_obs=d_obs,
        d_l2=np.array(G @ m_est).ravel(),
        d_inr=np.array(G @ m_inr).ravel(),
        sigma_d=np.array([sigma_d]),
        clim=np.array(clim),
        Nx=np.array([Nx]),
        Ny=np.array([Ny]),
    )
    save_npz(
        "metrics",
        ssi_l2=np.array([ssi_l2]),
        rpe_l2=np.array([rpe_l2]),
        ssi_inr=np.array([ssi_inr]),
        rpe_inr=np.array([rpe_inr]),
        best_ssi=np.array([best_ssi]),
        best_iter=np.array([best_iter]),
        inr_ssi_history=np.array(inr_ssi_history, dtype=float),
        inr_log_iters=np.array(log_iters, dtype=int),
    )

    # Performance summary
    collect_perf_summary({
        "L2 inversion": pt_l2,
        "CG inversion": pt_cg,
        "TV inversion": pt_tv,
        "INR":          pt_inr,
    })

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)

    _log_fh.close()