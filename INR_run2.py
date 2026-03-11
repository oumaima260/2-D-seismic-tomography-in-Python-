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

import os, sys, time, warnings, re, datetime, atexit
import tracemalloc, psutil

import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg as splinalg, eye, csr_matrix, kron, diags
from scipy.ndimage import median_filter, gaussian_filter
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 12})


# =============================================================================
# Setting up
# =============================================================================

TEST_ID = "002"

OUT_DIR = os.path.abspath(f"INR_results_{TEST_ID}")
os.makedirs(OUT_DIR, exist_ok=True)

log_file_path = os.path.join(OUT_DIR, "run.log")
_log_fh = open(log_file_path, "w", buffering=1)

class Tee:
    """Redirect stdout/stderr to both console and file."""
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

def _slugify(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^\w\s\-]+", "", s)
    s = re.sub(r"[\s\-]+", "_", s)
    return s[:120] if len(s) > 120 else s

FIG_COUNTER = 0

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

def _close_log():
    try:
        print("=" * 80)
        print("RUN END")
        print("Timestamp:", datetime.datetime.now().isoformat())
        print("=" * 80)
        _log_fh.close()
    except Exception:
        pass

atexit.register(_close_log)

print("=" * 80)
print("RUN START")
print("Timestamp:", datetime.datetime.now().isoformat())
print("OUT_DIR   :", OUT_DIR)
print("=" * 80)


# =============================================================================
# for tracking
# =============================================================================

class PerfTracker:
    """Context manager that tracks wall-clock time, CPU peak RAM, and GPU peak RAM."""
    def __init__(self, label):
        self.label       = label
        self.elapsed     = None
        self.peak_mb     = None
        self.gpu_peak_mb = 0.0

    def __enter__(self):
        tracemalloc.start()
        self._proc = psutil.Process(os.getpid())
        self._t0   = time.perf_counter()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
        except ImportError:
            pass
        return self

    def __exit__(self, *args):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                self.gpu_peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        except ImportError:
            pass
        self.elapsed = time.perf_counter() - self._t0
        _, peak      = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_mb = peak / 1024**2

    def report(self):
        print(f"  [{self.label}]  time = {self.elapsed:.3f} s  |  "
              f"CPU peak = {self.peak_mb:.1f} MB  |  GPU peak = {self.gpu_peak_mb:.1f} MB")


def collect_perf_summary(trackers: dict) -> None:
    """Print a tidy table of timing + memory for all methods."""
    print("\n" + "=" * 84)
    print("PERFORMANCE SUMMARY")
    print(f"  {'Method':<22s}  {'Time (s)':>10s}  {'CPU peak (MB)':>14s}  {'GPU peak (MB)':>14s}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*14}  {'-'*14}")
    for label, pt in trackers.items():
        if pt is not None and pt.elapsed is not None:
            gpu_mb = getattr(pt, 'gpu_peak_mb', 0.0)
            print(f"  {label:<22s}  {pt.elapsed:>10.3f}  {pt.peak_mb:>14.1f}  {gpu_mb:>14.1f}")
    print("=" * 84 + "\n")

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # e.g., if run in a notebook
    SCRIPT_DIR = os.getcwd()

shape   = (101, 101)
spacing = (10., 10.)
origin  = (0., 0.)
vp_bck  = 2.5
nbl     = 40  # (kept; not used below)

NX, NZ = shape
DX, DZ = spacing

MODEL_PATH = os.path.join(SCRIPT_DIR, 'model1.npy')
if not os.path.exists(MODEL_PATH):
    sys.exit(f"ERROR: model1.npy not found at {MODEL_PATH}\n"
             "Place model1.npy in the same folder as this script.")

v       = np.load(MODEL_PATH)
v       = np.transpose(v)
vp_true = v.astype(np.float32)

vp_init = (np.ones(shape, dtype=np.float32) * vp_bck)

ext = [
    origin[0],
    origin[0] + NX * DX * 1e-3,
    origin[1] + NZ * DZ * 1e-3,
    origin[1],
]


def plot_vp(vp, title, vmin=2.4, vmax=2.75, cmap='jet'):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(vp.T, origin='upper', cmap=cmap,
                   extent=ext, vmin=vmin, vmax=vmax, aspect='equal')
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Depth (km)')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Vp (km/s)', shrink=0.85)
    plt.tight_layout()
    save_current_fig(title)


def plot_borehole_geometry(vp, src_m, rec_m, title='Crosswell Borehole Geometry'):
    src_km_ = src_m * 1e-3
    rec_km_ = rec_m * 1e-3
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(vp.T, origin='upper', cmap='jet', extent=ext,
              vmin=2.4, vmax=2.75, aspect='equal', alpha=0.85)
    for si in range(len(src_km_)):
        for ri in range(len(rec_km_)):
            ax.plot([src_km_[si, 0], rec_km_[ri, 0]],
                    [src_km_[si, 1], rec_km_[ri, 1]],
                    'w-', lw=0.4, alpha=0.25)
    ax.scatter(src_km_[:, 0], src_km_[:, 1],
               c='yellow', marker='*', s=140, zorder=5, label=f'Sources ({len(src_km_)})')
    ax.scatter(rec_km_[:, 0], rec_km_[:, 1],
               c='cyan',   marker='v', s=70,  zorder=5, label=f'Receivers ({len(rec_km_)})')
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Depth (km)')
    ax.set_title(title)
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    save_current_fig(title)


plot_vp(vp_true, 'True velocity model')


# ===========================================================================
# 1.  Acquisition geometry
# ===========================================================================

NSRC = 11
NREC = 21


def borehole_coords(side, n):
    xmax = (NX - 1) * DX
    zmax = (NZ - 1) * DZ
    arr  = np.empty((n, 2), dtype=np.float32)
    arr[:, 1] = np.linspace(0, zmax, n)
    arr[:, 0] = 20. if side == 'left' else xmax - 20.
    return arr


def build_ray_matrix(src_km, rec_km, nx, nz, dx_km, dz_km, n_samp=120):
    n_src = src_km.shape[0]
    n_rec = rec_km.shape[0]
    rows, cols, data = [], [], []
    ray_idx = 0
    for si in range(n_src):
        x0, z0 = src_km[si]
        for ri in range(n_rec):
            x1, z1 = rec_km[ri]
            ddx, ddz = x1 - x0, z1 - z0
            rlen = np.sqrt(ddx**2 + ddz**2)
            if rlen < 1e-8:
                ray_idx += 1
                continue
            t_vals   = np.linspace(0, 1, n_samp)
            cell_len = {}
            for t in t_vals[:-1]:
                xi = x0 + t * ddx
                zi = z0 + t * ddz
                ci = int(xi / dx_km)
                cj = int(zi / dz_km)
                if 0 <= ci < nx and 0 <= cj < nz:
                    cid = ci * nz + cj
                    cell_len[cid] = cell_len.get(cid, 0.) + rlen / (n_samp - 1)
            for cid, ln in cell_len.items():
                rows.append(ray_idx); cols.append(cid); data.append(ln)
            ray_idx += 1
    return csr_matrix((data, (rows, cols)), shape=(n_src * n_rec, nx * nz))


print("\n" + "=" * 65)
print("STEP 1  —  FORWARD OPERATOR  (CROSSWELL BOREHOLE GEOMETRY)")
print(f"          {NSRC} sources (left well)  x  {NREC} receivers (right well)")
print("=" * 65)

src_km = borehole_coords('left',  NSRC) * 1e-3
rec_km = borehole_coords('right', NREC) * 1e-3

src_m  = borehole_coords('left',  NSRC)
rec_m  = borehole_coords('right', NREC)
plot_borehole_geometry(vp_true, src_m, rec_m,
                       title=f'Crosswell geometry — {NSRC} sources (left) | {NREC} receivers (right)')

t0 = time.time()
G  = build_ray_matrix(src_km, rec_km, NX, NZ, DX * 1e-3, DZ * 1e-3)

s_true  = (1.0 / vp_true).ravel()
s_init  = (1.0 / vp_init).ravel()
d_true  = G @ s_true
sigma_d = 1e-4 * np.max(np.abs(d_true))
np.random.seed(42)
d_obs   = d_true + sigma_d * np.random.randn(len(d_true))

save_npz(
    "setup_forward_operator",
    vp_true=vp_true, vp_init=vp_init,
    s_true=s_true, s_init=s_init,
    d_true=np.array(d_true).ravel(),
    d_obs=np.array(d_obs).ravel(),
    sigma_d=np.array([sigma_d], dtype=float),
    NX=np.array([NX], dtype=int), NZ=np.array([NZ], dtype=int),
    DX=np.array([DX], dtype=float), DZ=np.array([DZ], dtype=float),
)

plot_vp(vp_init, 'Starting model (homogeneous 2.5 km/s)')

print("\n" + "=" * 65)
print("STEP 2  —  L2 INVERSION")
print("=" * 65)


def laplacian_matrix(nx, nz):
    Dx  = diags([-1., 1.], [0, 1], shape=(nx-1, nx), dtype=float)
    Dz  = diags([-1., 1.], [0, 1], shape=(nz-1, nz), dtype=float)
    Inx = sp.eye(nx, format='csr')
    Inz = sp.eye(nz, format='csr')
    return kron(Dx.T @ Dx, Inz) + kron(Inx, Dz.T @ Dz)


t0_l2  = time.time()
L      = laplacian_matrix(NX, NZ)
lam_l2 = 1e2
Cd_inv = (1.0 / sigma_d**2) * eye(len(d_obs))
LtL    = lam_l2 * (L.T @ L)
GtCdi  = G.T @ Cd_inv
H_l2   = GtCdi @ G + LtL
rhs_l2 = GtCdi @ d_obs + LtL @ s_init

pt_l2 = PerfTracker("L2 inversion")
with pt_l2:
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpsp
        import cupyx.scipy.sparse.linalg as cplinalg
        G_gpu      = cpsp.csr_matrix(G)
        Cd_inv_gpu = cpsp.eye(len(d_obs), dtype=cp.float64) * (1.0 / sigma_d**2)
        LtL_gpu    = cpsp.csr_matrix(LtL)
        H_l2_gpu   = G_gpu.T @ Cd_inv_gpu @ G_gpu + LtL_gpu
        rhs_gpu    = G_gpu.T @ Cd_inv_gpu @ cp.array(d_obs) + LtL_gpu @ cp.array(s_init)
        s_l2       = cplinalg.spsolve(H_l2_gpu, rhs_gpu).get()
        print("  L2 solved on GPU (cupy)")
    except (ImportError, Exception):
        s_l2 = splinalg.spsolve(H_l2.tocsc(), rhs_l2)
        print("  L2 solved on CPU (scipy)")

vp_l2  = (1.0 / s_l2).reshape(NX, NZ).astype(np.float32)
pt_l2.report()

res_l2    = np.linalg.norm(G @ s_l2 - d_obs) / np.linalg.norm(d_obs)
rmse_l2   = np.sqrt(np.mean((s_l2 - s_true)**2))
misfit_l2 = 0.5 * np.sum((G @ s_l2 - d_obs)**2) / sigma_d**2

print(f"  L2: rel.res={res_l2:.4e} | RMSE={rmse_l2:.4e} | misfit={misfit_l2:.4e}")

plot_vp(vp_l2, 'L2 inversion')

save_npz(
    "l2_results",
    s_l2=s_l2.astype(np.float64),
    vp_l2=vp_l2,
    res_l2=np.array([res_l2], dtype=float),
    rmse_l2=np.array([rmse_l2], dtype=float),
    misfit_l2=np.array([misfit_l2], dtype=float),
    lam_l2=np.array([lam_l2], dtype=float),
)


# ===========================================================================
# INR INVERSION
# ===========================================================================

print("\n" + "=" * 65)
print("STEP 3  —  SIREN INVERSION")
print("=" * 65)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    HAS_TORCH = True
    print(f"  PyTorch device: {device}")
except ImportError:
    HAS_TORCH = False
    pt_siren  = None
    print("  PyTorch not found — skipping SIREN.")

if HAS_TORCH:

    class FourierEncoding(nn.Module):
        def __init__(self, n_freq=16, sigma=4.0):
            super().__init__()
            self.register_buffer('B', torch.randn(2, n_freq) * sigma)

        @property
        def out_features(self):
            return self.B.shape[1] * 2

        def forward(self, x):
            p = x @ self.B
            return torch.cat([torch.sin(p), torch.cos(p)], dim=-1)


    class SineLayer(nn.Module):
        def __init__(self, in_f, out_f, omega=6.0, is_first=False):
            super().__init__()
            self.omega  = omega
            self.linear = nn.Linear(in_f, out_f)
            with torch.no_grad():
                b = 1.0 / in_f if is_first else np.sqrt(6.0 / in_f) / omega
                self.linear.weight.uniform_(-b, b)
                self.linear.bias.zero_()

        def forward(self, x):
            return torch.sin(self.omega * self.linear(x))


    class SIREN(nn.Module):
        def __init__(self, hidden=128, n_layers=4, omega=6.0,
                     n_freq=24, sigma_enc=4.0):
            super().__init__()
            self.enc = FourierEncoding(n_freq, sigma_enc)
            fin      = self.enc.out_features
            layers   = [SineLayer(fin, hidden, omega, is_first=True)]
            for _ in range(n_layers):
                layers.append(SineLayer(hidden, hidden, omega))
            out = nn.Linear(hidden, 1)
            with torch.no_grad():
                b = np.sqrt(6.0 / hidden) / omega
                out.weight.uniform_(-b, b)
                out.bias.zero_()
            layers.append(out)
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(self.enc(x))

    # -- Slowness parameterisation ----------------------------------------
    s_bkg    = float(1.0 / vp_bck)
    s_min    = float(1.0 / vp_true.max())
    s_max    = float(1.0 / vp_true.min())
    ds_scale = max(s_bkg - s_min, s_max - s_bkg) * 1.05

    def to_s_t(raw):  return torch.clamp(s_bkg + ds_scale * torch.tanh(raw), s_min, s_max)
    def to_s_np(raw): return np.clip(s_bkg + ds_scale * np.tanh(raw), s_min, s_max)

    # -- Grid coords (normalised) -----------------------------------------
    x_max_km = (NX - 1) * DX * 1e-3
    z_max_km = (NZ - 1) * DZ * 1e-3
    xi_n = (np.arange(NX) + 0.5) * DX * 1e-3 / x_max_km
    zi_n = (np.arange(NZ) + 0.5) * DZ * 1e-3 / z_max_km
    XX_c, ZZ_c  = np.meshgrid(xi_n, zi_n, indexing='ij')
    coords_grid = torch.tensor(
        np.stack([XX_c.ravel(), ZZ_c.ravel()], axis=1).astype(np.float32),
        device=device)

    # -- Sparse G as torch COO --------------------------------------------
    G_coo   = G.tocoo()
    G_torch = torch.sparse_coo_tensor(
        torch.tensor(np.vstack([G_coo.row, G_coo.col]), dtype=torch.long),
        torch.tensor(G_coo.data, dtype=torch.float32),
        size=G.shape, device=device).coalesce()
    d_obs_t = torch.tensor(d_obs.astype(np.float32), device=device)

    # -- Hyperparameters --------------------------------------------------
    N_ITER      = 15000
    LR          = 6e-4
    LAM_TV      = 3e-3
    LAM_LAP     = 8e-7
    HUBER_DELTA = 1.0
    PRINT_EVERY = 500

    siren = SIREN(hidden=128, n_layers=5, omega=6.0,
                  n_freq=32, sigma_enc=5.0).to(device)

    with torch.no_grad():
        siren.net[-1].bias.zero_()

    optimizer = optim.Adam(siren.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_ITER, eta_min=2e-6)
    ld_hist, tv_hist = [], []

    best_misfit = np.inf
    best_state  = None
    best_iter   = 0

    # -- Training loop ----------------------------------------------------
    t0_siren = time.time()

    pt_siren = PerfTracker("INR (SIREN)")
    pt_siren.__enter__()

    for it in range(N_ITER):
        siren.train()
        optimizer.zero_grad(set_to_none=True)

        raw    = siren(coords_grid).squeeze()
        s_pred = to_s_t(raw)
        d_pred = torch.mv(G_torch, s_pred)

        # Huber data loss
        d_res  = (d_pred - d_obs_t) / sigma_d
        ld     = torch.mean(torch.where(
            d_res.abs() < HUBER_DELTA,
            0.5 * d_res**2,
            HUBER_DELTA * (d_res.abs() - 0.5 * HUBER_DELTA)))

        # Huber-TV regularisation
        S        = s_pred.reshape(NX, NZ)
        gx       = S[1:, :] - S[:-1, :]
        gz       = S[:, 1:] - S[:, :-1]
        gx_p     = torch.nn.functional.pad(gx, (0, 0, 0, 1))
        gz_p     = torch.nn.functional.pad(gz, (0, 1, 0, 0))
        grad_mag = torch.sqrt(gx_p**2 + gz_p**2 + 1e-8)
        tv_delta = 8e-4
        loss_tv  = torch.mean(torch.where(
            grad_mag < tv_delta,
            0.5 * grad_mag**2 / tv_delta,
            grad_mag - 0.5 * tv_delta))

        lx       = S[2:, :] - 2*S[1:-1, :] + S[:-2, :]
        lz       = S[:, 2:] - 2*S[:, 1:-1] + S[:, :-2]
        loss_lap = torch.mean(lx**2) + torch.mean(lz**2)

        loss = ld + LAM_TV * loss_tv + LAM_LAP * loss_lap
        loss.backward()
        torch.nn.utils.clip_grad_norm_(siren.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        ld_hist.append(ld.item())
        tv_hist.append(loss_tv.item())

        if (it + 1) % PRINT_EVERY == 0:
            siren.eval()
            with torch.no_grad():
                s_mon = to_s_np(siren(coords_grid).cpu().numpy().squeeze()).ravel()
            mis_mon = 0.5 * np.sum((G @ s_mon - d_obs)**2) / sigma_d**2
            if mis_mon < best_misfit:
                best_misfit = mis_mon
                best_iter   = it + 1
                best_state  = {k: v.clone() for k, v in siren.state_dict().items()}

            el  = time.time() - t0_siren
            rem = el / (it+1) * (N_ITER - it - 1)
            p   = (it+1) / N_ITER
            bar = '█' * int(30*p) + '░' * (30 - int(30*p))
            print(f"  [{bar}] {100*p:5.1f}% | iter {it+1:4d} | "
                  f"data {ld.item():.3e} | TV {loss_tv.item():.3e} | "
                  f"misfit {mis_mon:.3e} | best@{best_iter} | ETA {rem/60:.1f} min")

    pt_siren.__exit__(None, None, None)
    pt_siren.report()

    siren.load_state_dict(best_state)
    siren.eval()
    with torch.no_grad():
        raw_out = siren(coords_grid).cpu().numpy().squeeze()

    s_siren  = to_s_np(raw_out).reshape(NX, NZ)
    vp_siren = (1.0 / s_siren).astype(np.float32)

    # -- Post-processing --------------------------------------------------
    vp_clean   = median_filter(vp_siren, size=3)
    salt_mask  = vp_clean > 2.68
    vp_bg2     = gaussian_filter(vp_clean, sigma=0.8)
    vp_siren   = np.where(salt_mask, vp_clean, vp_bg2).astype(np.float32)
    s_siren    = (1.0 / vp_siren).astype(np.float32)

    res_siren    = np.linalg.norm(G @ s_siren.ravel() - d_obs) / np.linalg.norm(d_obs)
    rmse_siren   = np.sqrt(np.mean((s_siren.ravel() - s_true)**2))
    misfit_siren = 0.5 * np.sum((G @ s_siren.ravel() - d_obs)**2) / sigma_d**2

    print(f"  INR: rel.res={res_siren:.4e} | RMSE={rmse_siren:.4e} | misfit={misfit_siren:.4e}")
    plot_vp(vp_siren, 'INR inversion (SIREN)')

    save_npz(
        "inr_main_results",
        vp_siren=vp_siren,
        s_siren=s_siren,
        res_siren=np.array([res_siren], dtype=float),
        rmse_siren=np.array([rmse_siren], dtype=float),
        misfit_siren=np.array([misfit_siren], dtype=float),
        best_iter=np.array([best_iter], dtype=int),
        best_misfit=np.array([best_misfit], dtype=float),
        ld_hist=np.array(ld_hist, dtype=float),
        tv_hist=np.array(tv_hist, dtype=float),
        hyper=np.array([N_ITER, LR, LAM_TV, LAM_LAP, HUBER_DELTA, PRINT_EVERY], dtype=float),
    )


# ===========================================================================
# Results
# ===========================================================================

print("\n" + "=" * 65)
print("STEP 3b  —  DATA COMPARISON")
print("=" * 65)

d_pred_l2   = G @ s_l2
residual_l2 = d_pred_l2 - d_obs

if HAS_TORCH:
    d_pred_inr   = G @ s_siren.ravel()
    residual_inr = d_pred_inr - d_obs

ray_idx = np.arange(len(d_obs))

if HAS_TORCH:
    fig, axes = plt.subplots(2, 2, figsize=(16, 8),
                             gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})
    fig.suptitle("Data Comparison — Observed vs Predicted Traveltimes",
                 fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    ax.plot(ray_idx, d_obs,     'k.', ms=2.5, alpha=0.6, label='Observed')
    ax.plot(ray_idx, d_pred_l2, 'r-', lw=1.2,            label='Predicted (L2)')
    ax.set_title("L2 Inversion", fontweight='bold')
    ax.set_ylabel("Traveltime [s]")
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xticklabels([])
    ax.grid(True, ls='--', alpha=0.4)

    ax = axes[1, 0]
    ax.plot(ray_idx, residual_l2, 'k.', ms=2, alpha=0.6, label='Residual')
    ax.axhline( sigma_d, color='r', ls='--', lw=1.2, label=r'±$\sigma_d$')
    ax.axhline(-sigma_d, color='r', ls='--', lw=1.2)
    ax.axhline(0,        color='gray', ls='-', lw=0.8)
    ax.set_ylabel("Residual [s]")
    ax.set_xlabel("Ray path index")
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, ls='--', alpha=0.4)

    ax = axes[0, 1]
    ax.plot(ray_idx, d_obs,      'k.', ms=2.5, alpha=0.6, label='Observed')
    ax.plot(ray_idx, d_pred_inr, 'r-', lw=1.2,            label='Predicted (INR)')
    ax.set_title("INR / SIREN Inversion", fontweight='bold')
    ax.set_ylabel("Traveltime [s]")
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xticklabels([])
    ax.grid(True, ls='--', alpha=0.4)

    ax = axes[1, 1]
    ax.plot(ray_idx, residual_inr, 'k.', ms=2, alpha=0.6, label='Residual')
    ax.axhline( sigma_d, color='r', ls='--', lw=1.2, label=r'±$\sigma_d$')
    ax.axhline(-sigma_d, color='r', ls='--', lw=1.2)
    ax.axhline(0,        color='gray', ls='-', lw=0.8)
    ax.set_ylabel("Residual [s]")
    ax.set_xlabel("Ray path index")
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, ls='--', alpha=0.4)
else:
    fig, axes = plt.subplots(2, 1, figsize=(12, 6),
                             gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})
    fig.suptitle("Data Comparison — L2 Inversion: Observed vs Predicted",
                 fontsize=14, fontweight='bold')

    axes[0].plot(ray_idx, d_obs,     'k.', ms=2.5, alpha=0.6, label='Observed')
    axes[0].plot(ray_idx, d_pred_l2, 'r-', lw=1.2,            label='Predicted (L2)')
    axes[0].set_title("Traveltime fit — L2 inversion", fontweight='bold')
    axes[0].set_ylabel("Traveltime [s]")
    axes[0].legend(fontsize=9)
    axes[0].set_xticklabels([])
    axes[0].grid(True, ls='--', alpha=0.4)

    axes[1].plot(ray_idx, residual_l2, 'k.', ms=2, alpha=0.6, label='Residual')
    axes[1].axhline( sigma_d, color='r', ls='--', lw=1.2, label=r'±$\sigma_d$')
    axes[1].axhline(-sigma_d, color='r', ls='--', lw=1.2)
    axes[1].axhline(0,        color='gray', ls='-', lw=0.8)
    axes[1].set_ylabel("Residual [s]")
    axes[1].set_xlabel("Ray path index")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, ls='--', alpha=0.4)

plt.tight_layout()
save_current_fig("data_comparison_observed_vs_predicted")

save_npz(
    "data_comparison_arrays",
    d_obs=np.array(d_obs).ravel(),
    d_pred_l2=np.array(d_pred_l2).ravel(),
    residual_l2=np.array(residual_l2).ravel(),
    sigma_d=np.array([sigma_d], dtype=float),
    d_pred_inr=np.array(d_pred_inr).ravel() if HAS_TORCH else np.array([], dtype=float),
    residual_inr=np.array(residual_inr).ravel() if HAS_TORCH else np.array([], dtype=float),
)
print("\n" + "=" * 65)
print("STEP 4  —  COMPARISON")
print("=" * 65)

if HAS_TORCH:
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    models = [vp_true, vp_l2, vp_siren]
    titles = ['True model', 'L2', 'INR']
else:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    models = [vp_true, vp_l2]
    titles = ['True model', 'L2']

for ax, vp, title in zip(axes, models, titles):
    im = ax.imshow(vp.T, origin='upper', cmap='jet', aspect='equal',
                   extent=ext, vmin=2.4, vmax=2.75)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Depth (km)')
    plt.colorbar(im, ax=ax, label='Vp (km/s)', shrink=0.85)

plt.tight_layout()
save_current_fig("comparison_true_l2_inr")

if HAS_TORCH:
    winner = 'INR' if rmse_siren < rmse_l2 else 'L2 MAP'
    pct_w  = abs(1 - min(rmse_l2, rmse_siren) / max(rmse_l2, rmse_siren)) * 100
    print(f"\n  -> {winner} wins by {pct_w:.1f}% lower model RMSE")

collect_perf_summary({
    "L2 inversion" : pt_l2,
    "INR (SIREN)"  : pt_siren if HAS_TORCH else None,
})

save_npz(
    "summary_metrics_main",
    rmse_l2=np.array([rmse_l2], dtype=float),
    res_l2=np.array([res_l2], dtype=float),
    misfit_l2=np.array([misfit_l2], dtype=float),
    rmse_inr=np.array([rmse_siren], dtype=float) if HAS_TORCH else np.array([], dtype=float),
    res_inr=np.array([res_siren], dtype=float) if HAS_TORCH else np.array([], dtype=float),
    misfit_inr=np.array([misfit_siren], dtype=float) if HAS_TORCH else np.array([], dtype=float),
)


# ===========================================================================
# ===========================================================================
#  INR NOISE ANALYSIS
# ===========================================================================
# ===========================================================================

print("\n\n" + "=" * 65)
print("  PART II — INR NOISE STUDY (5% / 10% / 20%)")
print("=" * 65)

sigma_d_base = np.std(d_true)
print(f"  sigma_d_base (std of d_true) = {sigma_d_base:.6e} s")

NOISE_PCT    = [5, 10, 20]
sigma_d_loss = (5 / 100.0) * sigma_d_base  # fixed loss scaling

if not HAS_TORCH:
    print("  PyTorch not available — skipping noise study.")
else:
    # Hyperparameters for noise-study runs
    NS_N_ITER      = 15000
    NS_LR          = 6e-4
    NS_LAM_TV      = 3e-3
    NS_LAM_LAP     = 8e-7
    NS_HUBER_DELTA = 1.0
    NS_PRINT_EVERY = 500

    def run_inr_noise(d_obs_n, label, seed):
        """Train one INR run for the noise study."""
        print(f"\n{'='*65}")
        print(f"  INR NOISE STUDY  —  {label}  |  sigma_d_loss = {sigma_d_loss:.4e} s")
        print(f"{'='*65}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        d_obs_t_n = torch.tensor(d_obs_n.astype(np.float32), device=device)

        siren_n = SIREN(hidden=128, n_layers=5, omega=6.0,
                        n_freq=32, sigma_enc=5.0).to(device)
        with torch.no_grad():
            siren_n.net[-1].bias.zero_()

        opt_n   = optim.Adam(siren_n.parameters(), lr=NS_LR)
        sched_n = optim.lr_scheduler.CosineAnnealingLR(
            opt_n, T_max=NS_N_ITER, eta_min=2e-6)

        best_misfit_n = np.inf
        best_state_n  = None
        best_iter_n   = 0
        t0_n          = time.time()

        for it in range(NS_N_ITER):
            siren_n.train()
            opt_n.zero_grad(set_to_none=True)

            raw_n    = siren_n(coords_grid).squeeze()
            s_pred_n = to_s_t(raw_n)
            d_pred_n = torch.mv(G_torch, s_pred_n)

            d_res_n  = (d_pred_n - d_obs_t_n) / sigma_d_loss
            ld_n     = torch.mean(torch.where(
                d_res_n.abs() < NS_HUBER_DELTA,
                0.5 * d_res_n**2,
                NS_HUBER_DELTA * (d_res_n.abs() - 0.5 * NS_HUBER_DELTA)))

            S_n        = s_pred_n.reshape(NX, NZ)
            gx_n       = S_n[1:, :] - S_n[:-1, :]
            gz_n       = S_n[:, 1:] - S_n[:, :-1]
            gx_p_n     = torch.nn.functional.pad(gx_n, (0, 0, 0, 1))
            gz_p_n     = torch.nn.functional.pad(gz_n, (0, 1, 0, 0))
            grad_mag_n = torch.sqrt(gx_p_n**2 + gz_p_n**2 + 1e-8)
            tv_delta   = 8e-4
            loss_tv_n  = torch.mean(torch.where(
                grad_mag_n < tv_delta,
                0.5 * grad_mag_n**2 / tv_delta,
                grad_mag_n - 0.5 * tv_delta))

            lx_n       = S_n[2:, :] - 2*S_n[1:-1, :] + S_n[:-2, :]
            lz_n       = S_n[:, 2:] - 2*S_n[:, 1:-1] + S_n[:, :-2]
            loss_lap_n = torch.mean(lx_n**2) + torch.mean(lz_n**2)

            loss_n = ld_n + NS_LAM_TV * loss_tv_n + NS_LAM_LAP * loss_lap_n
            loss_n.backward()
            torch.nn.utils.clip_grad_norm_(siren_n.parameters(), 1.0)
            opt_n.step()
            sched_n.step()

            if (it + 1) % NS_PRINT_EVERY == 0:
                siren_n.eval()
                with torch.no_grad():
                    s_mon_n = to_s_np(siren_n(coords_grid).cpu().numpy().squeeze()).ravel()
                mis_n = 0.5 * np.sum((G @ s_mon_n - d_obs_n)**2) / sigma_d_loss**2
                if mis_n < best_misfit_n:
                    best_misfit_n = mis_n
                    best_iter_n   = it + 1
                    best_state_n  = {k: v.clone() for k, v in siren_n.state_dict().items()}

                el  = time.time() - t0_n
                rem = el / (it+1) * (NS_N_ITER - it - 1)
                p   = (it+1) / NS_N_ITER
                bar = '█' * int(30*p) + '░' * (30 - int(30*p))
                print(f"  [{bar}] {100*p:5.1f}% | iter {it+1:4d} | "
                      f"data {ld_n.item():.3e} | TV {loss_tv_n.item():.3e} | "
                      f"misfit {mis_n:.3e} | best@{best_iter_n} | ETA {rem/60:.1f} min")

        siren_n.load_state_dict(best_state_n)
        siren_n.eval()
        with torch.no_grad():
            raw_out_n = siren_n(coords_grid).cpu().numpy().squeeze()

        s_out_n  = to_s_np(raw_out_n).reshape(NX, NZ)
        vp_out_n = (1.0 / s_out_n).astype(np.float32)

        # post-process
        vp_clean_n = median_filter(vp_out_n, size=3)
        salt_msk_n = vp_clean_n > 2.68
        vp_bg_n    = gaussian_filter(vp_clean_n, sigma=0.8)
        vp_out_n   = np.where(salt_msk_n, vp_clean_n, vp_bg_n).astype(np.float32)
        s_out_n    = (1.0 / vp_out_n).astype(np.float32)

        rmse_n = np.sqrt(np.mean((s_out_n.ravel() - s_true)**2))
        res_n  = np.linalg.norm(G @ s_out_n.ravel() - d_obs_n) / np.linalg.norm(d_obs_n)

        print(f"\n  {label}  |  RMSE = {rmse_n:.4e}  |  rel.res = {res_n:.4e}"
              f"  |  time = {(time.time()-t0_n)/60:.1f} min")

        return vp_out_n, s_out_n, rmse_n, res_n, best_iter_n, best_misfit_n

    ns_results   = {}   # pct -> dict of results
    ns_d_obs_all = {}   # pct -> (d_obs_n, noise_std)

    for i, pct in enumerate(NOISE_PCT):
        noise_std_n = (pct / 100.0) * sigma_d_base
        seed = 100 + i * 37
        np.random.seed(seed)
        d_obs_n_pct = d_true + noise_std_n * np.random.randn(len(d_true))
        ns_d_obs_all[pct] = (d_obs_n_pct, noise_std_n)

        vp_n, s_n, rmse_n, res_n, bi, bm = run_inr_noise(d_obs_n_pct, f"{pct}% noise", seed=seed)
        ns_results[pct] = dict(vp=vp_n, s=s_n, rmse=rmse_n, res=res_n, best_iter=bi, best_misfit=bm)

        # save each run to npz
        save_npz(
            f"noise_study_{pct}pct",
            pct=np.array([pct], dtype=int),
            noise_std=np.array([noise_std_n], dtype=float),
            sigma_d_loss=np.array([sigma_d_loss], dtype=float),
            d_obs=d_obs_n_pct.astype(np.float64),
            vp=vp_n,
            s=s_n,
            rmse=np.array([rmse_n], dtype=float),
            res=np.array([res_n], dtype=float),
            best_iter=np.array([bi], dtype=int),
            best_misfit=np.array([bm], dtype=float),
        )

    # Combined figure: 3 rows x 4 cols
    ray_idx_ns = np.arange(NSRC * NREC)

    fig, axes_all = plt.subplots(3, 4, figsize=(28, 16))
    fig.suptitle(
        'INR Noise Study — velocity models and data fit (5% / 10% / 20%)',
        fontsize=15, fontweight='bold'
    )

    for row, pct in enumerate(NOISE_PCT):
        vp_n = ns_results[pct]["vp"]
        rmse_n = ns_results[pct]["rmse"]
        d_obs_n_p, sig_n = ns_d_obs_all[pct]
        d_pred_n = G @ (1.0 / vp_n).ravel()
        residual_n = d_pred_n - d_obs_n_p

        ax = axes_all[row, 0]
        im = ax.imshow(vp_true.T, origin='upper', cmap='jet', aspect='equal',
                       extent=ext, vmin=2.4, vmax=2.75)
        ax.set_title('True model', fontweight='bold', fontsize=10)
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Depth (km)')
        plt.colorbar(im, ax=ax, label='Vp (km/s)', shrink=0.85)

        ax = axes_all[row, 1]
        im = ax.imshow(vp_n.T, origin='upper', cmap='jet', aspect='equal',
                       extent=ext, vmin=2.4, vmax=2.75)
        ax.set_title(f'INR — {pct}% noise | RMSE={rmse_n:.4e}', fontweight='bold', fontsize=10)
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Depth (km)')
        plt.colorbar(im, ax=ax, label='Vp (km/s)', shrink=0.85)

        ax = axes_all[row, 2]
        ax.plot(ray_idx_ns, d_obs_n_p, 'k.', ms=2, alpha=0.5, label='Observed (noisy)')
        ax.plot(ray_idx_ns, d_pred_n,  'r-', lw=1.2, label='Predicted (INR)')
        ax.set_title(f'Traveltime fit — {pct}% noise', fontweight='bold', fontsize=10)
        ax.set_ylabel('Traveltime [s]')
        ax.set_xlabel('Ray index')
        ax.legend(fontsize=8)
        ax.grid(True, ls='--', alpha=0.4)

        ax = axes_all[row, 3]
        ax.plot(ray_idx_ns, residual_n, 'k.', ms=2, alpha=0.5, label='Residual')
        ax.axhline( sig_n, color='r', ls='--', lw=1.2, label=r'±$\sigma_d$')
        ax.axhline(-sig_n, color='r', ls='--', lw=1.2)
        ax.axhline(0, color='gray', ls='-', lw=0.8)
        ax.set_title(f'Residual — {pct}% noise', fontweight='bold', fontsize=10)
        ax.set_ylabel('Residual [s]')
        ax.set_xlabel('Ray index')
        ax.legend(fontsize=8)
        ax.grid(True, ls='--', alpha=0.4)

    plt.tight_layout()
    save_current_fig("noise_study_summary_grid")

    # Summary table
    print("\n" + "=" * 65)
    print("  FINAL SUMMARY — INR NOISE STUDY")
    print(f"  sigma_d_base (std of d_true) = {sigma_d_base:.4e} s")
    print(f"  sigma_d_loss (fixed, 5% level) = {sigma_d_loss:.4e} s")
    print("=" * 65)
    print(f"  {'Noise':8s}  {'noise_std':>12s}  {'RMSE':>14s}  {'Rel. res.':>12s}")
    print("  " + "-" * 52)
    for pct in NOISE_PCT:
        rmse_n = ns_results[pct]["rmse"]
        res_n  = ns_results[pct]["res"]
        _, noise_std_n = ns_d_obs_all[pct]
        print(f"  {str(pct)+'%':8s}  {noise_std_n:12.4e}  {rmse_n:14.4e}  {res_n:12.4e}")
    print("=" * 65)

    # Save an aggregated NPZ with all noise-study results
    save_npz(
        "noise_study_all",
        NOISE_PCT=np.array(NOISE_PCT, dtype=int),
        sigma_d_base=np.array([sigma_d_base], dtype=float),
        sigma_d_loss=np.array([sigma_d_loss], dtype=float),
        vp_5=ns_results[5]["vp"],  rmse_5=np.array([ns_results[5]["rmse"]], dtype=float),  res_5=np.array([ns_results[5]["res"]], dtype=float),
        vp_10=ns_results[10]["vp"], rmse_10=np.array([ns_results[10]["rmse"]], dtype=float), res_10=np.array([ns_results[10]["res"]], dtype=float),
        vp_20=ns_results[20]["vp"], rmse_20=np.array([ns_results[20]["rmse"]], dtype=float), res_20=np.array([ns_results[20]["res"]], dtype=float),
    )

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)