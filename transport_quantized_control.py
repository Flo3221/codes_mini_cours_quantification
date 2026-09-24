"""
transport_quantized_control.py
------------------------------
EDP de transport en strict-feedback

    u_t(x,t) = u_x(x,t) + g u(0,t),    x in [0,D],   u(D,t) = U(t)

stabilisee par backstepping avec MESURES QUANTIFIEES et strategie hybride
de zoom (chapitres 14-16 du cours).

Noyau explicite (g constant, gbar = 0) :  k(x,y) = -g exp(g (x-y)).
Schema numerique : upwind (decentre amont), CFL = dt/dx <= 1.
"""
import os, pathlib
_OUT = pathlib.Path(__file__).resolve().parent.parent / "figs"
_OUT.mkdir(parents=True, exist_ok=True)
def _p(name):
    """Chemin de sortie : toutes les figures et tables vont dans ../figs/."""
    return str(_OUT / name)

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Systeme, noyau, constantes de backstepping
# ----------------------------------------------------------------------
D, g = 1.0, 1.25

def k_kernel(x, y):
    return -g * np.exp(g * (x - y))

M1 = 1.0 + D * g * np.exp(g * D)      # ||u|| -> ||w||
M2 = 1.0 / (1.0 + D * g)              # ||w|| -> ||u||
M3 = D * g * np.exp(g * D)            # gain du controleur nominal
M1bar = M3 / (D * M2)                 # taux de croissance en boucle ouverte

# ----------------------------------------------------------------------
# 2. Quantificateur (M = portee, Delta = resolution)
# ----------------------------------------------------------------------
M, ratio = 2.0, 256.0                 # 8 bits
Delta = M / ratio
h = 2.0 * Delta                       # pas du quantificateur uniforme

def q_scalar(z):
    if z > M:   return M
    if z < -M:  return -M
    return h * np.floor(z / h + 0.5)

q_vec = np.vectorize(q_scalar)

def q_mu(z, mu):
    return mu * q_vec(np.asarray(z) / mu)

# ----------------------------------------------------------------------
# 3. Constantes de reglage (equations (14.3)-(14.7) du cours)
# ----------------------------------------------------------------------
lam, nu = 4.13, 0.16
M0   = (1 + lam) ** nu / (1 - (1 + lam) ** (nu - 1)) - 1.0
Omega = (1 + lam) * (1 + M0) ** 2 * M3 * Delta / (M2 * M)
T     = D * np.log((1 + M0) / Omega) / (nu * np.log(1 + lam))
theta = M2 / (M1 * (1 + M0))
delta = nu * np.log(1 + lam) / D

cond = Delta / M < M2 / ((1 + M0) * max((1 + lam) * (1 + M0) * M3, 2 * M1))
print(f"M1={M1:.3f} M2={M2:.3f} M3={M3:.3f}  M0={M0:.3f}")
print(f"Omega={Omega:.4f}  T={T:.3f}  theta={theta:.4f}  taux={np.log(1/Omega)/T:.4f}")
print("condition suffisante (15.1) :", "VERIFIEE" if cond else "VIOLEE (simulation quand meme)")

# phase 1
mu0, tau = 0.05, 0.2
c = np.exp(2.0 * M1bar * tau)         # ln c / tau > M1bar

# ----------------------------------------------------------------------
# 4. Maillage et donnee initiale
# ----------------------------------------------------------------------
N  = 400
dx = D / N
dt = 0.5 * dx                          # CFL = 0.5
t_end = 40.0
n_steps = int(t_end / dt)
xs = np.linspace(0.0, D, N + 1)

u = 3.0 * np.ones_like(xs) + 2.0 * np.sin(2 * np.pi * xs)   # profil initial
kD = k_kernel(D, xs)                    # k(D,x) sur le maillage

mu, phase, t_star, t_last = mu0, 1, None, 0.0
ts, norms, mus, Us = [], [], [], []
snapshots, snap_times = [], []

for n in range(n_steps):
    t = n * dt

    # ---- logique hybride ------------------------------------------------
    if phase == 1:
        if t - t_last >= tau:
            mu *= c
            t_last = t
        qu = q_mu(u, mu)
        if np.max(np.abs(qu)) <= (theta * M - Delta) * mu:
            phase, t_star, t_last = 2, t, t
        U = 0.0
    else:
        if t - t_last >= T:
            mu *= Omega
            t_last = t
        qu = q_mu(u, mu)
        U = np.trapezoid(kD * qu, xs)   # controleur backstepping quantifie

    # ---- schema upwind pour u_t = u_x + g u(0,t) ------------------------
    u_new = u.copy()
    u_new[:-1] = u[:-1] + (dt / dx) * (u[1:] - u[:-1]) + dt * g * u[0]
    u_new[-1] = U                       # condition au bord x = D
    u = u_new

    ts.append(t); norms.append(np.max(np.abs(u))); mus.append(mu); Us.append(U)
    if n % max(1, n_steps // 300) == 0:
        snapshots.append(u.copy()); snap_times.append(t)

ts = np.array(ts); norms = np.array(norms); mus = np.array(mus); Us = np.array(Us)
print(f"t1* = {t_star:.3f} s     ||u(T_end)||_inf = {norms[-1]:.3e}")

# ----------------------------------------------------------------------
# 5. Figures
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(11, 7))

ax1 = fig.add_subplot(2, 2, 1)
ax1.semilogy(ts, norms, label=r"$\|u(t)\|_\infty$")
ax1.semilogy(ts, mus, "--", label=r"$\mu(t)$")
ax1.axvline(t_star, color="k", ls=":", lw=0.8)
ax1.set_xlabel("t"); ax1.legend(); ax1.grid(alpha=.3)

ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(ts, Us, color="C3"); ax2.set_xlabel("t")
ax2.set_ylabel("U(t)"); ax2.grid(alpha=.3)

ax3 = fig.add_subplot(2, 1, 2)
Z = np.array(snapshots).T
im = ax3.pcolormesh(np.array(snap_times), xs, Z, shading="auto", cmap="viridis")
fig.colorbar(im, ax=ax3, label="u(x,t)")
ax3.set_xlabel("t"); ax3.set_ylabel("x")

plt.tight_layout()
plt.savefig(_p("transport_quantized_control.png"), dpi=130)
print("figure ecrite : transport_quantized_control.png")
