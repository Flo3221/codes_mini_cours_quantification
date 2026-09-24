"""
quantized_state_feedback.py
---------------------------
Retour d'etat quantifie avec strategie hybride de zoom (dimension finie).
Reproduit les figures : etats x(t), zoom mu(t), commande u(t).

Theorie : chapitres 5-6 du cours "Quantification en commande".
Auteur : F. Koudohode.  Licence : usage pedagogique libre.
"""
import os, pathlib
_OUT = pathlib.Path(__file__).resolve().parent.parent / "figs"
_OUT.mkdir(parents=True, exist_ok=True)
def _p(name):
    """Chemin de sortie : toutes les figures et tables vont dans ../figs/."""
    return str(_OUT / name)

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Systeme et gain nominal
# ----------------------------------------------------------------------
A = np.array([[0.0, 1.0],
              [2.0, 0.0]])
B = np.array([[0.0], [1.0]])
K = np.array([[-5.0, -4.0]])          # A + B K a pour valeurs propres -1, -3
Q = np.eye(2)

Ac = A + B @ K
P = la.solve_continuous_lyapunov(Ac.T, -Q)   # Ac^T P + P Ac = -Q
evP = np.linalg.eigvalsh(P)
lmin_P, lmax_P = evP[0], evP[-1]
kappa_P = np.sqrt(lmax_P / lmin_P)
lmin_Q = np.min(np.linalg.eigvalsh(Q))
gamma = 2.0 * np.linalg.norm(P @ B @ K, 2) / lmin_Q

# ----------------------------------------------------------------------
# 2. Quantificateur uniforme sature + version dynamique
# ----------------------------------------------------------------------
M = 6.0            # seuil de saturation
h = 0.25           # pas
Delta = h / 2.0    # borne d'erreur (P1)

def q_scalar(z):
    """Quantificateur uniforme mid-tread sature, scalaire."""
    if z > M:
        return M
    if z < -M:
        return -M
    return h * np.floor(z / h + 0.5)

def q_vec(z):
    return np.array([q_scalar(zi) for zi in np.ravel(z)])

def q_mu(z, mu):
    """Quantificateur dynamique q_mu(z) = mu q(z/mu)."""
    return mu * q_vec(np.ravel(z) / mu)

# ----------------------------------------------------------------------
# 3. Constantes de reglage certifiees (theoreme 6.4)
# ----------------------------------------------------------------------
eps = 3.0                                   # marge : voir table du cours
assert M / Delta > max(kappa_P * gamma * (1 + eps), 2 * kappa_P), \
    "condition (6.9) violee : pas assez de niveaux"

Omega = kappa_P * gamma * Delta * (1 + eps) / M       # facteur de contraction
alpha = eps / (1 + eps) * lmin_Q / lmax_P             # taux de Lyapunov
T_dwell = 2.0 / alpha * np.log(1.0 / Omega)           # temps de sejour
theta = 1.0 / kappa_P                                 # marge de l'evenement

# phase 1
mu0, tau = 0.05, 0.10
c = np.exp(2.0 * np.linalg.norm(A, 2) * tau)          # ln c > ||A|| tau

print(f"kappa_P = {kappa_P:.4f}   gamma = {gamma:.4f}   kappa*gamma = {kappa_P*gamma:.4f}")
print(f"Omega   = {Omega:.4f}   alpha = {alpha:.4f}   T = {T_dwell:.4f} s")
print(f"taux garanti = {np.log(1/Omega)/T_dwell:.4f} s^-1")

# ----------------------------------------------------------------------
# 4. Simulation (Runge-Kutta 4, commande maintenue sur le pas)
# ----------------------------------------------------------------------
dt, t_end = 1e-3, 12.0
n_steps = int(t_end / dt)

x = np.array([4.0, -3.0])
mu = mu0
phase = 1                 # 1 = agrandir la portee, 2 = affiner
t_star = None
t_last_update = 0.0

ts, xs, mus, us, phases = [], [], [], [], []

for n in range(n_steps):
    t = n * dt

    # --- logique hybride -------------------------------------------------
    if phase == 1:
        # mise a jour par paliers de duree tau
        if t - t_last_update >= tau:
            mu *= c
            t_last_update = t
        qx = q_mu(x, mu)
        if np.linalg.norm(qx) <= (theta * M - Delta) * mu:   # evenement (6.6)
            phase, t_star, t_last_update = 2, t, t
        u = np.zeros(1)
    else:
        if t - t_last_update >= T_dwell:
            mu *= Omega
            t_last_update = t
        qx = q_mu(x, mu)
        u = K @ qx

    # --- integration RK4 a commande gelee --------------------------------
    f = lambda z: A @ z + (B @ u).ravel()
    k1 = f(x); k2 = f(x + dt/2*k1); k3 = f(x + dt/2*k2); k4 = f(x + dt*k3)
    x = x + dt/6*(k1 + 2*k2 + 2*k3 + k4)

    ts.append(t); xs.append(x.copy()); mus.append(mu)
    us.append(u[0]); phases.append(phase)

ts = np.array(ts); xs = np.array(xs); mus = np.array(mus); us = np.array(us)
print(f"instant de capture t1* = {t_star:.3f} s")

# ----------------------------------------------------------------------
# 5. Figures
# ----------------------------------------------------------------------
fig, ax = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

ax[0].plot(ts, xs[:, 0], label=r"$x_1$")
ax[0].plot(ts, xs[:, 1], label=r"$x_2$")
ax[0].axvline(t_star, color="k", ls="--", lw=0.8)
ax[0].set_ylabel("etat"); ax[0].legend(); ax[0].grid(alpha=.3)

ax[1].semilogy(ts, mus, color="C2")
ax[1].axvline(t_star, color="k", ls="--", lw=0.8)
ax[1].set_ylabel(r"$\mu(t)$ (echelle log)"); ax[1].grid(alpha=.3)

ax[2].plot(ts, us, color="C3")
ax[2].axvline(t_star, color="k", ls="--", lw=0.8)
ax[2].set_ylabel("commande $u$"); ax[2].set_xlabel("t (s)"); ax[2].grid(alpha=.3)

# verification de la borne theorique
env = M * mus / Omega
ax[0].plot(ts, env, "k:", lw=0.8, label="borne $M\\mu/\\Omega$")
ax[0].plot(ts, -env, "k:", lw=0.8)
ax[0].legend()

plt.tight_layout()
plt.savefig(_p("quantized_state_feedback.png"), dpi=140)
print("figure ecrite : quantized_state_feedback.png")
