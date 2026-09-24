"""
transport_backstepping_demo.py
------------------------------
EDP de transport en strict-feedback, SANS quantification :

    u_t = u_x + g u(0,t),   u(D,t) = U(t),   D = 1,  g = 1.25.

Compare la boucle ouverte (U = 0, instable car gD > 1) au controleur
nominal de backstepping U = int_0^D k(D,x) u(x,t) dx, k(x,y) = -g e^{g(x-y)},
qui annule l'etat en temps fini D.

Depot : https://github.com/Flo3221
"""
import os, pathlib
_OUT = pathlib.Path(__file__).resolve().parent.parent / "figs"
_OUT.mkdir(parents=True, exist_ok=True)
def _p(name):
    """Chemin de sortie : toutes les figures et tables vont dans ../figs/."""
    return str(_OUT / name)

import numpy as np
import matplotlib.pyplot as plt

D, g = 1.0, 1.25
N, dt = 400, 0.5 / 400
xs = np.linspace(0.0, D, N + 1)
kD = -g * np.exp(g * (D - xs))          # noyau evalue en x = D
t_end = 8.0
n_steps = int(t_end / dt)

def run(closed_loop):
    u = 3.0 * np.ones_like(xs) + 2.0 * np.sin(2 * np.pi * xs)
    ts, nrm, Us = [], [], []
    for n in range(n_steps):
        U = np.trapezoid(kD * u, xs) if closed_loop else 0.0
        v = u.copy()
        v[:-1] = u[:-1] + (dt / (D / N)) * (u[1:] - u[:-1]) + dt * g * u[0]
        v[-1] = U
        u = v
        ts.append(n * dt); nrm.append(np.max(np.abs(u))); Us.append(U)
    return np.array(ts), np.array(nrm), np.array(Us)

t, n_ol, _ = run(False)
_, n_cl, U_cl = run(True)
print(f"boucle ouverte : ||u|| passe de {n_ol[0]:.2f} a {n_ol[-1]:.3e}")
print(f"boucle fermee  : ||u|| passe de {n_cl[0]:.2f} a {n_cl[-1]:.3e}")
print(f"purge attendue en t = D = {D}")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].semilogy(t, n_ol, label="boucle ouverte")
ax[0].semilogy(t, n_cl, label="backstepping nominal")
ax[0].axvline(D, ls=":", color="k", lw=.8)
ax[0].set_xlabel("t"); ax[0].set_ylabel(r"$\|u(\cdot,t)\|_\infty$")
ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(t, U_cl, color="C3"); ax[1].set_xlabel("t"); ax[1].set_ylabel("U(t)")
ax[1].set_xlim(0, 3); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(_p("transport_backstepping_demo.png"), dpi=130)

step = max(1, len(t) // 220)
for name, arr in [("bs_ol", n_ol), ("bs_cl", n_cl), ("bs_U", U_cl)]:
    d = np.c_[t[::step], arr[::step]]
    with open(_p(name + ".txt"), "w") as f:
        f.write(" ".join(f"({a:.3f},{max(b,1e-14) if name!='bs_U' else b:.5g})" for a, b in d))
