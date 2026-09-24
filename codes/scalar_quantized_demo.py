"""
scalar_quantized_demo.py
------------------------
Systeme scalaire instable  xdot = a x + u,  a > 0,  commande u = -k q_mu(x).

Illustre les trois regimes du chapitre 3 :
  (a) mu trop petit  -> saturation puis divergence ;
  (b) mu trop grand  -> pas de divergence mais bande ultime large ;
  (c) mu bien choisi -> bande ultime etroite, mais toujours pas de convergence
      vers zero : aucune loi a mu fige ne peut faire mieux.

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

a, k = 1.0, 3.0                 # systeme et gain nominal
M, Delta = 5.0, 0.1             # portee et resolution du quantifieur
h = 2 * Delta

def q(z):                       # quantifieur uniforme sature (mid-tread)
    return np.clip(h * np.floor(z / h + 0.5), -M, M)

def q_mu(z, mu):
    return mu * q(z / mu)

def simulate(mu, x0, t_end=12.0, dt=1e-3):
    n = int(t_end / dt)
    x, ts, xs = x0, np.arange(n) * dt, np.empty(n)
    for i in range(n):
        xs[i] = x
        f = lambda y: a * y - k * q_mu(y, mu)
        k1 = f(x); k2 = f(x + dt/2*k1); k3 = f(x + dt/2*k2); k4 = f(x + dt*k3)
        x = x + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        if abs(x) > 1e4:                       # divergence : on arrete
            xs[i+1:] = np.nan
            break
    return ts, xs

def rayon_ultime(mu):
    return k / (k - a) * Delta * mu           # eq. (3.3) du cours

cas = [(0.4, 8.0, "mu=0.4 : hors de portee, divergence"),
       (4.0, 8.0, "mu=4 : borne, bande ultime large"),
       (1.0, 8.0, "mu=1 : bande ultime etroite")]

fig, ax = plt.subplots(figsize=(8, 4.2))
for mu, x0, lab in cas:
    ts, xs = simulate(mu, x0)
    ax.plot(ts, np.abs(xs), label=lab)
    ax.axhline(rayon_ultime(mu), ls=":", lw=0.8, color="0.5")
ax.set_yscale("log"); ax.set_xlabel("t"); ax.set_ylabel(r"$|x(t)|$")
ax.set_ylim(1e-3, 1e3); ax.grid(alpha=.3); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(_p("scalar_quantized_demo.png"), dpi=130)

for mu, _, _ in cas:
    ts, xs = simulate(mu, 8.0)
    fini = np.isfinite(xs)
    print(f"mu={mu:>4}: portee M*mu={M*mu:6.2f}  rayon ultime={rayon_ultime(mu):7.4f}"
          f"  |x| final={abs(xs[fini][-1]):.4f}" if fini.any() else "diverge")
