"""
delay_quantized_predictor.py
----------------------------
Xdot = A X + B U(t-D)  avec MESURES QUANTIFIEES (etat + historique de commande)
et predictor feedback commute (partie V du cours).

Deux jeux de constantes sont simules :
  (a) CERTIFIE  : Omega et T donnes par (20.x), N = 19 bits ;
  (b) PRATIQUE  : Omega = 0.7, T = 0.8 s, N = 8 bits.
"""
import os, pathlib
_OUT = pathlib.Path(__file__).resolve().parent.parent / "figs"
_OUT.mkdir(parents=True, exist_ok=True)
def _p(name):
    """Chemin de sortie : toutes les figures et tables vont dans ../figs/."""
    return str(_OUT / name)

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov
import matplotlib.pyplot as plt

# --------------------------------------------------------------- systeme
A = np.array([[0.0, 1.0], [2.0, 0.0]])
B = np.array([[0.0], [1.0]])
K = np.array([[-4.25, -3.0]])                 # poles de A+BK en -1.5 (double)
D = 0.2
Acl = A + B @ K

nA, nAcl, nB, nK = 2.0, np.linalg.norm(Acl, 2), 1.0, np.linalg.norm(K, 2)
P = solve_continuous_lyapunov(Acl.T, -np.eye(2))
ev = np.linalg.eigvalsh(P)
Msig, sig = np.sqrt(ev.max() / ev.min()), 1.0 / (2 * ev.max())

M1 = 1 + nK * np.exp(nA * D) * max(1.0, D * nB)
M2 = 1.0 / (1 + nK * np.exp(nAcl * D) * max(1.0, D * nB))
M3 = nK * np.exp(nA * D) * (1 + nB * D)

def certified(delta_w=1e-3):
    gB = Msig * nB / (sig - delta_w)
    a  = 1 + gB
    lam = 3 * a - 1                            # proposition 22.1
    phi = a / (1 + lam)
    M0  = max(Msig, (1 + lam) * phi) / (1 - phi) - 1
    return gB, lam, phi, M0, delta_w

gB, lam, phi, M0, dlt = certified()
theta = M2 / (M1 * (1 + M0))
bound = M2 / ((1 + M0) * max((1 + lam) * (1 + M0) * M3, 2 * M1))
Nmin  = int(np.ceil(np.log2(1 / bound)))
print(f"M1={M1:.3f} M2={M2:.4f} M3={M3:.3f} Msig={Msig:.3f} sig={sig:.4f}")
print(f"gB={gB:.3f} lambda*={lam:.2f} phi={phi:.4f} M0={M0:.3f} theta={theta:.2e}")
print(f"Delta/M admissible < {bound:.3e}  ->  N >= {Nmin} bits")

# ------------------------------------------------------------ quantifieur
def make_quant(M, ratio):
    Delta = M / ratio
    h = 2 * Delta
    def q(z):
        z = np.asarray(z, dtype=float)
        return np.clip(h * np.floor(z / h + 0.5), -M, M)
    return q, Delta

# ------------------------------------------------------------- simulation
def simulate(ratio, Omega, T, tag, theta_ev, M=4.0, t_end=12.0, dt=1e-3, mu0=0.05, tau=0.05):
    q, Delta = make_quant(M, ratio)
    nb = int(round(D / dt))
    ys = np.linspace(0.0, D, nb + 1)
    expAy = np.array([expm(A * (D - y)) @ B for y in ys])[:, :, 0]   # (nb+1,2)

    X = np.array([2.0, -1.0])
    buf = np.zeros(nb + 1)                    # buf[k] = U(t - D + k*dt)
    mu, phase, tstar, tlast = mu0, 1, None, 0.0
    ts, nrm, mus, Us = [], [], [], []

    for n in range(int(t_end / dt)):
        t = n * dt
        if phase == 1:
            if t - tlast >= tau:
                mu *= np.exp(2 * nA * tau); tlast = t
            qX, qu = mu * q(X / mu), mu * q(buf / mu)
            if np.linalg.norm(qX) + np.max(np.abs(qu)) <= (theta_ev * M - Delta) * mu:
                phase, tstar, tlast = 2, t, t
            U = 0.0
        else:
            if t - tlast >= T:
                mu *= Omega; tlast = t
            qX, qu = mu * q(X / mu), mu * q(buf / mu)
            pred = expm(A * D) @ qX + np.trapezoid(expAy * qu[:, None], ys, axis=0)
            pred = np.asarray(pred).reshape(2)
            U = float((K @ pred).item() if hasattr(K @ pred, "item") else K @ pred)

        # integration RK4 de Xdot = A X + B U(t-D)   (entree retardee = buf[0])
        u_del = buf[0]
        f = lambda x: A @ x + B[:, 0] * u_del
        k1 = f(X); k2 = f(X + dt/2*k1); k3 = f(X + dt/2*k2); k4 = f(X + dt*k3)
        X = X + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        buf = np.roll(buf, -1); buf[-1] = U    # decalage du tampon circulaire

        ts.append(t); nrm.append(np.linalg.norm(X) + np.max(np.abs(buf)))
        mus.append(mu); Us.append(U)
    print(f"[{tag}] N={np.log2(ratio):.0f} bits  t1*={tstar}  ||(X,u)(T)||={nrm[-1]:.3e}")
    return map(np.array, (ts, nrm, mus, Us)), tstar

# (a) reglage certifie
Om_a = (1 + lam) * (1 + M0)**2 * M3 * (1 / 2**Nmin) / M2
T_a  = np.log((1 + M0) / Om_a) / dlt
print(f"certifie : Omega={Om_a:.4f}  T={T_a:.1f}s  taux={np.log(1/Om_a)/T_a:.2e}")

# (b) reglage pratique
(res_b, ts_b) = simulate(2**8, 0.70, 0.8, "pratique", 0.5)
(res_c, ts_c) = simulate(2**5, 0.70, 0.8, "5 bits", 0.5)
tb, nb_, mb, Ub = res_b
tc, nc_, mc, Uc = res_c

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].semilogy(tb, nb_, label=r"$\|(X,u)\|$"); ax[0].semilogy(tb, mb, "--", label=r"$\mu$")
ax[0].axvline(ts_b, color="k", ls=":", lw=.8); ax[0].legend(); ax[0].grid(alpha=.3)
ax[0].set_xlabel("t")
ax[0].semilogy(tc, nc_, color="C2", lw=.8, label="5 bits")
ax[1].plot(tb, Ub, color="C3"); ax[1].set_xlabel("t"); ax[1].set_ylabel("U(t)")
ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(_p("delay_quantized_predictor.png"), dpi=130)

np.savetxt(_p("d_ret_n.txt"), np.c_[tb[::20], np.maximum(nb_[::20], 1e-12)])
np.savetxt(_p("d_ret_mu.txt"), np.c_[tb[::20], mb[::20]])
np.savetxt(_p("d_ret_U.txt"), np.c_[tb[::20], Ub[::20]])
np.savetxt(_p("d_ret_n5.txt"), np.c_[tc[::20], np.maximum(nc_[::20],1e-12)])
print("figure ecrite : delay_quantized_predictor.png")
