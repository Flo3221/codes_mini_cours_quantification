"""Figure du compromis marge/vitesse pour l'exemple numerique (section 2.4).
Reprend exactement les constantes de quantized_state_feedback.py."""
import os, pathlib
_OUT = pathlib.Path(__file__).resolve().parent.parent / "figs"
_OUT.mkdir(parents=True, exist_ok=True)
def _p(name):
    """Chemin de sortie : toutes les figures et tables vont dans ../figs/."""
    return str(_OUT / name)

import numpy as np, scipy.linalg as la
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = np.array([[0.,1.],[2.,0.]]); B = np.array([[0.],[1.]])
K = np.array([[-5.,-4.]]); Q = np.eye(2)
Ac = A + B@K
P  = la.solve_continuous_lyapunov(Ac.T, -Q)
ev = np.linalg.eigvalsh(P); lmin_P, lmax_P = ev[0], ev[-1]
kappa = np.sqrt(lmax_P/lmin_P); lmin_Q = 1.0
gamma = 2*np.linalg.norm(P@B@K,2)/lmin_Q
M, Delta = 6.0, 0.125
eps_max = M/(Delta*kappa*gamma) - 1.0
print(f"kappa={kappa:.4f} gamma={gamma:.4f} kappa*gamma={kappa*gamma:.4f} eps_max={eps_max:.3f}")

eps = np.linspace(0.05, eps_max*0.999, 600)
Omega = kappa*gamma*Delta*(1+eps)/M
alpha = eps/(1+eps)*lmin_Q/lmax_P
T_exp = 2/alpha*np.log(1/Omega)
T_BL  = (lmin_P*M**2 - lmax_P*gamma**2*Delta**2*(1+eps)**2) / \
        (lmin_Q*gamma**2*Delta**2*eps*(1+eps))
T = np.minimum(T_exp, T_BL)
rate = np.log(1/Omega)/T

fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.1))
ax[0].plot(eps, Omega, lw=1.8, color="#143c78"); ax[0].set_ylabel(r"$\Omega$")
ax[0].set_title(r"facteur de contraction")
ax[1].semilogy(eps, T_exp, lw=1.6, color="#143c78", label=r"$T_{\exp}$ (exponentiel)")
ax[1].semilogy(eps, T_BL, lw=1.6, ls="--", color="#c86e14", label=r"$T_{\rm BL}$ (éq. orig.)")
ax[1].semilogy(eps, T, lw=2.4, color="#146e3c", alpha=.55, label=r"$T$ retenu")
ax[1].set_ylabel(r"$T$  (s)"); ax[1].set_title("temps de séjour"); ax[1].legend(fontsize=7)
ax[2].plot(eps, rate, lw=1.8, color="#822a28"); ax[2].set_ylabel(r"$\ln(1/\Omega)/T$  (s$^{-1}$)")
ax[2].set_title("taux garanti")
for a in ax:
    a.set_xlabel(r"marge $\varepsilon$"); a.grid(alpha=.25)
    a.axvline(3.0, color="k", ls=":", lw=1)
    a.axvline(eps_max, color="#822a28", ls=":", lw=1)
ax[0].annotate(r"$\varepsilon=3$", xy=(3.0, Omega[np.argmin(abs(eps-3))]),
               xytext=(1.1, 0.85), fontsize=8,
               arrowprops=dict(arrowstyle="->", lw=.8))
ax[2].annotate(r"$\varepsilon_{\max}$", xy=(eps_max, rate[-1]),
               xytext=(eps_max-1.6, rate.max()*0.55), fontsize=8, color="#822a28",
               arrowprops=dict(arrowstyle="->", lw=.8, color="#822a28"))
fig.tight_layout()
fig.savefig(_p("tradeoff_eps.png"), dpi=150)
print("figure ecrite : figs/tradeoff_eps.png")

for e in [0.5,1,2,3,4,4.4]:
    O = kappa*gamma*Delta*(1+e)/M
    al = e/(1+e)*lmin_Q/lmax_P
    Te = 2/al*np.log(1/O)
    Tb = (lmin_P*M**2 - lmax_P*gamma**2*Delta**2*(1+e)**2)/(lmin_Q*gamma**2*Delta**2*e*(1+e))
    Tm = min(Te,Tb)
    print(f"eps={e:>4}  Omega={O:.3f}  alpha={al:.3f}  T_exp={Te:7.2f}  T_BL={Tb:8.2f}  T={Tm:7.2f}  taux={np.log(1/O)/Tm:.3f}")
