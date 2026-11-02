"""Per-mode SNR of the path X_t = t z + (1 - t) xi on a 64^3 grid.

Course DFT convention: hat f = fftn(f) / N**3, so unit-variance white
noise has modal power 1 / N**3 at every wavenumber. Printed labels:
kappa_x is the crossover wavenumber, not a vector component.
"""
import numpy as np

N = 64
S_XI = 1.0 / N**3
K1 = np.fft.fftfreq(N, d=1.0 / N)
KX, KY, KZ = np.meshgrid(K1, K1, K1, indexing="ij")
KMAG = np.sqrt(KX**2 + KY**2 + KZ**2)
SHELL = np.rint(KMAG).astype(int)
NZ = KMAG > 0
TS = (0.1, 0.3, 0.5, 0.7, 0.9, 0.99)


def normalise(s):
    s = np.where(NZ, s, 0.0)
    return s / s.sum()  # unit variance per component (Parseval)


def pope(length=1.0, eta=1.0 / 20.0):
    """Modal power E(kappa) / (4 pi kappa^2) of essentials (14.9)-(14.10)."""
    k = np.where(NZ, KMAG, 1.0)
    f_l = (k * length / np.sqrt((k * length) ** 2 + 6.78)) ** (5 / 3 + 2)
    f_eta = np.exp(-5.2 * (((k * eta) ** 4 + 0.40**4) ** 0.25 - 0.40))
    return normalise(1.5 * k ** (-5 / 3) * f_l * f_eta / (4 * np.pi * k**2))


def snr(s, t):
    return t**2 * s / ((1 - t) ** 2 * S_XI)


def shell_mean(a):
    total = np.bincount(SHELL.ravel(), weights=a.ravel())
    return total / np.maximum(np.bincount(SHELL.ravel()), 1)


print("1. log10 SNR(kappa, t) on the axis; unit-variance data")
for lam in (5 / 3, 11 / 3):
    amp = 1.0 / np.sum(KMAG[NZ] ** (-lam))
    print(f"lambda = {lam:.4f}, A = {amp:.3e}")
    print("kappa " + "".join(f"  t={t:<5}" for t in TS))
    for k in (1, 2, 4, 8, 16, 32):
        row = [np.log10(t**2 * amp * k**-lam / ((1 - t) ** 2 * S_XI))
               for t in TS]
        print(f"{k:5d} " + "".join(f"{v:9.2f}" for v in row))
    k_cross = [(amp / S_XI) ** (1 / lam) * (t / (1 - t)) ** (2 / lam)
               for t in TS]
    print("kappa_x" + "".join(f"{v:9.2f}" for v in k_cross))
    k_cross_3 = (amp / S_XI * 1e3) ** (1 / lam)
    print(f"kappa_x at scalar SNR 1e3: {k_cross_3:.1f}")

print("2. Pope model, L = 1, eta = 1/20: signal-dominated window 1 - t*")
s_p = pope()
rel_p = shell_mean(s_p) * N**3
print("kappa  kappa*eta  N^3 S_pope  window_pope  window_k^-11/3")
for k in (2, 4, 8, 16, 24, 32):
    rel_11 = rel_p[4] * (k / 4) ** (-11 / 3)  # power law matched at 4
    wp = 1 / (1 + np.sqrt(1 / rel_p[k]))
    w11 = 1 / (1 + np.sqrt(1 / rel_11))
    print(f"{k:5d} {k / 20:10.3f} {rel_p[k]:11.3e} {wp:12.3e} {w11:14.3e}")
print(f"first shell noise-dominated at scalar SNR 1e3: "
      f"{np.argmax(rel_p[1:N // 2] * 1e3 < 1) + 1}")

print("3. Wiener (posterior-mean) denoising of one Pope sample")
rng = np.random.default_rng(2026)


def sample(s):
    w = rng.standard_normal((N, N, N))
    return np.fft.ifftn(np.sqrt(s * N**3) * np.fft.fftn(w)).real


t = 0.5
z, z_ref = sample(s_p), sample(s_p)
xi = rng.standard_normal((N, N, N))
g = snr(s_p, t) / (1 + snr(s_p, t))
x_hat = np.fft.fftn(t * z + (1 - t) * xi) / N**3
est_hat = g * x_hat / t
z_hat = np.fft.fftn(z) / N**3
ref_hat = np.fft.fftn(z_ref) / N**3
e_z = shell_mean(np.abs(z_hat) ** 2)
e_est = shell_mean(np.abs(est_hat) ** 2)
e_ref = shell_mean(np.abs(ref_hat) ** 2)
g_shell = shell_mean(s_p * g) / np.maximum(shell_mean(s_p), 1e-300)
print("shell  SNR      E_est/E_z  theory SNR/(1+SNR)")
for n in (2, 4, 8, 12, 16, 20, 24, 31):
    print(f"{n:5d} {shell_mean(snr(s_p, t))[n]:9.3e} "
          f"{e_est[n] / e_z[n]:10.4f} {g_shell[n]:10.4f}")
ns = np.arange(1, N // 2)
snr_sh = shell_mean(snr(s_p, t))[ns]
print(f"first shell with SNR < 1: {ns[np.argmax(snr_sh < 1)]}")
ratio = e_est[ns] / e_z[ns]
print(f"first shell with E_est/E_z < 1/2: {ns[np.argmax(ratio < 0.5)]}")
k2 = KMAG**2
print(f"variance ratio (TKE): "
      f"{np.sum(np.abs(est_hat)**2) / np.sum(np.abs(z_hat)**2):.4f}")
print(f"gradient-variance ratio (dissipation): "
      f"{np.sum(k2 * np.abs(est_hat)**2) / np.sum(k2 * np.abs(z_hat)**2):.4f}")
d_gen = np.sqrt(np.sum(np.log(e_est[ns] / e_ref[ns]) ** 2))
d_ref = np.sqrt(np.sum(np.log(e_z[ns] / e_ref[ns]) ** 2))
print(f"log-spectrum distance, shells 1-31: estimate {d_gen:.3f}, "
      f"reference pair {d_ref:.3f}")
modes = np.bincount(SHELL.ravel())[ns]
tol = 3 * np.sqrt(2) / np.sqrt(modes / 2)
out = np.abs(np.log(e_est[ns] / e_ref[ns])) > tol
stays = np.flip(np.cumprod(np.flip(out)))
print(f"departure shell (3-sigma, stays out to 31): {ns[np.argmax(stays)]}")
print(f"reference pair shells outside tolerance: "
      f"{np.sum(np.abs(np.log(e_z[ns] / e_ref[ns])) > tol)} of {ns.size}")

print("4. Power-law schedule, terminal B(1) held at 10.05, beta_min 0.1")
taus = np.linspace(0.0, 1.0, 200001)
for gam in (1.0, 3.0, 5.0):
    big_b = 0.1 * taus + (10.05 - 0.1) * taus ** (gam + 1)
    b_max = 0.1 + (10.05 - 0.1) * (gam + 1)
    out = []
    for k in (4, 16, 32):
        c = np.log1p(rel_p[k])
        out.append(taus[min(np.searchsorted(big_b, c), taus.size - 1)])
    fixed = np.exp(-(0.1 + 19.9 / (gam + 1)))
    print(f"gamma={gam:.0f}: beta_max={b_max:6.2f}, survival tau at "
          f"kappa 4/16/32 = {out[0]:.3f}/{out[1]:.3f}/{out[2]:.3f}; "
          f"abar(1) if beta_max=20: {fixed:.2e}")
