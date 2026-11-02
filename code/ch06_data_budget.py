"""Chapter 6: DNS training-set cost, effective samples and a lab budget.

Standard library and NumPy only. Reads only the published files
data/dns/params.json, stats.csv, spectrum.csv, corr.csv and rho_lags.csv;
writes no files. JHTDB numbers are the isotropic1024coarse README time
averages over 0 <= t_f <= 10.056, used as a cited comparison. GPU numbers are
single-run measurements on one NVIDIA RTX 3050 Laptop GPU (4096 MiB),
PyTorch 2.11, CUDA 12.8.
"""
import csv
import json
import math
from pathlib import Path
import numpy as np

DNS = Path(__file__).resolve().parents[1] / "data" / "dns"
J_NU, J_EPS, J_K, J_L11 = 0.000185, 0.103, 0.705, 1.364
J_TL, J_N = 1.99, 1024
J_DT, J_STORE, J_FRAMES = 0.0002, 0.002, 5028
GPU = [("32^3 b8 w32", 32, 8, 658, 79), ("64^3 b1 w32", 64, 1, 656, 79),
       ("64^3 b2 w32", 64, 2, 1281, 143), ("64^3 b4 w32", 64, 4, 2536, 380),
       ("64^3 b2 w48", 64, 2, 1924, 329)]
BLOCK = 3 * 4 * 64**3
TRAIN, TEST = 74, 20


def ar1_factor(n, r):
    """1 + 2 sum_{m=1}^{n-1} (1 - m/n) r^m in closed form (Prop. 6.3)."""
    s = r / (1 - r) - r * (1 - r**n) / (n * (1 - r) ** 2)
    return 1 + 2 * s


def n_eff(n, rho, stride=1):
    """n^2 / sum(rho_ij) for n equally spaced values (Prop. 6.2)."""
    lags = np.arange(1, n)
    return n / (1 + 2 * ((1 - lags / n) * rho[lags * stride]).sum())


def to_first_zero(rho):
    """Set rho to zero from its first non-positive value onward."""
    out = rho.copy()
    out[np.argmax(rho <= 0):] = 0
    return out


def ring_corr(d, box, ell):
    """Periodized exponential correlation on a ring of length box."""
    num = np.exp(-d / ell) + np.exp(-(box - d) / ell)
    return num / (1 + math.exp(-box / ell))


def n_eff_ring(p, box, ell):
    """Exact n^2 / sum(rho) for p equally spaced points on a ring."""
    d = np.abs(np.subtract.outer(np.arange(p), np.arange(p))) * box / p
    return p * p / ring_corr(d, box, ell).sum()


def grid_audit(name, nu, eps, k, l11, n, f, box):
    """(17.14) against the grid used, with the factors of Prop. 6.1."""
    re_l, eta, large = k**2 / (eps * nu), (nu**3 / eps) ** 0.25, k**1.5 / eps
    n3 = 4.4 * re_l**2.25
    keta = f * n * eta
    fac = (1.5 / keta, f / 0.5, 8 / (box / l11), 0.43 / (l11 / large))
    print(f"{name}: Re_L={re_l:.0f} eta={eta:.5f} L={large:.3f} "
          f"L/eta={large / eta:.0f} (17.14) N={n3 ** (1 / 3):.0f} used N={n} "
          f"kmax*eta={keta:.3f}")
    print(f"  ratio {n3 ** (1 / 3) / n:.3f} = resolution {fac[0]:.3f} x "
          f"fraction {fac[1]:.3f} x box {fac[2]:.3f} x length {fac[3]:.3f} "
          f"= {np.prod(fac):.3f}; box/L11={box / l11:.3f}")
    return re_l, large, n3, eta


par = json.loads((DNS / "params.json").read_text())
rows = list(csv.DictReader(open(DNS / "stats.csv")))
st = {c: np.array([float(r[c]) for r in rows]) for c in rows[0]}
win = st["t"] >= par["spinup"] - 1e-9
mean = {c: st[c][win].mean() for c in st}
NU, DT, N = par["nu"], par["dt"], par["n"]
FRAC = par["kmax"] / N
STEPS = int(st["step"][-1])

print("Part 1. Grid and cost")
grid_audit("JHTDB", J_NU, J_EPS, J_K, J_L11, J_N, math.sqrt(2) / 3,
           2 * math.pi)
re_l, large, n3, eta = grid_audit("course", NU, mean["eps"], mean["k"],
                                  mean["L"], N, FRAC, 2 * math.pi)
j_steps = round(J_FRAMES * J_STORE / J_DT)
j_modes = J_N**3 * j_steps
print(f"JHTDB stored segment: {j_steps} DNS steps, N^3 M={j_modes:.3e}, "
      f"T_G={1e3 * j_modes / 8.64e13:.0f} days at 1 gigaflop")
modes = N**3 * STEPS
hours = STEPS * mean["ms_per_step"] / 3.6e6
print(f"course run: t={st['t'][-1]:.0f}, {STEPS} steps, dt={DT}, "
      f"N^3 M={modes:.3e}, T_G={1e3 * modes / 8.64e13:.1f} days at 1 "
      f"gigaflop; measured {mean['ms_per_step']:.1f} ms/step = {hours:.2f} h")
print(f"  k^1/2 dt/dx={math.sqrt(mean['k']) * DT * N / (2 * math.pi):.4f}; "
      f"modes/s={N**3 / mean['ms_per_step'] * 1e3:.3e}")
m_model = 120 / math.pi * large / eta
dt_model = 4 * mean["k"] / mean["eps"] / m_model

print("Part 2. Storage")
j_frame = 4 * 4 * J_N**3
print(f"JHTDB u,v,w,p float32: {j_frame} bytes per frame; all frames "
      f"{j_frame * J_FRAMES:.4e} bytes")
print(f"64^3 float32 velocity block: {BLOCK} bytes ({BLOCK / 2**20:.0f} MiB)")
print(f"course: 100 filtered {100 * BLOCK} bytes, 800 crops {800 * BLOCK} "
      f"bytes, one 256^3 velocity {3 * 4 * N**3} bytes")

print("Part 3. Measured correlation of the velocity field")
lag, rho = np.loadtxt(DNS / "rho_lags.csv", delimiter=",", skiprows=1,
                      usecols=(0, 1)).T
cc = np.loadtxt(DNS / "corr.csv", delimiter=",")
fine = np.unique(cc[:, 1])
rf = np.array([cc[cc[:, 1] == x, 2].mean() for x in fine])
z = int(np.argmax(rho <= 0))
zero = lag[z - 1] + rho[z - 1] / (rho[z - 1] - rho[z])
tail = 0.5 * rho[z - 1] * (zero - lag[z - 1])
t_int = (np.trapezoid(np.r_[1, rf], np.r_[0, fine])
         + np.trapezoid(rho[1:z], lag[1:z]) + tail)
t_whole = np.trapezoid(rho[:z], lag[:z]) + tail
k_eps = mean["k"] / mean["eps"]
print(f"rho(1.0): corr.csv {rf[fine == 1.0][0]:.3f}, rho_lags.csv "
      f"{rho[1]:.3f}; rho(0.2)={rf[0]:.3f}")
print("rho at lags 1..8: " + " ".join(f"{x:.3f}" for x in rho[1:9]))
print(f"first zero at lag {zero:.2f}; min rho to lag 30 "
      f"{rho[:31].min():.3f}; T_int={t_int:.3f} (whole lags {t_whole:.3f})")
print(f"column TL (L/u') {mean['TL']:.3f}; k/eps {k_eps:.3f}; "
      f"T_int/TL={t_int / mean['TL']:.3f}; c=T_int/(k/eps)={t_int / k_eps:.3f}"
      f"; -1/ln rho(1)={-1 / math.log(rho[1]):.3f}")
direct = 1 + 2 * sum((1 - j / 1000) * 0.99**j for j in range(1, 1000))
assert abs(direct - ar1_factor(1000, 0.99)) < 1e-9
print(f"check, n=1000 rho=0.99: n_eff={1000 / direct:.8f}")
cut = to_first_zero(rho)
exp_r = math.exp(-1 / t_int)
for name, n in (("all 100", 100), ("train", TRAIN), ("test", TEST)):
    print(f"  {name:7s} n={n:3d}: n_eff={n_eff(n, cut):.2f} (rho to first "
          f"zero), {n_eff(n, rho):.2f} (all lags), "
          f"{n / ar1_factor(n, exp_r):.2f} (exponential)")
for s in (2, 3, 5, 10):
    n = 100 // s
    print(f"  every {s:2d} snapshots: n={n:3d} n_eff={n_eff(n, cut, s):.2f}")
print(f"record T={lag[-1]:.0f}: T/(2 T_int)={lag[-1] / (2 * t_int):.2f}")
print(f"steps per independent sample {2 * t_int / DT:.0f}, GPU "
      f"{2 * t_int / DT * mean['ms_per_step'] / 1e3:.0f} s; mode-steps "
      f"{2 * t_int / DT * N**3:.3e}, model 84.5 c Re_L^3="
      f"{84.5 * t_int / k_eps * re_l**3:.3e}")
print(f"  grid ratio (17.14)/N^3={n3 / N**3:.2f}; step ratio dt/dt_model="
      f"{DT / dt_model:.2f}")
ratio = t_int / mean["TL"]
j_t = J_TL * ratio
j_rec = J_FRAMES * J_STORE
br = 1 - j_t / j_rec * (1 - math.exp(-j_rec / j_t))
print(f"JHTDB if T_int/T_L were {ratio:.3f}: T_int={j_t:.2f}, "
      f"T/(2T_int)={j_rec / (2 * j_t):.2f}, "
      f"exact {j_rec / (2 * j_t) / br:.2f}")

print("Part 3b. Energy of one shell over the 100 snapshots")
spec = np.loadtxt(DNS / "spectrum.csv", delimiter=",")[:, 2:]
print("  shell  std/mean  T_int   n_eff(100)  n_eff(test)  s.e.(test)")
worst = 0.0
for s in range(1, 32):
    x = spec[:, s] - spec[:, s].mean()
    r = np.correlate(x, x, "full")[99:]
    r = to_first_zero(r / r[0])
    cv = spec[:, s].std(ddof=1) / spec[:, s].mean()
    nt = n_eff(TEST, r)
    worst = max(worst, cv / math.sqrt(nt))
    if s in (1, 2, 3, 5, 10, 16, 21, 31):
        print(f"  {s:5d} {cv:9.3f} {0.5 + r[1:].sum():7.2f} "
              f"{n_eff(100, r):9.1f} {nt:12.1f} {cv / math.sqrt(nt):11.3f}")
print(f"  largest s.e.(test), shells 1..31: {worst:.3f}; ln 1.25 = "
      f"{math.log(1.25):.3f}")
x = spec.sum(1) - spec.sum(1).mean()
r = np.correlate(x, x, "full")[99:]
r = r / r[0]
print("  k(t): std/mean "
      f"{x.std(ddof=1) / spec.sum(1).mean():.3f}, correlation at lags 1..8 "
      + " ".join(f"{v:.2f}" for v in r[1:9]))

print("Part 4. Separable periodic model for space, ell = L_11")
ell = mean["L"]
for p in (2, 4):
    ex = n_eff_ring(p, 2 * math.pi, ell)
    print(f"{p} positions per axis: n_eff per axis={ex:.3f}, cubed={ex**3:.2f}"
          f" of {p**3}")
print(f"box/L_11={2 * math.pi / ell:.2f}; crop side/L_11="
      f"{math.pi / 2 / ell:.2f}")

print("Part 5. The worked study's data budget")
for name, n in (("train", TRAIN), ("test", TEST)):
    ne = n_eff(n, cut)
    print(f"{name}: {n} fields, {n * BLOCK} bytes, n_eff={ne:.2f}, "
          f"std error={1 / math.sqrt(ne):.3f} of the snapshot std")
print(f"naive 1/sqrt({TEST})={1 / math.sqrt(TEST):.3f}; shown "
      f"{30000 * 2 / TRAIN:.0f} times each in 30000 steps at batch 2")

print("Part 6. GPU-hours from the measured ms/step")
for name, n, b, mib, ms in GPU:
    print(f"{name}: {mib:4d} MiB ({mib / 4096:.0%}), {ms / b:6.1f} ms/sample,"
          f" {b * n**3 / ms * 1e-3:.2f} Mvox/s, 30000 steps="
          f"{30000 * ms / 3.6e6:.2f} h")
print(f"generation bound, 32 samples, 50 Euler steps, batch 2: "
      f"{16 * 50 * 0.143:.0f} s")

print("Part 7. Memorization: 1 - rho at the lags that matter (measured rho)")
for g, r_g in ((0.2, rf[0]), (1.0, rho[1]), (6.0, rho[6]), (7.0, rho[7])):
    print(f"lag {g:4.1f}: rho={r_g:.3f}, 1-rho={1 - r_g:.3f}")
print(f"exponential rule T_int ln 20 = {t_int * math.log(20):.2f}")
