"""Energy spectrum of a discrete periodic velocity field (chapter 3).

Builds synthetic solenoidal isotropic fields with a prescribed model
spectrum on a periodic box [0, 2 pi)^3, computes E(kappa) by shell
summation of numpy.fft.fftn(v) / N**3 (the course convention), checks
Parseval, recovers the spectrum, fits the slope of the flattest band,
reads the three ranges, and measures window leakage and stride aliasing.
Inputs are set below. Output: printed numbers only; no files are read or
written. Requires Python 3.13 and NumPy.
"""
import numpy as np

C, BETA, C_L, C_ETA, P0 = 1.5, 5.2, 6.78, 0.40, 2.0  # essentials (14.10)
L_INT, ETA, EPS = 1.0, 0.01, 1.0  # shape parameters in box units
NU = (ETA**4 * EPS) ** (1 / 3)  # from eta = (nu^3 / eps)^(1/4)


def e_model(kap, eta=ETA):
    """Model spectrum (14.9)-(14.10) with prescribed L, eta and eps."""
    f_l = (kap * L_INT / np.sqrt((kap * L_INT) ** 2 + C_L)) ** (5 / 3 + P0)
    f_eta = np.exp(-BETA * (((kap * eta) ** 4 + C_ETA**4) ** 0.25 - C_ETA))
    return C * EPS ** (2 / 3) * kap ** (-5 / 3) * f_l * f_eta


def lattice(n):
    """Integer wavenumber vectors of an n^3 grid on a box of side 2 pi."""
    q = np.fft.fftfreq(n, 1 / n)
    return np.stack(np.meshgrid(q, q, q, indexing="ij"))


def synth(n, rng, gauss=False, eta=ETA):
    """Solenoidal field with E/(4 pi kappa^2) per mode, |n| < n/2 - 1/2."""
    kv = lattice(n)
    kk = np.sqrt((kv**2).sum(0))
    keep = (kk > 0) & (kk < n / 2 - 0.5)
    safe = np.where(keep, kk, 1.0)
    w = np.fft.fftn(rng.standard_normal((3, n, n, n)), axes=(1, 2, 3))
    w = w / n**3
    w = w - kv * (kv * w).sum(0) / safe**2  # projection P(kappa) w
    target = np.where(keep, e_model(safe, eta) / (4 * np.pi * safe**2), 0)
    if gauss:
        amp = np.sqrt(target * n**3)  # <|P w|^2> = 2 / n^3
    else:
        norm = np.sqrt((np.abs(w) ** 2).sum(0))
        amp = np.sqrt(2 * target) / np.where(norm > 0, norm, 1.0)
    vhat = amp * w
    v = np.fft.ifftn(vhat, axes=(1, 2, 3)) * n**3
    return v.real, np.abs(v.imag).max(), vhat, kv


def spectrum(v, box=2 * np.pi, window=None):
    """Shell sums E_m = (1/2) sum |v_hat|^2 over m - 1/2 <= |n| < m + 1/2."""
    n = v.shape[1]
    u = v - v.mean(axis=(1, 2, 3), keepdims=True)
    scale = 1.0
    if window is not None:
        u, scale = u * window, (window**2).mean()
    vhat = np.fft.fftn(u, axes=(1, 2, 3)) / n**3
    shell = np.rint(np.sqrt((lattice(n) ** 2).sum(0))).astype(int)
    e = 0.5 * (np.abs(vhat) ** 2).sum(0) / scale
    k0 = 2 * np.pi / box
    return np.bincount(shell.ravel(), e.ravel()) / k0, k0


def slope_fit(m, e):
    return np.polyfit(np.log(m), np.log(e), 1)[0]


rng = np.random.default_rng(2026)
print("Part 1. 64^3 random-phase solenoidal field, box 2 pi, kappa_0 = 1")
v, imag, vhat, kv = synth(64, rng)
kmag = np.sqrt((kv**2).sum(0))
amp = np.sqrt((np.abs(vhat) ** 2).sum(0))
div_x = np.fft.ifftn(1j * (kv * vhat).sum(0)) * 64**3
grad_rms = np.sqrt((kmag**2 * amp**2).sum())
print(f"  nu = {NU:.4e}; kappa*eta at shell 31 = {31 * ETA:.2f}")
print(f"  max|kappa.v_hat| / max(|kappa||v_hat|) = "
      f"{np.abs((kv * vhat).sum(0)).max() / (kmag * amp).max():.1e}")
print(f"  max|div v| / rms|grad v| = {np.abs(div_x).max() / grad_rms:.1e}")
print(f"  max|imag part of inverse FFT| = {imag:.1e}")

print("Part 2. Parseval: (1/2)<v.v> on the grid against the shell sum")
e64, k0 = spectrum(v)
k_grid = 0.5 * (v**2).sum(0).mean()
print(f"  k_grid = {k_grid:.12f}; sum_m E_m kappa_0 = {e64.sum() * k0:.12f}")
print(f"  relative Parseval error = {abs(e64.sum() * k0 / k_grid - 1):.1e}")

print("Part 3. Recovery: shell sum against the prescribed spectrum")
kk = np.sqrt((kv**2).sum(0))
m_of = np.rint(kk).astype(int).ravel()
inside = ((kk > 0) & (kk < 31.5)).ravel()
lat = np.bincount(m_of[inside], (e_model(kk.ravel()[inside])
                                 / (4 * np.pi * kk.ravel()[inside]**2)))
cnt = np.bincount(m_of[inside])
print(f"  max |E_m / lattice sum - 1|, m = 1..31: "
      f"{np.abs(e64[1:32] / lat[1:32] - 1).max():.1e}")
print("  m  N_m  4pi(m^2+1/12)  lattice/band  count-corrected/band")
for m in (1, 2, 3, 5, 10, 20, 31):
    s = np.linspace(m - 0.5, m + 0.5, 2001)
    band = np.trapezoid(e_model(s), s)
    vol = 4 * np.pi * (m * m + 1 / 12)
    print(f"  {m:2d} {cnt[m]:5d} {vol:10.1f} {lat[m] / band:12.4f}"
          f" {lat[m] * vol / cnt[m] / band:14.4f}")

print("Part 4. Gaussian amplitudes: shell scatter over 24 realizations")
runs = np.array([spectrum(synth(64, rng, gauss=True)[0])[0][:32]
                 for _ in range(24)])
rel = runs.std(axis=0, ddof=1) / runs.mean(axis=0)
for m in (1, 2, 5, 20):
    print(f"  m = {m:2d}: measured rel. std {rel[m]:.3f}; "
          f"N_m^(-1/2) = {cnt[m] ** -0.5:.3f}")

print("Part 5. Flattest band and fitted slope on the 64^3 field")
mm = np.arange(1, 32)
comp = mm ** (5 / 3) * e64[1:32] / (C * EPS ** (2 / 3))
band = mm[comp >= 0.95 * comp.max()]
a, b = band[0], band[-1]
sel = slice(a, b + 1)
inner = comp[a - 1:b]
print(f"  max of kappa^(5/3)E/(C eps^(2/3)) = {comp.max():.3f}"
      f" at m = {mm[comp.argmax()]}; span within 5%: m = {a}..{b}")
print(f"  {len(band)} of {b - a + 1} shells inside 5%: {band.tolist()}; "
      f"lowest {inner.min():.3f} at m = {a + inner.argmin()}")
cm = mm ** (5 / 3) * e_model(mm * 1.0) / C
mb = mm[cm >= 0.95 * cm.max()]
print(f"  model at integers: max {cm.max():.3f} at m = {mm[cm.argmax()]}; "
      f"within 5%: m = {mb[0]}..{mb[-1]}, {len(mb)} shells")
print(f"  fitted slope, random-phase field: "
      f"{slope_fit(mm[a - 1:b], e64[sel]):.3f}")
print(f"  fitted slope, one Gaussian field: "
      f"{slope_fit(mm[a - 1:b], runs[0][sel]):.3f}")
print(f"  fitted slope, model at integers:  "
      f"{slope_fit(mm[a - 1:b], e_model(mm[a - 1:b] * 1.0)):.3f}")
for m in (a, b):
    loc = (np.log(e_model(m * 1.01)) - np.log(e_model(m / 1.01)))
    print(f"  local slope of the model at m = {m}: "
          f"{loc / (2 * np.log(1.01)):.3f}")

print("Part 6. 128^3 parent field: the three ranges")
p, _, _, _ = synth(128, rng)
e128, _ = spectrum(p)
m2 = np.arange(1, 64)
e2 = e128[1:64]
eps_grid = (2 * NU * m2**2 * e2).sum()
s = np.logspace(-4, 3, 200001)
eps_inf = np.trapezoid(2 * NU * s**2 * e_model(s), s)
k_inf = np.trapezoid(e_model(s), s)
print(f"  k: grid {e2.sum():.4f}, model to infinity {k_inf:.4f}")
print(f"  eps: grid {eps_grid:.4f}, model to infinity {eps_inf:.4f}")
print(f"  model Re_lambda = {np.sqrt(20 * k_inf**2 / (3 * eps_inf * NU)):.1f}"
      f"; k^(3/2)/eps = {k_inf**1.5 / eps_inf:.3f}")
eps64 = (2 * NU * mm**2 * e64[1:32]).sum()
print(f"  resolved share of model k, eps: 64^3 {e64.sum() / k_inf:.3f},"
      f" {eps64 / eps_inf:.3f}; 128^3 {e2.sum() / k_inf:.3f},"
      f" {eps_grid / eps_inf:.3f}")
peak = (m2 ** (5 / 3) * e2).max() / C
print(f"  compensated max with eps = 1, eps_model, eps_grid: "
      f"{peak:.3f}, {peak / eps_inf ** (2 / 3):.3f},"
      f" {peak / eps_grid ** (2 / 3):.3f}")
comp2 = m2 ** (5 / 3) * e2 / (C * eps_grid ** (2 / 3))
band2 = m2[comp2 >= 0.95 * comp2.max()]
a2, b2 = band2[0], band2[-1]
print(f"  compensated max {comp2.max():.3f} at m = {m2[comp2.argmax()]};"
      f" span within 5%: m = {a2}..{b2}, {len(band2)} shells inside; "
      f"peak of E at m = {m2[e2.argmax()]}")
print(f"  fitted slope on band {slope_fit(m2[a2 - 1:b2], e2[a2 - 1:b2]):.3f}"
      f"; eta*kappa at band ends {a2 * ETA:.2f}, {b2 * ETA:.2f}")
print(f"  nominal kappa_DI = 2 pi/(60 eta) = {2 * np.pi / (60 * ETA):.1f};"
      f" kappa_EI = 12 pi/L = {12 * np.pi / L_INT:.1f}")
dis = 2 * NU * m2**2 * e2
for name, lo, hi in (("below band", 1, a2 - 1), ("band", a2, b2),
                     ("above band", b2 + 1, 63)):
    part = slice(lo - 1, hi)
    print(f"  {name:10s} m={lo:2d}..{hi:2d}: share of k "
          f"{e2[part].sum() / e2.sum():.3f}, of eps "
          f"{dis[part].sum() / dis.sum():.3f}")
print(f"  dissipation spectrum peaks at m = {m2[dis.argmax()]}")

print("Part 7. Eight contiguous 64^3 cutouts (box pi, kappa_0 = 2) of a")
print("  128^3 parent with eta = 0.02, averaged over the octants")
q, _, _, _ = synth(128, rng, eta=0.02)
e_par, _ = spectrum(q)
hann = np.sin(np.pi * np.arange(64) / 64) ** 2
win = hann[:, None, None] * hann[None, :, None] * hann[None, None, :]
octs = [q[:, i:i + 64, j:j + 64, k:k + 64] for i in (0, 64)
        for j in (0, 64) for k in (0, 64)]
e_rect = np.mean([spectrum(o, box=np.pi)[0] for o in octs], axis=0)
e_hann = np.mean([spectrum(o, box=np.pi, window=win)[0] for o in octs],
                 axis=0)
print(f"  mean(w^2) of the 3D Hann window = {(win**2).mean():.5f}")
print("  kappa  parent E  rectangle/parent  Hann/parent")
for mc in (1, 2, 5, 10, 20, 31):
    ref = (e_par[2 * mc - 1] + 2 * e_par[2 * mc] + e_par[2 * mc + 1]) / 4
    print(f"  {2 * mc:4d} {ref:9.2e} {e_rect[mc] / ref:14.3f}"
          f" {e_hann[mc] / ref:12.3f}")
print(f"  k: parent {e_par.sum():.4f}, rectangle {e_rect.sum() * 2:.4f},"
      f" Hann {e_hann.sum() * 2:.4f}")

print("Part 8. Stride-2 64^3 sample of the parent (full domain)")
ph = np.fft.fftn(p, axes=(1, 2, 3))
lat128 = lattice(128)
cube = (np.abs(lat128) < 32).all(0)
sharp = np.fft.ifftn(ph * cube, axes=(1, 2, 3)).real
boxed = 0.125 * sum(np.roll(p, (-i, -j, -k), axis=(1, 2, 3))
                    for i in (0, 1) for j in (0, 1) for k in (0, 1))
ref, _ = spectrum(sharp[:, ::2, ::2, ::2])
raw, _ = spectrum(p[:, ::2, ::2, ::2])
box_e, _ = spectrum(boxed[:, ::2, ::2, ::2])
true, _ = spectrum(sharp)
print(f"  sharp prefilter vs truncated parent, max rel. diff: "
      f"{np.abs(ref[1:32] / true[1:32] - 1).max():.1e}")
moved = raw - ref
print(f"  share of the relocated energy: shells 16..23 "
      f"{moved[16:24].sum() / moved.sum():.2f}, 24..31 "
      f"{moved[24:32].sum() / moved.sum():.2f}, above 31 "
      f"{moved[32:].sum() / moved.sum():.2f}")
print("  m   none/true  box-average/true")
for m in (4, 8, 16, 24, 31):
    print(f"  {m:2d} {raw[m] / ref[m]:10.3f} {box_e[m] / ref[m]:12.3f}")
print(f"  k: parent {0.5 * (p**2).sum(0).mean():.4f}, stride-2 "
      f"{0.5 * (p[:, ::2, ::2, ::2]**2).sum(0).mean():.4f}, truncated "
      f"{ref.sum():.4f}")
