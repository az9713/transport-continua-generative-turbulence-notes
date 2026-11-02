"""Compare the Fourier conventions of the source editions numerically.

Course (essentials) convention: forward transform carries (2 pi)^-d.
Modeling-edition convention: forward transform carries no prefactor.
Discrete periodic box: f_hat = numpy.fft.fftn(f) / N**3.
Standard library and NumPy only.
"""
import numpy as np

TWO_PI = 2.0 * np.pi

# (a) 1D Gaussian filter kernel of width delta, variance delta**2 / 12.
delta = 0.5
x = np.linspace(-10.0, 10.0, 20001)
dx = x[1] - x[0]
kernel = np.sqrt(6.0 / np.pi) / delta * np.exp(-6.0 * x**2 / delta**2)
print("(a) Gaussian kernel, delta = 0.5, Riemann sums on 20001 points")
print("kappa  course_fwd    modeling_fwd  ratio       exp(-k2 D2/24)")
for kappa in (0.0, 2.0, 5.0, 10.0):
    integral = np.sum(kernel * np.cos(kappa * x)) * dx
    course = integral / TWO_PI
    modeling = integral
    transfer = np.exp(-kappa**2 * delta**2 / 24.0)
    print(f"{kappa:5.1f}  {course:.10f}  {modeling:.10f}  "
          f"{modeling / course:.8f}  {transfer:.10f}")

# (b) Continuous 1D Parseval for f(x) = exp(-x**2 / 2) under both.
xs = np.linspace(-12.0, 12.0, 2401)
ks = np.linspace(-12.0, 12.0, 2401)
hx = xs[1] - xs[0]
hk = ks[1] - ks[0]
f = np.exp(-xs**2 / 2.0)
cosines = np.cos(np.outer(ks, xs))
f_course = cosines @ f * hx / TWO_PI
f_model = cosines @ f * hx
energy_x = np.sum(f**2) * hx
energy_course = TWO_PI * np.sum(f_course**2) * hk
energy_model = np.sum(f_model**2) * hk / TWO_PI
print("(b) Parseval, f = exp(-x^2/2); exact integral of f^2 = sqrt(pi)")
print(f"integral f^2 dx            = {energy_x:.12f}")
print(f"2 pi * int |f_course|^2    = {energy_course:.12f}")
print(f"(1/2 pi) * int |f_model|^2 = {energy_model:.12f}")
print(f"sqrt(pi)                   = {np.sqrt(np.pi):.12f}")
rel_exact = np.sum(f**2) * 0.01 / np.sqrt(np.pi) - 1
print(f"spacing error, hx - 0.01      = {hx - 0.01:.1e}")
print(f"relative gap at spacing 0.01  = {rel_exact:.1e}")
print(f"f(0) = 1; inverse of f_course = {np.sum(f_course) * hk:.12f}")
print(f"mismatch: inverse of f_model  = {np.sum(f_model) * hk:.12f}")

# (c) Discrete periodic box [0, 2 pi)^3: a known two-mode field.
n_small = 16
grid = np.arange(n_small) * TWO_PI / n_small
gx, gy, gz = np.meshgrid(grid, grid, grid, indexing="ij")
field = 2.0 * np.cos(3.0 * gx) + np.sin(5.0 * gy)
coef = np.fft.fftn(field) / n_small**3
c300 = np.round(coef[3, 0, 0], 12) + 0j  # + 0j turns -0.0 into 0.0
c050 = np.round(coef[0, 5, 0], 12) + 0j
n_fine = 48
fine = np.arange(n_fine) * TWO_PI / n_fine
fx, fy, _ = np.meshgrid(fine, fine, fine, indexing="ij")
fine_field = 2.0 * np.cos(3.0 * fx) + np.sin(5.0 * fy)
box_sum = np.sum(fine_field * np.exp(-3j * fx)) * (TWO_PI / n_fine)**3
print("(c) f = 2 cos(3x) + sin(5y) on a 16^3 grid, box side 2 pi")
print(f"course f_hat(3,0,0)  = {c300.real:+.12f} {c300.imag:+.12f}i")
print(f"course f_hat(0,5,0)  = {c050.real:+.12f} {c050.imag:+.12f}i")
print(f"modeling box integral at (3,0,0) = {box_sum.real:.10f}")
print(f"mean f^2 = {np.mean(field**2):.12f}; "
      f"sum |f_hat|^2 = {np.sum(np.abs(coef)**2):.12f}")

# (d) Random three-component field on 32^3: Parseval and shell sum.
n_big = 32
rng = np.random.default_rng(20261101)
velocity = rng.standard_normal((3, n_big, n_big, n_big))
v_hat = np.fft.fftn(velocity, axes=(1, 2, 3)) / n_big**3
v_raw = np.fft.fftn(velocity, axes=(1, 2, 3))
k_physical = 0.5 * np.mean(np.sum(velocity**2, axis=0))
k_spectral = 0.5 * np.sum(np.abs(v_hat)**2)
wave = np.fft.fftfreq(n_big, d=1.0 / n_big)
kx, ky, kz = np.meshgrid(wave, wave, wave, indexing="ij")
shell = np.rint(np.sqrt(kx**2 + ky**2 + kz**2)).astype(int)
mode_energy = 0.5 * np.sum(np.abs(v_hat)**2, axis=0)
spectrum = np.bincount(shell.ravel(), weights=mode_energy.ravel())
raw_ratio = np.sum(np.abs(v_raw)**2) / np.sum(np.abs(v_hat)**2)
print("(d) random velocity, 32^3, seed 20261101")
print(f"k = 0.5 mean(v.v)          = {k_physical:.12f}")
print(f"k = 0.5 sum |v_hat|^2      = {k_spectral:.12f}")
print(f"k = sum over shells of E   = {np.sum(spectrum):.12f}")
print(f"relative Parseval error    = "
      f"{abs(k_spectral - k_physical) / k_physical:.3e}")
print(f"raw fftn / course ratio    = {raw_ratio:.6e} (N^6 = {n_big**6:.6e})")
