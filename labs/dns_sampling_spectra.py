"""Spectra of the two 64^3 samples of the DNS, for chapter 3, section 3.8.

Reads the snapshots of labs/dns_isotropic.py. Writes sampling_spectra.csv in
the same folder, with columns kind,kappa,E, for all shells 0 to 63. Spectra
are shell sums of chapter 3, equation (3.3), each with its own mean removed.
  crop_rect, crop_hann  mean over all crop/ blocks, box pi/2, kappa_0 = 4,
                        without a window and with the 3D Hann window divided
                        by mean(w^2) = 27/512
  stride4_none          every 4th point of the full/ snapshots, no prefilter
  stride4_sharp         filtered/ fields of the same snapshots
Run:  python labs/dns_sampling_spectra.py [--dir data/dns]
"""
import argparse
import glob
import pathlib
import numpy as np


def spectrum(v, box, window=None):
    n = v.shape[1]
    u = v - v.mean(axis=(1, 2, 3), keepdims=True)
    scale = 1.0
    if window is not None:
        u, scale = u * window, (window**2).mean()
    e = 0.5 * (np.abs(np.fft.fftn(u, axes=(1, 2, 3)) / n**3)**2).sum(0) / scale
    q = np.fft.fftfreq(n, 1 / n)
    kk = sum(a**2 for a in np.meshgrid(q, q, q, indexing='ij'))
    shell = np.rint(np.sqrt(kk)).astype(int)
    return np.bincount(shell.ravel(), e.ravel(), n) * box / (2 * np.pi)


ap = argparse.ArgumentParser()
ap.add_argument('--dir', default='data/dns')
d = pathlib.Path(ap.parse_args().dir)
crops = sorted(glob.glob(str(d / 'crop' / '*.npy')))
full = sorted(glob.glob(str(d / 'full' / '*.npy')))
hann = np.sin(np.pi * np.arange(64) / 64)**2
win = hann[:, None, None] * hann[None, :, None] * hann[None, None, :]
rect, hanned = np.zeros(64), np.zeros(64)
for f in crops:
    v = np.load(f).astype(np.float64)
    rect += spectrum(v, np.pi / 2) / len(crops)
    hanned += spectrum(v, np.pi / 2, win) / len(crops)
none, sharp = np.zeros(64), np.zeros(64)
for f in full:
    v = np.load(f)[:, ::4, ::4, ::4].astype(np.float64)
    none += spectrum(v, 2 * np.pi) / len(full)
    name = pathlib.Path(f).name
    v = np.load(d / 'filtered' / name).astype(np.float64)
    sharp += spectrum(v, 2 * np.pi) / len(full)
with open(d / 'sampling_spectra.csv', 'w') as f:
    f.write('kind,kappa,E\n')
    for name, e, k0 in (('crop_rect', rect, 4), ('crop_hann', hanned, 4),
                        ('stride4_none', none, 1),
                        ('stride4_sharp', sharp, 1)):
        for m in range(64):
            f.write(f'{name},{m * k0},{e[m]:.6e}\n')
print(f'{len(crops)} crops, {len(full)} full snapshots; wrote '
      f'{d / "sampling_spectra.csv"}')
