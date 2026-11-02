"""Time correlation of the DNS at lags of whole snapshot intervals (chapter 6).

corr.csv from labs/dns_isotropic.py stops at lag 1.0. This script continues it
from the 64^3 filtered fields, which hold almost all of the energy: for each
lag l it averages <u(t_s) . u(t_s + l)> / sqrt(<u.u>(t_s) <u.u>(t_s + l)) over
all snapshot pairs, and writes lag,rho,pairs to rho_lags.csv in that folder.
Run:  python labs/dns_correlation.py [--dir data/dns]
"""
import argparse
import glob
import json
import pathlib
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--dir', default='data/dns')
args = ap.parse_args()
d = pathlib.Path(args.dir)
every = json.loads((d / 'params.json').read_text())['every']
files = sorted(glob.glob(str(d / 'filtered' / '*.npy')))
x = np.stack([np.load(f).ravel() for f in files]).astype(np.float64)
gram = x @ x.T
norm = np.sqrt(np.diag(gram))
rho = gram / np.outer(norm, norm)
with open(d / 'rho_lags.csv', 'w') as f:
    f.write('lag,rho,pairs\n')
    for lag in range(len(files)):
        r = np.diag(rho, lag)
        f.write(f'{lag * every:.1f},{r.mean():.6f},{len(r)}\n')
print(f'{len(files)} snapshots; wrote {d / "rho_lags.csv"}')
