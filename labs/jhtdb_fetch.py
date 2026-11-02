"""Download velocity cubes from JHTDB isotropic1024coarse for Lab 2.

Dataset: forced isotropic turbulence, DNS on a 1024^3 periodic grid,
domain [0, 2 pi)^3, 5028 stored time steps.

Token: set the environment variable JHTDB_TOKEN to a personal token
(request one by email from turbulence@lists.johnshopkins.edu). Without it
the public testing token is used, which allows at most 4096 points per
request, i.e. 16^3 cubes only.

Two sampling modes (chapter 3 explains the trade-off):
  contiguous  an n^3 block at grid spacing 2 pi / 1024; resolves the
              smallest scales but covers only n/1024 of the domain and is
              not periodic (window before taking an FFT).
  strided     n^3 points spread over the whole domain with stride 1024/n;
              periodic and covers all large scales, but aliases the
              unresolved small scales (no prefilter is applied).

Requires: pip install givernylocal  (tested with 3.6.4, in a separate venv)
Run:  python labs/jhtdb_fetch.py --n 16 --count 8 --mode contiguous
"""
import argparse
import os
import pathlib
import numpy as np

TESTING = 'edu.jhu.pha.turbulence.testing-201406'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--count', type=int, default=8)
    ap.add_argument('--mode', choices=['contiguous', 'strided'],
                    default='contiguous')
    ap.add_argument('--out', default='data/jhtdb')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    from givernylocal.turbulence_dataset import turb_dataset
    from givernylocal.turbulence_toolkit import getCutout
    token = os.environ.get('JHTDB_TOKEN', TESTING)
    if token == TESTING and args.n**3 > 4096:
        raise SystemExit('n^3 > 4096 needs a personal token in JHTDB_TOKEN')
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cube = turb_dataset(dataset_title='isotropic1024coarse',
                        output_path=str(out), auth_token=token)
    rng = np.random.default_rng(args.seed)
    stride = 1024 // args.n if args.mode == 'strided' else 1
    span = args.n * stride
    for i in range(args.count):
        # Separate snapshots by 100 stored steps to reduce time correlation.
        step = 1 + 100 * i
        x0, y0, z0 = rng.integers(1, 1025 - span + 1, 3) if stride == 1 \
            else (1, 1, 1)
        ranges = np.array([[x0, x0 + span - 1], [y0, y0 + span - 1],
                           [z0, z0 + span - 1], [step, step]])
        res = getCutout(cube, 'velocity', ranges,
                        np.array([stride, stride, stride, 1]),
                        verbose=False)
        # givernylocal returns dims (zcoor, ycoor, xcoor, values);
        # store as (component, x, y, z) to match the course convention.
        v = np.asarray(res[list(res.data_vars)[0]], dtype=np.float32)
        v = v.transpose(3, 2, 1, 0)
        np.save(out / f'iso_{args.mode}_{args.n}_{i:04d}.npy', v)
        print(f'saved cube {i}: time step {step}, origin {x0},{y0},{z0}')


if __name__ == '__main__':
    main()
