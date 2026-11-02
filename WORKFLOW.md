# Textbook production workflow

This independent course joins chapters of five existing study editions with
six new chapters and two laboratories. Eleven of the fourteen weeks link to the
source editions by their public Pages addresses and copy nothing from them.

Production applies rigorous-explainer for the chapters and science-editor for
the reviews. Representative chapters of the Turbulence Modeling and MIT 6.S184
editions were read end to end as the model of structure, tone and exercises.

1. Fix the course notation first. The five editions use the same symbols for
   different objects, so chapter 1 records every conflict, and all new chapters
   use one table and one Fourier convention.
2. Author each new chapter as a body fragment. Each one has an executed program
   with its exact printed output, at least six exercises with complete
   solutions, and a closing that states what the chapter does not establish.
   Distinguish derived results, results quoted from a paper, numerical checks
   and unverified claims.
3. Generate the data. The course has no database token, so `labs/dns_isotropic.py`
   runs a $256^3$ pseudo-spectral DNS on one laptop GPU. Two production runs were
   discarded after measurement: the forcing amplified a roundoff fault to an
   energy of 0.17 by the end of the record, against a total of about 0.6. The
   cause, the rate and the fix are in the docstring of `clean()`. `labs/check_dns.py` is the acceptance check; it
   passes on the published run and fails on the discarded one.
4. Review twice, independently. Round 1 and round 2 of each chapter go to
   different agents, neither of which wrote the chapter. An agent never accepts
   a fix it wrote. Only the final reviewer writes the acceptance, bound to the
   SHA-256 digest of the exact authored body in the sidecar. Any later edit of
   the body voids the acceptance.
5. Build root-level HTML with local CSS, JavaScript and MathJax. The build
   refuses a body whose digest disagrees with its sidecar.
6. Check the code, the rendering and the links. Then publish the public files,
   enable GitHub Pages, and check the served pages. A successful push is not a
   live check.

## Evidence limits

Reviews are independent AI scientific reviews, not human expert certification.
Review reports stay in the ignored `work/` folder; the public record is the
digest-bound status in each sidecar and `editorial-review-summary.json`. The
DNS is one run at $\mathrm{Re}_\lambda=115$, and its spectrum has no inertial
band (chapter 3, Section 3.8). The Lab 2 outputs are single stochastic runs on
one GPU. Chapter 1 and the study route record what a read-only depth check of
the linked chapters did and did not establish.

## Data

Published in `data/dns/`: `params.json`, `stats.csv`, `spectrum.csv`,
`corr.csv`, `rho_lags.csv`, `sampling_spectra.csv`. Chapter programs read only
these files. The snapshots, 3.6 GB, are git-ignored; regenerate them with:

    python -u labs/dns_isotropic.py --n 256 --spinup 12 --every 1.0 --snaps 100 --log 50 --out data/dns
    python labs/check_dns.py data/dns
    python labs/dns_correlation.py --dir data/dns
    python labs/dns_sampling_spectra.py --dir data/dns

The first command took 2.13 hours at 276.5 ms per step on an NVIDIA RTX 3050
Laptop GPU with 4096 MiB. Turbulence is chaotic: another machine or library
gives a different realization with the same statistics, not the same digits.

## Public checks

Run from the repository root:

    python tools/build_book.py
    python tools/check_code.py
    python tools/browser_check.py

`check_code.py` requires each chapter listing to equal a file of `code/` and its
displayed output to equal the printed output under Python 3.13, and runs
pycodestyle on `code/` and `labs/`. Roundoff-level digits can differ on another
NumPy build. `browser_check.py` opens every public page at 1440 and 390 pixels
in installed Chrome through Playwright and writes `reader-validation.json`; pass
`--base <URL>` to check the live site. The validators of the rigorous-explainer
skill (checktex, checklt, check_links, verify_dom) were also run on the pages;
they are not part of this repository.

Chapter programs need Python 3.13 and NumPy. The laboratories add PyTorch. See
`requirements.txt`.
