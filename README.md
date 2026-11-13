# Two instability intervals for a switched planar system with one non-Hurwitz matrix

Yitzchak Shmalo

This repository contains a paper on the randomly switched planar linear system dX/dt = A_{I_t} X_t, where I jumps
between 0 and 1 at rate r, together with the programs and data behind its computer-assisted steps.

Paper: [paper/main.pdf](paper/main.pdf) (source [paper/main.tex](paper/main.tex)).

## Summary

Malrieu (Open question 3 of his survey on piecewise deterministic Markov processes) asked whether two 2 x 2
matrices can be chosen so that the set of jump rates r giving unstable processes is a union of several intervals.
For the Hurwitz matrix A_0 = [[-3/4, -1], [1, -3/4]] and the non-Hurwitz matrix A_1 = diag(5/4, -11/4) the paper
proves that the top Lyapunov exponent lambda(r) is continuous on (0, infinity) and that

* lambda(r) > 9/50 for 0 < r <= 1/10,
* -0.0756 <= lambda(r) <= -0.0055 for 4.997 <= r <= 5.003,
* 0.0361 <= lambda(r) <= 0.1192 for 19.95 <= r <= 20.05,

so the unstable set has at least two connected components. The bounds near r = 5 and r = 20 come from
trigonometric polynomials with exact rational coefficients that solve the Poisson equation of the angular process up
to an explicit error. Numerically lambda changes sign twice, near r = 1.2228 and r = 7.7616. The case in which both
matrices are Hurwitz remains open; Section 6 of the paper explains why shifts and normal matrices do not reach it and
records a numerical search that found no example.

## Code

Python 3 with NumPy, SciPy and mpmath (versions in `data/software_versions.json`); `code/verify_certificate.py` uses
only the standard library. Run the scripts from the repository root, for example `python code/verify_certificate.py`.

Certificates (Proposition 3.2 and Corollary 3.3 of the paper):

* `code/poisson_certificate.py` builds the rational test functions at r = 5 (K = 128) and r = 20 (K = 32) and writes
  `data/poisson_certificates.json`.
* `code/verify_certificate.py` recomputes the bounds of Table 1 in exact rational arithmetic in the sine-cosine
  basis and writes `data/verification.json`.
* `code/independent_certificates.py` recomputes them with Laurent polynomials in z = exp(2 i theta) and writes
  `data/independent/certificate_audit.json`.
* `code/interval_audit.py` encloses the same values in interval arithmetic at fifty digits and writes
  `data/interval_audit.json`.
* `code/certified_neighborhoods.py` extends the signs to the rate intervals around 5 and 20 and writes
  `data/certified_neighborhoods.json`.

Numerics (Section 5):

* `code/candidate_density.py` and `code/numerics.py` compute lambda(r) from the explicit stationary density and from
  a discretised generator; `code/validate_candidate.py` runs the grid in `data/candidate_grid.csv`, the cross-checks
  in `data/crosschecks.json` and the two crossings in `data/numerical_roots.json`.
* `code/monte_carlo.c` (C99, `cc -O2 -o mc code/monte_carlo.c -lm`) simulates the process; block means are in
  `data/monte_carlo_blocks.csv`.
* `code/independent_density.py` is a separate stationary-flux calculation on its own 151-rate grid
  (`data/independent/`).
* `code/audit_grid.py` compares the grid with a refined one (`data/candidate_grid_refined.csv`,
  `data/grid_audit.json`); `code/plot_candidate.py` draws `paper/figures/curve.pdf`.

The Hurwitz case (Section 6):

* `code/search_ctmc.py` and `code/search_hurwitz.py` run the search over traceless pairs and their scalar shifts and
  write `data/search_ctmc.json` and `data/hurwitz_search.json`.

`SHA256SUMS` lists the checksums of every file in the repository.

## License

The text is released under CC BY 4.0 and the code under the MIT license.
