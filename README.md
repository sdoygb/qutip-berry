# qutip-berry

**Geometric phases and topological invariants for QuTiP.**

A lightweight extension package for [QuTiP](https://github.com/qutip/qutip) that computes Berry phases, Berry curvature, and Chern numbers on discretized parameter grids. Implements the gauge-invariant Fukui–Hatsugai–Suzuki (FHS) discretization for single bands and the multi-band Wilson-loop determinant formulation, with built-in integer-quantization verification.

[![License](https://img.shields.io/badge/license-BSD--3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://pypi.org/project/qutip-berry/)
[![QuTiP](https://img.shields.io/badge/QuTiP-5.0+-green.svg)](https://github.com/qutip/qutip)

## Features

- **Berry curvature** on 2D parameter grids (sphere, torus, or arbitrary mesh)
- **Chern number** with exact integer quantization on any grid size (FHS formulation)
- **Zak phase / Berry phase** along closed 1D parameter paths
- **Multi-band support** via Wilson-loop determinant (valid when bands do not mix)
- **Periodic boundary conditions** per dimension (needed for azimuthal direction on a sphere, or Brillouin-zone torii)
- **Gauge-invariant** outputs — invariant under local U(1) re-gauging of eigenstates
- Accepts both NumPy arrays and lists of QuTiP `Qobj` kets as input

## Installation

```bash
pip install qutip-berry
```

Or install from source:

```bash
git clone https://github.com/sdoygb/qutip-berry.git
cd qutip-berry
pip install -e .
```

## Quick Start

### Chern number of a two-level monopole on S²

```python
import numpy as np
from qutip import sigmax, sigmay, sigmaz
import qutip_berry as berry

# Parameter grid on the unit sphere
n_theta, n_phi = 40, 40
thetas = np.linspace(0, np.pi, n_theta)
phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)

# Compute lower-band eigenstates of H = n(θ,φ)·σ
eigfs = np.zeros((n_theta, n_phi, 2), dtype=complex)
for i, th in enumerate(thetas):
    for j, ph in enumerate(phis):
        n = np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])
        H = n[0]*sigmax() + n[1]*sigmay() + n[2]*sigmaz()
        eigfs[i, j] = H.eigenstates()[1][0].full().reshape(-1)

# Chern number: θ is non-periodic (poles), φ is periodic
C = berry.chern_number(eigfs, periodic=(False, True))
print(f"C = {C:.6f}")  # C = -1.000000 (exact to machine precision)
```

### Zak phase of the SSH model

```python
import numpy as np
from qutip import sigmax, sigmay
import qutip_berry as berry

n = 100
ks = np.linspace(0, 2*np.pi, n, endpoint=False)
eigfs = np.zeros((n, 2), dtype=complex)
for i, k in enumerate(ks):
    H = (1.0 + 2.0*np.cos(k))*sigmax() + 2.0*np.sin(k)*sigmay()
    eigfs[i] = H.eigenstates()[1][0].full().reshape(-1)

zak = berry.berry_phase(eigfs)
print(f"Zak phase = {zak:.4f}")  # π in topological phase (v < w)
```

## API Reference

### `berry_curvature(eigfs, periodic=(False, False))`

Computes the discretized Berry curvature (plaquette flux) on a 2D parameter grid.

**Parameters:**
- `eigfs`: eigenstates as a 4D array `(n0, n1, nocc, hilbert)` or 3D `(n0, n1, hilbert)` for a single band. Also accepts nested lists of `Qobj` kets.
- `periodic`: tuple of bools indicating whether each grid dimension is closed (periodic boundary).

**Returns:** 2D array of plaquette fluxes, shape `(m0, m1)` where `m = n` for periodic, `m = n-1` otherwise.

### `chern_number(eigfs, periodic=(False, False))`

Sums the Berry curvature over the grid and divides by 2π. With the single-band FHS formulation the result is exactly integral up to floating-point roundoff on any grid size.

### `berry_phase(eigfs)`

Computes the Berry (Zak) phase accumulated along a closed 1D path, via the determinant of the Wilson loop. Accepts a 3D array `(n, nocc, hilbert)` or 2D `(n, hilbert)`.

## Verification

The package ships with a test suite (`pytest tests/`) covering:

| Test | Model | Expected |
|------|-------|----------|
| Ground-state monopole | H = n·σ on S² | C = −1 |
| Excited-state monopole | H = n·σ on S² | C = +1 |
| SSH model (topological) | v=1, w=2 | Zak = π |
| SSH model (trivial) | v=2, w=1 | Zak = 0 |
| Multi-band block-diagonal | 6 levels (3 monopoles) | total C ≈ 0 |
| Qobj list input | SSH model | Zak = π |
| Bad input shapes | — | raises `ValueError` |

All tests pass. The single-band FHS Chern number is exact to machine precision (~10⁻¹⁴) for grid sizes from 16×16 to 96×96.

## How it relates to QuTiP

This package originated as a proposed submodule for QuTiP (see [qutip/qutip#2972](https://github.com/qutip/qutip/issues/2972)). The QuTiP maintainers concluded that, while the module is well-developed and useful for researchers working on geometric and topological phenomena, it is a specialized tool that is better maintained as a standalone extension package rather than merged into the QuTiP core. This repository is that standalone package.

It depends on QuTiP for quantum object representation and eigenstate solving, and is designed to compose naturally with QuTiP workflows.

## References

1. T. Fukui, Y. Hatsugai, and H. Suzuki, "Chern Numbers in Discretized Brillouin Zone: Efficient Method of Computing (Spin) Hall Conductances", *J. Phys. Soc. Jpn.* **74**, 1674 (2005).
2. M. V. Berry, "Quantal phase factors accompanying adiabatic changes", *Proc. R. Soc. Lond. A* **392**, 45 (1984).
3. J. R. Johansson, P. D. Nation, and F. Nori, "QuTiP: An open-source Python framework for the dynamics of open quantum systems", *Comp. Phys. Comm.* **183**, 1760 (2012).

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
