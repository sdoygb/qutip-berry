# qutip-berry

**Geometric phases and topological invariants for QuTiP.**

A lightweight extension package for [QuTiP](https://github.com/qutip/qutip) that computes Berry phases, Berry curvature, and Chern numbers on discretized parameter grids. Implements the gauge-invariant Fukui–Hatsugai–Suzuki (FHS) discretization for single bands and the multi-band Wilson-loop determinant formulation, with built-in integer-quantization verification.

[![License](https://img.shields.io/badge/license-BSD--3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://pypi.org/project/qutip-berry/)
[![QuTiP](https://img.shields.io/badge/QuTiP-5.0+-green.svg)](https://github.com/qutip/qutip)
[![PyPI](https://img.shields.io/pypi/v/qutip-berry.svg)](https://pypi.org/project/qutip-berry/)

## Features

- **Two API levels**:
  - *Low-level*: pass pre-computed eigenstate arrays (`berry_curvature`, `chern_number`, `berry_phase`)
  - *High-level*: pass a parameterized Hamiltonian `H(θ)` and the package sweeps the parameter space automatically, with **adiabatic tracking** (maximum-overlap matching via the Hungarian algorithm) to keep bands continuous through avoided crossings
- **Berry curvature** on 2D parameter grids (sphere, torus, or arbitrary mesh)
- **Chern number** with exact integer quantization on any grid size (FHS formulation)
- **Zak phase / Berry phase** along closed 1D parameter paths
- **Multi-band support** via Wilson-loop determinant (valid when bands do not mix)
- **Periodic boundary conditions** per dimension (needed for azimuthal direction on a sphere, or Brillouin-zone torii)
- **Gauge-invariant** outputs — invariant under local U(1) re-gauging of eigenstates
- Accepts both NumPy arrays and QuTiP `Qobj` kets as input

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

### High-level API: Chern number from a Hamiltonian (recommended)

Just define your parameterized Hamiltonian and pass it in — the package handles eigenstate solving, adiabatic band tracking, and curvature computation:

```python
import numpy as np
from qutip import sigmax, sigmay, sigmaz
import qutip_berry as berry

# Define H(θ, φ) = n(θ,φ) · σ  (magnetic monopole on S²)
def H(th, ph):
    n = np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])
    return n[0]*sigmax() + n[1]*sigmay() + n[2]*sigmaz()

# Parameter grid
thetas = np.linspace(0, np.pi, 30)
phis = np.linspace(0, 2*np.pi, 30, endpoint=False)

# Ground-state Chern number (θ non-periodic at poles, φ periodic)
C = berry.chern_number_from_hamiltonian(
    H, (thetas, phis), bands=0, periodic=(False, True)
)
print(f"C = {C:.6f}")  # C = -1.000000 (exact to machine precision)

# Excited state has C = +1
C_exc = berry.chern_number_from_hamiltonian(
    H, (thetas, phis), bands=1, periodic=(False, True)
)
print(f"C_exc = {C_exc:.6f}")  # C_exc = +1.000000
```

### High-level API: Zak phase of the SSH model

```python
import numpy as np
from qutip import sigmax, sigmay
import qutip_berry as berry

# SSH Hamiltonian H(k) = (v + w cos k) σ_x + w sin k σ_y
def H_ssh(k, v=1.0, w=2.0):
    return (v + w*np.cos(k))*sigmax() + w*np.sin(k)*sigmay()

ks = np.linspace(0, 2*np.pi, 50, endpoint=False)

# Topological phase (v < w): Zak = π
zak_topo = berry.berry_phase_from_hamiltonian(H_ssh, ks, bands=0)
print(f"Zak (topological) = {zak_topo:.4f}")  # π

# Trivial phase (v > w): Zak = 0
zak_triv = berry.berry_phase_from_hamiltonian(
    lambda k: H_ssh(k, v=2.0, w=1.0), ks, bands=0
)
print(f"Zak (trivial) = {zak_triv:.4f}")  # 0
```

### Low-level API: from pre-computed eigenstates

```python
import numpy as np
from qutip import sigmax, sigmay, sigmaz
import qutip_berry as berry

n = 30
thetas = np.linspace(0, np.pi, n)
phis = np.linspace(0, 2*np.pi, n, endpoint=False)

# Manually compute eigenstates at each grid point
eigfs = np.zeros((n, n, 2), dtype=complex)
for i, th in enumerate(thetas):
    for j, ph in enumerate(phis):
        B = np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])
        H = B[0]*sigmax() + B[1]*sigmay() + B[2]*sigmaz()
        eigfs[i, j] = H.eigenstates()[1][0].full().reshape(-1)

# Berry curvature field
F = berry.berry_curvature(eigfs, periodic=(False, True))
print(f"Curvature shape: {F.shape}")  # (29, 30)

# Chern number
C = berry.chern_number(eigfs, periodic=(False, True))
print(f"C = {C:.6f}")  # -1.000000
```

## API Reference

### High-level (from Hamiltonian)

#### `chern_number_from_hamiltonian(H, params, bands=0, periodic=(False, False))`

Compute the Chern number directly from a parameterized Hamiltonian.

- `H`: callable `H(theta0, theta1)` returning a Qobj or Hermitian NumPy array
- `params`: tuple `(params0, params1)` of 1D arrays defining the parameter grid
- `bands`: int or list of int (0 = ground state). Single band uses FHS (exact integer); multiple bands use Wilson-loop determinant
- `periodic`: tuple of bool for each dimension

#### `berry_curvature_from_hamiltonian(H, params, bands=0, periodic=(False, False))`

Compute the discretized Berry curvature field directly from a Hamiltonian. Same parameters as above. Returns a 2D array of plaquette fluxes.

#### `berry_phase_from_hamiltonian(H, params, bands=0)`

Compute the Berry (Zak) phase along a closed 1D path.

- `H`: callable `H(theta)` returning a Qobj
- `params`: 1D array of parameter values (path is closed: last connects to first)
- `bands`: int or list of int

### Low-level (from eigenstates)

#### `berry_curvature(eigfs, periodic=(False, False))`

Computes the discretized Berry curvature (plaquette flux) on a 2D grid of eigenstates. `eigfs` is a 4D array `(n0, n1, nocc, hilbert)` or 3D `(n0, n1, hilbert)` for a single band. Also accepts nested lists of Qobj kets.

#### `chern_number(eigfs, periodic=(False, False))`

Sums the Berry curvature over the grid and divides by 2π. With the single-band FHS formulation this is exactly integral up to floating-point roundoff.

#### `berry_phase(eigfs)`

Computes the Berry (Zak) phase along a closed 1D path via the Wilson-loop determinant. `eigfs` is a 3D array `(n, nocc, hilbert)` or 2D `(n, hilbert)`.

## Adiabatic Tracking

The high-level API automatically tracks bands across the parameter grid using **maximum-overlap matching** (Hungarian algorithm via `scipy.optimize.linear_sum_assignment`). At each grid point, the newly solved eigenstates are reordered to maximize overlap with the previously computed eigenstates (left neighbor, or top neighbor for the first column). This ensures that the same band is followed continuously even through avoided crossings, which is essential for correct Chern number computation in systems with near-degenerate bands.

## Verification

The package ships with a test suite (`pytest tests/`) covering 15 tests:

| Category | Test | Expected |
|----------|------|----------|
| Low-level | Ground-state monopole | C = −1 |
| Low-level | Excited-state monopole | C = +1 |
| Low-level | SSH model (topological) | Zak = π |
| Low-level | SSH model (trivial) | Zak = 0 |
| Low-level | Multi-band block-diagonal | total C ≈ 0 |
| Low-level | Qobj list input | Zak = π |
| Low-level | Bad input shapes | raises ValueError |
| **High-level** | **Monopole ground state** | **C = −1** |
| **High-level** | **Monopole excited state** | **C = +1** |
| **High-level** | **Curvature shape** | **(n0−1, n1)** |
| **High-level** | **SSH topological** | **Zak = π** |
| **High-level** | **SSH trivial** | **Zak = 0** |
| **High-level** | **High-level = low-level** | **match to 1e-12** |
| **High-level** | **Multi-band (both bands)** | **total C ≈ 0** |
| **High-level** | **Avoided crossing tracking** | **integral C** |

All tests pass. The single-band FHS Chern number is exact to machine precision (~10⁻¹⁴) for grid sizes from 16×16 to 96×96.

## How it relates to QuTiP

This package originated as a proposed submodule for QuTiP (see [qutip/qutip#2972](https://github.com/qutip/qutip/issues/2972)). The QuTiP maintainers concluded that, while the module is well-developed and useful for researchers working on geometric and topological phenomena, it is a specialized tool that is better maintained as a standalone extension package rather than merged into the QuTiP core. This repository is that standalone package.

It depends on QuTiP for quantum object representation and eigenstate solving, and is designed to compose naturally with QuTiP workflows — especially geometric quantum computation, where the high-level `H(θ)` interface integrates directly with `qutip.control` and `qutip.qip`.

## References

1. T. Fukui, Y. Hatsugai, and H. Suzuki, "Chern Numbers in Discretized Brillouin Zone: Efficient Method of Computing (Spin) Hall Conductances", *J. Phys. Soc. Jpn.* **74**, 1674 (2005).
2. M. V. Berry, "Quantal phase factors accompanying adiabatic changes", *Proc. R. Soc. Lond. A* **392**, 45 (1984).
3. J. R. Johansson, P. D. Nation, and F. Nori, "QuTiP: An open-source Python framework for the dynamics of open quantum systems", *Comp. Phys. Comm.* **183**, 1760 (2012).

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
