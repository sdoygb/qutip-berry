"""
qutip-berry: Geometric phases and topological invariants for QuTiP.

A lightweight extension package for QuTiP that computes Berry phases,
Berry curvature, and Chern numbers on discretized parameter grids,
with gauge-invariant discretization (Fukui-Hatsugai-Suzuki) and
multi-band Wilson-loop formulations.

Two API levels are provided:

- **Low-level** (:func:`berry_curvature`, :func:`chern_number`,
  :func:`berry_phase`): accept pre-computed eigenstate arrays.
- **High-level** (:func:`berry_curvature_from_hamiltonian`,
  :func:`chern_number_from_hamiltonian`,
  :func:`berry_phase_from_hamiltonian`): accept a parameterized
  Hamiltonian ``H(theta)`` and sweep the parameter space automatically,
  with adiabatic tracking to keep bands continuous.
"""

from .berry import (
    berry_phase,
    berry_curvature,
    chern_number,
    berry_phase_from_hamiltonian,
    berry_curvature_from_hamiltonian,
    chern_number_from_hamiltonian,
)

__all__ = [
    "berry_phase",
    "berry_curvature",
    "chern_number",
    "berry_phase_from_hamiltonian",
    "berry_curvature_from_hamiltonian",
    "chern_number_from_hamiltonian",
]
__version__ = "0.2.0"
