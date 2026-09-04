"""
qutip-berry: Geometric phases and topological invariants for QuTiP.

A lightweight extension package for QuTiP that computes Berry phases,
Berry curvature, and Chern numbers on discretized parameter grids,
with gauge-invariant discretization (Fukui-Hatsugai-Suzuki) and
multi-band Wilson-loop formulations.
"""

from .berry import berry_phase, berry_curvature, chern_number

__all__ = ["berry_phase", "berry_curvature", "chern_number"]
__version__ = "0.1.0"
