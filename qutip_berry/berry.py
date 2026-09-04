"""
Geometric phases and topological invariants on discretized parameter grids.

This module restores and modernizes ``qutip.topology.berry_curvature``
(removed in QuTiP 5.0) and extends it with:

- a Fukui--Hatsugai--Suzuki (FHS) U(1) formulation for a single band, in
  which the Chern number is exactly integral for any grid size,
- the multi-band determinant (Wilson loop) formulation of the original
  implementation, valid when bands do not mix,
- a one-dimensional Wilson loop for Berry/Zak phases along a closed path,
- periodic-grid support: dimensions flagged in ``periodic`` are closed,
  with the link crossing the grid boundary included (needed, e.g., for
  the azimuthal direction of a spherical grid).

References
----------
T. Fukui, Y. Hatsugai and H. Suzuki,
"Chern Numbers in Discretized Brillouin Zone: Efficient Method of
Computing (Spin) Hall Conductances",
J. Phys. Soc. Jpn. 74, 1674 (2005).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = [
    "berry_phase", "berry_curvature", "chern_number",
    "berry_phase_from_hamiltonian",
    "berry_curvature_from_hamiltonian",
    "chern_number_from_hamiltonian",
]


def _to_grid(eigfs):
    """Normalize ``eigfs`` to a complex ndarray.

    Accepts an ndarray in one of the layouts below, or a nested list of
    Qobj kets (anything exposing a ``full()`` method) in the same layout.
    """
    if isinstance(eigfs, np.ndarray):
        return np.asarray(eigfs, dtype=complex)

    def convert(x):
        if hasattr(x, "full"):
            return np.asarray(x.full(), dtype=complex).reshape(-1)
        if isinstance(x, (list, tuple)):
            return [convert(e) for e in x]
        return np.asarray(x, dtype=complex).reshape(-1)

    return np.asarray(convert(eigfs), dtype=complex)


def _check_2d_grid(grid):
    if grid.ndim == 3:
        grid = grid[:, :, None, :]
    if grid.ndim != 4:
        raise ValueError(
            "eigfs must be a 4d array (n0, n1, nocc, hilbert) or a 3d "
            "array (n0, n1, hilbert); got shape {}".format(grid.shape)
        )
    return grid


def berry_curvature(eigfs, periodic=(False, False)):
    """Computes the discretized Berry curvature on a two-dimensional grid of
    parameters.

    Parameters
    ----------
    eigfs : ndarray or nested list of Qobj
        - a 4d array of shape ``(n0, n1, nocc, hilbert)`` holding the
          eigenstates, where the first two indices run over the discrete
          values of the two parameters, the third indexes the occupied
          bands, and the fourth spans the Hilbert space; or
        - a 3d array of shape ``(n0, n1, hilbert)`` for a single band.

    periodic : tuple of bool, default (False, False)
        Flags whether each grid dimension is closed.  For a closed
        dimension the link crossing the grid boundary (index ``n - 1`` to
        index 0) is included and the returned curvature has ``n``
        plaquettes along that dimension; otherwise it has ``n - 1``.

    Returns
    -------
    b_curv : ndarray
        A two dimensional array of the discretized Berry curvature
        (plaquette flux) of shape ``(m0, m1)``, where ``m = n`` for a
        periodic dimension and ``m = n - 1`` otherwise.

    Notes
    -----
    For a single band the plaquette flux is assembled from the
    phase-field link variables in the Fukui--Hatsugai--Suzuki (FHS)
    formulation: each flux is wrapped into :math:`(-\\pi, \\pi]`, which
    makes the summed Chern number exactly integral on any grid size.

    For multiple bands the determinant of the Wilson loop around each
    plaquette is used, as in the original ``qutip.topology``
    implementation.  This is valid when the bands do not mix; the summed
    Chern number then converges to an integer as the grid is refined.
    """
    grid = _check_2d_grid(_to_grid(eigfs))
    n0, n1, nocc, _ = grid.shape
    per0, per1 = bool(periodic[0]), bool(periodic[1])
    m0 = n0 if per0 else n0 - 1
    m1 = n1 if per1 else n1 - 1
    b_curv = np.zeros((m0, m1), dtype=float)

    if nocc == 1:
        psi = grid[:, :, 0, :]
        if per0:
            th0 = np.angle(
                np.einsum("ijh,ijh->ij", psi.conj(), np.roll(psi, -1, axis=0))
            )
        else:
            th0 = np.angle(np.einsum("ijh,ijh->ij", psi[:-1].conj(), psi[1:]))
        if per1:
            th1 = np.angle(
                np.einsum("ijh,ijh->ij", psi.conj(), np.roll(psi, -1, axis=1))
            )
        else:
            th1 = np.angle(np.einsum("ijh,ijh->ij", psi[:, :-1].conj(), psi[:, 1:]))
        for i in range(m0):
            for j in range(m1):
                flux = (
                    th0[i, j]
                    + th1[(i + 1) % n0, j]
                    - th0[i, (j + 1) % n1]
                    - th1[i, j]
                )
                # wrap the plaquette phase into (-pi, pi]
                b_curv[i, j] = (flux + np.pi) % (2 * np.pi) - np.pi
        return b_curv

    edges = [((0, 0), (1, 0)), ((1, 0), (1, 1)), ((1, 1), (0, 1)), ((0, 1), (0, 0))]
    for i in range(m0):
        for j in range(m1):
            rect_prd = np.eye(nocc, dtype=complex)
            for (di0, dj0), (di1, dj1) in edges:
                s = np.einsum(
                    "kh,lh->kl",
                    grid[(i + di0) % n0, (j + dj0) % n1].conj(),
                    grid[(i + di1) % n0, (j + dj1) % n1],
                )
                rect_prd = rect_prd @ s
            b_curv[i, j] = np.angle(np.linalg.det(rect_prd))
    return b_curv


def chern_number(eigfs, periodic=(False, False)):
    """Sums the discretized Berry curvature over the grid to give the
    (first) Chern number.

    Parameters
    ----------
    eigfs : ndarray or nested list of Qobj
        Eigenstates on a two-dimensional grid, as for
        :func:`berry_curvature`.

    periodic : tuple of bool, default (False, False)
        Flags whether each grid dimension is closed; see
        :func:`berry_curvature`.

    Returns
    -------
    chern : float
        The summed Berry curvature divided by :math:`2\\pi`.  With the
        single-band FHS formulation this is exactly integral up to
        floating point roundoff; with the multi-band determinant
        formulation it converges to an integer as the grid is refined.
    """
    return np.sum(berry_curvature(eigfs, periodic=periodic)) / (2 * np.pi)


def berry_phase(eigfs):
    """Computes the Berry (Zak) phase accumulated along a closed
    one-dimensional path.

    Parameters
    ----------
    eigfs : ndarray or nested list of Qobj
        - a 3d array of shape ``(n, nocc, hilbert)`` holding the
          eigenstates along the discrete values of the parameter; or
        - a 2d array of shape ``(n, hilbert)`` for a single band.

        The path is closed: eigenstate at index ``n`` is identified with
        the one at index 0.

    Returns
    -------
    phase : float
        The gauge-invariant phase in :math:`(-\\pi, \\pi]`, given by the
        determinant of the Wilson loop around the path.
    """
    grid = _to_grid(eigfs)
    if grid.ndim == 2:
        grid = grid[:, None, :]
    if grid.ndim != 3:
        raise ValueError(
            "eigfs must be a 3d array (n, nocc, hilbert) or a 2d array "
            "(n, hilbert); got shape {}".format(grid.shape)
        )
    n, nocc, _ = grid.shape
    w = np.eye(nocc, dtype=complex)
    for i in range(n):
        s = np.einsum("kh,lh->kl", grid[i].conj(), grid[(i + 1) % n])
        w = w @ s
    return np.angle(np.linalg.det(w))


# ---------------------------------------------------------------------------
# High-level interfaces: compute directly from a parameterized Hamiltonian
# ---------------------------------------------------------------------------

def _solve_eigenstates(H, args):
    """Solve ``H(*args)`` and return ``(eigenvalues, eigenvectors)``.

    Eigenvectors are returned as an array of shape ``(n_bands, hilbert_dim)``
    and sorted by ascending eigenvalue.
    """
    Hmat = H(*args)
    if hasattr(Hmat, "eigenstates"):
        evals, evecs = Hmat.eigenstates()
        evecs_arr = np.array([e.full().reshape(-1) for e in evecs])
        return np.asarray(evals, dtype=float), evecs_arr
    H_np = np.asarray(Hmat, dtype=complex)
    evals, evecs = np.linalg.eigh(H_np)
    return evals, evecs.T  # evecs.T[i] is the i-th eigenstate


def _adiabatic_sort(evecs_new, evecs_ref):
    """Reorder ``evecs_new`` so it is adiabatically continuous with ``evecs_ref``.

    Uses maximum-overlap matching solved by the Hungarian algorithm
    (``scipy.optimize.linear_sum_assignment``) on the matrix of
    ``|<ref_i | new_j>|``.

    Both inputs have shape ``(n_bands, hilbert_dim)``.
    """
    S = np.abs(np.einsum("ih,jh->ij", evecs_ref.conj(), evecs_new))
    _, col_ind = linear_sum_assignment(-S)
    return evecs_new[col_ind]


def _eigfs_from_hamiltonian_2d(H, params0, params1, bands):
    """Sweep a 2D parameter grid and return eigenstates with adiabatic tracking.

    Returns an array of shape ``(n0, n1, n_bands, hilbert_dim)``.
    """
    params0 = np.asarray(params0, dtype=float)
    params1 = np.asarray(params1, dtype=float)
    n0, n1 = len(params0), len(params1)

    if isinstance(bands, (int, np.integer)):
        bands = [int(bands)]
    bands = list(bands)
    n_bands = len(bands)

    eigfs = None
    for i in range(n0):
        for j in range(n1):
            evals, evecs = _solve_eigenstates(H, (params0[i], params1[j]))

            if eigfs is None:
                hilbert_dim = evecs.shape[1]
                eigfs = np.zeros((n0, n1, n_bands, hilbert_dim), dtype=complex)
                eigfs[i, j] = evecs[bands]  # first point: energy order
            else:
                # Adiabatic reference: left neighbor if available, else top
                if j > 0:
                    ref = eigfs[i, j - 1]
                else:
                    ref = eigfs[i - 1, j]
                evecs_sorted = _adiabatic_sort(evecs, ref)
                eigfs[i, j] = evecs_sorted  # already n_bands matched

    return eigfs


def _eigfs_from_hamiltonian_1d(H, params, bands):
    """Sweep a 1D closed parameter path and return eigenstates with tracking.

    Returns an array of shape ``(n, n_bands, hilbert_dim)``.
    """
    params = np.asarray(params, dtype=float)
    n = len(params)

    if isinstance(bands, (int, np.integer)):
        bands = [int(bands)]
    bands = list(bands)
    n_bands = len(bands)

    eigfs = None
    for i in range(n):
        evals, evecs = _solve_eigenstates(H, (params[i],))

        if eigfs is None:
            hilbert_dim = evecs.shape[1]
            eigfs = np.zeros((n, n_bands, hilbert_dim), dtype=complex)
            eigfs[i] = evecs[bands]
        else:
            evecs_sorted = _adiabatic_sort(evecs, eigfs[i - 1])
            eigfs[i] = evecs_sorted  # already n_bands matched

    return eigfs


def berry_curvature_from_hamiltonian(H, params, bands=0, periodic=(False, False)):
    """Compute the discretized Berry curvature directly from a parameterized
    Hamiltonian.

    This is a high-level convenience wrapper: it sweeps the parameter space,
    solves for eigenstates at every grid point (with adiabatic tracking to
    keep bands continuous across avoided crossings), and then delegates to
    :func:`berry_curvature`.

    Parameters
    ----------
    H : callable
        Hamiltonian function ``H(theta0, theta1)`` returning a QuTiP
        ``Qobj`` (or any object with an ``eigenstates()`` method, or a
        Hermitian NumPy array).
    params : tuple of array_like
        ``(params0, params1)`` — two 1D arrays defining the parameter grid.
    bands : int or list of int, default 0
        Band index (or list of band indices) to include, with ``0`` the
        ground state. A single band uses the FHS formulation (exact integer
        Chern number); multiple bands use the Wilson-loop determinant.
    periodic : tuple of bool, default (False, False)
        Whether each parameter dimension is periodic (closed grid).

    Returns
    -------
    b_curv : ndarray
        Discretized Berry curvature (plaquette flux), shape ``(m0, m1)``
        where ``m = n`` for a periodic dimension and ``m = n - 1`` otherwise.

    Examples
    --------
    >>> from qutip import sigmax, sigmay, sigmaz
    >>> import numpy as np
    >>> def H(th, ph):
    ...     n = [np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)]
    ...     return n[0]*sigmax() + n[1]*sigmay() + n[2]*sigmaz()
    >>> thetas = np.linspace(0, np.pi, 20)
    >>> phis = np.linspace(0, 2*np.pi, 20, endpoint=False)
    >>> F = berry_curvature_from_hamiltonian(H, (thetas, phis),
    ...                                        bands=0, periodic=(False, True))
    """
    params0, params1 = params
    eigfs = _eigfs_from_hamiltonian_2d(H, params0, params1, bands)
    return berry_curvature(eigfs, periodic=periodic)


def chern_number_from_hamiltonian(H, params, bands=0, periodic=(False, False)):
    """Compute the Chern number directly from a parameterized Hamiltonian.

    High-level wrapper around :func:`chern_number`; see
    :func:`berry_curvature_from_hamiltonian` for the parameter description.

    Returns
    -------
    chern : float
        The summed Berry curvature divided by ``2*pi``. With a single band
        (FHS) this is exactly integral up to floating-point roundoff.
    """
    params0, params1 = params
    eigfs = _eigfs_from_hamiltonian_2d(H, params0, params1, bands)
    return chern_number(eigfs, periodic=periodic)


def berry_phase_from_hamiltonian(H, params, bands=0):
    """Compute the Berry (Zak) phase directly from a parameterized Hamiltonian.

    High-level wrapper that sweeps a 1D closed parameter path, solves for
    eigenstates with adiabatic tracking, and calls :func:`berry_phase`.

    Parameters
    ----------
    H : callable
        Hamiltonian function ``H(theta)`` returning a Qobj or Hermitian
        NumPy array.
    params : array_like
        1D array of parameter values along the closed path. The path is
        closed: the last point connects back to the first.
    bands : int or list of int, default 0
        Band index (or list of band indices).

    Returns
    -------
    phase : float
        The gauge-invariant Berry phase in ``(-pi, pi]``.

    Examples
    --------
    >>> from qutip import sigmax, sigmay
    >>> import numpy as np
    >>> def H(k):
    ...     return (1 + 2*np.cos(k))*sigmax() + 2*np.sin(k)*sigmay()
    >>> ks = np.linspace(0, 2*np.pi, 50, endpoint=False)
    >>> zak = berry_phase_from_hamiltonian(H, ks, bands=0)
    """
    eigfs = _eigfs_from_hamiltonian_1d(H, params, bands)
    return berry_phase(eigfs)
