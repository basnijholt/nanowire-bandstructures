import cmath
import functools
import math
import re
import types
from typing import Union

import kwant
import kwant.continuum
import numpy as np
import scipy.constants
from scipy.constants import physical_constants
from shapely.geometry import Point, Polygon
from shapely import speedups

import peierls

speedups.enable()


def _V(*x):
    return 0


def memoize(obj):
    cache = obj.cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return cache[key]

    return memoizer


# (Fundamental) constants definitions in nm and meV.
constants = {
    "m_eff": 0.015 * scipy.constants.m_e / scipy.constants.eV / (1e9) ** 2 * 1e3,
    "phi_0": 2 * physical_constants["mag. flux quantum"][0] * (1e9) ** 2,
    "mu_B": physical_constants["Bohr magneton in eV/T"][0] * 1e3,
    "hbar": scipy.constants.hbar / scipy.constants.eV * 1e3,
    "exp": cmath.exp,
}


# Hamiltonian and system definition


@memoize
def get_sympy_hamiltonian(subs, dim=1, with_holes=True):
    ham = (
        "(0.5 * hbar**2 * (k_x**2 + k_y**2 + k_z**2) / m_eff - mu + V) * kron(sigma_0, sigma_z)"
        "+ alpha * (k_y * kron(sigma_x, sigma_z) - k_x * kron(sigma_y, sigma_z))"
        "+ 0.5 * g * mu_B * (B_x * kron(sigma_x, sigma_0) + B_y * kron(sigma_y, sigma_0) + B_z * kron(sigma_z, sigma_0))"
        "+ Delta * kron(sigma_0, sigma_x)"
    )

    if dim == 1:
        ham = ham.replace("k_y", "0")
        ham = ham.replace("k_z", "0")
    elif dim == 2:
        ham = ham.replace("k_z", "0")

    if not with_holes:
        ham = re.sub(r"kron\((sigma_[xyz0]), sigma_[xzy0]\)", r"\1", ham)

    ham = kwant.continuum.sympify(ham, locals=subs)
    return ham


def get_template(a, subs=None, dim=1, with_holes=True):
    ham = get_sympy_hamiltonian(subs if subs is not None else {}, dim, with_holes)
    tb_ham, coords = kwant.continuum.discretize_symbolic(ham)
    if dim > 1:
        if dim == 2:
            vector_potential = "[-B_z * y, 0]"
        elif dim == 3:
            vector_potential = "[B_y * z - B_z * y, 0, B_x * y]"
        # TODO: check if signs are correc
        signs = [1, 1] if not with_holes else [1, 1, -1, -1]
        tb_ham = peierls.apply(tb_ham, coords, A=vector_potential, signs=signs)
    template = kwant.continuum.build_discretized(tb_ham, grid=a, coords=coords)
    return template


def get_triangle(R):
    return Polygon([(-R, -R), (R, -R), (0, R)])


def get_shape(R, L0=0, L1=None, shape: Union[str, Polygon] = "hexagon", dim=3):
    if L1 is None:
        start_coords = (L0, 0, 0)[:dim]
    else:
        start_coords = ((L0 + L1) / 2, 0, 0)[:dim]

    def _shape(site):
        if dim == 1:
            x = site.pos[0]
            is_in_shape = True
        elif dim == 2:
            (x, y) = site.pos
            is_in_shape = abs(y) < R
        elif dim == 3:
            (x, y, z) = site.pos
            if shape == "hexagon":
                is_in_shape = (
                    y > -R
                    and y < R
                    and y > -2 * (R - z)
                    and y < -2 * (z - R)
                    and y < 2 * (z + R)
                    and y > -2 * (z + R)
                )
            elif shape == "square":
                is_in_shape = abs(z) < R and abs(y) < R
            elif isinstance(shape, Polygon):
                is_in_shape = shape.buffer(1e-10).contains(Point(y, z))
            else:
                raise ValueError(
                    "Only 'hexagon' and 'square', and `shapely.geometry.Polygon` allowed."
                )
        return is_in_shape and ((L1 is None) or (x >= 0 and x < L1))

    return _shape, start_coords


@memoize
def make_wire(
    a,
    L,
    r=None,
    shape="hexagon",
    dim=1,
    left_lead=True,
    right_lead=True,
    subs_right=None,
    subs_left=None,
    subs_mid=None,
):
    """Create a 3D wire with one SC lead.

    Parameters
    ----------
    a : int
        Discretization constant in nm.
    L : int
        Length of wire in nm.
    r : int
        Radius of the wire in nm.
    shape : str
        Either `hexagon` or `square` shaped cross section.

    Returns
    -------
    syst : kwant.builder.FiniteSystem
        The finilized kwant system.

    Examples
    --------
    This doesn't use default parameters because the variables need to be saved,
    to a file. So I create a dictionary that is passed to the function.

    >>> syst_params = dict(a=10, L=100, ...)
    >>> syst, hopping = make_3d_wire(**syst_params)

    """
    subs_right = subs_right or {"V": "V_left", "Delta": "0"}
    subs_left = subs_left or {"V": "V_right"}
    subs_mid = subs_mid or {}

    syst = kwant.Builder()

    template = get_template(a, subs_mid, dim)
    shape_coords = get_shape(r, 0, L, shape, dim)
    syst.fill(template, *shape_coords)

    if left_lead:
        lead = make_lead_SC(a, r, shape, dim, subs=subs_left)
        syst.attach_lead(lead.reversed())
    if right_lead:
        lead = make_lead_normal(a, r, shape, dim, subs=subs_right)
        syst.attach_lead(lead)

    return syst.finalized()


def make_lead_normal(a, r=None, shape="hexagon", dim=1, with_holes=True, subs={}):
    sz = np.array([[1, 0], [0, -1]])
    cons_law = np.kron(np.eye(2), -sz)
    shape_lead = get_shape(r, shape=shape, dim=dim)
    symmetry = kwant.TranslationalSymmetry((a, 0, 0)[:dim])
    lead = kwant.Builder(symmetry, conservation_law=cons_law if with_holes else None)
    template = get_template(a, dict(subs, Delta=0), dim, with_holes)
    lead.fill(template, *shape_lead)
    return lead


def make_lead_SC(a, r=None, shape="hexagon", dim=1, with_holes=True, subs={}):
    shape_lead = get_shape(r, shape=shape, dim=dim)
    symmetry = kwant.TranslationalSymmetry((a, 0, 0)[:dim])
    lead = kwant.Builder(symmetry)
    template = get_template(a, subs, dim, with_holes)
    lead.fill(template, *shape_lead)
    return lead


def andreev_conductance(smatrix, normal_lead=1):
    """The Andreev conductance is N - R_ee + R_he."""
    r_ee = smatrix.transmission((normal_lead, 0), (normal_lead, 0))
    r_he = smatrix.transmission((normal_lead, 1), (normal_lead, 0))
    N_e = smatrix.submatrix((normal_lead, 0), (normal_lead, 0)).shape[0]
    return N_e - r_ee + r_he


def bands(lead, params, ks=None):
    if ks is None:
        ks = np.linspace(-3, 3)

    bands = kwant.physics.Bands(lead, params=params)

    if isinstance(ks, (float, int)):
        return bands(ks)
    else:
        return np.array([bands(k) for k in ks])


def translation_ev(h, t, tol=1e6):
    """Compute the eigenvalues of the translation operator of a lead.

    Adapted from kwant.physics.leads.modes.

    Parameters
    ----------
    h : numpy array, real or complex, shape (N, N) The unit cell
        Hamiltonian of the lead unit cell.
    t : numpy array, real or complex, shape (N, M)
        The hopping matrix from a lead cell to the one on which self-energy
        has to be calculated (and any other hopping in the same direction).
    tol : float
        Numbers and differences are considered zero when they are smaller
        than `tol` times the machine precision.

    Returns
    -------
    ev : numpy array
        Eigenvalues of the translation operator in the form lambda=r*exp(i*k),
        for |r|=1 they are propagating modes.
    """
    a, b = kwant.physics.leads.setup_linsys(h, t, tol, None).eigenproblem
    ev = kwant.physics.leads.unified_eigenproblem(a, b, tol=tol)[0]
    return ev


def cell_mats(lead, params, bias=0):
    h = lead.cell_hamiltonian(params=params)
    h -= bias * np.identity(len(h))
    t = lead.inter_cell_hopping(params=params)
    return h, t


@memoize
def gap_minimizer(lead, params, energy):
    """Function that minimizes a function to find the band gap.
    This objective function checks if there are progagating modes at a
    certain energy. Returns zero if there is a propagating mode.

    Parameters
    ----------
    lead : kwant.builder.InfiniteSystem object
        The finalized infinite system.
    params : dict
        A dict that is used to store Hamiltonian parameters.
    energy : float
        Energy at which this function checks for propagating modes.

    Returns
    -------
    minimized_scalar : float
        Value that is zero when there is a propagating mode.
    """
    h, t = cell_mats(lead, params, bias=energy)
    ev = translation_ev(h, t)
    norm = (ev * ev.conj()).real
    return np.min(np.abs(norm - 1))


@memoize
def find_gap(lead, params, tol=1e-6):
    """Finds the gapsize by peforming a binary search of the modes with a
    tolarance of tol.

    Parameters
    ----------
    lead : kwant.builder.InfiniteSystem object
        The finalized infinite system.
    params : dict
        A dict that is used to store Hamiltonian parameters.
    tol : float
        The precision of the binary search.

    Returns
    -------
    gap : float
        Size of the gap.
    """
    lim = [0, np.abs(bands(lead, params, ks=0)).min()]
    if gap_minimizer(lead, params, energy=0) < 1e-15:
        # No band gap
        gap = 0
    else:
        while lim[1] - lim[0] > tol:
            energy = sum(lim) / 2
            par = gap_minimizer(lead, params, energy)
            if par < 1e-10:
                lim[1] = energy
            else:
                lim[0] = energy
        gap = sum(lim) / 2
    return gap


def get_cross_section(syst, pos, direction):
    coord = np.array([s.pos for s in syst.sites if s.pos[direction] == pos])
    cross_section = np.delete(coord, direction, 1)
    return cross_section


def get_densities(lead, k, params):
    xy = get_cross_section(lead, pos=0, direction=0)
    h, t = lead.cell_hamiltonian(params=params), lead.inter_cell_hopping(params=params)
    h_k = h + t * np.exp(1j * k) + t.T.conj() * np.exp(-1j * k)

    vals, vecs = np.linalg.eigh(h_k)
    indxs = np.argsort(np.abs(vals))
    vecs = vecs[:, indxs]
    vals = vals[indxs]

    norbs = lat_from_syst(lead).norbs
    densities = np.linalg.norm(vecs.reshape(-1, norbs, len(vecs)), axis=1) ** 2
    return xy, vals, densities.T


def plot_wfs_in_cross_section(lead, params, k, num_bands=40):
    import holoviews as hv

    xy, energies, densities = get_densities(lead, k, params)
    wfs = [
        kwant.plotter.mask_interpolate(xy, density, oversampling=1)[0]
        for density in densities[:num_bands]
    ]
    ims = {E: hv.Image(wf) for E, wf in zip(energies, wfs)}
    return hv.HoloMap(ims, kdims=[hv.Dimension("E", unit="meV")])


def get_h_k(lead, params):
    h, t = cell_mats(lead, params)
    h_k = lambda k: h + t * np.exp(1j * k) + t.T.conj() * np.exp(-1j * k)  # noqa: E731
    return h_k


def lat_from_syst(syst):
    lats = {s.family for s in syst.sites}
    if len(lats) > 1:
        raise Exception("No unique lattice in the system.")
    return list(lats)[0]


# Scales

SI_constants = types.SimpleNamespace(
    hbar=scipy.constants.hbar,
    meV=scipy.constants.eV * 1e-3,
    m_eff=scipy.constants.m_e * 0.015,
)


def calc_k_F(mu, SI_constants=SI_constants):
    c = SI_constants
    return np.sqrt(2 * mu * c.m_eff / c.hbar ** 2)


def calc_v_F(mu, SI_constants=SI_constants):
    c = SI_constants
    k_F = calc_k_F(mu, SI_constants)
    return c.hbar * k_F / c.m_eff


def calc_fermi_wavelength(mu, SI_constants=SI_constants):
    k_F = calc_k_F(mu, SI_constants)
    return 2 * np.pi / k_F


# Below, non-default arguments have units of meV and nm


def calc_alpha(l_R, SI_constants=SI_constants):
    c = SI_constants
    l_R *= 1e-9
    return c.hbar ** 2 / (c.m_eff * l_R) / (1e-9 * c.meV)


def calc_mfp_analytic(mu, a, disorder, dim=1, SI_constants=SI_constants):
    c = SI_constants
    mu *= c.meV
    a *= 1e-9
    w = disorder * c.meV / 2
    v_F = calc_v_F(mu, c)
    if dim == 1:
        rho = a * np.sqrt(c.m_eff / (2 * mu)) / (np.pi * c.hbar)
    elif dim == 2:
        rho = a ** 2 * c.m_eff / (np.pi * c.hbar ** 2)
    mfp = c.hbar * v_F / (2 * np.pi) * (rho * w ** 2 / 3) ** (-1)
    return mfp * 1e9


def calc_disorder_from_mfp(mfp, a, mu, dim=1, SI_constants=SI_constants):
    if math.isinf(mfp):
        return 0
    c = SI_constants
    mfp *= 1e-9
    a *= 1e-9
    mu *= c.meV
    v_F = calc_v_F(mu, c)
    if dim == 1:
        rho = a * np.sqrt(c.m_eff / (2 * mu)) / (np.pi * c.hbar)
    elif dim == 2:
        rho = a ** 2 * c.m_eff / (np.pi * c.hbar ** 2)

    return np.sqrt((3 * c.hbar * v_F) / (2 * np.pi * mfp * rho)) / (c.meV / 2)
