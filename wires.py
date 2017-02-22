import discretizer
import kwant
import numpy as np
from scipy.constants import hbar, eV
import sympy
from sympy.physics.quantum import TensorProduct as kr

sx, sy, sz = [sympy.physics.matrices.msigma(i) for i in range(1, 4)]
s0 = sympy.eye(2)
s0sz = np.kron(s0, sz)
s0s0 = np.kron(s0, s0)


def make_1d_wire(a=10, L=1, verbose=False):
    """Makes a hexagonal shaped 3D wire.

    Parameters:
    -----------
    a : int
        Lattice constant in nm.
    L : int
        Length of the wire in units of the lattice constant, L=None if infinite
        wire.
    vebose : bool
        Prints the discretized Hamiltonian.

    Returns:
    --------
    sys : kwant.builder.(In)finiteSystem object
        The finalized (in)finite system.
    """
    k_x, k_y, k_z = discretizer.momentum_operators
    t, B_x, B_y, B_z, mu_B, Delta, mu, alpha, g = sympy.symbols('t B_x B_y B_z mu_B Delta mu alpha g', real=True)

    hamiltonian = ((t * k_x**2 - mu) * kr(s0, sz) +
                   alpha * (- k_x * kr(sy, sz)) +
                   0.5 * g * mu_B * (B_x * kr(sx, s0) + B_y * kr(sy, s0) + B_z * kr(sz, s0)) +
                   Delta * kr(s0, sx))

    tb = discretizer.Discretizer(hamiltonian, lattice_constant=a, verbose=verbose)

    if L is None:
        sys = kwant.Builder(kwant.TranslationalSymmetry((-a, )))
        L = 1
    else:
        sys = kwant.Builder()

    sys[[tb.lattice(x) for x in range(L)]] = tb.onsite

    for hop, val in tb.hoppings.items():
        sys[hop] = val

    return sys.finalized()


def make_2d_wire(a=10, W=10, L=1, holes=True, verbose=False):
    """Makes a hexagonal shaped 3D wire.

    Parameters:
    -----------
    a : int
        Lattice constant in nm.
    W : int
        Width of the wire in units of the lattice constant.
    L : int
        Length of the wire in units of the lattice constant, L=None if infinite
        wire.
    vebose : bool
        Prints the discretized Hamiltonian.

    Returns:
    --------
    sys : kwant.builder.(In)finiteSystem object
        The finalized (in)finite system.
    """
    k_x, k_y, k_z = discretizer.momentum_operators
    t, B_x, B_y, B_z, mu_B, Delta, mu, alpha, g = sympy.symbols('t B_x B_y B_z mu_B Delta mu alpha g', real=True)
    k =  sympy.sqrt(k_x**2+k_y**2)
    hamiltonian = ((t * k**2 - mu) * kr(s0, sz) +
                   alpha * (k_y * kr(sx, sz) - k_x * kr(sy, sz)) +
                   0.5 * g * mu_B * (B_x * kr(sx, s0) + B_y * kr(sy, s0) + B_z * kr(sz, s0)) +
                   Delta * kr(s0, sx))

    tb = discretizer.Discretizer(hamiltonian, lattice_constant=a, verbose=verbose)

    if L is None:
        sys = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)))
        L = 1
    else:
        sys = kwant.Builder()


    sys[[tb.lattice(x, y) for x in range(L) for y in range(W)]] = tb.onsite

    def peierls(val, ind):
        def phase(s1, s2, p):
            x, y = s1.pos
            A = lambda p, x, y: [-p.B_z * y, 0, p.B_x * y]
            A_site = A(p, x, y)[ind]
            A_site *= a * 1e-18 * eV / hbar
            return np.cos(A_site) * s0s0 - 1j * np.sin(A_site) * s0sz

        def with_phase(s1, s2, p):
            if p.orbital:
                return phase(s1, s2, p).dot(val(s1, s2, p))
            else:
                return val(s1, s2, p)
        return with_phase


    for hop, val in tb.hoppings.items():
        ind = np.argmax(hop.delta)
        sys[hop] = peierls(val, ind)
    return sys.finalized()


def make_3d_wire(a=10, R=50, L=None, holes=True, verbose=False):
    """Makes a hexagonal shaped 3D wire.

    Parameters:
    -----------
    a : int
        Lattice constant in nm.
    R : int
        Radius of the wire in units in units of nm.
    L : int
        Length of the wire in units of nm, L=None if infinite wire.
    holes : bool
        True if PHS, False if no holes and only in spin space.
    vebose : bool
        Prints the discretized Hamiltonian.

    Returns:
    --------
    syst : kwant.builder.(In)finiteSystem object
        The finalized (in)finite system.
    """
    k_x, k_y, k_z = discretizer.momentum_operators
    x, y, z = discretizer.coordinates
    t, B_x, B_y, B_z, mu_B, Delta, mu, alpha, g, V = sympy.symbols('t B_x B_y B_z mu_B Delta mu alpha g V', real=True)
    k =  sympy.sqrt(k_x**2+k_y**2+k_z**2)
    if holes:
        hamiltonian = ((t * k**2 - mu - V(x, y, z)) * kr(s0, sz) +
                       alpha * (k_y * kr(sx, sz) - k_x * kr(sy, sz)) +
                       0.5 * g * mu_B * (B_x * kr(sx, s0) + B_y * kr(sy, s0) + B_z * kr(sz, s0)) +
                       Delta * kr(s0, sx))
    else:
        hamiltonian = ((t * k**2 - mu - V(x, y, z)) * s0 + alpha * (k_y * sx - k_x * sy) +
                       0.5 * g * mu_B * (B_x * sx + B_y * sy + B_z * sz) +
                       Delta * s0)

    tb = discretizer.Discretizer(hamiltonian, lattice_constant=a,
                     verbose=verbose)
    syst = kwant.Builder(kwant.TranslationalSymmetry((-a, 0, 0)))

    if L is None:
        L = 1

    def hexagon(pos):
        (x, y, z) = pos
        return (y > -R and y < R and y > -2 * (R - z) and y < -2 *
                (z - R) and y < 2 * (z + R) and y > -2 *
                (z + R) and x >= 0 and x < L)

    syst[tb.lattice.shape(hexagon, (0, 0, 0))] = tb.onsite

    def peierls(val, ind):
        def phase(s1, s2, p):
            x, y, z = s1.pos
            A = lambda p, x, y, z: [p.B_y * z - p.B_z * y, 0, p.B_x * y]
            A_site = A(p, x, y, z)[ind]
            A_site *= a * 1e-18 * eV / hbar
            if holes:
                return np.cos(A_site) * s0s0 - 1j * np.sin(A_site) * s0sz
            else:
                return np.exp(-1j * A_site)

        def with_phase(s1, s2, p):
            if p.orbital:
                try:
                    return phase(s1, s2, p).dot(val(s1, s2, p))
                except AttributeError:
                    return phase(s1, s2, p) * val(s1, s2, p)
            else:
                return val(s1, s2, p)
        return with_phase

    for hop, val in tb.hoppings.items():
        ind = np.argmax(hop.delta)
        syst[hop] = peierls(val, ind)
    return syst.finalized()



def make_3d_wire_external_sc(a=10, r1=50, r2=70, phi=135, angle=45, finalized=True):
    """Makes a hexagonal shaped 3D wire with external superconductor.

    Parameters:
    -----------
    a : int
        Lattice constant in nm.
    r1 : float
        Diameter of wire part in nm.
    r2 : float
        Diameter of wire plus superconductor part in nm.
    phi : float
        Coverage angle of superconductor in degrees.
    angle : float
        Angle of the superconductor w.r.t. the y-axis in degrees.
    finalized : bool
        Return a finalized system if True or kwant.Builder object
        if False.

    Returns:
    --------
    syst : kwant.builder.InfiniteSystem object
        The finalized infinite system.
    """
    k_x, k_y, k_z = discretizer.momentum_operators
    x, y, z = discretizer.coordinates
    t, B_x, B_y, B_z, mu_B, Delta, mu, alpha, g, V = sympy.symbols('t B_x B_y B_z mu_B Delta mu alpha g V', real=True)
    t_interface = sympy.symbols('t_interface', real=True)
    k =  sympy.sqrt(k_x**2+k_y**2+k_z**2)

    hamiltonian = ((t * k**2 - mu - V(x, y, z)) * kr(s0, sz) +
                   alpha * (k_y * kr(sx, sz) - k_x * kr(sy, sz)) +
                   0.5 * g * mu_B * (B_x * kr(sx, s0) + B_y * kr(sy, s0) + B_z * kr(sz, s0)) +
                   Delta * kr(s0, sx))

    def cylinder_sector(r1, r2=0, phi=360, angle=angle):
        phi *= np.pi / 360
        angle *= np.pi / 180
        r1sq, r2sq = r1 ** 2, r2 ** 2
        def sector(pos):
            x, y, z = pos
            n = (y + 1j * z) * np.exp(1j * angle)
            y, z = n.real, n.imag
            rsq = y ** 2 + z ** 2
            return r2sq <= rsq < r1sq and z >= np.cos(phi) * np.sqrt(rsq)
        r_mid = (r1 + r2) / 2
        return sector, (0, r_mid * np.sin(angle), r_mid * np.cos(angle))

    args = dict(lattice_constant=a)
    tb_normal = discretizer.Discretizer(hamiltonian.subs(Delta, 0), **args)
    tb_sc = discretizer.Discretizer(hamiltonian, **args)
    tb_interface = discretizer.Discretizer(hamiltonian.subs(t, t_interface), **args)
    lat = tb_normal.lattice
    syst = kwant.Builder(kwant.TranslationalSymmetry((-a, 0, 0)))

    shape_normal = cylinder_sector(r1=r1, angle=angle)
    shape_sc = cylinder_sector(r1=r2, r2=r1, phi=phi, angle=angle)

    syst[lat.shape(*shape_normal)] = tb_normal.onsite
    syst[lat.shape(*shape_sc)] = tb_sc.onsite
    sc_sites = list(syst.expand(lat.shape(*shape_sc)))

    def peierls(val, ind, a):
        def phase(s1, s2, p):
            x, y, z = s1.pos
            A = lambda p, x, y, z: [p.B_y * z - p.B_z * y, 0, p.B_x * y]
            A_site = A(p, x, y, z)[ind]
            if p.A_correction:
                A_sc = [A(p, *site.pos) for site in sc_sites]
                A_site -= np.mean(A_sc, axis=0)[ind]
            A_site *= a * 1e-18 * eV / hbar
            return np.cos(A_site) * s0s0 - 1j * np.sin(A_site) * s0sz

        def with_phase(s1, s2, p):
            if p.orbital:
                return phase(s1, s2, p).dot(val(s1, s2, p))
            else:
                return val(s1, s2, p)
        return with_phase


    def hoppingkind_in_shape(hop, shape, syst):
        """Returns an HoppingKind iterator for hoppings in shape."""
        def in_shape(site1, site2, shape):
            return shape[0](site1.pos) and shape[0](site2.pos)
        hoppingkind = kwant.HoppingKind(hop.delta, hop.family_a)(syst)
        return ((i, j) for (i, j) in hoppingkind if in_shape(i, j, shape))


    def hoppingkind_at_interface(hop, shape1, shape2, syst):
        """Returns an HoppingKind iterator for hoppings at an interface between
           shape1 and shape2."""
        def at_interface(site1, site2, shape1, shape2):
            return ((shape1[0](site1.pos) and shape2[0](site2.pos)) or 
                    (shape2[0](site1.pos) and shape1[0](site2.pos)))
        hoppingkind = kwant.HoppingKind(hop.delta, hop.family_a)(syst)
        return ((i, j) for (i, j) in hoppingkind if at_interface(i, j, shape1, shape2))


    for hop, func in tb_normal.hoppings.items():
        # Add hoppings in normal parts of wire and lead with Peierls substitution
        ind = np.argmax(hop.delta) # Index of direction of hopping
        syst[hoppingkind_in_shape(hop, shape_normal, syst)] = peierls(func, ind, a)

        
    for hop, func in tb_sc.hoppings.items():
        # Add hoppings in superconducting parts of wire and lead with Peierls substitution
        ind = np.argmax(hop.delta) # Index of direction of hopping
        syst[hoppingkind_in_shape(hop, shape_sc, syst)] = peierls(func, ind, a)


    for hop, func in tb_interface.hoppings.items():
        # Add hoppings at the interface of superconducting parts and normal parts of wire and lead
        ind = np.argmax(hop.delta) # Index of direction of hopping
        syst[hoppingkind_at_interface(hop, shape_sc, shape_normal, syst)] = peierls(func, ind, a)

    
    if finalized:
        return syst.finalized()
    else:
        return syst