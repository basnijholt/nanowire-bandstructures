import kwant
import sympy

a = sympy.symbols("a")
phi_0 = sympy.symbols("phi_0")
ri = sympy.symbols("x_i y_i z_i")
rj = sympy.symbols("x_j y_j z_j")


def get_phase(A, dim):
    """Calculate Peierl's phase phi_ij

    Parameters
    ----------
    A : string
        String representing vector potential. For example: "[-B_z * y, 0, 0]"
        Must be in 2D/3D cartesian coords.

    Returns:
    phase : sympy.Expr
        (pi/phi_0) * integrate_{i->j} (A(r).r)
    """
    A = kwant.continuum.sympify(A)
    x, y, z = kwant.continuum.position_operators

    xi, yi, zi = ri
    xj, yj, zj = rj

    t = sympy.symbols("_t_internal_for_integration")

    subs = {
        x: (1 - t) * xi + t * xj,
        y: (1 - t) * yi + t * yj,
        z: (1 - t) * zi + t * zj,
    }

    output = [xj - xi, yj - yi, zj - zi][:dim]
    for i, Ai in enumerate(A):
        if isinstance(Ai, sympy.Expr):
            Ai = Ai.subs(subs)
        output[i] *= sympy.integrate(Ai, (t, 0, 1))

    return (2 * sympy.pi / phi_0) * sum(output)


def apply(tb_hamiltonian, coords, *, A, signs=None):
    """Modify tight-binding Hamiltonian to include Peierl's substitution.


    Parameters
    ----------
    A : string
        String representing vector potential. For example: "[-B_z * y, 0, 0]"
        A must be in 2D/3D cartesian coords.

    coords : sequence of strings
        Discrete coordinates.

    signs : sequence of integers
        The relative signs of the phase-factors for the different orbitals.

    Returns
    -------
    discrete_hamiltonian: dict
        Discrete Hamiltonian after with Peierl's substitution.
    """
    tb_hamiltonian = tb_hamiltonian.copy()

    if not isinstance(A, str):
        raise ValueError("Vector potential should be a string.")

    phase_ij = get_phase(A, dim=len(coords))
    if signs:
        phase_factors = [sympy.exp(s * sympy.I * phase_ij) for s in signs]
        phase_factors = sympy.diag(*phase_factors)
    else:
        phase_factors = sympy.exp(sympy.I * phase_ij)

    target = [kwant.continuum.sympify(c) for c in sorted(coords)]
    target_subs = {xi: pos for xi, pos in zip(ri, target)}

    for offset, hopping in tb_hamiltonian.items():
        hopping = hopping * phase_factors
        source = [c + n * a for c, n in zip(target, offset)]
        source_subs = {xj: pos for xj, pos in zip(rj, source)}
        tb_hamiltonian[offset] = hopping.subs(target_subs).subs(source_subs)

    return tb_hamiltonian
