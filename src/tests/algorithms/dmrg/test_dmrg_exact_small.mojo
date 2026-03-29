"""Exact small-system DMRG tests.

Runs DMRG on tiny chains where the ground state energy is known exactly,
so we can assert convergence without C reference data. Used for regression
and to validate the solver on minimal instances.
"""

from collections.list import List
from gpu.host import DeviceContext
from src.state.mps_state import create_product_mps
from src.state.hamiltonians import create_heisenberg_xxz_mpo
from src.algorithms.dmrg import dmrg_two_site, DMRGParams
from src.tests.test_utils import assert_close
from testing import TestSuite


# 4-site Heisenberg XXX (J=1, D=1, h=0): H = -sum(XX+YY+ZZ), 3 bonds.
# Fully polarized state has E = -3.0 (3 bonds × -1); ground state (ferromagnetic).
alias E0_4SITE_XXX: Float64 = -3.0
alias EXACT_ATOL: Float64 = 0.02   # allow ~0.3% error for float32
alias EXACT_RTOL: Float64 = 5e-3


fn neel_basis(nsites: Int, d: Int) -> List[Int]:
    """Néel product state [0,1,0,1,...] for initial MPS."""
    var basis = List[Int](capacity=nsites)
    for i in range(nsites):
        basis.append(i % d)
    return basis^


fn test_dmrg_4site_heisenberg_xxx_exact() raises:
    """DMRG on 4-site Heisenberg XXX must converge to exact E0 = -3.0.
    
    No C reference; validates solver against known exact ground state energy.
    """
    with DeviceContext() as ctx:
        var num_sites = 4
        var d = 2
        var J = 1.0
        var D = 1.0
        var h = 0.0
        var H = create_heisenberg_xxz_mpo[DType.float32](ctx, num_sites, J=J, D=D, h=h)
        var basis = neel_basis(num_sites, d)
        var psi_initial = create_product_mps[DType.float32](ctx, d, basis^)

        var params = DMRGParams(
            num_sweeps=6,
            chi_max=8,
            eps_trunc=1e-10,
            max_krylov_iter=20,
            krylov_tol=1e-8,
            energy_tol=1e-8,
            two_site=True,
            verbose=False,
        )

        var result = dmrg_two_site[DType.float32](ctx, H^, psi_initial^, params)
        var E = result[0]

        assert_close(
            E, E0_4SITE_XXX,
            rtol=EXACT_RTOL, atol=EXACT_ATOL,
            label="4-site XXX energy vs exact"
        )


fn main() raises:
    try:
        TestSuite.discover_tests[__functions_in_module()]().run()
    except e:
        print("Tests failed: " + String(e))
        raise e
