"""
Observable tests for MPS when MPS is involved in contractions (norms, overlaps,
MPS-MPO inner product, apply_mpo, mps_overlap).

These tests use the same setup as the contraction benchmark (random MPS with
1/sqrt(nelem) scaling, Ising MPO) and assert that observables stay in O(1) ranges.
They would fail if e.g. random MPS were unscaled (huge norms/overlaps) or if
environment/contraction logic were wrong.
"""

from collections.list import List
from gpu.host import DeviceContext
from src.state.mps_state import (
    MatrixProductState,
    create_uniform_mps,
    mps_norm,
)
from src.state.mpo_state import MatrixProductOperator
from src.state.hamiltonians import create_ising_1d_mpo
from src.state.environments import (
    expectation_value_two_mps,
    mps_overlap,
    apply_mpo,
)
from src.tests.test_utils import assert_close
from testing import TestSuite


# Same default params as bench_contractions for consistency
alias NSITES = 6
alias D = 2
alias CHI_MAX = 16

# Bounds for O(1) observables (generous; would catch unscaled random MPS or wrong contractions)
alias NORM_MAX = 100.0
alias OVERLAP_ABS_MAX = 50.0
alias INNER_RESULT_ABS_MAX = 50.0
alias APPLY_NORM_MAX = 100.0


fn _bond_dims_list(nsites: Int, chi_max: Int) -> List[Int]:
    """Return [1, chi_max, ..., chi_max, 1] of length nsites+1."""
    var bonds = List[Int](capacity=nsites + 1)
    bonds.append(1)
    for _ in range(nsites - 1):
        bonds.append(chi_max)
    bonds.append(1)
    return bonds^


fn test_random_mps_norm_bounded() raises:
    """Random MPS (scaled by 1/sqrt(nelem) per site) has norm O(1)."""
    var ctx = DeviceContext()
    var bond_dims = _bond_dims_list(NSITES, CHI_MAX)
    var psi = create_uniform_mps[DType.float32](ctx, NSITES, D, bond_dims^)
    var n = mps_norm[DType.float32](psi, ctx)
    if n < 1e-6 or n > NORM_MAX:
        raise Error("Random MPS norm out of range: " + String(n) + " (expected O(1), max " + String(NORM_MAX) + ")")
    print("  ✓ random MPS norm = " + String(n) + " (bounded)")


fn test_random_mps_overlap_bounded() raises:
    """Overlap <chi|psi> of two random MPS is O(1)."""
    var ctx = DeviceContext()
    var bond_dims_psi = _bond_dims_list(NSITES, CHI_MAX)
    var bond_dims_chi = _bond_dims_list(NSITES, CHI_MAX)
    var psi = create_uniform_mps[DType.float32](ctx, NSITES, D, bond_dims_psi^)
    var chi = create_uniform_mps[DType.float32](ctx, NSITES, D, bond_dims_chi^)
    var ov = mps_overlap[DType.float32](chi, psi, ctx)
    if abs(ov) > OVERLAP_ABS_MAX:
        raise Error("MPS-MPS overlap out of range: " + String(ov) + " (expected O(1), max " + String(OVERLAP_ABS_MAX) + ")")
    print("  ✓ mps_overlap(chi, psi) = " + String(ov) + " (bounded)")


fn test_mps_mpo_inner_bounded() raises:
    """<chi|MPO|psi> with random MPS and Ising MPO is O(1)."""
    var ctx = DeviceContext()
    var mpo = create_ising_1d_mpo[DType.float32](ctx, num_sites=NSITES, J=1.0, h_longitudinal=0.0, g_transverse=0.0)
    var bond_dims_psi = _bond_dims_list(NSITES, CHI_MAX)
    var bond_dims_chi = _bond_dims_list(NSITES, CHI_MAX)
    var psi = create_uniform_mps[DType.float32](ctx, NSITES, D, bond_dims_psi^)
    var chi = create_uniform_mps[DType.float32](ctx, NSITES, D, bond_dims_chi^)
    var inner = expectation_value_two_mps[DType.float32](chi, mpo, psi, ctx)
    if abs(inner) > INNER_RESULT_ABS_MAX:
        raise Error("MPS-MPO inner product out of range: " + String(inner) + " (expected O(1), max " + String(INNER_RESULT_ABS_MAX) + ")")
    print("  ✓ expectation_value_two_mps(chi, mpo, psi) = " + String(inner) + " (bounded)")


fn test_mps_mpo_apply_norm_bounded() raises:
    """Norm of apply_mpo(mpo, psi) with random MPS and Ising MPO is O(1)."""
    var ctx = DeviceContext()
    var mpo = create_ising_1d_mpo[DType.float32](ctx, num_sites=NSITES, J=1.0, h_longitudinal=0.0, g_transverse=0.0)
    var bond_dims = _bond_dims_list(NSITES, CHI_MAX)
    var psi = create_uniform_mps[DType.float32](ctx, NSITES, D, bond_dims^)
    var op_psi = apply_mpo[DType.float32](mpo, psi, ctx)
    var n = mps_norm[DType.float32](op_psi, ctx)
    if n < 1e-6 or n > APPLY_NORM_MAX:
        raise Error("apply_mpo norm out of range: " + String(n) + " (expected O(1), max " + String(APPLY_NORM_MAX) + ")")
    print("  ✓ mps_norm(apply_mpo(mpo, psi)) = " + String(n) + " (bounded)")


fn test_self_overlap_equals_norm_sq() raises:
    """<psi|psi> should equal ||psi||^2 for any MPS."""
    var ctx = DeviceContext()
    var bond_dims = _bond_dims_list(NSITES, CHI_MAX)
    var psi = create_uniform_mps[DType.float32](ctx, NSITES, D, bond_dims^)
    var n = mps_norm[DType.float32](psi, ctx)
    var ov = mps_overlap[DType.float32](psi, psi, ctx)
    assert_close(ov, n * n, rtol=1e-4, atol=1e-5, label="<psi|psi> vs ||psi||^2")
    print("  ✓ <psi|psi> = " + String(ov) + " = ||psi||^2 = " + String(n * n))


fn main() raises:
    try:
        TestSuite.discover_tests[__functions_in_module()]().run()
    except e:
        print("Tests failed: " + String(e))
        raise e
