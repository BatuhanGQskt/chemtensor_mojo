"""
Observable-based comparison tests: C DMRG vs Mojo DMRG.

Same pattern as test_mps_c_comparison and test_mpo_c_comparison:
- Load C reference from JSON (test_data/dmrg_results_c_*.json)
- Run Mojo DMRG with the same model parameters (Heisenberg XXZ)
- Compare gauge-invariant observables: energy_final, norm, bond_dims length

C uses random initial MPS (seed 42), Mojo uses product state |0...0>;
so we use tolerance-based comparison for energy. Norm must be ~1 for both.
"""

from collections.list import List
from gpu.host import DeviceContext
from src.state.mps_state import create_product_mps
from src.state.hamiltonians import create_heisenberg_xxz_mpo
from src.algorithms.dmrg import dmrg_two_site, DMRGParams
from src.tests.test_utils import assert_close, assert_equal
from src.tests.algorithms.dmrg_json_loader import load_dmrg_reference, print_dmrg_reference_info
from testing import TestSuite


fn print_separator(title: String) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


fn print_subsection(title: String) -> None:
    print("\n--- " + title + " ---")


# Relaxed energy rtol: C uses random initial MPS, Mojo uses product state.
alias energy_rtol: Float64 = 1e-3
alias norm_atol: Float64 = 1e-5


fn test_dmrg_singlesite_vs_c_reference() raises:
    """Compare Mojo single-site proxy (two-site with chi_max=16) vs C reference.

    C reference: test_data/dmrg_results_c_singlesite.json
    Model: Heisenberg XXZ (J=1, D=1, h=0), nsites=7, d=2, 6 sweeps, chi_max=16.
    """
    print_separator("Test 1: DMRG Single-Site Proxy vs C Reference")

    var ref_path = String("test_data/dmrg_results_c_singlesite.json")
    var dmrg_ref = load_dmrg_reference(ref_path)
    print_dmrg_reference_info(dmrg_ref)

    with DeviceContext() as ctx:
        var basis = List[Int](capacity=dmrg_ref.nsites)
        for _ in range(dmrg_ref.nsites):
            basis.append(0)
        var psi_initial = create_product_mps[DType.float32](ctx, dmrg_ref.d, basis^)
        var H = create_heisenberg_xxz_mpo[DType.float32](
            ctx, dmrg_ref.nsites, J=dmrg_ref.J, D=dmrg_ref.D, h=dmrg_ref.h
        )

        var params = DMRGParams(
            num_sweeps=dmrg_ref.num_sweeps,
            chi_max=dmrg_ref.chi_max,
            eps_trunc=1e-10,
            max_krylov_iter=dmrg_ref.maxiter_lanczos,
            krylov_tol=1e-8,
            energy_tol=1e-8,
            two_site=True,
            verbose=False,
        )

        var result = dmrg_two_site[DType.float32](ctx, H^, psi_initial^, params)
        var E_mojo = result[0]
        var psi = result[1]

        print_subsection("Comparing with C reference")
        assert_close(
            E_mojo, dmrg_ref.energy_final,
            rtol=energy_rtol, atol=1e-6,
            label="energy_final"
        )
        print("  ✓ energy_final: " + String(E_mojo) + " vs C " + String(dmrg_ref.energy_final))

        assert_close(
            Float64(1.0), dmrg_ref.norm,
            rtol=1e-6, atol=norm_atol,
            label="norm (C)"
        )
        # Mojo MPS should be normalized
        assert_equal(len(psi.bond_dims), len(dmrg_ref.bond_dims), "bond_dims length")
        print("  ✓ bond_dims length: " + String(len(psi.bond_dims)))
        print("\n✓ Single-site proxy DMRG matches C reference (observables)")


fn test_dmrg_twosite_vs_c_reference() raises:
    """Compare Mojo two-site DMRG vs C reference.

    C reference: test_data/dmrg_results_c_twosite.json
    Model: Heisenberg XXZ (J=1, D=0.5, h=0.2), nsites=11, d=2, 4 sweeps, chi_max=32.
    """
    print_separator("Test 2: DMRG Two-Site vs C Reference")

    var ref_path = String("test_data/dmrg_results_c_twosite.json")
    var dmrg_ref = load_dmrg_reference(ref_path)
    print_dmrg_reference_info(dmrg_ref)

    with DeviceContext() as ctx:
        var basis = List[Int](capacity=dmrg_ref.nsites)
        for _ in range(dmrg_ref.nsites):
            basis.append(0)
        var psi_initial = create_product_mps[DType.float32](ctx, dmrg_ref.d, basis^)
        var H = create_heisenberg_xxz_mpo[DType.float32](
            ctx, dmrg_ref.nsites, J=dmrg_ref.J, D=dmrg_ref.D, h=dmrg_ref.h
        )

        var params = DMRGParams(
            num_sweeps=dmrg_ref.num_sweeps,
            chi_max=dmrg_ref.chi_max,
            eps_trunc=1e-10,
            max_krylov_iter=dmrg_ref.maxiter_lanczos,
            krylov_tol=1e-8,
            energy_tol=1e-8,
            two_site=True,
            verbose=False,
        )

        var result = dmrg_two_site[DType.float32](ctx, H^, psi_initial^, params)
        var E_mojo = result[0]
        var psi = result[1]

        print_subsection("Comparing with C reference")
        assert_close(
            E_mojo, dmrg_ref.energy_final,
            rtol=energy_rtol, atol=1e-6,
            label="energy_final"
        )
        print("  ✓ energy_final: " + String(E_mojo) + " vs C " + String(dmrg_ref.energy_final))

        assert_equal(len(psi.bond_dims), len(dmrg_ref.bond_dims), "bond_dims length")
        print("  ✓ bond_dims length: " + String(len(psi.bond_dims)))
        print("\n✓ Two-site DMRG matches C reference (observables)")


fn main() raises:
    try:
        TestSuite.discover_tests[__functions_in_module()]().run()
    except e:
        print("Tests failed: " + String(e))
        raise e
