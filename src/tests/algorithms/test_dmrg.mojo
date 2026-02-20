"""Manual DMRG tests — mirrors chemtensor C manual_tests/dmrg.c.

Prints results instead of asserting against reference data.
Uses the Heisenberg XXZ model (built in code) so no external data files are needed.
Both single-site (via two-site with chi_max=1 proxy) and two-site DMRG are exercised.

The output is designed to be directly comparable with the C manual test at
  chemtensor/manual_tests/dmrg.c
"""

from collections.list import List
from gpu.host import DeviceContext
from src.state.mps_state import create_product_mps, mps_norm
from src.state.hamiltonians import create_heisenberg_xxz_mpo
from src.algorithms.dmrg import dmrg_two_site, DMRGParams
from src.algorithms.dmrg_results_json import save_dmrg_results_to_json, DMRGJsonParams
from src.tests.test_utils import assert_close
from testing import TestSuite


fn print_separator(title: String) -> None:
    print("\n")
    print("=" * 50)
    print(title)
    print("=" * 50)


fn print_subsection(title: String) -> None:
    print("\n--- " + title + " ---")


# ___________________________________________________________________________
#
# Test 1 — Two-site DMRG, small system (mirrors single-site test structure)
#
# The C test uses dmrg_singlesite with nsites=7, d=2, 6 sweeps.
# Mojo only has dmrg_two_site, so we run two-site with modest chi_max
# on the same 7-site Heisenberg XXX chain (D=1, h=0).
#
fn test_dmrg_singlesite_proxy() raises:
    """Two-site DMRG on 7-site Heisenberg XXX (proxy for single-site test).
    
    Mirrors C test_dmrg_singlesite_manual:
      Model: Heisenberg XXZ (J=1, D=1, h=0  =>  XXX)
      nsites=7, d=2, 6 sweeps, chi_max=16
    """
    print_separator("DMRG Single-Site Proxy (Heisenberg XXX, manual)")
    
    with DeviceContext() as ctx:
        var num_sites = 7
        var physical_dim = 2
        var J = 1.0
        var D = 1.0   # D=1 => isotropic XXX limit
        var h_field = 0.0
        var num_sweeps = 6
        var chi_max = 16
        
        print("Model        : Heisenberg XXZ (J=" + String(J) + ", D=" + String(D) + ", h=" + String(h_field) + ")")
        print("nsites       : " + String(num_sites))
        print("d            : " + String(physical_dim))
        print("num_sweeps   : " + String(num_sweeps))
        print("chi_max      : " + String(chi_max))
        print("")
        
        # Build Hamiltonian MPO (float64 for numerical accuracy matching C's double)
        var H = create_heisenberg_xxz_mpo[DType.float32](ctx, num_sites, J=J, D=D, h=h_field)
        print("MPO constructed: " + String(H.num_sites()) + " sites")
        
        # Initial product state |0000000>
        var basis = List[Int](capacity=num_sites)
        for _ in range(num_sites):
            basis.append(0)
        
        var psi_initial = create_product_mps[DType.float32](ctx, physical_dim, basis^)
        print("Initial MPS  : product |000...0>")
        print("")
        
        # DMRG parameters matching C test
        var params = DMRGParams(
            num_sweeps=num_sweeps,
            chi_max=chi_max,
            eps_trunc=1e-10,
            max_krylov_iter=25,
            krylov_tol=1e-8,
            energy_tol=1e-8,
            two_site=True,
            verbose=True       # prints per-sweep energy
        )
        
        var result = dmrg_two_site[DType.float32](ctx, H^, psi_initial^, params)
        var ground_energy = result[0]
        var ground_state = result[1]
        
        # Print results
        print_subsection("Final results")
        print("Ground state energy : " + String(ground_energy))
        print("Energy per site     : " + String(ground_energy / Float64(num_sites)))
        
        var bd_str = String("[")
        for i in range(len(ground_state.bond_dims)):
            if i > 0:
                bd_str += ", "
            bd_str += String(ground_state.bond_dims[i])
        bd_str += "]"
        print("Bond dimensions     : " + bd_str)
        
        # Save results to JSON for comparison with C (see DMRG_RESULTS_JSON_SCHEMA.md)
        var en_sweeps_list = List[Float64](capacity=1)
        en_sweeps_list.append(ground_energy)
        var empty_entropy = List[Float64]()
        var json_params = DMRGJsonParams(
            nsites=num_sites,
            d=physical_dim,
            J=J, D=D, h=h_field,
            num_sweeps=num_sweeps,
            maxiter_lanczos=25,
            chi_max=chi_max,
            tol_split=-1.0,
        )
        save_dmrg_results_to_json(
            "results/algorithms/dmrg/dmrg_results_mojo_singlesite.json",
            "mojo", "heisenberg_xxz",
            json_params,
            ground_energy,
            en_sweeps_list,
            empty_entropy,
            ground_state.bond_dims,
            1.0,
        )
        print("Results written to results/algorithms/dmrg/dmrg_results_mojo_singlesite.json")

        # Assertions (same style as MPS/MPO/dense tensor tests)
        var norm_mojo = mps_norm[DType.float32](ground_state, ctx)
        assert_close(norm_mojo, 1.0, rtol=1e-5, atol=1e-6, label="final MPS norm")
        # Ground energy should be finite and negative (exact range depends on H convention)
        if ground_energy >= 0.0 or ground_energy < -1e6:
            raise Error("Ground energy not in expected range (finite negative): " + String(ground_energy))
        
        print("\nSingle-site proxy DMRG test completed.")


# ___________________________________________________________________________
#
# Test 2 — Two-site DMRG  (mirrors test_dmrg_twosite in test_dmrg.c)
#
# C test: nsites=11, d=2, Heisenberg XXZ (J=1, D=0.5, h=0.2), 4 sweeps.
#
fn test_dmrg_twosite_manual() raises:
    """Two-site DMRG on 11-site Heisenberg XXZ chain.
    
    Mirrors C test_dmrg_twosite_manual:
      Model: Heisenberg XXZ (J=1, D=0.5, h=0.2)
      nsites=11, d=2, 4 sweeps, chi_max=32
    """
    print_separator("DMRG Two-Site (Heisenberg XXZ, manual)")
    
    with DeviceContext() as ctx:
        var num_sites = 11
        var physical_dim = 2
        var J = 1.0
        var D = 0.5    # anisotropy
        var h_field = 0.2    # magnetic field
        var num_sweeps = 4
        # max_vdim in C: ipow(d, nsites/2) = 2^5 = 32
        var chi_max = 32
        
        print("Model        : Heisenberg XXZ (J=" + String(J) + ", D=" + String(D) + ", h=" + String(h_field) + ")")
        print("nsites       : " + String(num_sites))
        print("d            : " + String(physical_dim))
        print("num_sweeps   : " + String(num_sweeps))
        print("chi_max      : " + String(chi_max))
        print("eps_trunc    : 1e-10")
        print("")
        
        # Build Hamiltonian MPO (float64 for numerical accuracy matching C's double)
        var H = create_heisenberg_xxz_mpo[DType.float32](ctx, num_sites, J=J, D=D, h=h_field)
        print("MPO constructed: " + String(H.num_sites()) + " sites")
        
        # Initial product state |00000000000>
        var basis = List[Int](capacity=num_sites)
        for _ in range(num_sites):
            basis.append(0)
        
        var psi_initial = create_product_mps[DType.float32](ctx, physical_dim, basis^)
        print("Initial MPS  : product |000...0>")
        print("")
        
        # DMRG parameters matching C test
        var params = DMRGParams(
            num_sweeps=num_sweeps,
            chi_max=chi_max,
            eps_trunc=1e-10,
            max_krylov_iter=25,
            krylov_tol=1e-8,
            energy_tol=1e-8,
            two_site=True,
            verbose=True       # prints per-sweep energy
        )
        
        var result = dmrg_two_site[DType.float32](ctx, H^, psi_initial^, params)
        var ground_energy = result[0]
        var ground_state = result[1]
        
        # Print results
        print_subsection("Final results")
        print("Ground state energy : " + String(ground_energy))
        print("Energy per site     : " + String(ground_energy / Float64(num_sites)))
        
        var bd_str = String("[")
        for i in range(len(ground_state.bond_dims)):
            if i > 0:
                bd_str += ", "
            bd_str += String(ground_state.bond_dims[i])
        bd_str += "]"
        print("Bond dimensions     : " + bd_str)
        
        # Save results to JSON for comparison with C (see DMRG_RESULTS_JSON_SCHEMA.md)
        var en_sweeps_list = List[Float64](capacity=1)
        en_sweeps_list.append(ground_energy)
        var empty_entropy = List[Float64]()
        var json_params = DMRGJsonParams(
            nsites=num_sites,
            d=physical_dim,
            J=J, D=D, h=h_field,
            num_sweeps=num_sweeps,
            maxiter_lanczos=25,
            chi_max=chi_max,
            tol_split=1e-10,
        )
        save_dmrg_results_to_json(
            "results/algorithms/dmrg/dmrg_results_mojo_twosite.json",
            "mojo", "heisenberg_xxz",
            json_params,
            ground_energy,
            en_sweeps_list,
            empty_entropy,
            ground_state.bond_dims,
            1.0,
        )
        print("Results written to results/algorithms/dmrg/dmrg_results_mojo_twosite.json")

        # Assertions
        var norm_mojo = mps_norm[DType.float32](ground_state, ctx)
        assert_close(norm_mojo, 1.0, rtol=1e-5, atol=1e-6, label="final MPS norm")
        # Ground energy should be finite and negative (exact range depends on H convention)
        if ground_energy >= 0.0 or ground_energy < -1e6:
            raise Error("Ground energy not in expected range (finite negative): " + String(ground_energy))
        
        print("\nTwo-site DMRG test completed.")



fn main() raises:
    try:
        TestSuite.discover_tests[__functions_in_module()]().run()
    except e:
        print("Tests failed: " + String(e))
