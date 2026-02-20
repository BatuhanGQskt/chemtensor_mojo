"""
Element-wise comparison tests between Mojo and C MPO implementations.

These tests validate that the Mojo MPO implementation produces identical tensor
data as the C ground truth implementation by:
- Loading reference data from JSON files exported by C
- Creating equivalent MPOs in Mojo
- Comparing tensor data element-wise with tight tolerances

This catches implementation bugs in:
- Tensor layout and indexing
- Operator placement and coefficients
- Bond dimension construction
"""

from collections.list import List
from gpu.host import DeviceContext
from src.state.mpo_state import MatrixProductOperator
from src.state.hamiltonians import create_ising_1d_mpo, create_heisenberg_xxz_mpo, create_bose_hubbard_mpo
from src.tests.test_utils import assert_equal, compare_tensors
from src.tests.state.mpo.mpo_json_loader import load_mpo_reference, print_mpo_reference_info
from testing import TestSuite
from pathlib import Path


fn print_separator(title: String) -> None:
    """Print a separator line with title."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


fn print_subsection(title: String) -> None:
    """Print a subsection header."""
    print("\n--- " + title + " ---")


fn test_ising_2site_vs_c_reference() raises:
    """Compare 2-site pure Ising MPO (J=1, h=0, g=0) with C reference.
    
    This is the simplest case with analytical solution, making it ideal
    for validating the comparison infrastructure.
    """
    print_separator("Test 1: 2-Site Ising vs C Reference (J=1.0, h=0, g=0)")
    
    var ctx = DeviceContext()
    
    # Load C reference data
    var ref_path = String("test_data/ising_1d_J1.0_h0.0_g0.0_n2.json")
    print("Loading C reference from: " + ref_path)
    var mpo_ref = load_mpo_reference[DType.float32](ref_path, ctx)
    
    print_mpo_reference_info(mpo_ref)
    
    # Create Mojo MPO with same parameters
    print_subsection("Creating Mojo MPO with same parameters")
    var mpo = create_ising_1d_mpo[DType.float32](
        ctx, num_sites=2, J=1.0, h_longitudinal=0.0, g_transverse=0.0
    )
    
    # Compare metadata
    print_subsection("Comparing Metadata")
    assert_equal(mpo.num_sites(), mpo_ref.nsites, "Number of sites")
    assert_equal(mpo.physical_in_dim, mpo_ref.d, "Physical input dimension")
    assert_equal(mpo.physical_out_dim, mpo_ref.d, "Physical output dimension")
    
    print("  ✓ nsites: " + String(mpo.num_sites()))
    print("  ✓ physical_dim: " + String(mpo.physical_in_dim))
    
    # Compare bond dimensions
    print_subsection("Comparing Bond Dimensions")
    for i in range(len(mpo.bond_dims)):
        assert_equal(mpo.bond_dims[i], mpo_ref.bond_dims[i], 
                    "Bond dimension " + String(i))
        print("  ✓ bond_dim[" + String(i) + "]: " + String(mpo.bond_dims[i]))
    
    # Compare site tensors element-wise
    print_subsection("Comparing Site Tensors Element-Wise")
    for i in range(mpo.num_sites()):
        print("\nSite " + String(i) + ":")
        var mojo_tensor = mpo.sites[i].tensor
        var c_tensor = mpo_ref.site_tensors[i]
        
        print("  Mojo shape: [" + String(mojo_tensor.shape[0]) + ", " + 
              String(mojo_tensor.shape[1]) + ", " + 
              String(mojo_tensor.shape[2]) + ", " + 
              String(mojo_tensor.shape[3]) + "]")
        
        # Element-wise comparison with tight tolerance
        compare_tensors(
            mojo_tensor, c_tensor, ctx,
            rtol=1e-10, atol=1e-12,
            site_index=i, max_errors_to_print=5
        )
        print("  ✓ All elements match within tolerance")
    
    print("\n✓ 2-site Ising MPO matches C reference perfectly!")


fn test_ising_4site_vs_c_reference() raises:
    """Compare 4-site Ising MPO (J=1.0, h=0.5, g=0.3) with C reference.
    
    This tests a more complex case with non-zero field terms.
    """
    print_separator("Test 2: 4-Site Ising vs C Reference (J=1.0, h=0.5, g=0.3)")
    
    var ctx = DeviceContext()
    
    # Load C reference data
    var ref_path = String("test_data/ising_1d_J1.0_h0.5_g0.3_n4.json")
    print("Loading C reference from: " + ref_path)
    var mpo_ref = load_mpo_reference[DType.float32](ref_path, ctx)
    
    print_mpo_reference_info(mpo_ref)
    
    # Create Mojo MPO with same parameters
    print_subsection("Creating Mojo MPO with same parameters")
    var mpo = create_ising_1d_mpo[DType.float32](
        ctx, num_sites=4, J=1.0, h_longitudinal=0.5, g_transverse=0.3
    )
    
    # Compare metadata
    print_subsection("Comparing Metadata")
    assert_equal(mpo.num_sites(), mpo_ref.nsites, "Number of sites")
    assert_equal(mpo.physical_in_dim, mpo_ref.d, "Physical dimension")
    
    # Compare bond dimensions
    print_subsection("Comparing Bond Dimensions")
    for i in range(len(mpo.bond_dims)):
        assert_equal(mpo.bond_dims[i], mpo_ref.bond_dims[i], 
                    "Bond dimension " + String(i))
        print("  ✓ bond_dim[" + String(i) + "]: " + String(mpo.bond_dims[i]))
    
    # Compare site tensors
    print_subsection("Comparing Site Tensors Element-Wise")
    for i in range(mpo.num_sites()):
        print("\nSite " + String(i) + ":")
        compare_tensors(
            mpo.sites[i].tensor, mpo_ref.site_tensors[i], ctx,
            rtol=1e-10, atol=1e-12,
            site_index=i, max_errors_to_print=5
        )
        print("  ✓ All elements match within tolerance")
    
    print("\n✓ 4-site Ising MPO matches C reference perfectly!")


fn test_heisenberg_xxz_3site_vs_c_reference() raises:
    """Compare 3-site Heisenberg XXZ MPO with C reference.
    
    This tests the Heisenberg model which has a different MPO structure
    and larger bond dimension (5 vs 3 for Ising).
    """
    print_separator("Test 3: 3-Site Heisenberg XXZ vs C Reference")
    
    var ctx = DeviceContext()
    
    # Load C reference data
    var ref_path = String("test_data/heisenberg_xxz_J1.0_D0.5_h0.2_n3.json")
    print("Loading C reference from: " + ref_path)
    var mpo_ref = load_mpo_reference[DType.float32](ref_path, ctx)
    
    print_mpo_reference_info(mpo_ref)
    
    # Create Mojo MPO with same parameters
    print_subsection("Creating Mojo MPO with same parameters")
    var mpo = create_heisenberg_xxz_mpo[DType.float32](
        ctx, num_sites=3, J=1.0, D=0.5, h=0.2
    )
    
    # Compare metadata
    print_subsection("Comparing Metadata")
    assert_equal(mpo.num_sites(), mpo_ref.nsites, "Number of sites")
    assert_equal(mpo.physical_in_dim, mpo_ref.d, "Physical dimension")
    
    print("  ✓ nsites: " + String(mpo.num_sites()))
    print("  ✓ physical_dim: " + String(mpo.physical_in_dim))
    
    # Compare bond dimensions
    print_subsection("Comparing Bond Dimensions")
    for i in range(len(mpo.bond_dims)):
        assert_equal(mpo.bond_dims[i], mpo_ref.bond_dims[i], 
                    "Bond dimension " + String(i))
        print("  ✓ bond_dim[" + String(i) + "]: " + String(mpo.bond_dims[i]))
    
    # Compare site tensors
    print_subsection("Comparing Site Tensors Element-Wise")
    for i in range(mpo.num_sites()):
        print("\nSite " + String(i) + ":")
        compare_tensors(
            mpo.sites[i].tensor, mpo_ref.site_tensors[i], ctx,
            rtol=1e-10, atol=1e-12,
            site_index=i, max_errors_to_print=5
        )
        print("  ✓ All elements match within tolerance")
    
    print("\n✓ 3-site Heisenberg XXZ MPO matches C reference perfectly!")


fn test_bose_hubbard_3site_vs_c_reference() raises:
    """Compare 3-site Bose-Hubbard MPO with C reference.
    
    This tests a model with different physical dimension (d=3 instead of d=2),
    providing validation that the implementation works for arbitrary local
    Hilbert space dimensions.
    """
    print_separator("Test 4: 3-Site Bose-Hubbard vs C Reference (d=3)")
    
    var ctx = DeviceContext()
    
    # Load C reference data
    var ref_path = String("test_data/bose_hubbard_d3_t1.0_u2.0_mu0.5_n3.json")
    print("Loading C reference from: " + ref_path)
    var mpo_ref = load_mpo_reference[DType.float32](ref_path, ctx)
    
    print_mpo_reference_info(mpo_ref)
    
    # Create Mojo MPO with same parameters
    print_subsection("Creating Mojo MPO with same parameters")
    var mpo = create_bose_hubbard_mpo[DType.float32](
        ctx, num_sites=3, physical_dim=3, t=1.0, u=2.0, mu=0.5
    )
    
    # Compare metadata
    print_subsection("Comparing Metadata")
    assert_equal(mpo.num_sites(), mpo_ref.nsites, "Number of sites")
    assert_equal(mpo.physical_in_dim, mpo_ref.d, "Physical dimension")
    
    print("  ✓ nsites: " + String(mpo.num_sites()))
    print("  ✓ physical_dim: " + String(mpo.physical_in_dim))
    
    # Compare bond dimensions
    print_subsection("Comparing Bond Dimensions")
    for i in range(len(mpo.bond_dims)):
        assert_equal(mpo.bond_dims[i], mpo_ref.bond_dims[i], 
                    "Bond dimension " + String(i))
        print("  ✓ bond_dim[" + String(i) + "]: " + String(mpo.bond_dims[i]))
    
    # Compare site tensors
    print_subsection("Comparing Site Tensors Element-Wise")
    for i in range(mpo.num_sites()):
        print("\nSite " + String(i) + ":")
        compare_tensors(
            mpo.sites[i].tensor, mpo_ref.site_tensors[i], ctx,
            rtol=1e-10, atol=1e-12,
            site_index=i, max_errors_to_print=5
        )
        print("  ✓ All elements match within tolerance")
    
    print("\n✓ 3-site Bose-Hubbard MPO matches C reference perfectly!")


fn main() raises:
    """Run all C comparison tests."""
    try:
        TestSuite.discover_tests[__functions_in_module()]().run()
    except e:
        print("Tests failed: " + String(e))
        raise e
