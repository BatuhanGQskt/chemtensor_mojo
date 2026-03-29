"""
Observable-based comparison tests: C MPS vs Mojo MPS.

Tests compare norms and overlaps only (using test_utils.assert_close),
not inner dense tensor representations, so they tolerate implementation
differences and small numerical error.
"""

from collections.list import List
from gpu.host import DeviceContext
from src.state.mps_state import (
    MatrixProductState,
    create_product_mps,
    mps_norm,
    mps_to_statevector,
)
from src.tests.test_utils import assert_close, assert_equal, tensor_to_host
from src.tests.state.mps.mps_json_loader import load_mps_reference, print_mps_reference_info
from testing import TestSuite


fn print_separator(title: String) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


fn print_subsection(title: String) -> None:
    print("\n--- " + title + " ---")


fn test_product_state_norm_no_c() raises:
    """Product state |0,0,0> has norm 1. No C reference; observable-only check."""
    print_separator("Test 1: Product state norm (no C reference)")
    var ctx = DeviceContext()
    var basis = List[Int](0, 0, 0)
    var psi = create_product_mps[DType.float32](ctx, 2, basis^)
    var n = mps_norm[DType.float32](psi, ctx)
    print("Product state |0,0,0>, d=2: norm = " + String(n))
    assert_close(n, 1.0, rtol=1e-5, atol=1e-6, label="norm")
    print("  ✓ Norm = 1.0 within tolerance")


fn test_product_state_2sites_norm() raises:
    """Two-site product state norm."""
    print_separator("Test 2: Two-site product state norm")
    var ctx = DeviceContext()
    var basis = List[Int](1, 0)
    var psi = create_product_mps[DType.float32](ctx, 2, basis^)
    var n = mps_norm[DType.float32](psi, ctx)
    assert_close(n, 1.0, rtol=1e-5, atol=1e-6, label="norm")
    print("  ✓ |1,0> norm = 1.0")


fn test_mps_vs_c_reference_product_000() raises:
    """Compare Mojo product state |0,0,0> with C reference (observables only).

    Requires test_data/mps_product_3sites_d2_basis_000.json from C export.
    Compares: nsites, d, bond_dims, norm, and overlap with C state vector.
    """
    print_separator("Test 3: Mojo vs C reference (product |0,0,0>)")
    var ref_path = String("test_data/mps_product_3sites_d2_basis_000.json")
    # Check file exists (Python open will raise if not)
    var ctx = DeviceContext()

    var mps_ref = load_mps_reference(ref_path)
    print_mps_reference_info(mps_ref)

    var basis = List[Int](0, 0, 0)
    var psi = create_product_mps[DType.float32](ctx, 2, basis^)

    print_subsection("Metadata")
    assert_equal(psi.num_sites(), mps_ref.nsites, "nsites")
    assert_equal(psi.physical_dim, mps_ref.d, "d")
    print("  ✓ nsites: " + String(psi.num_sites()))
    print("  ✓ d: " + String(psi.physical_dim))

    print_subsection("Bond dimensions")
    for i in range(len(psi.bond_dims)):
        assert_equal(psi.bond_dims[i], mps_ref.bond_dims[i], "bond_dim[" + String(i) + "]")
    print("  ✓ bond_dims match")

    print_subsection("Norm (observable)")
    var norm_mojo = mps_norm[DType.float32](psi, ctx)
    assert_close(norm_mojo, mps_ref.norm, rtol=1e-5, atol=1e-6, label="norm")
    print("  ✓ norm: " + String(norm_mojo) + " vs C " + String(mps_ref.norm))

    print_subsection("Overlap with C state vector")
    var vec_mojo = mps_to_statevector[DType.float32](psi, ctx)
    var host_mojo = tensor_to_host(ctx, vec_mojo)
    if len(host_mojo) != len(mps_ref.state_vector):
        raise Error("State vector length mismatch: Mojo " + String(len(host_mojo)) + " vs ref " + String(len(mps_ref.state_vector)))
    var overlap: Float64 = 0.0
    for i in range(len(host_mojo)):
        overlap += Float64(host_mojo[i]) * mps_ref.state_vector[i]
    # Same state => |overlap| ≈ 1
    assert_close(abs(overlap), 1.0, rtol=1e-5, atol=1e-6, label="|overlap|")
    print("  ✓ |overlap| = " + String(abs(overlap)))
    print("\n✓ C vs Mojo observables match (not glued to tensor layout)")


fn test_mps_vs_c_reference_product_101() raises:
    """Compare Mojo product state |1,0,1> with C reference."""
    print_separator("Test 4: Mojo vs C reference (product |1,0,1>)")
    var ref_path = String("test_data/mps_product_3sites_d2_basis_101.json")
    var ctx = DeviceContext()

    var mps_ref = load_mps_reference(ref_path)
    print_mps_reference_info(mps_ref)

    var basis = List[Int](1, 0, 1)
    var psi = create_product_mps[DType.float32](ctx, 2, basis^)

    assert_equal(psi.num_sites(), mps_ref.nsites, "nsites")
    assert_equal(psi.physical_dim, mps_ref.d, "d")
    for i in range(len(psi.bond_dims)):
        assert_equal(psi.bond_dims[i], mps_ref.bond_dims[i], "bond_dim[" + String(i) + "]")

    var norm_mojo = mps_norm[DType.float32](psi, ctx)
    assert_close(norm_mojo, mps_ref.norm, rtol=1e-5, atol=1e-6, label="norm")

    var vec_mojo = mps_to_statevector[DType.float32](psi, ctx)
    var host_mojo = tensor_to_host(ctx, vec_mojo)
    if len(host_mojo) != len(mps_ref.state_vector):
        raise Error("State vector length mismatch")
    var overlap: Float64 = 0.0
    for i in range(len(host_mojo)):
        overlap += Float64(host_mojo[i]) * mps_ref.state_vector[i]
    assert_close(abs(overlap), 1.0, rtol=1e-5, atol=1e-6, label="|overlap|")
    print("  ✓ Norm and overlap match C reference")
    print("\n✓ All observable checks passed")


fn main() raises:
    try:
        TestSuite.discover_tests[__functions_in_module()]().run()
    except e:
        print("Tests failed: " + String(e))
        raise e
