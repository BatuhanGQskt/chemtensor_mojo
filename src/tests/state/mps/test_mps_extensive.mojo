"""
Extensive MPS tests: product states, norms, state vectors, overlaps.

No C reference data required. Exercises create_product_mps, mps_norm,
mps_to_statevector, and observable checks.
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
from testing import TestSuite


fn print_sep(title: String) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


fn print_sub(title: String) -> None:
    print("\n--- " + title + " ---")


fn test_product_state_single_site() raises:
    """Single-site product state |0> has norm 1 and state vector length d."""
    print_sep("Single-site product state")
    var ctx = DeviceContext()
    var basis = List[Int](0)
    var psi = create_product_mps[DType.float32](ctx, 2, basis^)
    assert_equal(psi.num_sites(), 1, "num_sites")
    assert_equal(psi.physical_dim, 2, "physical_dim")
    var n = mps_norm[DType.float32](psi, ctx)
    assert_close(n, 1.0, rtol=1e-5, atol=1e-6, label="norm")
    var vec = mps_to_statevector[DType.float32](psi, ctx)
    assert_equal(len(tensor_to_host(ctx, vec)), 2, "state_vector_length")
    print("  ✓ single site |0> norm=1, len(vec)=2")


fn test_product_state_two_sites() raises:
    """Two-site product states: |00>, |11>, |01>, |10> all have norm 1."""
    print_sep("Two-site product states")
    var ctx = DeviceContext()
    var cases = List[List[Int]]()
    cases.append(List[Int](0, 0))
    cases.append(List[Int](1, 1))
    cases.append(List[Int](0, 1))
    cases.append(List[Int](1, 0))
    for i in range(len(cases)):
        var psi = create_product_mps[DType.float32](ctx, 2, cases[i].copy())
        assert_equal(psi.num_sites(), 2, "num_sites")
        var n = mps_norm[DType.float32](psi, ctx)
        assert_close(n, 1.0, rtol=1e-5, atol=1e-6, label="norm")
    print("  ✓ |00> |11> |01> |10> norm=1")


fn test_product_state_four_sites_d2() raises:
    """Four-site product state (d=2): norm 1, state vector length 16."""
    print_sep("Four-site product state d=2")
    var ctx = DeviceContext()
    var basis = List[Int](1, 0, 1, 0)
    var psi = create_product_mps[DType.float32](ctx, 2, basis^)
    assert_equal(psi.num_sites(), 4, "num_sites")
    assert_equal(psi.physical_dim, 2, "physical_dim")
    var n = mps_norm[DType.float32](psi, ctx)
    assert_close(n, 1.0, rtol=1e-5, atol=1e-6, label="norm")
    var vec = mps_to_statevector[DType.float32](psi, ctx)
    var host = tensor_to_host(ctx, vec)
    assert_equal(len(host), 16, "state_vector_length")
    print("  ✓ 4 sites d=2 norm=1 len(vec)=16")


fn test_product_state_three_sites_d3() raises:
    """Three-site product state with d=3: norm 1, state vector length 27."""
    print_sep("Three-site product state d=3")
    var ctx = DeviceContext()
    var basis = List[Int](0, 1, 2)
    var psi = create_product_mps[DType.float32](ctx, 3, basis^)
    assert_equal(psi.num_sites(), 3, "num_sites")
    assert_equal(psi.physical_dim, 3, "physical_dim")
    var n = mps_norm[DType.float32](psi, ctx)
    assert_close(n, 1.0, rtol=1e-5, atol=1e-6, label="norm")
    var vec = mps_to_statevector[DType.float32](psi, ctx)
    assert_equal(len(tensor_to_host(ctx, vec)), 27, "state_vector_length")
    print("  ✓ 3 sites d=3 norm=1 len(vec)=27")


fn test_bond_dims_product_state() raises:
    """Product state has bond dimensions all 1."""
    print_sep("Bond dimensions (product state)")
    var ctx = DeviceContext()
    var basis = List[Int](0, 1, 0, 1, 0)
    var psi = create_product_mps[DType.float32](ctx, 2, basis^)
    for i in range(len(psi.bond_dims)):
        assert_equal(psi.bond_dims[i], 1, "bond_dim[" + String(i) + "]")
    print("  ✓ bond_dims = [1,1,1,1,1,1]")


fn test_state_vector_product_00() raises:
    """Product |00> has state vector [1,0,0,0] (row-major)."""
    print_sep("State vector content |00>")
    var ctx = DeviceContext()
    var basis = List[Int](0, 0)
    var psi = create_product_mps[DType.float32](ctx, 2, basis^)
    var vec = mps_to_statevector[DType.float32](psi, ctx)
    var host = tensor_to_host(ctx, vec)
    assert_equal(len(host), 4, "length")
    assert_close(Float64(host[0]), 1.0, rtol=1e-5, atol=1e-6, label="vec[0]")
    assert_close(Float64(host[1]), 0.0, rtol=1e-5, atol=1e-6, label="vec[1]")
    assert_close(Float64(host[2]), 0.0, rtol=1e-5, atol=1e-6, label="vec[2]")
    assert_close(Float64(host[3]), 0.0, rtol=1e-5, atol=1e-6, label="vec[3]")
    print("  ✓ |00> -> [1,0,0,0]")


fn test_state_vector_product_11() raises:
    """Product |11> has state vector [0,0,0,1] (index 3 = 1+1*2)."""
    print_sep("State vector content |11>")
    var ctx = DeviceContext()
    var basis = List[Int](1, 1)
    var psi = create_product_mps[DType.float32](ctx, 2, basis^)
    var vec = mps_to_statevector[DType.float32](psi, ctx)
    var host = tensor_to_host(ctx, vec)
    assert_equal(len(host), 4, "length")
    assert_close(Float64(host[3]), 1.0, rtol=1e-5, atol=1e-6, label="vec[3]")
    for i in range(3):
        assert_close(Float64(host[i]), 0.0, rtol=1e-5, atol=1e-6, label="vec[" + String(i) + "]")
    print("  ✓ |11> -> [0,0,0,1]")


fn test_overlap_same_state() raises:
    """Overlap of product state with itself (via state vector) is 1."""
    print_sep("Overlap same state")
    var ctx = DeviceContext()
    var basis = List[Int](1, 0, 1)
    var psi = create_product_mps[DType.float32](ctx, 2, basis^)
    var vec = mps_to_statevector[DType.float32](psi, ctx)
    var host = tensor_to_host(ctx, vec)
    var overlap: Float64 = 0.0
    for i in range(len(host)):
        overlap += Float64(host[i]) * Float64(host[i])
    assert_close(overlap, 1.0, rtol=1e-5, atol=1e-6, label="self_overlap")
    print("  ✓ <psi|psi> = 1")


fn test_overlap_orthogonal_states() raises:
    """|000> and |111> are orthogonal: overlap of state vectors = 0."""
    print_sep("Overlap orthogonal product states")
    var ctx = DeviceContext()
    var b000 = List[Int](0, 0, 0)
    var b111 = List[Int](1, 1, 1)
    var psi_000 = create_product_mps[DType.float32](ctx, 2, b000^)
    var psi_111 = create_product_mps[DType.float32](ctx, 2, b111^)
    var v000 = tensor_to_host(ctx, mps_to_statevector[DType.float32](psi_000, ctx))
    var v111 = tensor_to_host(ctx, mps_to_statevector[DType.float32](psi_111, ctx))
    var overlap: Float64 = 0.0
    for i in range(len(v000)):
        overlap += Float64(v000[i]) * Float64(v111[i])
    assert_close(abs(overlap), 0.0, rtol=1e-5, atol=1e-6, label="overlap")
    print("  ✓ <000|111> = 0")


fn test_describe_no_crash() raises:
    """MatrixProductState.describe() runs without error."""
    print_sep("MPS describe")
    var ctx = DeviceContext()
    var basis = List[Int](0, 1, 0)
    var psi = create_product_mps[DType.float32](ctx, 2, basis^)
    psi.describe()
    print("  ✓ describe() OK")


fn main() raises:
    try:
        TestSuite.discover_tests[__functions_in_module()]().run()
    except e:
        print("Tests failed: " + String(e))
        raise e
