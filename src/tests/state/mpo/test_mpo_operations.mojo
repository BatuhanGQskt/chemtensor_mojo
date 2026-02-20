"""
Tests for MPO tensor operations (merge, transpose, full contraction).

These tests validate the correctness of tensor operations on MPO structures,
which are critical for DMRG and other algorithms. Tests include:
- Merging adjacent MPO site tensors
- Full MPO contraction to dense matrix
- Transpose operations on MPO sites
- Reshape operations

Operations are tested against known properties and C reference behavior.
"""

from collections.list import List
from gpu.host import DeviceContext
from src.m_tensor.dense_tensor import DenseTensor, create_dense_tensor, dense_tensor_dot
from src.state.mpo_state import MatrixProductOperator, MPOSite
from src.state.hamiltonians import create_ising_1d_mpo
from src.tests.test_utils import assert_equal, assert_close, compare_tensors
from src.tests.state.mpo.mpo_test_helpers import mpo_to_full_matrix, check_hermiticity
from testing import TestSuite


fn print_separator(title: String) -> None:
    """Print a separator line with title."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


fn print_subsection(title: String) -> None:
    """Print a subsection header."""
    print("\n--- " + title + " ---")


fn merge_mpo_tensor_pair(
    site0: MPOSite[DType.float32],
    site1: MPOSite[DType.float32],
    ctx: DeviceContext,
) raises -> DenseTensor[DType.float32]:
    """Merge two adjacent MPO site tensors by contracting the shared bond.
    
    This matches the C implementation which:
    1. Contracts a0 and a1 over the bond (Wr0 == Wl1)
       Result shape: [Wl0, d_in0, d_out0, d_in1, d_out1, Wr1]
    2. Permutes with [0, 1, 3, 2, 4, 5] to group physical indices
       Result shape: [Wl0, d_in0, d_in1, d_out0, d_out1, Wr1]
    3. Flattens input and output dimensions
       Result shape: [Wl0, d_in0*d_in1, d_out0*d_out1, Wr1]
    
    Args:
        site0: First MPO site tensor, shape `[Wl0, d_in, d_out, Wr0]`.
        site1: Second MPO site tensor, shape `[Wl1, d_in, d_out, Wr1]`.
        ctx: Device context.
    
    Returns:
        Merged tensor with shape `[Wl0, d_in*d_in, d_out*d_out, Wr1]`.
        Physical indices are properly ordered: combined inputs then combined outputs.
    """
    var W0 = site0.tensor
    var W1 = site1.tensor
    
    var W0_shape = W0.shape.copy()
    var W1_shape = W1.shape.copy()
    
    var wl0 = W0_shape[0]
    var d_in0 = W0_shape[1]
    var d_out0 = W0_shape[2]
    var wr0 = W0_shape[3]
    
    var wl1 = W1_shape[0]
    var d_in1 = W1_shape[1]
    var d_out1 = W1_shape[2]
    var wr1 = W1_shape[3]
    
    if wr0 != wl1:
        raise Error("Bond dimension mismatch")
    if d_in0 != d_in1 or d_out0 != d_out1:
        raise Error("Physical dimension mismatch")
    
    # Step 1: Contract over bond
    var W0_temp = W0.reshape(List[Int](wl0 * d_in0 * d_out0, wr0))
    var W1_temp = W1.reshape(List[Int](wl1, d_in1 * d_out1 * wr1))
    
    var contracted = create_dense_tensor[DType.float32](
        ctx, List[Int](wl0 * d_in0 * d_out0, d_in1 * d_out1 * wr1),
        init_value=Scalar[DType.float32](0.0)
    )
    dense_tensor_dot(contracted, W0_temp^, W1_temp^, ctx)
    
    # Step 2: Reshape to 6D
    var intermediate = contracted^.reshape(List[Int](wl0, d_in0, d_out0, d_in1, d_out1, wr1))
    
    # Step 3: Permute to group physical indices
    var permuted = intermediate^.transpose(List[Int](0, 1, 3, 2, 4, 5), ctx)
    
    # Step 4: Flatten physical dimensions
    return permuted^.reshape(List[Int](wl0, d_in0 * d_in1, d_out0 * d_out1, wr1))


fn test_mpo_merge_2site_ising() raises:
    """Test MPO tensor merge operation on 2-site Ising model.
    
    For a 2-site system, merging both sites should give a [1, 4, 4, 1] tensor
    (with virtual bonds of dimension 1 at boundaries).
    """
    print_separator("Test 1: MPO Merge Operation - 2-Site Ising")
    
    var ctx = DeviceContext()
    var mpo = create_ising_1d_mpo[DType.float32](
        ctx, num_sites=2, J=1.0, h_longitudinal=0.0, g_transverse=0.0
    )
    
    print("Parameters: nsites=2, J=1.0, h=0.0, g=0.0")
    print("Site 0 shape: [" + String(mpo.sites[0].tensor.shape[0]) + ", " +
          String(mpo.sites[0].tensor.shape[1]) + ", " +
          String(mpo.sites[0].tensor.shape[2]) + ", " +
          String(mpo.sites[0].tensor.shape[3]) + "]")
    print("Site 1 shape: [" + String(mpo.sites[1].tensor.shape[0]) + ", " +
          String(mpo.sites[1].tensor.shape[1]) + ", " +
          String(mpo.sites[1].tensor.shape[2]) + ", " +
          String(mpo.sites[1].tensor.shape[3]) + "]")
    
    # Merge sites
    print_subsection("Merging Sites 0 and 1")
    var merged = merge_mpo_tensor_pair(mpo.sites[0], mpo.sites[1], ctx)
    
    # Check shape
    var merged_shape = merged.shape.copy()
    print("Merged shape: [" + String(merged_shape[0]) + ", " +
          String(merged_shape[1]) + ", " +
          String(merged_shape[2]) + ", " +
          String(merged_shape[3]) + "]")
    
    # For 2-site Ising: expect [1, 4, 4, 1]
    assert_equal(merged_shape[0], 1, "Merged Wl (left boundary)")
    assert_equal(merged_shape[1], 4, "Merged d_in (2*2)")
    assert_equal(merged_shape[2], 4, "Merged d_out (2*2)")
    assert_equal(merged_shape[3], 1, "Merged Wr (right boundary)")
    
    print("✓ Merged shape is correct: [1, 4, 4, 1]")
    
    # Verify the merged tensor represents -Z⊗Z by checking it's the same
    # as the full matrix (after removing virtual bonds)
    var H_full = mpo_to_full_matrix(mpo, ctx)
    
    # Extract [4, 4] matrix from merged [1, 4, 4, 1]
    var merged_matrix = merged.reshape(List[Int](4, 4))
    
    print_subsection("Comparing Merged Tensor with Full Matrix")
    compare_tensors(merged_matrix, H_full, ctx, rtol=1e-10, atol=1e-12)
    print("✓ Merged tensor matches full matrix contraction")
    
    print("\n✓ MPO merge operation validated!")


fn test_mpo_full_contraction() raises:
    """Test full MPO contraction to dense matrix.
    
    Verifies that contracting all sites produces the correct Hamiltonian matrix
    with proper Hermiticity and expected dimensions.
    """
    print_separator("Test 2: Full MPO Contraction to Dense Matrix")
    
    var ctx = DeviceContext()
    
    # Test with 3-site Ising
    print_subsection("3-Site Ising Model")
    var mpo3 = create_ising_1d_mpo[DType.float32](
        ctx, num_sites=3, J=1.0, h_longitudinal=0.2, g_transverse=0.1
    )
    
    var H3 = mpo_to_full_matrix(mpo3, ctx)
    var shape3 = H3.shape.copy()
    
    print("Contracted matrix shape: [" + String(shape3[0]) + ", " + String(shape3[1]) + "]")
    print("Expected: [8, 8] (2^3 = 8)")
    
    assert_equal(shape3[0], 8, "Matrix rows")
    assert_equal(shape3[1], 8, "Matrix columns")
    print("✓ Dimensions correct")
    
    # Verify Hermiticity
    var is_hermitian3 = check_hermiticity(H3, ctx, tol=1e-10)
    if not is_hermitian3:
        raise Error("3-site Hamiltonian is not Hermitian!")
    print("✓ Matrix is Hermitian")
    
    # Test with 4-site Ising
    print_subsection("4-Site Ising Model")
    var mpo4 = create_ising_1d_mpo[DType.float32](
        ctx, num_sites=4, J=1.0, h_longitudinal=0.3, g_transverse=0.2
    )
    
    var H4 = mpo_to_full_matrix(mpo4, ctx)
    var shape4 = H4.shape.copy()
    
    print("Contracted matrix shape: [" + String(shape4[0]) + ", " + String(shape4[1]) + "]")
    print("Expected: [16, 16] (2^4 = 16)")
    
    assert_equal(shape4[0], 16, "Matrix rows")
    assert_equal(shape4[1], 16, "Matrix columns")
    print("✓ Dimensions correct")
    
    # Verify Hermiticity
    var is_hermitian4 = check_hermiticity(H4, ctx, tol=1e-10)
    if not is_hermitian4:
        raise Error("4-site Hamiltonian is not Hermitian!")
    print("✓ Matrix is Hermitian")
    
    print("\n✓ Full contraction operation validated!")


fn test_mpo_site_transpose() raises:
    """Test transpose operation on MPO site tensors.
    
    Swapping physical in/out indices [Wl, d_in, d_out, Wr] -> [Wl, d_out, d_in, Wr]
    corresponds to taking the Hermitian conjugate of the operator.
    """
    print_separator("Test 3: MPO Site Tensor Transpose")
    
    var ctx = DeviceContext()
    var mpo = create_ising_1d_mpo[DType.float32](
        ctx, num_sites=2, J=1.0, h_longitudinal=0.0, g_transverse=0.0
    )
    
    print("Original site 0 shape: [" + String(mpo.sites[0].tensor.shape[0]) + ", " +
          String(mpo.sites[0].tensor.shape[1]) + ", " +
          String(mpo.sites[0].tensor.shape[2]) + ", " +
          String(mpo.sites[0].tensor.shape[3]) + "]")
    
    # Transpose: swap physical indices [Wl, d_in, d_out, Wr] -> [Wl, d_out, d_in, Wr]
    var site0_transposed = mpo.sites[0].tensor.transpose(List[Int](0, 2, 1, 3), ctx)
    
    print("Transposed shape: [" + String(site0_transposed.shape[0]) + ", " +
          String(site0_transposed.shape[1]) + ", " +
          String(site0_transposed.shape[2]) + ", " +
          String(site0_transposed.shape[3]) + "]")
    
    # Shape should be unchanged (physical dimensions are equal for qubits)
    assert_equal(site0_transposed.shape[0], mpo.sites[0].tensor.shape[0], "Wl")
    assert_equal(site0_transposed.shape[1], mpo.sites[0].tensor.shape[2], "d_out (was d_in)")
    assert_equal(site0_transposed.shape[2], mpo.sites[0].tensor.shape[1], "d_in (was d_out)")
    assert_equal(site0_transposed.shape[3], mpo.sites[0].tensor.shape[3], "Wr")
    
    print("✓ Transpose dimensions correct")
    
    # For Hermitian operators, transposing back should give the original
    var site0_double_transposed = site0_transposed.transpose(List[Int](0, 2, 1, 3), ctx)
    compare_tensors(site0_double_transposed, mpo.sites[0].tensor, ctx, rtol=1e-10, atol=1e-12)
    print("✓ Double transpose returns to original")
    
    print("\n✓ Transpose operation validated!")


fn test_mpo_site_reshape() raises:
    """Test reshape operation on MPO site tensors.
    
    Flattening to matrix form is needed for various operations.
    """
    print_separator("Test 4: MPO Site Tensor Reshape")
    
    var ctx = DeviceContext()
    var mpo = create_ising_1d_mpo[DType.float32](
        ctx, num_sites=2, J=1.0, h_longitudinal=0.0, g_transverse=0.0
    )
    
    var site0 = mpo.sites[0].tensor
    var orig_shape = site0.shape.copy()
    
    print("Original shape: [" + String(orig_shape[0]) + ", " +
          String(orig_shape[1]) + ", " +
          String(orig_shape[2]) + ", " +
          String(orig_shape[3]) + "]")
    
    # Reshape [1, 2, 2, 3] -> [4, 3]
    var site0_reshaped = site0.reshape(List[Int](orig_shape[0] * orig_shape[1] * orig_shape[2], orig_shape[3]))
    
    print("Reshaped: [" + String(site0_reshaped.shape[0]) + ", " +
          String(site0_reshaped.shape[1]) + "]")
    
    assert_equal(site0_reshaped.shape[0], 4, "Flattened rows")
    assert_equal(site0_reshaped.shape[1], 3, "Wr columns")
    print("✓ Reshape dimensions correct")
    
    # Verify element count preserved
    var orig_size = orig_shape[0] * orig_shape[1] * orig_shape[2] * orig_shape[3]
    var reshaped_size = site0_reshaped.shape[0] * site0_reshaped.shape[1]
    assert_equal(reshaped_size, orig_size, "Element count")
    print("✓ Element count preserved: " + String(orig_size))
    
    # Reshape back and verify data unchanged
    var site0_back = site0_reshaped.reshape(List[Int](orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3]))
    compare_tensors(site0_back, site0, ctx, rtol=1e-10, atol=1e-12)
    print("✓ Reshaping back preserves data")
    
    print("\n✓ Reshape operation validated!")


fn main() raises:
    """Run all MPO operation tests."""
    try:
        TestSuite.discover_tests[__functions_in_module()]().run()
    except e:
        print("Tests failed: " + String(e))
        raise e
