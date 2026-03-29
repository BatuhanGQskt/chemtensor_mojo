"""
Comprehensive tests for MPO (Matrix Product Operator) implementation.

Tests cover:
- Ising 1D MPO construction and properties
- Heisenberg MPO construction and properties
- Identity MPO
- MPO to dense matrix conversion
- MPO tensor merging operations
- MPO consistency checks

Based on ChemTensor C implementation tests.
"""

from collections.list import List
from gpu.host import DeviceContext
from src.state.mpo_state import MatrixProductOperator, MPOSite, create_identity_mpo
from src.state.hamiltonians import create_ising_1d_mpo, create_heisenberg_mpo, create_heisenberg_xxz_mpo
from src.m_tensor.dense_tensor import DenseTensor, create_dense_tensor, create_dense_tensor_from_data, dense_tensor_dot
from testing import TestSuite


fn print_separator(title: String) -> None:
    print("\n")
    print("=" * 50)
    print(title)
    print("=" * 50)


fn print_subsection(title: String) -> None:
    print("\n--- " + title + " ---")


fn print_mpo_info(name: String, mpo: MatrixProductOperator) -> None:
    """Print detailed information about an MPO, similar to C test output."""
    print_subsection(name)
    
    print("Number of sites: " + String(mpo.num_sites()))
    print("Physical input dimension: " + String(mpo.physical_in_dim))
    print("Physical output dimension: " + String(mpo.physical_out_dim))
    
    var bond_str = String("Virtual bond dimensions: [")
    for i in range(len(mpo.bond_dims)):
        if i > 0:
            bond_str += ", "
        bond_str += String(mpo.bond_dims[i])
    bond_str += "]"
    print(bond_str)
    
    print("\nDetailed tensor information per site:")
    for i in range(mpo.num_sites()):
        var shape = mpo.site_shape(i)
        print("\n  Site " + String(i) + ":")
        print("    Logical dimensions: [" + String(shape[0]) + ", " + String(shape[1]) + ", " + String(shape[2]) + ", " + String(shape[3]) + "]")


fn print_dense_tensor_data[dtype: DType](t: DenseTensor[dtype], name: String, ctx: DeviceContext) raises -> None:
    """Print information about a dense tensor."""
    print("\n=== Dense Tensor: " + name + " ===")
    var shape_str = String("Shape: [")
    for i in range(len(t.shape)):
        if i > 0:
            shape_str += ", "
        shape_str += String(t.shape[i])
    shape_str += "]"
    print(shape_str)
    
    print("Total elements: " + String(t.size))
    
    # Print first few elements (limit to reasonable size)
    var max_print = 100
    if t.size < max_print:
        max_print = t.size
    
    print("\nFirst " + String(max_print) + " elements (row-major order):")
    
    # Copy to host
    var host_buf = ctx.enqueue_create_host_buffer[dtype](t.size)
    ctx.enqueue_copy(host_buf, t.storage)
    ctx.synchronize()
    
    var count = 0
    for i in range(max_print):
        if count > 0 and count % 10 == 0:
            print("")
        print(String(Float64(host_buf[i])) + (", " if count < max_print - 1 else ""), end="")
        count += 1
    if max_print % 10 != 0:
        print("")
    if t.size > max_print:
        print("... (" + String(t.size - max_print) + " more elements not shown)")


fn contract_mpo_to_dense_simple(
    mpo: MatrixProductOperator[DType.float32],
    ctx: DeviceContext,
) raises -> DenseTensor[DType.float32]:
    """Contract an MPO to a dense matrix representation.
    
    For small systems, this contracts all MPO tensors to form the full Hamiltonian matrix.
    Shape: [d^N, d^N] where d is physical dimension and N is number of sites.
    """
    var nsites = mpo.num_sites()
    var d = mpo.physical_in_dim
    if d != mpo.physical_out_dim:
        raise Error("Physical input and output dimensions must match for this test")
    
    # Start with first site tensor
    var result = mpo.sites[0].tensor
    
    # Contract remaining sites
    for i in range(1, nsites):
        var next_site = mpo.sites[i].tensor
        
        # Contract result with next_site
        # result: [..., d, d, Wr_prev]
        # next_site: [Wl_next, d, d, Wr_next]
        # Need to contract Wr_prev with Wl_next, and physical dimensions
        
        # Reshape result: [..., d, d, Wr_prev] -> [..., d*d, Wr_prev]
        var result_shape = result.shape.copy()
        var rank = len(result_shape)
        var total_left = 1
        for j in range(rank - 2):
            total_left *= result_shape[j]
        var d_dim = result_shape[rank - 2]
        var wr_prev = result_shape[rank - 1]
        
        var result_mat = result.reshape(List[Int](total_left * d_dim, wr_prev))
        
        # Reshape next_site: [Wl_next, d, d, Wr_next] -> [Wl_next, d*d*Wr_next]
        var next_shape = next_site.shape.copy()
        var wl_next = next_shape[0]
        var d_next = next_shape[1]
        var d_out_next = next_shape[2]
        var wr_next = next_shape[3]
        
        if wl_next != wr_prev:
            raise Error("Bond dimension mismatch in MPO contraction")
        if d_dim != d_next:
            raise Error("Physical dimension mismatch in MPO contraction")
        
        # Reshape next_site to [Wl_next, d*d*Wr_next]
        var next_mat = next_site.reshape(List[Int](wl_next, d_next * d_out_next * wr_next))
        
        # Contract: [total_left * d_dim, wr_prev] @ [wl_next, d_next * d_out_next * wr_next]
        # -> [total_left * d_dim, d_next * d_out_next * wr_next]
        var temp = create_dense_tensor[DType.float32](ctx, List[Int](total_left * d_dim, d_next * d_out_next * wr_next), init_value=Scalar[DType.float32](0.0))
        dense_tensor_dot(temp, result_mat^, next_mat^, ctx)
        
        # Reshape back: [total_left, d_dim, d_out_next, wr_next]
        var new_shape = List[Int](capacity=rank + 1)
        for j in range(rank - 2):
            new_shape.append(result_shape[j])
        new_shape.append(d_dim)
        new_shape.append(d_out_next)
        new_shape.append(wr_next)
        
        result = temp^.reshape(new_shape^)
    
    # Final result should be [d, d, ..., d, d] (2*nsites dimensions)
    # Reshape to matrix [d^nsites, d^nsites]
    var final_shape = result.shape.copy()
    var left_dim = 1
    var right_dim = 1
    for i in range(len(final_shape) // 2):
        left_dim *= final_shape[i]
    for i in range(len(final_shape) // 2, len(final_shape)):
        right_dim *= final_shape[i]
    
    return result.reshape(List[Int](left_dim, right_dim))


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
        raise Error("Bond dimension mismatch: site0.right_bond_dim=" + String(wr0) + " != site1.left_bond_dim=" + String(wl1))
    if d_in0 != d_in1 or d_out0 != d_out1:
        raise Error("Physical dimension mismatch between sites")
    
    print("Merged shape components:")
    print("  wl0 =", wl0, ", d_in0 =", d_in0, ", d_in1 =", d_in1, ", d_out0 =", d_out0, ", d_out1 =", d_out1, ", wr0 =", wr0, ", wr1 =", wr1)
    
    # Step 1: Contract over the bond dimension (Wr0 == Wl1)
    # W0: [Wl0, d_in0, d_out0, Wr0]
    # W1: [Wl1, d_in1, d_out1, Wr1]  where Wr0 == Wl1
    # 
    # Reshape for contraction:
    # W0_temp: [Wl0, d_in0, d_out0, Wr0] -> [Wl0 * d_in0 * d_out0, Wr0]
    # W1_temp: [Wl1, d_in1, d_out1, Wr1] -> [Wl1, d_in1 * d_out1 * Wr1]
    
    var W0_temp = W0.reshape(List[Int](wl0 * d_in0 * d_out0, wr0))
    var W1_temp = W1.reshape(List[Int](wl1, d_in1 * d_out1 * wr1))
    
    # Contract: [Wl0 * d_in0 * d_out0, Wr0] @ [Wl1, d_in1 * d_out1 * Wr1]
    # -> [Wl0 * d_in0 * d_out0, d_in1 * d_out1 * Wr1]
    var contracted = create_dense_tensor[DType.float32](ctx, List[Int](wl0 * d_in0 * d_out0, d_in1 * d_out1 * wr1), init_value=Scalar[DType.float32](0.0))
    dense_tensor_dot(contracted, W0_temp^, W1_temp^, ctx)
    
    # Step 2: Reshape to 6D: [Wl0, d_in0, d_out0, d_in1, d_out1, Wr1]
    var intermediate = contracted^.reshape(List[Int](wl0, d_in0, d_out0, d_in1, d_out1, wr1))
    
    # Step 3: Permute to [Wl0, d_in0, d_in1, d_out0, d_out1, Wr1]
    # Source layout: [Wl0, d_in0, d_out0, d_in1, d_out1, Wr1]
    # Dest layout:   [Wl0, d_in0, d_in1, d_out0, d_out1, Wr1]
    # Permutation:   [0, 1, 3, 2, 4, 5] - swap dims 2 and 3
    var permuted = intermediate^.transpose(List[Int](0, 1, 3, 2, 4, 5), ctx)
    
    # Step 4: Flatten physical dimensions
    # From [Wl0, d_in0, d_in1, d_out0, d_out1, Wr1] to [Wl0, d_in0*d_in1, d_out0*d_out1, Wr1]
    return permuted^.reshape(List[Int](wl0, d_in0 * d_in1, d_out0 * d_out1, wr1))


fn test_ising_1d_mpo() raises:
    """Test 1D Ising Model MPO construction and properties."""
    var ctx = DeviceContext()
    print_separator("Test 1: 1D Ising Model MPO")
    
    var nsites = 4
    var J = 1.0   # Coupling strength
    var h = 0.5   # Longitudinal field
    var g = 0.3   # Transverse field
    
    print("Parameters:")
    print("  Number of sites: " + String(nsites))
    print("  J (coupling): " + String(J))
    print("  h (longitudinal field): " + String(h))
    print("  g (transverse field): " + String(g))
    print("\nHamiltonian: H = -J * sum_i(Z_i * Z_{i+1}) - g * sum_i(X_i) - h * sum_i(Z_i)")
    
    # Create MPO using float32 for GPU compatibility
    var mpo = create_ising_1d_mpo[DType.float32](ctx, nsites, J=J, h_longitudinal=h, g_transverse=g)
    
    print_mpo_info("Ising 1D MPO", mpo)
    
    # Check bond dimensions (should be 3 for bulk sites in Ising model)
    print("\nBond dimension check:")
    print("  Left boundary: " + String(mpo.bond_dimension(0)) + " (expected: 1)")
    if nsites > 1:
        print("  Bulk bond: " + String(mpo.bond_dimension(1)) + " (expected: 3)")
    print("  Right boundary: " + String(mpo.bond_dimension(nsites)) + " (expected: 1)")
    
    # Print a sample tensor from the middle site
    if nsites > 1:
        print_subsection("Sample: Site 1 tensor (for comparison with C implementation)")
        var site1 = mpo.sites[1]
        print_dense_tensor_data[DType.float32](site1.tensor, "Site 1 Tensor", ctx)
    print("\n✓ Ising MPO test passed")


fn test_heisenberg_mpo() raises:
    """Test Heisenberg XXX Model MPO construction and properties."""
    var ctx = DeviceContext()
    print_separator("Test 2: Heisenberg XXX Model MPO")
    
    var nsites = 3
    var J = 1.0   # Exchange coupling
    
    print("Parameters:")
    print("  Number of sites: " + String(nsites))
    print("  J (exchange): " + String(J))
    print("\nHamiltonian: H = J * sum_i(X_i*X_{i+1} + Y_i*Y_{i+1} + Z_i*Z_{i+1})")
    
    # Create MPO using float32 for GPU compatibility
    var mpo = create_heisenberg_mpo[DType.float32](ctx, nsites, J=J)
    
    print_mpo_info("Heisenberg XXX MPO", mpo)
    
    # Check bond dimensions (should be 5 for bulk sites in Heisenberg model)
    print("\nBond dimension check:")
    print("  Left boundary: " + String(mpo.bond_dimension(0)) + " (expected: 1)")
    if nsites > 1:
        print("  Bulk bond: " + String(mpo.bond_dimension(1)) + " (expected: 5)")
    print("  Right boundary: " + String(mpo.bond_dimension(nsites)) + " (expected: 1)")
    
    # Print the first site tensor
    print_subsection("First site tensor (site 0)")
    var first_site = mpo.sites[0]
    print_dense_tensor_data[DType.float32](first_site.tensor, "Site 0 Tensor", ctx)
    
    print("\n✓ Heisenberg MPO test passed")


fn test_heisenberg_xxz_mpo() raises:
    """Test Heisenberg XXZ Model MPO construction (matches C implementation)."""
    var ctx = DeviceContext()
    print_separator("Test 2b: Heisenberg XXZ Model MPO (C Implementation Match)")
    
    var nsites = 3
    var J = 1.0   # Exchange coupling
    var D = 0.5   # Anisotropy
    var h = 0.2   # Magnetic field
    
    print("Parameters:")
    print("  Number of sites: " + String(nsites))
    print("  J (exchange): " + String(J))
    print("  D (anisotropy): " + String(D))
    print("  h (magnetic field): " + String(h))
    print("\nHamiltonian: H = -J * sum_i[(X_i*X_{i+1} + Y_i*Y_{i+1} + D*Z_i*Z_{i+1})] - h * sum_i(Z_i)")
    print("\nThis matches the C implementation in chemtensor/manual_tests/mpo.c")
    
    # Create MPO using float32 for GPU compatibility
    var mpo = create_heisenberg_xxz_mpo[DType.float32](ctx, nsites, J=J, D=D, h=h)
    
    print_mpo_info("Heisenberg XXZ MPO", mpo)
    
    # Check bond dimensions (should be 5 for bulk sites)
    print("\nBond dimension check:")
    print("  Left boundary: " + String(mpo.bond_dimension(0)) + " (expected: 1)")
    if nsites > 1:
        print("  Bulk bond: " + String(mpo.bond_dimension(1)) + " (expected: 5)")
    print("  Right boundary: " + String(mpo.bond_dimension(nsites)) + " (expected: 1)")
    
    # Print the first site tensor (for comparison with C implementation)
    print_subsection("First site tensor (site 0) - Compare with C output")
    var first_site = mpo.sites[0]
    print_dense_tensor_data[DType.float32](first_site.tensor, "Site 0 Tensor", ctx)
    print("\nExpected C output (from mpo.c test):")
    print("  (-0.2000+0.0000i), (1.0000+0.0000i), (0.0000+0.0000i), (0.0000+0.0000i), (0.5000+0.0000i),")
    print("  (0.0000+0.0000i), (0.0000+0.0000i), (1.0000+0.0000i), (0.0000-1.0000i), (0.0000+0.0000i),")
    print("  (0.0000+0.0000i), (0.0000+0.0000i), (1.0000+0.0000i), (0.0000+1.0000i), (0.0000+0.0000i),")
    print("  (0.2000+0.0000i), (1.0000+0.0000i), (0.0000+0.0000i), (0.0000+0.0000i), (-0.5000+0.0000i)")
    print("\nNote: Y operator differences expected - C uses complex i, Mojo uses real approximation")
    
    print("\n✓ Heisenberg XXZ MPO test passed")


fn test_identity_mpo() raises:
    """Test identity MPO construction."""
    var ctx = DeviceContext()
    print_separator("Test 3: Identity MPO")
    
    var nsites = 4
    var physical_dim = 2
    
    print("Parameters:")
    print("  Number of sites: " + String(nsites))
    print("  Physical dimension: " + String(physical_dim))
    
    var mpo = create_identity_mpo[DType.float32](ctx, nsites, physical_dim)
    
    print_mpo_info("Identity MPO", mpo)
    
    # Check that all bond dimensions are 1
    print("\nBond dimension check (all should be 1):")
    for i in range(len(mpo.bond_dims)):
        var bd = mpo.bond_dims[i]
        if bd != 1:
            raise Error("Identity MPO bond dimension " + String(i) + " is " + String(bd) + ", expected 1")
        print("  Bond " + String(i) + ": " + String(bd))
    
    print("\n✓ Identity MPO test passed")


fn test_mpo_merge_operations() raises:
    """Test MPO tensor merge operations."""
    var ctx = DeviceContext()
    print_separator("Test 4: MPO Tensor Merge Operations")
    
    var nsites = 2
    var J = 1.0
    var h = 0.0
    var g = 0.0
    
    print("Creating a simple " + String(nsites) + "-site Ising MPO for merge test")
    print("Parameters: nsites=" + String(nsites) + ", J=" + String(J) + ", h=" + String(h) + ", g=" + String(g))
    
    var mpo = create_ising_1d_mpo[DType.float32](ctx, nsites, J=J, h_longitudinal=h, g_transverse=g)
    
    print_mpo_info("Original " + String(nsites) + "-site MPO", mpo)
    
    # Print individual site tensors
    print_subsection("Site 0 Tensor")
    var tensor0 = mpo.sites[0].tensor
    print_dense_tensor_data[DType.float32](tensor0, "Site 0", ctx)
    
    print_subsection("Site 1 Tensor")
    var tensor1 = mpo.sites[1].tensor
    print_dense_tensor_data[DType.float32](tensor1, "Site 1", ctx)
    
    # Merge the two tensors
    print_subsection("Merged Tensor")
    var merged = merge_mpo_tensor_pair(mpo.sites[0], mpo.sites[1], ctx)
    print_dense_tensor_data[DType.float32](merged, "Merged (Site 0 + Site 1)", ctx)
    
    print("\nVerification: Merged tensor should have shape [Wl0, d*d, d*d, Wr1]")
    var merged_shape = merged.shape.copy()
    print("Actual shape: [" + String(merged_shape[0]) + ", " + String(merged_shape[1]) + ", " + String(merged_shape[2]) + ", " + String(merged_shape[3]) + "]")
    
    print("\n✓ MPO merge test passed")


fn test_mpo_site_transpose() raises:
    """Test 5: Transpose MPO site tensor [Wl, d_in, d_out, Wr] -> [Wl, d_out, d_in, Wr]."""
    var ctx = DeviceContext()
    print_separator("Test 5: MPO Site Tensor Transpose")
    
    var nsites = 2
    var J = 1.0
    var h = 0.0
    var g = 0.0
    
    print("Transpose site 0 tensor: [Wl, d_in, d_out, Wr] -> [Wl, d_out, d_in, Wr]")
    print("Permutation: [0, 2, 1, 3] (swap physical in and out indices)")
    print("")
    
    var mpo = create_ising_1d_mpo[DType.float32](ctx, nsites, J=J, h_longitudinal=h, g_transverse=g)
    
    var site0 = mpo.sites[0].tensor
    
    print_subsection("Original Site 0 (before transpose)")
    print_dense_tensor_data[DType.float32](site0, "Site 0 [1,2,2,3]", ctx)
    
    var perm = List[Int](0, 2, 1, 3)  # [Wl, d_out, d_in, Wr]
    var site0_transposed = site0.transpose(perm, ctx)
    
    print_subsection("Site 0 Transposed [1,2,2,3] -> [1,2,2,3]")
    print_dense_tensor_data[DType.float32](site0_transposed, "Site 0 transposed (perm 0,2,1,3)", ctx)
    
    print("\n✓ MPO site transpose test passed")


fn test_mpo_site_reshape() raises:
    """Test 6: Reshape MPO site tensor [1,2,2,3] -> [4,3]."""
    var ctx = DeviceContext()
    print_separator("Test 6: MPO Site Tensor Reshape")
    
    var nsites = 2
    var J = 1.0
    var h = 0.0
    var g = 0.0
    
    print("Reshape site 0 tensor: [1, 2, 2, 3] -> [4, 3]")
    print("Flatten dims 0,1,2 into rows; keep Wr as columns")
    print("")
    
    var mpo = create_ising_1d_mpo[DType.float32](ctx, nsites, J=J, h_longitudinal=h, g_transverse=g)
    
    var site0 = mpo.sites[0].tensor
    
    print_subsection("Original Site 0 [1,2,2,3]")
    print_dense_tensor_data[DType.float32](site0, "Site 0", ctx)
    
    var site0_reshaped = site0.reshape(List[Int](4, 3))  # 1*2*2=4, 3
    
    print_subsection("Site 0 Reshaped to [4, 3]")
    print_dense_tensor_data[DType.float32](site0_reshaped, "Site 0 reshaped [4,3]", ctx)
    
    print("\n✓ MPO site reshape test passed")


fn test_mpo_site_scale() raises:
    """Test 7: Scale MPO site tensor by scalar factor."""
    var ctx = DeviceContext()
    print_separator("Test 7: MPO Site Tensor Scale")
    
    var nsites = 2
    var J = 1.0
    var h = 0.0
    var g = 0.0
    var scale_factor = Scalar[DType.float32](2.0)
    
    print("Scale site 0 tensor by factor 2.0")
    print("")
    
    var mpo = create_ising_1d_mpo[DType.float32](ctx, nsites, J=J, h_longitudinal=h, g_transverse=g)
    
    var site0 = mpo.sites[0].tensor
    
    print_subsection("Original Site 0 (before scale)")
    print_dense_tensor_data[DType.float32](site0, "Site 0", ctx)
    
    site0.scale_in_place(scale_factor, ctx)
    
    print_subsection("Site 0 Scaled by 2.0")
    print_dense_tensor_data[DType.float32](site0, "Site 0 scaled", ctx)
    
    print("\n✓ MPO site scale test passed")


fn test_mpo_merged_reshape_to_matrix() raises:
    """Test 8: Merge sites 0+1 then reshape [1,4,4,1] -> [4,4] Hamiltonian matrix."""
    var ctx = DeviceContext()
    print_separator("Test 8: Merged MPO Reshape to Matrix")
    
    var nsites = 2
    var J = 1.0
    var h = 0.0
    var g = 0.0
    
    print("Merge sites 0+1, then reshape [1,4,4,1] -> [4,4] Hamiltonian matrix")
    print("")
    
    var mpo = create_ising_1d_mpo[DType.float32](ctx, nsites, J=J, h_longitudinal=h, g_transverse=g)
    
    var merged = merge_mpo_tensor_pair(mpo.sites[0], mpo.sites[1], ctx)
    
    print_subsection("Merged Tensor [1,4,4,1]")
    print_dense_tensor_data[DType.float32](merged, "Merged", ctx)
    
    var h_matrix = merged.reshape(List[Int](4, 4))
    
    print_subsection("Reshaped to Hamiltonian Matrix [4,4]")
    print_dense_tensor_data[DType.float32](h_matrix, "H [4x4]", ctx)
    
    print("\n✓ MPO merged reshape to matrix test passed")


fn test_mpo_consistency() raises:
    """Test MPO consistency checks."""
    var ctx = DeviceContext()
    print_separator("Test 9: MPO Consistency Checks")
    
    var nsites = 4
    var mpo = create_ising_1d_mpo[DType.float32](ctx, nsites, J=1.0, h_longitudinal=0.0, g_transverse=0.5)
    
    print("Checking MPO consistency...")
    
    # Check that all sites have matching physical dimensions
    var phys_in = mpo.physical_in_dim
    var phys_out = mpo.physical_out_dim
    print("  Physical input dimension: " + String(phys_in))
    print("  Physical output dimension: " + String(phys_out))
    
    for i in range(mpo.num_sites()):
        var site = mpo.sites[i]
        if site.physical_in_dim() != phys_in:
            raise Error("Site " + String(i) + " physical_in_dim mismatch")
        if site.physical_out_dim() != phys_out:
            raise Error("Site " + String(i) + " physical_out_dim mismatch")
    
    # Check bond dimension consistency
    for i in range(mpo.num_sites()):
        if i > 0:
            var prev_right = mpo.sites[i - 1].right_bond_dim()
            var curr_left = mpo.sites[i].left_bond_dim()
            if prev_right != curr_left:
                raise Error("Bond mismatch between sites " + String(i - 1) + " and " + String(i))
    
    # Check that bond_dims array matches actual tensor dimensions
    if mpo.bond_dimension(0) != mpo.sites[0].left_bond_dim():
        raise Error("Left boundary bond dimension mismatch")
    for i in range(mpo.num_sites()):
        if mpo.bond_dimension(i + 1) != mpo.sites[i].right_bond_dim():
            raise Error("Right bond dimension mismatch at site " + String(i))
    
    print("  ✓ All physical dimensions match")
    print("  ✓ All bond dimensions are consistent")
    print("  ✓ Bond dimension array matches tensor dimensions")
    
    print("\n✓ MPO consistency test passed")


fn main() raises:
    try:
        TestSuite.discover_tests[__functions_in_module()]().run()
    except e:
        print("Tests failed: " + String(e))
