from sys import has_accelerator
from src.m_tensor.static_tensor import create_static_tensor, create_static_tensor_with_stride
from src.m_tensor.dense_tensor import DenseTensor, create_dense_tensor, create_dense_tensor_from_data, dense_tensor_dot, dense_tensor_qr, dense_tensor_svd_trunc_lapack_f64
from src.m_tensor.complex_tensor import ComplexDenseTensor, create_complex_tensor, create_complex_tensor_from_data, complex_matmul, create_complex_identity
from layout.layout import DimList, Layout
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext
from math import ceildiv
from time import perf_counter_ns
from collections.optional import Optional
from collections.list import List
from python import Python
from layout import IntTuple, LayoutTensor, RuntimeTuple
from layout.runtime_layout import RuntimeLayout, make_layout
from complex import ComplexSIMD
from src.state.mps_state import create_product_mps, create_uniform_mps, mps_orthogonalize_qr
from src.state.hamiltonians import create_ising_1d_mpo, create_transverse_ising_mpo
from src.algorithms.dmrg import DMRGParams, dmrg_two_site

alias dtype = DType.float32

fn create_runtime_layout_from_user_input[layout: Layout](
    shape: List[Int], 
    stride: List[Int], 
    rank: Int
) -> Optional[RuntimeLayout[layout]]:
    """Create a RuntimeLayout from user-provided shape and stride.
    
    Args:
        shape: List of dimensions.
        stride: List of strides.
        rank: Number of dimensions (1-4 supported).
    
    Returns:
        Optional RuntimeLayout - None if rank is unsupported.
    """
    if rank == 1:
        var rt_shape = RuntimeTuple[layout.shape](shape[0])
        var rt_stride = RuntimeTuple[layout.stride](stride[0])
        var rt_layout = RuntimeLayout[layout](shape=rt_shape, stride=rt_stride)
        return rt_layout
    elif rank == 2:
        var rt_shape = RuntimeTuple[layout.shape](shape[0], shape[1])
        var rt_stride = RuntimeTuple[layout.stride](stride[0], stride[1])
        var rt_layout = RuntimeLayout[layout](shape=rt_shape, stride=rt_stride)
        return rt_layout
    elif rank == 3:
        var rt_shape = RuntimeTuple[layout.shape](shape[0], shape[1], shape[2])
        var rt_stride = RuntimeTuple[layout.stride](stride[0], stride[1], stride[2])
        var rt_layout = RuntimeLayout[layout](shape=rt_shape, stride=rt_stride)
        return rt_layout
    elif rank == 4:
        var rt_shape = RuntimeTuple[layout.shape](shape[0], shape[1], shape[2], shape[3])
        var rt_stride = RuntimeTuple[layout.stride](stride[0], stride[1], stride[2], stride[3])
        var rt_layout = RuntimeLayout[layout](shape=rt_shape, stride=rt_stride)
        return rt_layout
    else:
        print("Error: Rank", rank, "is not supported. Only 1-4 are supported")
        return None

alias SIZE = 1024
alias MAX_RANK = 4
alias TPB = 16  # TODO: Cannot alocate dynamic memory in the shared memory space at dense_tensor_dot. Dynamic allocation not allow both for types and size which kills the point of optimizing tensor size and fitting it to the GPU space. This is from p13 of gpu quizes
alias THREADS_PER_BLOCK = (TPB, 2) # 16*2 = 32 for better memory utilization on Warps for threads/block
alias BLOCK_PER_GRID = (1, 1)
alias my_layout: Layout = Layout.row_major(THREADS_PER_BLOCK[0], THREADS_PER_BLOCK[1])
alias res_layout: Layout = Layout.row_major(THREADS_PER_BLOCK[0], THREADS_PER_BLOCK[0])

fn list_to_dimlist(dims: List[Int]) -> DimList:
    """Convert List[Int] to DimList (assumes MAX_RANK=4).
    
    Args:
        dims: List of integers that represents dimensions.

    Returns:
        DimList of size 2 or 4.
    
    Descriptions:
        var dims = List[Int](2, 4)
        var m_dimlist = list_to_dimlist(dims)
        m_dimlist = [2, 4, 1, 1] # if MAX_RANK == 4.
    """
    @parameter
    if MAX_RANK == 4:
        return DimList(dims[0], dims[1], dims[2], dims[3])
    else:
        # Fallback for other ranks - would need to be extended
        return DimList(dims[0], dims[1])


# Look into the dynamic implementation
fn get_dims_from_user(prompt: String, rank: Int) raises -> List[Int]:
    """Get dimensions from user input and pad to MAX_RANK."""
    var input = Python.import_module("builtins").input
    var int_fn = Python.import_module("builtins").int
    
    print(prompt)
    var dims = List[Int]()
    
    for i in range(rank):
        var dim_prompt = "  Enter dimension " + String(i) + ": "
        var dim_str = input(dim_prompt)
        var dim_val = int_fn(dim_str)
        var dim_int = Int(dim_val)
        dims.append(dim_int)
    
    # Pad remaining dimensions with 1
    for _ in range(MAX_RANK - rank):
        dims.append(1)
    
    return dims.copy()

fn test_complex_tensors() raises:
    """Test complex tensor operations.
    
    Demonstrates:
    - Creating complex tensors
    - Complex matrix multiplication
    - Identity matrix operations
    """
    @parameter
    if not has_accelerator():
        print("No compatible GPU found - skipping complex tensor tests")
        return
    
    print("\n" + "="*60)
    print("TESTING COMPLEX TENSOR OPERATIONS")
    print("="*60 + "\n")
    
    with DeviceContext() as ctx:
        # Test 1: Create and print a simple complex matrix
        print("Test 1: Creating a 2×2 complex matrix...")
        var data = List[ComplexSIMD[DType.float32, 1]]()
        data.append(ComplexSIMD[DType.float32, 1](1.0, 0.5))  # 1.0 + 0.5i
        data.append(ComplexSIMD[DType.float32, 1](2.0, 0.5))  # 2.0 + 0.5i
        data.append(ComplexSIMD[DType.float32, 1](3.0, 0.5))  # 3.0 + 0.5i
        data.append(ComplexSIMD[DType.float32, 1](4.0, 0.5))  # 4.0 + 0.5i
        
        var A_shape = List[Int](2, 2)
        var A = create_complex_tensor_from_data[DType.float32](
            ctx, data, A_shape^
        )
        print("Matrix A:")
        A.print_tensor(ctx)
        
        # Test 2: Create identity matrix
        print("\n" + "-"*60)
        print("Test 2: Creating 3×3 identity matrix...")
        var I = create_complex_identity[DType.float32](ctx, 3)
        print("Identity matrix:")
        I.print_tensor(ctx)
        
        # Test 3: Complex matrix multiplication
        print("\n" + "-"*60)
        print("Test 3: Complex matrix multiplication (2×3) × (3×2)...")
        var data_A2 = List[ComplexSIMD[DType.float32, 1]]()
        data_A2.append(ComplexSIMD[DType.float32, 1](1.0, 0.1))
        data_A2.append(ComplexSIMD[DType.float32, 1](2.0, 0.2))
        data_A2.append(ComplexSIMD[DType.float32, 1](3.0, 0.3))
        data_A2.append(ComplexSIMD[DType.float32, 1](4.0, 0.4))
        data_A2.append(ComplexSIMD[DType.float32, 1](5.0, 0.5))
        data_A2.append(ComplexSIMD[DType.float32, 1](6.0, 0.6))
        var A2_shape = List[Int](2, 3)
        var A2 = create_complex_tensor_from_data[DType.float32](
            ctx, data_A2, A2_shape^
        )
        
        var data_B2 = List[ComplexSIMD[DType.float32, 1]]()
        data_B2.append(ComplexSIMD[DType.float32, 1](1.0, 0.0))
        data_B2.append(ComplexSIMD[DType.float32, 1](0.0, 0.0))
        data_B2.append(ComplexSIMD[DType.float32, 1](0.0, 0.0))
        data_B2.append(ComplexSIMD[DType.float32, 1](1.0, 0.0))
        data_B2.append(ComplexSIMD[DType.float32, 1](1.0, 0.0))
        data_B2.append(ComplexSIMD[DType.float32, 1](1.0, 0.0))
        var B2_shape = List[Int](3, 2)
        var B2 = create_complex_tensor_from_data[DType.float32](
            ctx, data_B2, B2_shape^
        )
        
        print("Matrix A (2×3):")
        print("  Shape: [", end="")
        var shape_A2 = A2.shape()
        for i in range(len(shape_A2)):
            if i > 0:
                print(", ", end="")
            print(shape_A2[i], end="")
        print("]")
        
        print("Matrix B (3×2):")
        print("  Shape: [", end="")
        var shape_B2 = B2.shape()
        for i in range(len(shape_B2)):
            if i > 0:
                print(", ", end="")
            print(shape_B2[i], end="")
        print("]")
        var C2 = create_complex_tensor[DType.float32](ctx, List[Int](2, 2))
        complex_matmul[DType.float32](C2, A2^, B2^, ctx)
        
        print("Result C = A × B (2×2):")
        C2.print_tensor(ctx)
        
        # Test 4: Larger matrix multiplication
        print("\n" + "-"*60)
        print("Test 4: Larger matrix multiplication (4×4) × (4×4)...")
        var A4 = create_complex_tensor[DType.float32](
            ctx, List[Int](4, 4), real_init=1.0, imag_init=0.0
        )
        var B4 = create_complex_tensor[DType.float32](
            ctx, List[Int](4, 4), real_init=0.5, imag_init=0.25
        )
        var C4 = create_complex_tensor[DType.float32](ctx, List[Int](4, 4))
        
        complex_matmul[DType.float32](C4, A4^, B4^, ctx)
        print("Result C = A × B (showing first few elements):")
        C4.print_tensor(ctx)
        
        # Test 5: Test tensor properties
        print("\n" + "-"*60)
        print("Test 5: Tensor properties...")
        var test_tensor = create_complex_tensor[DType.float32](
            ctx, List[Int](3, 4, 5)
        )
        print("Tensor shape: [", end="")
        var shape_test = test_tensor.shape()
        for i in range(len(shape_test)):
            if i > 0:
                print(", ", end="")
            print(shape_test[i], end="")
        print("]")
        print("Tensor rank:", test_tensor.rank())
        print("Tensor size:", test_tensor.size())
        print("Is contiguous:", test_tensor.is_contiguous())
        
        print("\n" + "="*60)
        print("COMPLEX TENSOR TESTS COMPLETED")
        print("="*60 + "\n")


fn test_dense_qr() raises:
    """Test QR decomposition using MAX API.
    
    Demonstrates:
    - QR factorization of a 2D matrix
    - Verification that A = Q @ R
    - Verification that Q is orthogonal (Q^T @ Q = I)
    """    
    print("\n" + "="*60)
    print("TESTING QR DECOMPOSITION (MAX API)")
    print("="*60 + "\n")
    
    with DeviceContext() as ctx:
        print("Test 1: QR decomposition of 3×3 matrix...")
        var data = List[Float32]()
        # Create a well-conditioned 3×3 test matrix
        # [12, -51,   4]
        # [ 6, 167, -68]
        # [-4,  24, -41]
        data.append(12.0)
        data.append(-51.0)
        data.append(4.0)
        data.append(6.0)
        data.append(167.0)
        data.append(-68.0)
        data.append(-4.0)
        data.append(24.0)
        data.append(-41.0)
        
        var shape = List[Int](3, 3)
        var A = create_dense_tensor_from_data[DType.float32](ctx, data, shape^)
        
        print("Matrix A:")
        A.print_tensor(ctx)
        
        # Compute QR decomposition
        print("\nComputing QR decomposition...")
        var result = dense_tensor_qr[DType.float32](A, ctx)
        var Q = result[0]
        var R = result[1]
        
        print("\nMatrix Q (orthogonal):")
        Q.print_tensor(ctx)
        
        print("\nMatrix R (upper triangular):")
        R.print_tensor(ctx)
        
        # Verification 1: A ≈ Q @ R
        print("\n" + "-"*60)
        print("Verification 1: Computing Q @ R (should equal A)...")
        
        # Need to create new QR for A since we moved it
        var A2 = create_dense_tensor_from_data[DType.float32](ctx, data, List[Int](3, 3)^)
        var result2 = dense_tensor_qr[DType.float32](A2, ctx)
        var Q2 = result2[0]
        var R2 = result2[1]
        
        var A_reconstructed = create_dense_tensor[DType.float32](
            ctx, List[Int](3, 3)^, init_value=Scalar[DType.float32](0.0)
        )
        dense_tensor_dot[DType.float32](A_reconstructed, Q2^, R2^, ctx)
        ctx.synchronize()
        
        print("\nReconstructed A = Q @ R:")
        A_reconstructed.print_tensor(ctx)
        
        print("\nOriginal A (for comparison):")
        var A_original = create_dense_tensor_from_data[DType.float32](ctx, data, List[Int](3, 3)^)
        A_original.print_tensor(ctx)
        
        # Verification 2: Q^T @ Q ≈ I
        print("\n" + "-"*60)
        print("Verification 2: Computing Q^T @ Q (should be identity)...")
        
        # Need another copy of Q for transpose
        var A3 = create_dense_tensor_from_data[DType.float32](ctx, data, List[Int](3, 3)^)
        var result3 = dense_tensor_qr[DType.float32](A3, ctx)
        var Q3 = result3[0]
        var Q4 = result3[1]  # Not used, just to consume the tuple
        
        # Transpose Q3
        var perm = List[Int](1, 0)
        var Q_T = Q3^.transpose(perm, ctx)
        
        # Get another Q for the multiplication
        var A4 = create_dense_tensor_from_data[DType.float32](ctx, data, List[Int](3, 3)^)
        var result4 = dense_tensor_qr[DType.float32](A4, ctx)
        var Q5 = result4[0]
        
        var Q_T_Q = create_dense_tensor[DType.float32](
            ctx, List[Int](3, 3)^, init_value=Scalar[DType.float32](0.0)
        )
        dense_tensor_dot[DType.float32](Q_T_Q, Q_T^, Q5^, ctx)
        ctx.synchronize()
        
        print("\nQ^T @ Q (should be identity matrix):")
        Q_T_Q.print_tensor(ctx)
        
        print("\nExpected identity matrix:")
        print("[0,0] = 1.0, [1,1] = 1.0, [2,2] = 1.0")
        print("All off-diagonal elements should be ≈ 0.0")
        
        # Test 2: Rectangular matrix (more rows than columns)
        print("\n" + "="*60)
        print("Test 2: QR decomposition of 4×3 rectangular matrix...")
        
        var data_rect = List[Float32]()
        # Create a 4×3 matrix
        for i in range(12):
            data_rect.append(Float32(i + 1))
        
        var shape_rect = List[Int](4, 3)
        var A_rect = create_dense_tensor_from_data[DType.float32](
            ctx, data_rect, shape_rect^
        )
        
        print("\nMatrix A (4×3):")
        A_rect.print_tensor(ctx)
        
        var result_rect = dense_tensor_qr[DType.float32](A_rect, ctx)
        var Q_rect = result_rect[0]
        var R_rect = result_rect[1]
        
        print("\nMatrix Q (4×4):")
        Q_rect.print_tensor(ctx)
        
        print("\nMatrix R (4×3):")
        R_rect.print_tensor(ctx)
        
        # Verify reconstruction
        var A_rect_recon = create_dense_tensor[DType.float32](
            ctx, List[Int](4, 3)^, init_value=Scalar[DType.float32](0.0)
        )
        dense_tensor_dot[DType.float32](A_rect_recon, Q_rect^, R_rect^, ctx)
        
        print("\nReconstructed A (should match original):")
        A_rect_recon.print_tensor(ctx)
    
    print("\n" + "="*60)
    print("QR DECOMPOSITION TEST COMPLETED")
    print("="*60 + "\n")


fn test_mps_creation() raises:
    """Smoke-test the MPS helpers by building a few small states."""
    @parameter
    if not has_accelerator():
        print("No compatible GPU found - skipping MPS creation tests")
        return

    print("\n" + "="*60)
    print("TESTING MPS CREATION HELPERS")
    print("="*60 + "\n")

    with DeviceContext() as ctx:
        print("Test 1: create_uniform_mps with varying bond dimensions...")
        var num_sites = 4
        var physical_dim = 2
        var expected_bonds = List[Int](1, 2, 3, 2, 1)

        var uniform_mps = create_uniform_mps[dtype](
            ctx,
            num_sites,
            physical_dim,
            expected_bonds.copy()^,
            init_value=Scalar[dtype](0.25),
        )
        uniform_mps.describe()

        if uniform_mps.num_sites() != num_sites:
            raise Error(
                "Uniform MPS length mismatch: expected "
                + String(num_sites)
                + ", got "
                + String(uniform_mps.num_sites())
            )

        for idx in range(num_sites):
            var site_shape = uniform_mps.site_shape(idx)
            if len(site_shape) != 3:
                raise Error(
                    "Uniform MPS site "
                    + String(idx)
                    + " expected rank-3 tensor, got rank "
                    + String(len(site_shape))
                )

            if site_shape[0] != expected_bonds[idx] or site_shape[2] != expected_bonds[idx + 1]:
                raise Error(
                    "Uniform MPS bond mismatch at site "
                    + String(idx)
                    + ": expected ["
                    + String(expected_bonds[idx])
                    + ", "
                    + String(expected_bonds[idx + 1])
                    + "], got ["
                    + String(site_shape[0])
                    + ", "
                    + String(site_shape[2])
                    + "]"
                )

            if site_shape[1] != physical_dim:
                raise Error(
                    "Uniform MPS physical dim mismatch at site "
                    + String(idx)
                )

        print("Uniform MPS creation passed all bond/shape checks.\n")

        print("Test 2: create_product_mps from a local basis pattern...")
        var basis = List[Int](0, 1, 0, 1)
        var product_mps = create_product_mps[dtype](ctx, physical_dim, basis.copy()^)
        product_mps.describe()

        for idx in range(product_mps.num_sites()):
            var site_shape = product_mps.site_shape(idx)
            if site_shape[0] != 1 or site_shape[2] != 1:
                raise Error(
                    "Product MPS internal bonds must both be 1 at site "
                    + String(idx)
                )

            if site_shape[1] != physical_dim:
                raise Error(
                    "Product MPS physical dim mismatch at site "
                    + String(idx)
                )

        if product_mps.bond_dimension(product_mps.num_sites()) != 1:
            raise Error("Product MPS final bond dimension must be 1")

        print("Product-state MPS creation passed bond checks.\n")

        print("Test 3: mps_orthogonalize_qr via QR sweep...")
        var dense_shape = List[Int](physical_dim, physical_dim, physical_dim)
        var dense_state = create_dense_tensor[dtype](ctx, dense_shape.copy())
        var sweep_mps = mps_orthogonalize_qr[dtype](ctx, dense_state^)
        sweep_mps.describe()

        if sweep_mps.num_sites() != len(dense_shape):
            raise Error(
                "Dense-to-MPS conversion length mismatch: expected "
                + String(len(dense_shape))
                + ", got "
                + String(sweep_mps.num_sites())
            )

        if sweep_mps.bond_dimension(0) != 1 or sweep_mps.bond_dimension(sweep_mps.num_sites()) != 1:
            raise Error("Dense-to-MPS conversion must keep dummy bonds of size 1")

        for idx in range(sweep_mps.num_sites()):
            var site_shape = sweep_mps.site_shape(idx)
            if site_shape[1] != physical_dim:
                raise Error(
                    "Dense-to-MPS site "
                    + String(idx)
                    + " expected physical dimension "
                    + String(physical_dim)
                )

        print("Dense tensor sweep decomposition passed structural checks.\n")

    print("="*60)
    print("MPS CREATION TEST COMPLETED")
    print("="*60 + "\n")


fn test_dense_svd_trunc() raises:
    """Test truncated SVD implementation using NuMojo's optimized SVD."""
    print("\\n" + "="*60)
    print("TESTING TRUNCATED SVD (Lapack-based)")
    print("="*60 + "\\n")
    
    with DeviceContext() as ctx:
        print("Test 1: SVD of 4x3 matrix...")
        
        # Create a simple test matrix with known structure
        var data = List[Float64]()
        # Matrix with rank 2 (two non-zero singular values)
        # [2, 0, 0]
        # [0, 3, 0] 
        # [0, 0, 0]
        # [0, 0, 0]
        data.append(2.0)
        data.append(0.0)
        data.append(0.0)
        data.append(0.0)
        data.append(3.0)
        data.append(0.0)
        data.append(0.0)
        data.append(0.0)
        data.append(0.0)
        data.append(0.0)
        data.append(0.0)
        data.append(0.0)
        
        var shape = List[Int](4, 3)
        var A = create_dense_tensor_from_data[DType.float32](ctx, data, shape^)
        
        print("Matrix A:")
        A.print_tensor(ctx)
        
        # Compute truncated SVD using Lapack
        print("\nComputing truncated SVD with Lapack (chi_max=2, eps_trunc=1e-10)...")
        var svd_result = dense_tensor_svd_trunc_lapack_f64[DType.float32](A^, ctx, chi_max=2, eps_trunc=1e-10)
        var U = svd_result[0]
        var S = svd_result[1] 
        var Vt = svd_result[2]
        var chi_kept = svd_result[3]
        
        print("\nNumber of singular values kept:", chi_kept)
        
        print("\nSingular values S:")
        S.print_tensor(ctx)
        
        print("\nLeft singular vectors U:")
        U.print_tensor(ctx)
        
        print("\nRight singular vectors Vt:")
        Vt.print_tensor(ctx)
        
        print("\n✓ Lapack-based truncated SVD test completed")


fn run_dmrg_tfim_comparison() raises:
    """Run a Python-like TFIM DMRG example to compare outputs.
    
    Matches the user's Python example as closely as the current Mojo API allows.
    
    Note:
    - `state.hamiltonians.create_transverse_ising_mpo` implements:
        H = -J * Σ Z_i Z_{i+1} - h * Σ X_i
      which corresponds to the *transverse* field only.
    - The Python example also has a longitudinal field term (`h`) and a separate
      transverse-field strength (`g`). Here we map `g -> h` and ignore the
      longitudinal field since it's not implemented yet.
    - Entropy output is not implemented in this Mojo codebase yet.
    """
    @parameter
    if not has_accelerator():
        print("No compatible GPU found - skipping DMRG TFIM comparison run")
        return

    # IMPORTANT: Use float32 so MAX can compile GPU matmul on your setup.
    # (Float64 GPU matmul/GEMV currently fails in your MAX nightly with an offload error.)
    alias dmrg_dtype = DType.float32

    test_dmrg[dmrg_dtype](10, 1.0, 0.0, 1.0)
    test_dmrg[dmrg_dtype](5, 1.0, 0.0, 1.0)


fn test_dmrg[dmrg_dtype: DType](nsites: Int, J: Float64, h_longitudinal: Float64, g_transverse: Float64):
    # Model parameters (same as the Python snippet)
    var num_sweeps = 6
    var max_vdim = 64
    var tol_split: Float64 = 1e-10

    print("\n" + "=" * 60)
    print("DMRG TFIM COMPARISON RUN (Mojo)")
    print("=" * 60)
    print("nsites=", nsites, ", J=", J, ", h(longitudinal)=", h_longitudinal, ", g(transverse)=", g_transverse)
    print("num_sweeps=", num_sweeps, ", max_vdim=", max_vdim, ", tol_split=", tol_split)

    try:
        with DeviceContext() as ctx:
            # Build Hamiltonian MPO template (we'll copy it per sweep because dmrg_two_site consumes its args)
            var H_template = create_ising_1d_mpo[dmrg_dtype](
                ctx,
                nsites,
                J=J,
                h_longitudinal=h_longitudinal,
                g_transverse=g_transverse,
            )

            # Initial product state |000...0>
            var basis = List[Int](capacity=nsites)
            for _ in range(nsites):
                basis.append(0)
            var psi = create_product_mps[dmrg_dtype](ctx, 2, basis^)

            var en_sweeps = List[Float64](capacity=num_sweeps)

            # Run one sweep at a time so we can collect energies per sweep.
            for sweep in range(num_sweeps):
                var params = DMRGParams(
                    num_sweeps=1,
                    chi_max=max_vdim,
                    eps_trunc=tol_split,
                    max_krylov_iter=20,
                    krylov_tol=1e-10,
                    energy_tol=0.0,          # disable early stopping; user wants fixed sweep count
                    two_site=True,
                    verbose=False,
                )

                var H = H_template  # copy so we can move it into dmrg_two_site
                print("H: ", H)
                var res = dmrg_two_site[dmrg_dtype](ctx, H^, psi^, params)
                var E = res[0]
                psi = res[1]

                en_sweeps.append(E)
                print("Sweep ", sweep + 1, " energy: ", E)

            print("\nEnergies per sweep:", end=" ")
            print("[", end="")
            for i in range(len(en_sweeps)):
                if i > 0:
                    print(", ", end="")
                print(en_sweeps[i], end="")
            print("]")

            print("Final energy:", en_sweeps[len(en_sweeps) - 1])
            print("Sites:", psi.num_sites())
            print("Bond dimensions:", end=" ")
            print("[", end="")
            for i in range(len(psi.bond_dims)):
                if i > 0:
                    print(", ", end="")
                print(psi.bond_dims[i], end="")
            print("]")
            print("Entropies: (not implemented in Mojo yet)")
    except e:
        print("Error during DMRG test: " + String(e))


def main():
    # Keep the tensor/unit demos available, but run the DMRG comparison by default.
    run_dmrg_tfim_comparison()
    return


#     @parameter
#     if not has_accelerator():
#         print("No compatible GPU found")
#     else:
#         print("\n" + "="*60)
#         print("DYNAMIC TENSOR CALCULATION DEMO (GPU)")
#         print("="*60 + "\n")
        
#         # Get user input
#         var use_input = input("Use custom dimensions? (y/n): ")
#         var str_fn = Python.import_module("builtins").str
        
#         var dims_A: List[Int]
#         var dims_B: List[Int]
#         var dims_C: List[Int]

#         if String(str_fn(use_input)).lower() == "y":
#             print("\nMatrix multiplication: C = A @ B")
#             print("For A[M, K] @ B[K, N] = C[M, N]\n")
            
#             # Get dimensions for matrix A
#             var input_fn = Python.import_module("builtins").input
#             var int_fn = Python.import_module("builtins").int
            
#             var m_str = input_fn("Enter M (rows of A): ")
#             var k_str = input_fn("Enter K (cols of A / rows of B): ")
#             var n_str = input_fn("Enter N (cols of B): ")
            
#             var M = Int(int_fn(m_str))
#             var K = Int(int_fn(k_str))
#             var N = Int(int_fn(n_str))
            
#             dims_A = List[Int](M, K)
#             dims_B = List[Int](K, N)
#             dims_C = List[Int](M, N)
            
#             print("\nConfiguration:")
#             print("  Matrix A: [", M, ", ", K, "]")
#             print("  Matrix B: [", K, ", ", N, "]")
#             print("  Result C: [", M, ", ", N, "]")
#         else:
#             # Use default values
#             dims_A = List[Int](3, 4)
#             dims_B = List[Int](4, 3)
#             dims_C = List[Int](3, 3)
#             print("Using default dimensions: A=[3,4], B=[4,3], Result=[3,3]")

#         print("\n" + "-"*60)
#         var start_ns = perf_counter_ns()
        
#         with DeviceContext() as ctx:
#             print("Creating Dynamic Tensors with sequential data patterns...")
            
#             # Create tensor A with sequential values: [1, 2, 3, 4, 5, 6, ...]
#             var M = dims_A[0]
#             var K = dims_A[1]
#             var N = dims_B[1]
            
#             var data_A = List[Float32]()
#             for i in range(M * K):
#                 data_A.append(Float32(i + 1))
            
#             # Create tensor B with pattern: [0.5, 1.0, 1.5, 2.0, ...]
#             var data_B = List[Float32]()
#             for i in range(K * N):
#                 data_B.append(Float32(i + 1) * 0.5)
            
#             # Create dynamic tensors from data
#             var tensor_A = create_dense_tensor_from_data[DType.float32](ctx, data_A, dims_A^)
#             var tensor_B = create_dense_tensor_from_data[DType.float32](ctx, data_B, dims_B^)
#             var tensor_C = create_dense_tensor[DType.float32](ctx, dims_C^, init_value=0.0)
            
#             print("\nTensor A created: ", tensor_A)
#             print("Tensor B created: ", tensor_B)
#             print("Tensor C (output): ", tensor_C)
            
#             # Check if tensors are contiguous
#             print("\n" + "-"*60)
#             print("Checking tensor properties...")
#             print("  Tensor A is contiguous: ", tensor_A.is_contiguous())
#             print("  Tensor B is contiguous: ", tensor_B.is_contiguous())
            
#             # Print tensor A contents
#             print("\n" + "-"*60)
#             print("Tensor A contents:")
#             tensor_A.print_tensor(ctx)
            
#             # Print tensor B contents
#             print("\n" + "-"*60)
#             print("Tensor B contents:")
#             tensor_B.print_tensor(ctx)
            
#             # Perform matrix multiplication C = A @ B
#             print("\n" + "-"*60)
#             print("Performing matrix multiplication: C = A @ B")
#             dense_tensor_dot[DType.float32](tensor_C, tensor_A^, tensor_B^, ctx)
#             ctx.synchronize()
            
#             # Print result C
#             print("\n" + "-"*60)
#             print("Result C contents:")
#             tensor_C.print_tensor(ctx)
            
#             # Demonstrate transpose operation on result
#             print("\n" + "-"*60)
#             print("Demonstrating transpose operation on C...")
#             var perm = List[Int](1, 0)  # Transpose: swap rows and columns
#             var tensor_C_T = tensor_C^.transpose(perm, ctx)
#             print("Transposed C:", tensor_C_T)
#             print("Transposed C contents:")
#             tensor_C_T.print_tensor(ctx)
            
#             # Demonstrate creating a 3D tensor and flattening
#             print("\n" + "-"*60)
#             print("Demonstrating 3D tensor operations...")
#             var shape_3d = List[Int](2, 3, 4)
#             var data_3d = List[Float32]()
#             for i in range(24):
#                 data_3d.append(Float32(i))
#             var tensor_3d = create_dense_tensor_from_data[DType.float32](ctx, data_3d, shape_3d^)
#             print("3D Tensor: ", tensor_3d)
#             print("Is contiguous: ", tensor_3d.is_contiguous())
            
#             # Flatten last two dimensions
#             var tensor_flat = tensor_3d^.flatten_dims(1, 3, ctx)
#             print("After flattening dims [1:3]: ", tensor_flat)
#             print("Flattened tensor contents:")
#             tensor_flat.print_tensor(ctx)
                    
#         var finish_ns = perf_counter_ns()
#         var res_ns = finish_ns - start_ns
#         var res_s = Float64(res_ns) / 1_000_000_000.0
        
#         print("\n" + "-"*60)
#         print("All Dynamic Tensor calculations completed!")
#         print("Execution time: ", res_s, " seconds (", res_ns, " ns)")
#         print("="*60 + "\n")


# fn test_func(val: Optional[Int]) -> Int:
#     var new_val = val.or_else(0)
#     print("After or else", new_val)
#     return val.or_else(0)


# fn test_nd_tensor_dot() raises:
#     """Test ND tensor multiplication with the example from the user.
    
#     Tests A(4,3,2,1) @ B(1,2,5,7) with ndim_mult=2 should give C(4,3,5,7).
#     """
#     @parameter
#     if not has_accelerator():
#         print("No compatible GPU found - skipping ND tensor test")
#         return
    
#     print("\n" + "="*60)
#     print("TESTING ND TENSOR DOT PRODUCT")
#     print("="*60 + "\n")
    
#     with DeviceContext() as ctx:
#         # Test case from user: A(4,3,2,1) @ B(1,2,5,7) -> C(4,3,5,7)
#         print("Test 1: A(4,3,2,1) @ B(1,2,5,7) with ndim_mult=2")
#         print("  Expected output shape: (4,3,5,7)")
#         print("  Contracting A's last 2 dims (2,1) with B's first 2 dims (1,2)")
        
#         var shape_A = List[Int](4, 3, 2, 1)
#         var shape_B = List[Int](1, 2, 5, 7)
#         var shape_C = List[Int](4, 3, 5, 7)  # Result shape
        
#         var A = create_dense_tensor[DType.float32](ctx, shape_A^, init_value=2.0)
#         var B = create_dense_tensor[DType.float32](ctx, shape_B^, init_value=3.0)
#         var C = create_dense_tensor[DType.float32](ctx, shape_C^, init_value=0.0)
        
#         print("  A shape:", A)
#         print("  B shape:", B)
#         print("  C shape:", C)
        
#         # Perform contraction
#         dense_tensor_dot[DType.float32](C, A^, B^, ctx, ndim_mult=2)
#         ctx.synchronize()
        
#         print("\n  Result C after contraction:")
#         C.print_tensor(ctx)
        
#         # Verify: Each element in C should be 2.0 * 3.0 * (2*1) = 12.0
#         # Because we're summing over 2*1=2 elements
#         print("  Expected value per element: 2.0 * 3.0 * 2 = 12.0")
        
#         # Test case 2: 3D tensors
#         print("\n" + "-"*60)
#         print("Test 2: A(2,3,4) @ B(4,5,6) with ndim_mult=1")
#         print("  Expected output shape: (2,3,5,6)")
        
#         var shape_A2 = List[Int](2, 3, 4)
#         var shape_B2 = List[Int](4, 5, 6)
#         var shape_C2 = List[Int](2, 3, 5, 6)
        
#         var A2 = create_dense_tensor[DType.float32](ctx, shape_A2^, init_value=1.0)
#         var B2 = create_dense_tensor[DType.float32](ctx, shape_B2^, init_value=0.5)
#         var C2 = create_dense_tensor[DType.float32](ctx, shape_C2^, init_value=0.0)
        
#         print("  A2 shape:", A2)
#         print("  B2 shape:", B2)
#         print("  C2 shape:", C2)
        
#         dense_tensor_dot[DType.float32](C2, A2^, B2^, ctx, ndim_mult=1)
#         ctx.synchronize()
        
#         print("\n  Result C2 after contraction:")
#         C2.print_tensor(ctx)
#         print("  Expected value per element: 1.0 * 0.5 * 4 = 2.0")
        
#         print("\n" + "="*60)
#         print("ND TENSOR DOT TEST COMPLETED")
#         print("="*60 + "\n")