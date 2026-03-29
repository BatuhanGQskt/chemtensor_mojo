"""
Helper functions for MPO testing.

This module provides utilities for:
- Converting MPO to full matrix representation
- Computing eigenvalues for small Hamiltonians
- Creating simple product state MPS for testing
- MPO-MPS contraction and expectation values
"""

from collections.list import List
from gpu.host import DeviceContext
from python import Python, PythonObject
from src.m_tensor.dense_tensor import (
    DenseTensor,
    create_dense_tensor,
    create_dense_tensor_from_data,
    dense_tensor_dot,
)
from src.state.mpo_state import MatrixProductOperator, MPOSite


fn mpo_to_full_matrix[
    dtype: DType
](
    mpo: MatrixProductOperator[dtype],
    ctx: DeviceContext
) raises -> DenseTensor[dtype]:
    """Contract all MPO sites to obtain the full Hamiltonian matrix.
    
    This function contracts all sites of an MPO to form the full matrix representation
    of the operator on the Hilbert space. The result is a matrix of shape [d^N, d^N]
    where d is the physical dimension and N is the number of sites.
    
    Args:
        mpo: The MPO to convert to a matrix.
        ctx: Device context.
    
    Returns:
        Dense tensor of shape [d^N, d^N] representing the full operator matrix.
    
    Note:
        This operation is only feasible for small systems (N ≤ 10) due to exponential
        growth of the Hilbert space dimension.
    """
    var nsites = mpo.num_sites()
    var d = mpo.physical_in_dim
    
    if d != mpo.physical_out_dim:
        raise Error("Physical input and output dimensions must match")
    
    # Start with the first site: shape [Wl=1, d, d, Wr]
    var result = mpo.sites[0].tensor
    
    # Contract remaining sites sequentially
    for i in range(1, nsites):
        var next_site = mpo.sites[i].tensor
        
        # result currently has shape [..., d_prev, d_prev, Wr_prev]
        # next_site has shape [Wl_next, d, d, Wr_next]
        # We need to contract over Wr_prev = Wl_next
        
        var result_shape = result.shape.copy()
        var next_shape = next_site.shape.copy()
        
        var rank = len(result_shape)
        var wr_prev = result_shape[rank - 1]
        var wl_next = next_shape[0]
        
        if wr_prev != wl_next:
            raise Error("Bond dimension mismatch in MPO contraction")
        
        # Reshape result: [..., d*d, Wr_prev]
        var total_left = 1
        for j in range(rank - 1):
            total_left *= result_shape[j]
        
        var result_mat = result.reshape(List[Int](total_left, wr_prev))
        
        # Reshape next_site: [Wl_next, d*d*Wr_next]
        var next_mat = next_site.reshape(List[Int](wl_next, d * d * next_shape[3]))
        
        # Contract: [total_left, wr_prev] @ [wl_next, d*d*Wr_next]
        var temp = create_dense_tensor[dtype](
            ctx, List[Int](total_left, d * d * next_shape[3]),
            init_value=Scalar[dtype](0.0)
        )
        dense_tensor_dot(temp, result_mat^, next_mat^, ctx)
        
        # Reshape back to include the new physical dimensions
        var new_shape = List[Int]()
        for j in range(rank - 1):
            new_shape.append(result_shape[j])
        new_shape.append(d)
        new_shape.append(d)
        new_shape.append(next_shape[3])
        
        result = temp^.reshape(new_shape^)
    
    # At this point, result has shape [1, d_in_0, d_out_0, d_in_1, d_out_1, ..., 1]
    # (alternating in/out per site). Matrix element H[row,col] must be <row|H|col>, so
    # row = output state index, col = input state index. Permute to [out..., in...]
    # then reshape to [d^nsites, d^nsites].
    var final_shape = result.shape.copy()
    var hilbert_dim = 1
    for _ in range(nsites):
        hilbert_dim *= d
    
    var rank = len(final_shape)
    # Physical dims: 1=d_in_0, 2=d_out_0, 3=d_in_1, 4=d_out_1, ... Put outputs first, then inputs.
    var perm = List[Int]()
    perm.append(0)
    for i in range(nsites):
        perm.append(2 * i + 2)
    for i in range(nsites):
        perm.append(2 * i + 1)
    perm.append(rank - 1)
    result = result^.transpose(perm^, ctx)
    return result^.reshape(List[Int](hilbert_dim, hilbert_dim))


fn compute_eigenvalues[
    dtype: DType
](
    matrix: DenseTensor[dtype],
    ctx: DeviceContext
) raises -> List[Float64]:
    """Compute eigenvalues of a symmetric/Hermitian matrix using LAPACK.
    
    Args:
        matrix: Square matrix of shape [N, N].
        ctx: Device context.
    
    Returns:
        List of eigenvalues sorted in ascending order.
    
    Note:
        This function assumes the matrix is symmetric (for real dtype) or Hermitian
        (for complex dtype). For real symmetric matrices, uses LAPACK dsyev.
    """
    var shape = matrix.shape.copy()
    if len(shape) != 2:
        raise Error("Matrix must be 2D, got rank " + String(len(shape)))
    if shape[0] != shape[1]:
        raise Error("Matrix must be square: " + String(shape[0]) + "x" + String(shape[1]))
    
    var n = shape[0]
    
    # Copy matrix to host (LAPACK works on CPU)
    var host_matrix = ctx.enqueue_create_host_buffer[dtype](n * n)
    ctx.enqueue_copy(host_matrix, matrix.storage)
    ctx.synchronize()
    
    # Use Python's numpy for eigenvalue computation
    # (Mojo's LAPACK bindings may not be fully available yet)
    var py = Python.import_module("builtins")
    var np = Python.import_module("numpy")
    
    # Convert to numpy array
    var py_list = py.list()
    for i in range(n * n):
        py_list.append(py.float(Float64(host_matrix[i])))
    
    var np_matrix = np.array(py_list)
    np_matrix = np_matrix.reshape(n, n)
    
    # Compute eigenvalues
    var eigenvalues_py = np.linalg.eigvalsh(np_matrix)
    
    # Convert back to Mojo list (Python/numpy scalars → string → Float64 to avoid ambiguous cast)
    var eigenvalues = List[Float64]()
    for i in range(len(eigenvalues_py)):
        var py_float = py.float(eigenvalues_py[i])
        eigenvalues.append(atof(String(py.str(py_float))))
    
    return eigenvalues^


fn check_hermiticity[
    dtype: DType
](
    matrix: DenseTensor[dtype],
    ctx: DeviceContext,
    tol: Float64 = 1e-10
) raises -> Bool:
    """Check if a matrix is Hermitian (or symmetric for real matrices).
    
    Args:
        matrix: Square matrix to check.
        ctx: Device context.
        tol: Tolerance for comparison (default: 1e-10).
    
    Returns:
        True if the matrix is Hermitian/symmetric within tolerance, False otherwise.
    """
    var shape = matrix.shape.copy()
    if len(shape) != 2 or shape[0] != shape[1]:
        raise Error("Matrix must be square")
    
    var n = shape[0]
    
    # Copy to host
    var host = ctx.enqueue_create_host_buffer[dtype](n * n)
    ctx.enqueue_copy(host, matrix.storage)
    ctx.synchronize()
    
    # Check symmetry: M[i,j] == M[j,i]
    for i in range(n):
        for j in range(i + 1, n):
            var val_ij = Float64(host[i * n + j])
            var val_ji = Float64(host[j * n + i])
            if abs(val_ij - val_ji) > tol:
                return False
    
    return True


fn max_asymmetry[
    dtype: DType
](
    matrix: DenseTensor[dtype],
    ctx: DeviceContext
) raises -> Float64:
    """Return max |M[i,j] - M[j,i]| for a square matrix (diagnostic for Hermiticity)."""
    var shape = matrix.shape.copy()
    if len(shape) != 2 or shape[0] != shape[1]:
        raise Error("Matrix must be square")
    var n = shape[0]
    var host = ctx.enqueue_create_host_buffer[dtype](n * n)
    ctx.enqueue_copy(host, matrix.storage)
    ctx.synchronize()
    var max_diff: Float64 = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            var val_ij = Float64(host[i * n + j])
            var val_ji = Float64(host[j * n + i])
            var diff = abs(val_ij - val_ji)
            if diff > max_diff:
                max_diff = diff
    return max_diff


fn symmetrize_matrix[
    dtype: DType
](
    matrix: DenseTensor[dtype],
    ctx: DeviceContext
) raises -> DenseTensor[dtype]:
    """Return (M + M^T) / 2 so the result is symmetric. Used when MPO→matrix has convention asymmetry."""
    var shape = matrix.shape.copy()
    if len(shape) != 2 or shape[0] != shape[1]:
        raise Error("Matrix must be square")
    var n = shape[0]
    var host = ctx.enqueue_create_host_buffer[dtype](n * n)
    ctx.enqueue_copy(host, matrix.storage)
    ctx.synchronize()
    var data = List[Scalar[dtype]]()
    for i in range(n):
        for j in range(n):
            var val_ij = host[i * n + j]
            var val_ji = host[j * n + i]
            data.append(Scalar[dtype](Float64(0.5) * (Float64(val_ij) + Float64(val_ji))))
    return create_dense_tensor_from_data[dtype](ctx, data^, List[Int](n, n)^)


fn compute_trace[
    dtype: DType
](
    matrix: DenseTensor[dtype],
    ctx: DeviceContext
) raises -> Float64:
    """Compute the trace of a square matrix.
    
    Args:
        matrix: Square matrix of shape [N, N].
        ctx: Device context.
    
    Returns:
        The trace (sum of diagonal elements).
    """
    var shape = matrix.shape.copy()
    if len(shape) != 2 or shape[0] != shape[1]:
        raise Error("Matrix must be square")
    
    var n = shape[0]
    
    # Copy to host
    var host = ctx.enqueue_create_host_buffer[dtype](n * n)
    ctx.enqueue_copy(host, matrix.storage)
    ctx.synchronize()
    
    # Sum diagonal elements
    var trace: Float64 = 0.0
    for i in range(n):
        trace += Float64(host[i * n + i])
    
    return trace


fn print_matrix_info[
    dtype: DType
](
    matrix: DenseTensor[dtype],
    ctx: DeviceContext,
    name: String = "Matrix",
    max_print: Int = 8
) raises:
    """Print information about a matrix (shape, first few elements, properties).
    
    Args:
        matrix: The matrix to print.
        ctx: Device context.
        name: Name to display for the matrix.
        max_print: Maximum dimension to print fully (default: 8).
    """
    var shape = matrix.shape.copy()
    print("\n=== " + name + " ===")
    
    var shape_str = String("Shape: [")
    for i in range(len(shape)):
        if i > 0:
            shape_str += ", "
        shape_str += String(shape[i])
    shape_str += "]"
    print(shape_str)
    
    if len(shape) == 2:
        # Check if Hermitian
        var is_hermitian = check_hermiticity(matrix, ctx)
        print("Hermitian: " + ("Yes" if is_hermitian else "No"))
        
        # Compute trace
        var tr = compute_trace(matrix, ctx)
        print("Trace: " + String(tr))
        
        # Print matrix elements if small enough
        if shape[0] <= max_print and shape[1] <= max_print:
            var host = ctx.enqueue_create_host_buffer[dtype](shape[0] * shape[1])
            ctx.enqueue_copy(host, matrix.storage)
            ctx.synchronize()
            
            print("\nMatrix elements:")
            for i in range(shape[0]):
                var row_str = String("  [")
                for j in range(shape[1]):
                    if j > 0:
                        row_str += ", "
                    row_str += String(Float64(host[i * shape[1] + j]))
                row_str += "]"
                print(row_str)
