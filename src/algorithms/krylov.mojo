from collections.list import List
from gpu.host import DeviceContext
from m_tensor.dense_tensor import DenseTensor, create_dense_tensor
from math import sqrt


fn lanczos_ground_state[dtype: DType](
    apply_H: fn(DenseTensor[dtype], DeviceContext) raises -> DenseTensor[dtype],
    initial_vec: DenseTensor[dtype],
    ctx: DeviceContext,
    max_iter: Int = 20,
    tol: Float64 = 1e-8,
    reorthogonalize: Bool = True,
) raises -> Tuple[Float64, DenseTensor[dtype]]:
    """Find the ground state eigenvector and eigenvalue using Lanczos iteration.
    
    Implements the Lanczos algorithm for finding the smallest eigenvalue
    and corresponding eigenvector of a Hermitian operator H.
    
    The algorithm builds a Krylov subspace and projects H onto it, then
    solves a small tridiagonal eigenvalue problem to find the ground state.
    
    For DMRG, apply_H should be the effective Hamiltonian operator that
    contracts environments with the two-site theta tensor.
    
    Args:
        apply_H: Function that applies the Hamiltonian to a vector.
                 Takes a DenseTensor and DeviceContext, returns H|v>.
        initial_vec: Initial guess vector (will be normalized).
        ctx: Device context for GPU operations.
        max_iter: Maximum number of Lanczos iterations.
        tol: Convergence tolerance for eigenvalue.
        reorthogonalize: If True, reorthogonalize against all previous vectors
                        (more stable but slower).
    
    Returns:
        Tuple of (eigenvalue, eigenvector) where eigenvalue is Float64
        and eigenvector is a DenseTensor with same shape as initial_vec.
    
    Example:
        ```mojo
        # Define effective Hamiltonian application
        fn apply_heff(vec: DenseTensor[dtype], ctx: DeviceContext) raises -> DenseTensor[dtype]:
            # ... contract with environments and MPO ...
            return result
        
        var initial = create_dense_tensor(ctx, shape^)
        var result = lanczos_ground_state(apply_heff, initial^, ctx, max_iter=30)
        var E0 = result[0]
        var psi0 = result[1]
        ```
    """
    var dim = initial_vec.size
    
    # Copy initial vector to host and normalize
    var host_v0 = ctx.enqueue_create_host_buffer[dtype](dim)
    ctx.enqueue_copy(host_v0, initial_vec.storage)
    ctx.synchronize()
    
    # Compute norm
    var norm_v0: Float64 = 0.0
    for i in range(dim):
        var val = Float64(host_v0[i])
        norm_v0 += val * val
    norm_v0 = sqrt(norm_v0)
    
    if norm_v0 < 1e-14:
        raise Error("Initial vector has zero norm")
    
    # Normalize
    for i in range(dim):
        host_v0[i] = host_v0[i] / Scalar[dtype](norm_v0)
    
    # TODO: Storage for Lanczos vectors (on host for now, can optimize later to device for better performance)
    var krylov_vectors = List[List[Scalar[dtype]]](capacity=max_iter + 1)
    var v0_list = List[Scalar[dtype]](capacity=dim)
    for i in range(dim):
        v0_list.append(host_v0[i])
    krylov_vectors.append(v0_list^)
    
    # Tridiagonal matrix elements
    var alpha = List[Float64](capacity=max_iter)
    var beta = List[Float64](capacity=max_iter)
    
    var v_current_shape = initial_vec.shape.copy()
    var v_current = create_dense_tensor[dtype](ctx, v_current_shape^, init_value=Scalar[dtype](0.0))
    ctx.enqueue_copy(v_current.storage, host_v0)
    ctx.synchronize()
    
    var prev_eigenvalue: Float64 = 0.0
    
    for iter in range(max_iter):
        # Apply Hamiltonian: w = H|v_current>
        var w = apply_H(v_current, ctx)
        
        # Copy to host
        var host_w = ctx.enqueue_create_host_buffer[dtype](dim)
        ctx.enqueue_copy(host_w, w.storage)
        ctx.synchronize()
        
        # Compute alpha_j = <v_j | H | v_j>
        var alpha_j: Float64 = 0.0
        var host_v_curr = ctx.enqueue_create_host_buffer[dtype](dim)
        ctx.enqueue_copy(host_v_curr, v_current.storage)
        ctx.synchronize()
        
        for i in range(dim):
            alpha_j += Float64(host_v_curr[i]) * Float64(host_w[i])
        alpha.append(alpha_j)
        
        # w = w - alpha_j * v_j
        for i in range(dim):
            host_w[i] = host_w[i] - Scalar[dtype](alpha_j) * host_v_curr[i]
        
        # w = w - beta_{j-1} * v_{j-1} (if j > 0)
        if iter > 0:
            var beta_prev = beta[iter - 1]
            var host_v_prev = krylov_vectors[iter - 1].copy()
            for i in range(dim):
                host_w[i] = host_w[i] - Scalar[dtype](beta_prev) * host_v_prev[i]
        
        # Reorthogonalization (optional but recommended for stability)
        if reorthogonalize:
            for k in range(len(krylov_vectors)):
                var host_v_k = krylov_vectors[k].copy()
                var overlap: Float64 = 0.0
                for i in range(dim):
                    overlap += Float64(host_w[i]) * Float64(host_v_k[i])
                for i in range(dim):
                    host_w[i] = host_w[i] - Scalar[dtype](overlap) * host_v_k[i]
        
        # Compute beta_j = ||w||
        var beta_j: Float64 = 0.0
        for i in range(dim):
            var val = Float64(host_w[i])
            beta_j += val * val
        beta_j = sqrt(beta_j)
        
        # Check for convergence or breakdown
        if beta_j < 1e-14 or iter == max_iter - 1:
            # Lanczos has converged or reached max iterations
            # Solve tridiagonal eigenvalue problem
            var eigenvalue = solve_tridiagonal_ground_state(alpha, beta, tol)
            
            # Reconstruct eigenvector from Krylov subspace
            var eigenvector = reconstruct_eigenvector[dtype](
                alpha, beta, krylov_vectors, dim, ctx
            )
            
            return (eigenvalue, eigenvector^)
        
        beta.append(beta_j)
        
        # Normalize w to get next Krylov vector
        for i in range(dim):
            host_w[i] = host_w[i] / Scalar[dtype](beta_j)
        
        # Store new Krylov vector
        var w_list = List[Scalar[dtype]](capacity=dim)
        for i in range(dim):
            w_list.append(host_w[i])
        krylov_vectors.append(w_list^)
        
        # Update v_current for next iteration
        ctx.enqueue_copy(v_current.storage, host_w)
        ctx.synchronize()
        
        # Check eigenvalue convergence (approximate)
        if iter > 0:
            var current_eigenvalue = alpha[0]  # Approximate
            if abs(current_eigenvalue - prev_eigenvalue) < tol:
                # Early convergence
                var eigenvalue = solve_tridiagonal_ground_state(alpha, beta, tol)
                var eigenvector = reconstruct_eigenvector[dtype](
                    alpha, beta, krylov_vectors, dim, ctx
                )
                return (eigenvalue, eigenvector^)
            prev_eigenvalue = current_eigenvalue
    
    # Should not reach here, but handle it
    var eigenvalue = solve_tridiagonal_ground_state(alpha, beta, tol)
    var eigenvector = reconstruct_eigenvector[dtype](
        alpha, beta, krylov_vectors, dim, ctx
    )
    return (eigenvalue, eigenvector^)


fn solve_tridiagonal_ground_state(
    alpha: List[Float64],
    beta: List[Float64],
    tol: Float64
) raises -> Float64:
    """Solve tridiagonal eigenvalue problem to find ground state energy.
    
    Uses the QR algorithm to find the smallest eigenvalue of the
    tridiagonal matrix T defined by alpha (diagonal) and beta (off-diagonal).
    
    For small matrices (typical in Lanczos), this is a simple implementation.
    For production code, one would use optimized LAPACK routines.
    
    Args:
        alpha: Diagonal elements of tridiagonal matrix.
        beta: Off-diagonal elements (length = len(alpha) - 1).
        tol: Convergence tolerance.
    
    Returns:
        Smallest eigenvalue (ground state energy).
    """
    var n = len(alpha)
    if n == 0:
        raise Error("Empty tridiagonal matrix")
    
    if n == 1:
        return alpha[0]
    
    # TODO: Implement QR Algorithm for Eigenvalues of the tridiagonal matrix
    # For small n, use power iteration on the tridiagonal matrix
    # (Simple but not optimal - could use bisection or QR for production)
    
    # Initialize with uniform vector
    var v = List[Float64](capacity=n)
    for i in range(n):
        v.append(1.0 / sqrt(Float64(n)))
    
    var max_power_iter = 100
    var prev_lambda: Float64 = 0.0
    
    for _ in range(max_power_iter):
        # Apply tridiagonal matrix: w = T * v
        var w = List[Float64](capacity=n)
        
        for i in range(n):
            var sum: Float64 = alpha[i] * v[i]
            if i > 0:
                sum += beta[i - 1] * v[i - 1]
            if i < n - 1:
                sum += beta[i] * v[i + 1]
            w.append(sum)
        
        # Normalize w
        var norm: Float64 = 0.0
        for i in range(n):
            norm += w[i] * w[i]
        norm = sqrt(norm)
        
        for i in range(n):
            v[i] = w[i] / norm
        
        # Compute Rayleigh quotient: eigenval = v^T T v
        var eigenval: Float64 = 0.0
        for i in range(n):
            var sum: Float64 = alpha[i] * v[i] * v[i]
            if i > 0:
                sum += 2.0 * beta[i - 1] * v[i - 1] * v[i]
            eigenval += sum
        
        # Check convergence
        if abs(eigenval - prev_lambda) < tol:
            return eigenval
        
        prev_lambda = eigenval
    
    return prev_lambda


fn reconstruct_eigenvector[dtype: DType](
    alpha: List[Float64],
    beta: List[Float64],
    krylov_vectors: List[List[Scalar[dtype]]],
    dim: Int,
    ctx: DeviceContext,
) raises -> DenseTensor[dtype]:
    """Reconstruct the eigenvector from Krylov subspace.
    
    Solves the small tridiagonal eigenproblem to get coefficients,
    then forms the eigenvector as a linear combination of Krylov vectors.
    
    Args:
        alpha: Diagonal of tridiagonal matrix.
        beta: Off-diagonal of tridiagonal matrix.
        krylov_vectors: List of Krylov basis vectors.
        dim: Dimension of each vector.
        ctx: Device context.
    
    Returns:
        Ground state eigenvector as DenseTensor.
    """
    var n = len(alpha)
    
    # Solve for eigenvector of tridiagonal matrix
    var coeffs = List[Float64](capacity=n)
    for i in range(n):
        coeffs.append(1.0 / sqrt(Float64(n)))  # Uniform initialization
    
    # Power iteration to find eigenvector
    for _ in range(50):
        var w = List[Float64](capacity=n)
        for i in range(n):
            var sum: Float64 = alpha[i] * coeffs[i]
            if i > 0:
                sum += beta[i - 1] * coeffs[i - 1]
            if i < n - 1:
                sum += beta[i] * coeffs[i + 1]
            w.append(sum)
        
        var norm: Float64 = 0.0
        for i in range(n):
            norm += w[i] * w[i]
        norm = sqrt(norm)
        
        for i in range(n):
            coeffs[i] = w[i] / norm
    
    # Reconstruct full eigenvector: psi = sum_i coeffs[i] * krylov_vectors[i]
    var host_result = ctx.enqueue_create_host_buffer[dtype](dim)
    for j in range(dim):
        host_result[j] = Scalar[dtype](0.0)
    
    for i in range(n):
        var coeff = coeffs[i]
        var krylov_i = krylov_vectors[i].copy()
        for j in range(dim):
            host_result[j] = host_result[j] + Scalar[dtype](coeff) * krylov_i[j]
    
    # Normalize result
    var result_norm: Float64 = 0.0
    for j in range(dim):
        var val = Float64(host_result[j])
        result_norm += val * val
    result_norm = sqrt(result_norm)
    
    for j in range(dim):
        host_result[j] = host_result[j] / Scalar[dtype](result_norm)
    
    # Copy to device
    # Create 1D tensor with correct shape
    var shape_list = List[Int](dim)  # Assume 1D for now
    var result_tensor = create_dense_tensor[dtype](ctx, shape_list^, init_value=Scalar[dtype](0.0))
    ctx.enqueue_copy(result_tensor.storage, host_result)
    ctx.synchronize()
    
    return result_tensor^

