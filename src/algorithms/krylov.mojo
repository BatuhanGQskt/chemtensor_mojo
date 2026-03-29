from collections.list import List
from gpu.host import DeviceContext
from m_tensor.dense_tensor import DenseTensor, create_dense_tensor
from math import sqrt, abs as math_abs


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
    var original_shape = initial_vec.shape.copy()

    # --- Normalize initial vector on GPU ---
    var norm_v0 = initial_vec.norm(ctx)
    if norm_v0 < 1e-14:
        raise Error("Initial vector has zero norm")

    var v0_shape = initial_vec.shape.copy()
    var v_current = create_dense_tensor[dtype](ctx, v0_shape^, init_value=Scalar[dtype](0.0))
    ctx.enqueue_copy(v_current.storage, initial_vec.storage)
    v_current.scale_in_place(Scalar[dtype](1.0 / norm_v0), ctx)

    # Krylov vectors stored ON DEVICE (flat 1-D copies for dot / axpy)
    var krylov_dev = List[DenseTensor[dtype]](capacity=max_iter + 1)
    var v0_flat_shape = List[Int](dim)
    var v0_copy = create_dense_tensor[dtype](ctx, v0_flat_shape^, init_value=Scalar[dtype](0.0))
    ctx.enqueue_copy(v0_copy.storage, v_current.storage)
    krylov_dev.append(v0_copy^)

    # We still keep host copies for the final eigenvector reconstruction,
    # since the Krylov space is tiny (max_iter <= ~30 vectors of size dim).
    var krylov_host = List[List[Scalar[dtype]]](capacity=max_iter + 1)
    var host_v0 = ctx.enqueue_create_host_buffer[dtype](dim)
    ctx.enqueue_copy(host_v0, v_current.storage)
    ctx.synchronize()
    var v0_list = List[Scalar[dtype]](capacity=dim)
    for i in range(dim):
        v0_list.append(host_v0[i])
    krylov_host.append(v0_list^)

    # Tridiagonal matrix elements
    var alpha_list = List[Float64](capacity=max_iter)
    var beta_list = List[Float64](capacity=max_iter)

    var prev_eigenvalue: Float64 = 0.0

    for iter_idx in range(max_iter):
        # Apply Hamiltonian: w = H|v_current>
        var w = apply_H(v_current, ctx)

        # alpha_j = <v_current, w>  (GPU inner product)
        var alpha_j = v_current.dot_product(w, ctx)
        alpha_list.append(alpha_j)

        # w -= alpha_j * v_current   (GPU axpy)
        w.axpy_in_place(Scalar[dtype](-alpha_j), v_current, ctx)

        # w -= beta_{j-1} * v_{j-1}  (GPU axpy)
        if iter_idx > 0:
            var beta_prev = beta_list[iter_idx - 1]
            w.axpy_in_place(Scalar[dtype](-beta_prev), krylov_dev[iter_idx - 1], ctx)

        # Full reorthogonalization (GPU axpy per Krylov vector)
        if reorthogonalize:
            for k in range(len(krylov_dev)):
                var overlap = w.dot_product(krylov_dev[k], ctx)
                w.axpy_in_place(Scalar[dtype](-overlap), krylov_dev[k], ctx)

        # beta_j = ||w||
        var beta_j = w.norm(ctx)

        # Check convergence / breakdown
        if beta_j < 1e-14 or iter_idx == max_iter - 1:
            var eigenvalue = solve_tridiagonal_ground_state(alpha_list, beta_list, tol)
            var eigenvector = reconstruct_eigenvector[dtype](
                alpha_list, beta_list, krylov_host, dim, ctx
            )
            # Reshape back to original tensor shape
            var out = eigenvector^.reshape(original_shape^)
            return (eigenvalue, out^)

        beta_list.append(beta_j)

        # Normalize w to get next Krylov vector
        w.scale_in_place(Scalar[dtype](1.0 / beta_j), ctx)

        # Store Krylov vector on device
        var w_dev_shape = List[Int](dim)
        var w_dev_copy = create_dense_tensor[dtype](ctx, w_dev_shape^, init_value=Scalar[dtype](0.0))
        ctx.enqueue_copy(w_dev_copy.storage, w.storage)
        krylov_dev.append(w_dev_copy^)

        # Store host copy for eigenvector reconstruction
        var host_w = ctx.enqueue_create_host_buffer[dtype](dim)
        ctx.enqueue_copy(host_w, w.storage)
        ctx.synchronize()
        var w_list = List[Scalar[dtype]](capacity=dim)
        for i in range(dim):
            w_list.append(host_w[i])
        krylov_host.append(w_list^)

        # Update v_current for next iteration
        var v_next_shape = v_current.shape.copy()
        v_current = create_dense_tensor[dtype](ctx, v_next_shape^, init_value=Scalar[dtype](0.0))
        ctx.enqueue_copy(v_current.storage, w.storage)
        ctx.synchronize()

        # Check eigenvalue convergence
        if iter_idx > 0:
            var current_eigenvalue = solve_tridiagonal_ground_state(alpha_list, beta_list, tol)
            if math_abs(current_eigenvalue - prev_eigenvalue) < tol:
                var eigenvector = reconstruct_eigenvector[dtype](
                    alpha_list, beta_list, krylov_host, dim, ctx
                )
                var out = eigenvector^.reshape(original_shape^)
                return (current_eigenvalue, out^)
            prev_eigenvalue = current_eigenvalue

    # Should not reach here
    var eigenvalue = solve_tridiagonal_ground_state(alpha_list, beta_list, tol)
    var eigenvector = reconstruct_eigenvector[dtype](
        alpha_list, beta_list, krylov_host, dim, ctx
    )
    var out = eigenvector^.reshape(original_shape^)
    return (eigenvalue, out^)


fn solve_tridiagonal_ground_state(
    alpha: List[Float64],
    beta: List[Float64],
    tol: Float64
) raises -> Float64:
    """Solve tridiagonal eigenvalue problem to find ground state energy.
    
    Uses the implicit QR algorithm with Wilkinson shift to find the smallest
    eigenvalue of the tridiagonal matrix T defined by alpha (diagonal) and
    beta (off-diagonal).  O(n^2) per sweep, converges cubically.
    
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

    var result = _tridiag_qr_eig(alpha, beta, tol)
    var eigenvalues = result[0]

    # Return smallest eigenvalue
    var min_eval = eigenvalues[0]
    for i in range(1, len(eigenvalues)):
        if eigenvalues[i] < min_eval:
            min_eval = eigenvalues[i]
    return min_eval


fn _tridiag_qr_eig(
    alpha: List[Float64],
    beta: List[Float64],
    tol: Float64,
    max_iter: Int = 200,
) raises -> Tuple[List[Float64], List[List[Float64]]]:
    """Implicit QR algorithm with Wilkinson shift for symmetric tridiagonal matrices.
    
    Computes ALL eigenvalues and eigenvectors of the symmetric tridiagonal matrix T
    defined by diagonal `alpha` and off-diagonal `beta`.
    
    The algorithm:
      1. Apply Wilkinson shift (cubic convergence near eigenvalues).
      2. Implicit QR step via Givens rotations.
      3. Accumulate rotations into eigenvector matrix Q.
      4. Deflate converged off-diagonal elements.
    
    Complexity: O(n^2) per sweep, typically O(n) sweeps -> O(n^3) total
    (much better than power iteration's O(n * max_iter) with poor convergence).
    
    Args:
        alpha: Diagonal elements (length n).
        beta: Off-diagonal elements (length n-1).
        tol: Convergence tolerance for off-diagonal elements.
        max_iter: Maximum number of QR iterations.
    
    Returns:
        Tuple of (eigenvalues, eigenvectors) where eigenvectors[j] is the j-th
        eigenvector stored as a List[Float64] of length n.
    """
    var n = len(alpha)

    # Working copies of diagonal and off-diagonal
    var d = List[Float64](capacity=n)
    for i in range(n):
        d.append(alpha[i])

    var e = List[Float64](capacity=n)
    for i in range(n - 1):
        e.append(beta[i])
    e.append(0.0)  # sentinel

    # Eigenvector accumulator Q (n x n identity)
    # Stored as Q[row][col]; i.e. Q[i] is row i.
    var Q = List[List[Float64]](capacity=n)
    for i in range(n):
        var row = List[Float64](capacity=n)
        for j in range(n):
            row.append(1.0 if i == j else 0.0)
        Q.append(row^)

    # Main QR iteration
    var m = n  # active matrix size
    var total_iter = 0

    while m > 1 and total_iter < max_iter * n:
        total_iter += 1

        # Find the largest unreduced sub-block [l, m)
        var l = m - 2
        while l >= 0:
            if math_abs(e[l]) <= tol * (math_abs(d[l]) + math_abs(d[l + 1])):
                break
            l -= 1
        l += 1  # l is now the start of the unreduced block

        if l == m - 1:
            # 1x1 block converged — deflate
            m -= 1
            continue

        # Wilkinson shift: eigenvalue of trailing 2x2 block closest to d[m-1]
        var dm1 = d[m - 1]
        var dm2 = d[m - 2]
        var em2 = e[m - 2]
        var delta = (dm2 - dm1) * 0.5
        var sign_delta: Float64 = 1.0 if delta >= 0.0 else -1.0
        var shift = dm1 - em2 * em2 / (delta + sign_delta * sqrt(delta * delta + em2 * em2))

        # Implicit QR step with Givens rotations
        var x = d[l] - shift
        var z = e[l]

        for k in range(l, m - 1):
            # Compute Givens rotation to zero out z
            var r = sqrt(x * x + z * z)
            var c: Float64
            var s: Float64
            if r < 1e-30:
                c = 1.0
                s = 0.0
            else:
                c = x / r
                s = -z / r

            # Apply rotation to tridiagonal elements
            if k > l:
                e[k - 1] = r

            var d_k = d[k]
            var d_k1 = d[k + 1]
            var e_k = e[k]

            var w = c * d_k - s * e_k
            d[k + 1] = s * s * d_k + c * c * d_k1 + 2.0 * c * s * e_k
            d[k] = c * c * d_k + s * s * d_k1 - 2.0 * c * s * e_k
            # Correction: d[k] = w  would be the Golub-Kahan form.
            # More precise:
            d[k] = d_k * c * c + d_k1 * s * s - 2.0 * e_k * c * s
            d[k + 1] = d_k * s * s + d_k1 * c * c + 2.0 * e_k * c * s
            e[k] = (d_k - d_k1) * c * s + e_k * (c * c - s * s)

            # Bulge chase
            if k < m - 2:
                x = e[k]
                z = -s * e[k + 1]
                e[k + 1] = c * e[k + 1]

            # Accumulate eigenvector rotation: Q <- Q * G(k, k+1, theta)
            for i in range(n):
                var qi_k = Q[i][k]
                var qi_k1 = Q[i][k + 1]
                Q[i][k] = c * qi_k - s * qi_k1
                Q[i][k + 1] = s * qi_k + c * qi_k1

        # Check for deflation
        for k in range(l, m - 1):
            if math_abs(e[k]) <= tol * (math_abs(d[k]) + math_abs(d[k + 1])):
                e[k] = 0.0

    # d now contains eigenvalues
    var eigenvalues = d^

    # Q columns are eigenvectors — extract as Q_col[j] = [Q[0][j], Q[1][j], ... Q[n-1][j]]
    var eigenvectors = List[List[Float64]](capacity=n)
    for j in range(n):
        var v = List[Float64](capacity=n)
        for i in range(n):
            v.append(Q[i][j])
        eigenvectors.append(v^)

    return (eigenvalues^, eigenvectors^)


fn reconstruct_eigenvector[dtype: DType](
    alpha: List[Float64],
    beta: List[Float64],
    krylov_vectors: List[List[Scalar[dtype]]],
    dim: Int,
    ctx: DeviceContext,
) raises -> DenseTensor[dtype]:
    """Reconstruct the eigenvector from Krylov subspace.
    
    Solves the small tridiagonal eigenproblem via QR to get the ground-state
    coefficients, then forms the eigenvector as a linear combination of
    Krylov vectors.
    
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

    # Solve full tridiagonal eigenproblem (returns eigenvalues + eigenvectors)
    var result = _tridiag_qr_eig(alpha, beta, 1e-12)
    var eigenvalues = result[0]
    var eigenvectors = result[1]

    # Find index of smallest eigenvalue
    var min_idx = 0
    var min_eval = eigenvalues[0]
    for i in range(1, len(eigenvalues)):
        if eigenvalues[i] < min_eval:
            min_eval = eigenvalues[i]
            min_idx = i

    var coeffs = eigenvectors[min_idx]  # Krylov-space coefficients for ground state

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
    
    if result_norm > 1e-14:
        for j in range(dim):
            host_result[j] = host_result[j] / Scalar[dtype](result_norm)
    
    # Copy to device
    var shape_list = List[Int](dim)
    var result_tensor = create_dense_tensor[dtype](ctx, shape_list^, init_value=Scalar[dtype](0.0))
    ctx.enqueue_copy(result_tensor.storage, host_result)
    ctx.synchronize()
    
    return result_tensor^

