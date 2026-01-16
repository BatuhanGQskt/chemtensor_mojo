from collections.list import List
from gpu.host import DeviceContext
from math import sqrt, atan2, sin, cos
from src.m_tensor.dense_tensor import (
    DenseTensor,
    create_dense_tensor,
    dense_tensor_dot,
    dense_tensor_svd_trunc,
)
from src.state.mps_state import MPSSite, MatrixProductState
from src.state.mpo_state import MPOSite, MatrixProductOperator
from src.state.environments import (
    update_left_environment,
    update_right_environment,
    build_right_environments,      
    expectation_value,
    expectation_value_normalized,
)
from src.algorithms.krylov import lanczos_ground_state


@fieldwise_init
struct DMRGParams:
    """Parameters for DMRG algorithm.
    
    Controls convergence, truncation, and algorithmic choices.
    """
    var num_sweeps: Int
    var chi_max: Int            # Maximum bond dimension
    var eps_trunc: Float64      # Discarded weight threshold
    var max_krylov_iter: Int    # Max Lanczos iterations per site
    var krylov_tol: Float64     # Lanczos convergence tolerance
    var energy_tol: Float64     # Energy convergence threshold for early stopping
    var two_site: Bool          # Two-site DMRG (True) or single-site (False, not implemented)
    var reorthogonalize: Bool   # Reorthogonalize Krylov vectors
    var verbose: Bool           # Print progress


struct DMRGWorkspace[dtype: DType]:
    """Workspace for DMRG algorithm to avoid repeated allocations.
    
    Stores left and right environments that are updated incrementally
    during sweeps.
    """
    var L_env: List[DenseTensor[dtype]]  # Left environments, length N+1
    var R_env: List[DenseTensor[dtype]]  # Right environments, length N+1
    var num_sites: Int
    
    fn __init__(
        out self,
        mps: MatrixProductState[dtype],
        mpo: MatrixProductOperator[dtype],
        ctx: DeviceContext,
    ) raises:
        """Initialize workspace with boundary conditions.
        
        Builds all right environments and sets up left boundary.
        """
        self.num_sites = mps.num_sites()
        
        if mpo.num_sites() != self.num_sites:
            raise Error("MPS and MPO must have the same number of sites")
        
        # Build right environments from scratch
        self.R_env = build_right_environments[dtype](mps, mpo, ctx)
        
        # Initialize left environments (will be built during sweep)
        self.L_env = List[DenseTensor[dtype]](capacity=self.num_sites + 1)
        
        # Left boundary: L[0] is identity
        var wL_boundary = mpo.bond_dimension(0)
        var Dl_boundary = mps.bond_dimension(0)
        
        var L_boundary_shape = List[Int](wL_boundary, Dl_boundary, Dl_boundary)
        var L_boundary = create_dense_tensor[dtype](ctx, L_boundary_shape^, init_value=Scalar[dtype](0.0))
        
        # Set to identity
        var host_L = ctx.enqueue_create_host_buffer[dtype](wL_boundary * Dl_boundary * Dl_boundary)
        for w in range(wL_boundary):
            for d in range(Dl_boundary):
                var idx = w * (Dl_boundary * Dl_boundary) + d * Dl_boundary + d
                host_L[idx] = Scalar[dtype](1.0)
        ctx.enqueue_copy(L_boundary.storage, host_L)
        ctx.synchronize()
        
        # Fill L_env with boundary (will update as we sweep)
        for _ in range(self.num_sites + 1):
            self.L_env.append(L_boundary)
        
        self.L_env[0] = L_boundary^


fn build_two_site_theta[dtype: DType](
    A_i: MPSSite[dtype],
    A_ip1: MPSSite[dtype],
    ctx: DeviceContext,
) raises -> DenseTensor[dtype]:
    """Build two-site wavefunction theta from adjacent MPS tensors.
    
    Contracts the middle bond to form theta[Dl, d_i, d_{i+1}, Dr].
    
    Args:
        A_i: MPS tensor at site i, shape [Dl, d, Dm].
        A_ip1: MPS tensor at site i+1, shape [Dm, d, Dr].
        ctx: Device context.
    
    Returns:
        Two-site tensor theta with shape [Dl, d, d, Dr].
    """
    var A_i_shape = A_i.tensor.shape.copy()
    var A_ip1_shape = A_ip1.tensor.shape.copy()
    
    var Dl = A_i_shape[0]
    var d_i = A_i_shape[1]
    var Dm = A_i_shape[2]
    var d_ip1 = A_ip1_shape[1]
    var Dr = A_ip1_shape[2]
    
    if A_ip1_shape[0] != Dm:
        raise Error("Bond dimension mismatch in build_two_site_theta")
    
    if d_i != d_ip1:
        raise Error("Physical dimensions must match")
    
    var d = d_i
    
    # Reshape A_i: [Dl, d, Dm] -> [Dl*d, Dm]
    var A_i_mat = A_i.tensor.reshape(List[Int](Dl * d, Dm))
    
    # Reshape A_ip1: [Dm, d, Dr] -> [Dm, d*Dr]
    var A_ip1_mat = A_ip1.tensor.reshape(List[Int](Dm, d * Dr))
    
    # Contract: [Dl*d, Dm] @ [Dm, d*Dr] -> [Dl*d, d*Dr]
    var theta_mat = create_dense_tensor[dtype](ctx, List[Int](Dl * d, d * Dr), init_value=Scalar[dtype](0.0))
    dense_tensor_dot(theta_mat, A_i_mat^, A_ip1_mat^, ctx)
    
    # Reshape to [Dl, d, d, Dr]
    var theta = theta_mat^.reshape(List[Int](Dl, d, d, Dr))
    
    return theta^


fn split_two_site_theta_svd[dtype: DType](
    theta: DenseTensor[dtype],
    chi_max: Int,
    eps_trunc: Float64,
    ctx: DeviceContext,
    left_to_right: Bool = True,
) raises -> Tuple[MPSSite[dtype], MPSSite[dtype], Float64]:
    """Split optimized two-site tensor using SVD with truncation.
    
    Decomposes theta[Dl, d, d, Dr] -> A_i[Dl, d, chi] and A_{i+1}[chi, d, Dr]
    where chi is determined by chi_max and eps_trunc.
    
    Args:
        theta: Two-site tensor, shape [Dl, d, d, Dr].
        chi_max: Maximum bond dimension to keep.
        eps_trunc: Maximum allowed discarded weight.
        ctx: Device context.
        left_to_right: If True, canonicalize left site (left-to-right sweep).
                      If False, canonicalize right site (right-to-left sweep).
    
    Returns:
        Tuple of (A_i, A_{i+1}, discarded_weight) where:
        - A_i: New left site tensor
        - A_{i+1}: New right site tensor
        - discarded_weight: Fraction of discarded singular values squared
    """
    var theta_shape = theta.shape.copy()
    var Dl = theta_shape[0]
    var d = theta_shape[1]
    var Dr = theta_shape[3]
    
    # Reshape theta to matrix: [Dl*d, d*Dr]
    var theta_mat = theta.reshape(List[Int](Dl * d, d * Dr))
    
    # Compute truncated SVD
    var svd_result = dense_tensor_svd_trunc[dtype](theta_mat^, ctx, chi_max, eps_trunc)
    var U = svd_result[0]      # [Dl*d, chi]
    var S = svd_result[1]      # [chi]
    var Vt = svd_result[2]     # [chi, d*Dr]
    var chi_kept = svd_result[3]
    
    # Compute discarded weight
    var host_S = ctx.enqueue_create_host_buffer[dtype](chi_kept)
    ctx.enqueue_copy(host_S, S.storage)
    ctx.synchronize()
    
    var total_norm_sq: Float64 = 0.0
    var kept_norm_sq: Float64 = 0.0
    for i in range(chi_kept):
        var s_val = Float64(host_S[i])
        var s_sq = s_val * s_val
        total_norm_sq += s_sq
        kept_norm_sq += s_sq
    
    var discarded_weight: Float64 = 0.0
    if total_norm_sq > 0.0:
        discarded_weight = (total_norm_sq - kept_norm_sq) / total_norm_sq
    
    # Distribute singular values based on sweep direction
    var A_i_new: DenseTensor[dtype]
    var A_ip1_new: DenseTensor[dtype]
    
    if left_to_right:
        # Left-to-right: absorb S into right tensor
        # A_i = U, A_{i+1} = S * Vt
        
        # U: [Dl*d, chi] -> [Dl, d, chi]
        A_i_new = U^.reshape(List[Int](Dl, d, chi_kept))
        
        # S * Vt: [chi, d*Dr]
        # First multiply S into Vt
        var host_Vt = ctx.enqueue_create_host_buffer[dtype](chi_kept * d * Dr)
        ctx.enqueue_copy(host_Vt, Vt.storage)
        ctx.synchronize()
        
        for i in range(chi_kept):
            var s_val = host_S[i]
            for j in range(d * Dr):
                var idx = i * (d * Dr) + j
                host_Vt[idx] = host_Vt[idx] * s_val
        
        var SVt = create_dense_tensor[dtype](ctx, List[Int](chi_kept, d * Dr), init_value=Scalar[dtype](0.0))
        ctx.enqueue_copy(SVt.storage, host_Vt)
        ctx.synchronize()
        
        # Reshape to [chi, d, Dr]
        A_ip1_new = SVt^.reshape(List[Int](chi_kept, d, Dr))
    else:
        # Right-to-left: absorb S into left tensor
        # A_i = U * S, A_{i+1} = Vt
        
        # U * S: [Dl*d, chi]
        var host_U = ctx.enqueue_create_host_buffer[dtype](Dl * d * chi_kept)
        ctx.enqueue_copy(host_U, U.storage)
        ctx.synchronize()
        
        for i in range(Dl * d):
            for j in range(chi_kept):
                var idx = i * chi_kept + j
                host_U[idx] = host_U[idx] * host_S[j]
        
        var US = create_dense_tensor[dtype](ctx, List[Int](Dl * d, chi_kept), init_value=Scalar[dtype](0.0))
        ctx.enqueue_copy(US.storage, host_U)
        ctx.synchronize()
        
        A_i_new = US^.reshape(List[Int](Dl, d, chi_kept))
        A_ip1_new = Vt^.reshape(List[Int](chi_kept, d, Dr))
    
    return (MPSSite[dtype](A_i_new^), MPSSite[dtype](A_ip1_new^), discarded_weight)

fn apply_two_site_heff[dtype: DType](
    theta: DenseTensor[dtype],
    L_env: DenseTensor[dtype],
    R_env: DenseTensor[dtype],
    W_i: MPOSite[dtype],
    W_ip1: MPOSite[dtype],
    ctx: DeviceContext,
) raises -> DenseTensor[dtype]:
    """Apply effective two-site Hamiltonian to theta without forming Heff explicitly.
    
    This is the matrix-free application H_eff|theta> used in Lanczos.
    
    Convention:
        theta[Dl, d, d, Dr]
        L[wL, Dl, Dl']
        R[wR, Dr, Dr']
        W_i[wL, d, d', wM]   (Modified MPO layout)
        W_{i+1}[wM, d, d', wR]
    
    Contraction network:
        L[wL, Dl, Dl'] -- theta[Dl, d_i, d_{i+1}, Dr] -- R[wR, Dr, Dr']
                |                  |         |                |
              W_i[wL, d_i, d_i', wM]    W_{i+1}[wM, d_{i+1}, d_{i+1}', wR]
    
    Result: theta_out[Dl', d_i', d_{i+1}', Dr']
    
    Args:
        theta: Two-site wavefunction, shape [Dl, d, d, Dr].
        L_env: Left environment, shape [wL, Dl, Dl'].
        R_env: Right environment, shape [wR, Dr, Dr'].
        W_i: MPO tensor at site i, shape [wL, d, d', wM].
        W_ip1: MPO tensor at site i+1, shape [wM, d, d', wR].
        ctx: Device context.
    
    Returns:
        H_eff|theta> with same shape as theta.
    """
    # This is a complex multi-index contraction
    # Break it down into sequential contractions
    
    var theta_shape = theta.shape.copy()
    var Dl = theta_shape[0]
    var d = theta_shape[1]
    var Dr = theta_shape[3]
    
    var L_shape = L_env.shape.copy()
    var R_shape = R_env.shape.copy()
    var Wi_shape = W_i.tensor.shape.copy()
    var Wip1_shape = W_ip1.tensor.shape.copy()
    
    var wL = L_shape[0]
    var Dl_bra = L_shape[2]
    var wR = R_shape[0]
    var Dr_bra = R_shape[2]
    var wM = Wi_shape[3]
    var d_out = Wi_shape[2]
    
    # Step 1: Contract L with theta
    # L[wL, Dl, Dl'] × theta[Dl, d_i, d_{i+1}, Dr] -> temp1[wL, Dl', d_i, d_{i+1}, Dr]
    
    # Reshape L: [wL, Dl, Dl'] -> [wL*Dl, Dl']
    # Wait, need to contract axes correctly.
    # Transpose L to [Dl, wL, Dl']
    var L_trans = L_env.transpose(List[Int](1, 0, 2), ctx)
    var L_flat = L_trans^.reshape(List[Int](Dl, wL * Dl_bra))
    var theta_flat = theta.reshape(List[Int](Dl, d * d * Dr))
    
    var temp1_contract = create_dense_tensor[dtype](ctx, List[Int](wL * Dl_bra, d * d * Dr), init_value=Scalar[dtype](0.0))
    # Contract leading axis of L_flat (Dl) with leading axis of theta_flat (Dl).
    dense_tensor_dot(temp1_contract, L_flat^, theta_flat^, ctx, ndim_mult=1, axrange_A=True, axrange_B=True)
    
    var temp1 = temp1_contract^.reshape(List[Int](wL, Dl_bra, d, d, Dr))
    
    # Step 2: Contract temp1 with W_i on physical index of site i
    # temp1[wL, Dl', d_i, d_{i+1}, Dr] × W_i[wL, d_i, d_i', wM] -> temp2[Dl', d_{i+1}, Dr, d_i', wM]
    # Contract wL and d_i
    
    # Transpose temp1 to [Dl', d_{i+1}, Dr, wL, d_i]
    var temp1_perm = temp1^.transpose(List[Int](1, 3, 4, 0, 2), ctx)
    var temp1_mat2 = temp1_perm^.reshape(List[Int](Dl_bra * d * Dr, wL * d))
    
    # W_i [wL, d_i, d_i', wM] -> reshape [wL*d_i, d_i'*wM]
    var Wi_mat = W_i.tensor.reshape(List[Int](wL * d, d_out * wM))
    
    var temp2_mat = create_dense_tensor[dtype](ctx, List[Int](Dl_bra * d * Dr, d_out * wM), init_value=Scalar[dtype](0.0))
    dense_tensor_dot(temp2_mat, temp1_mat2^, Wi_mat^, ctx)
    
    var temp2 = temp2_mat^.reshape(List[Int](Dl_bra, d, Dr, d_out, wM))
    
    # Step 3: Contract temp2 with W_{i+1} on physical index of site i+1
    # temp2[Dl', d_{i+1}, Dr, d_i', wM] × W_{i+1}[wM, d_{i+1}, d_{i+1}', wR] -> temp3[Dl', Dr, d_i', d_{i+1}', wR]
    # Contract wM and d_{i+1}
    
    # Transpose temp2 to [Dl', Dr, d_i', wM, d_{i+1}]
    var temp2_perm = temp2^.transpose(List[Int](0, 2, 3, 4, 1), ctx)
    var temp2_mat3 = temp2_perm^.reshape(List[Int](Dl_bra * Dr * d_out, wM * d))
    
    # W_{i+1} [wM, d_{i+1}, d_{i+1}', wR] -> reshape [wM*d_{i+1}, d_{i+1}'*wR]
    var Wip1_mat = W_ip1.tensor.reshape(List[Int](wM * d, d_out * wR))
    
    var temp3_mat = create_dense_tensor[dtype](ctx, List[Int](Dl_bra * Dr * d_out, d_out * wR), init_value=Scalar[dtype](0.0))
    dense_tensor_dot(temp3_mat, temp2_mat3^, Wip1_mat^, ctx)
    
    var temp3 = temp3_mat^.reshape(List[Int](Dl_bra, Dr, d_out, d_out, wR))
    
    # Step 4: Contract temp3 with R
    # temp3[Dl', Dr, d_i', d_{i+1}', wR] × R[wR, Dr, Dr'] -> result[Dl', d_i', d_{i+1}', Dr']
    # Contract wR and Dr
    
    # Transpose temp3 to [Dl', d_i', d_{i+1}', wR, Dr]
    var temp3_perm = temp3^.transpose(List[Int](0, 2, 3, 4, 1), ctx)
    var temp3_mat4 = temp3_perm^.reshape(List[Int](Dl_bra * d_out * d_out, wR * Dr))
    
    # Transpose R to [wR, Dr, Dr'] -> Reshape [wR*Dr, Dr']
    var R_mat = R_env.reshape(List[Int](wR * Dr, Dr_bra))
    
    var result_mat = create_dense_tensor[dtype](ctx, List[Int](Dl_bra * d_out * d_out, Dr_bra), init_value=Scalar[dtype](0.0))
    dense_tensor_dot(result_mat, temp3_mat4^, R_mat^, ctx)
    
    var result = result_mat^.reshape(List[Int](Dl_bra, d_out, d_out, Dr_bra))
    
    return result^


fn dmrg_two_site[dtype: DType](
    ctx: DeviceContext,
    mpo: MatrixProductOperator[dtype],
    var mps: MatrixProductState[dtype],
    params: DMRGParams,
) raises -> Tuple[Float64, MatrixProductState[dtype]]:
    """Two-site DMRG optimization to find ground state.
    
    Performs sweeps of local two-site optimizations using Lanczos,
    then splits with SVD and truncation. Continues until convergence
    or maximum sweeps reached.
    
    Args:
        ctx: Device context for GPU operations.
        mpo: Matrix product operator (Hamiltonian).
        mps: Initial matrix product state (will be optimized).
        params: DMRG parameters (sweeps, chi_max, tolerances, etc.).
    
    Returns:
        Tuple of (ground_energy, optimized_mps).
    
    Example:
        ```mojo
        var params = DMRGParams(
            num_sweeps=10,
            chi_max=100,
            eps_trunc=1e-10,
            max_krylov_iter=20,
            krylov_tol=1e-8,
            energy_tol=1e-6,
            two_site=True,
            reorthogonalize=True,
            verbose=True
        )
        
        var result = dmrg_two_site(ctx, hamiltonian^, initial_mps^, params)
        var E0 = result[0]
        var ground_state = result[1]
        ```
    """
    if not params.two_site:
        raise Error("Only two-site DMRG is currently implemented")
    
    var N = mps.num_sites()
    if N < 2:
        raise Error("DMRG requires at least 2 sites")
    
    if params.verbose:
        print("=== Starting DMRG ===")
        print("  Sites:", N)
        print("  Max sweeps:", params.num_sweeps)
        print("  Max bond dim:", params.chi_max)
        print("  Truncation threshold:", params.eps_trunc)
    
    # Initialize workspace
    var workspace = DMRGWorkspace[dtype](mps, mpo, ctx)
    
    var energy: Float64 = 0.0
    var prev_energy: Float64 = 0.0
    
    for sweep in range(params.num_sweeps):
        if params.verbose:
            print("\n--- Sweep", sweep + 1, "/", params.num_sweeps, "---")
        
        # Left-to-right sweep
        for i in range(N - 1):
            energy = dmrg_local_update_two_site[dtype](
                i, mps, mpo, workspace, params, ctx, left_to_right=True
            )
            
            if params.verbose and i % max(1, (N - 1) // 10) == 0:
                print("  L->R site", i, "energy:", energy)
        
        # Right-to-left sweep
        for i in range(N - 2, -1, -1):
            energy = dmrg_local_update_two_site[dtype](
                i, mps, mpo, workspace, params, ctx, left_to_right=False
            )
            
            if params.verbose and i % max(1, (N - 1) // 10) == 0:
                print("  R->L site", i, "energy:", energy)
        
        # IMPORTANT: `energy` from local updates is a *local two-site Ritz value*,
        # not the global <psi|H|psi>. For comparisons (and for convergence checks),
        # compute the full expectation after each sweep.
        var sweep_energy = expectation_value_normalized[dtype](mps, mpo, ctx)
        energy = sweep_energy

        if params.verbose:
            print("Sweep", sweep + 1, "completed. Energy:", sweep_energy)
        
        # Check convergence
        if sweep > 0:
            var energy_change = abs(sweep_energy - prev_energy)
            if energy_change < params.energy_tol:
                if params.verbose:
                    print("Converged! Energy change:", energy_change)
                break
        
        prev_energy = sweep_energy
    
    if params.verbose:
        print("\n=== DMRG Complete ===")
        print("Final energy:", energy)
        mps.describe()
    
    return (energy, mps^)


fn lanczos_two_site_optimize[dtype: DType](
    initial_theta: DenseTensor[dtype],
    L_env: DenseTensor[dtype],
    R_env: DenseTensor[dtype],
    W_i: MPOSite[dtype],
    W_ip1: MPOSite[dtype],
    ctx: DeviceContext,
    max_iter: Int,
    tol: Float64,
    reorthogonalize: Bool = True,
) raises -> Tuple[Float64, DenseTensor[dtype]]:
    """Specialized Lanczos for two-site DMRG optimization without closures.
    
    We build a Krylov subspace with repeated applications of the effective two-site
    Hamiltonian via `apply_two_site_heff`, then solve the small projected eigenproblem
    to get the lowest Ritz pair. The heavy contractions are still done with GPU
    matmul inside `apply_two_site_heff`.
    """
    var v0 = initial_theta
    var dim = v0.size
    var shape = v0.shape.copy()

    # ---- helpers: Jacobi eigen solver for small symmetric matrices ----
    fn _jacobi_smallest_eigpair(
        mut A: List[List[Float64]],
        tol: Float64,
        max_sweeps: Int = 200,
    ) raises -> Tuple[Float64, List[Float64]]:
        var n = len(A)
        if n == 0:
            raise Error("Empty matrix in Jacobi eigensolver")
        if n == 1:
            return (A[0][0], List[Float64](1.0))

        # Eigenvector accumulator V (identity)
        var V = List[List[Float64]](capacity=n)
        for i in range(n):
            var row = List[Float64](capacity=n)
            for j in range(n):
                row.append(1.0 if i == j else 0.0)
            V.append(row^)

        for _ in range(max_sweeps):
            # Find largest off-diagonal element
            var p = 0
            var q = 1
            var max_val = abs(A[0][1])
            for i in range(n):
                for j in range(i + 1, n):
                    var v = abs(A[i][j])
                    if v > max_val:
                        max_val = v
                        p = i
                        q = j

            if max_val < tol:
                break

            var app = A[p][p]
            var aqq = A[q][q]
            var apq = A[p][q]

            var phi = 0.5 * atan2(2.0 * apq, (aqq - app))
            var c = cos(phi)
            var s = sin(phi)

            # Rotate A: update rows/cols p,q
            for k in range(n):
                if k != p and k != q:
                    var aik = A[k][p]
                    var akq = A[k][q]
                    var new_kp = c * aik - s * akq
                    var new_kq = s * aik + c * akq
                    A[k][p] = new_kp
                    A[p][k] = new_kp
                    A[k][q] = new_kq
                    A[q][k] = new_kq

            var new_pp = c * c * app - 2.0 * s * c * apq + s * s * aqq
            var new_qq = s * s * app + 2.0 * s * c * apq + c * c * aqq
            A[p][p] = new_pp
            A[q][q] = new_qq
            A[p][q] = 0.0
            A[q][p] = 0.0

            # Rotate V
            for k in range(n):
                var vip = V[k][p]
                var viq = V[k][q]
                V[k][p] = c * vip - s * viq
                V[k][q] = s * vip + c * viq

        # Pick smallest eigenvalue
        var min_i = 0
        var min_val = A[0][0]
        for i in range(1, n):
            if A[i][i] < min_val:
                min_val = A[i][i]
                min_i = i

        # Eigenvector is column min_i of V
        var vec = List[Float64](capacity=n)
        for i in range(n):
            vec.append(V[i][min_i])

        # Normalize vec
        var norm: Float64 = 0.0
        for i in range(n):
            norm += vec[i] * vec[i]
        norm = sqrt(norm)
        if norm > 0.0:
            for i in range(n):
                vec[i] = vec[i] / norm

        return (min_val, vec^)

    # ---- normalize initial vector ----
    var host_v0 = ctx.enqueue_create_host_buffer[dtype](dim)
    ctx.enqueue_copy(host_v0, v0.storage)
    ctx.synchronize()

    var norm0: Float64 = 0.0
    for i in range(dim):
        var val = Float64(host_v0[i])
        norm0 += val * val
    norm0 = sqrt(norm0)
    if norm0 < 1e-14:
        raise Error("Initial vector has zero norm")
    for i in range(dim):
        host_v0[i] = host_v0[i] / Scalar[dtype](norm0)

    # IMPORTANT: don't consume `shape` here; we need it again later.
    var v_current = create_dense_tensor[dtype](ctx, shape.copy()^, init_value=Scalar[dtype](0.0))
    ctx.enqueue_copy(v_current.storage, host_v0)
    ctx.synchronize()

    # Store Krylov basis on host (each is length dim)
    var krylov = List[List[Scalar[dtype]]](capacity=max_iter + 1)
    var v0_list = List[Scalar[dtype]](capacity=dim)
    for i in range(dim):
        v0_list.append(host_v0[i])
    krylov.append(v0_list^)

    var alpha = List[Float64](capacity=max_iter)
    var beta = List[Float64](capacity=max_iter)

    for iter in range(max_iter):
        var w = apply_two_site_heff[dtype](v_current^, L_env, R_env, W_i, W_ip1, ctx)

        var host_w = ctx.enqueue_create_host_buffer[dtype](dim)
        var host_v = ctx.enqueue_create_host_buffer[dtype](dim)
        ctx.enqueue_copy(host_w, w.storage)
        ctx.enqueue_copy(host_v, v_current.storage)
        ctx.synchronize()

        # alpha = <v|w>
        var a: Float64 = 0.0
        for i in range(dim):
            a += Float64(host_v[i]) * Float64(host_w[i])
        alpha.append(a)

        # w <- w - a v - b_prev v_prev
        for i in range(dim):
            host_w[i] = host_w[i] - Scalar[dtype](a) * host_v[i]
        if iter > 0:
            var bprev = beta[iter - 1]
            var vprev = krylov[iter - 1].copy()
            for i in range(dim):
                host_w[i] = host_w[i] - Scalar[dtype](bprev) * vprev[i]

        if reorthogonalize:
            for k in range(len(krylov)):
                var vk = krylov[k].copy()
                var overlap: Float64 = 0.0
                for i in range(dim):
                    overlap += Float64(host_w[i]) * Float64(vk[i])
                for i in range(dim):
                    host_w[i] = host_w[i] - Scalar[dtype](overlap) * vk[i]

        # beta = ||w||
        var b: Float64 = 0.0
        for i in range(dim):
            var val = Float64(host_w[i])
            b += val * val
        b = sqrt(b)

        if b < 1e-12:
            break

        beta.append(b)

        # v_{next} = w / b
        for i in range(dim):
            host_w[i] = host_w[i] / Scalar[dtype](b)

        var vnext_list = List[Scalar[dtype]](capacity=dim)
        for i in range(dim):
            vnext_list.append(host_w[i])
        krylov.append(vnext_list^)

        ctx.enqueue_copy(v_current.storage, host_w)
        ctx.synchronize()

    var nK = len(alpha)
    if nK == 0:
        raise Error("Lanczos produced empty Krylov basis")

    # Build dense T from alpha/beta
    var T = List[List[Float64]](capacity=nK)
    for _ in range(nK):
        var row = List[Float64](capacity=nK)
        for _ in range(nK):
            row.append(0.0)
        T.append(row^)
    for i in range(nK):
        T[i][i] = alpha[i]
    for i in range(nK - 1):
        T[i][i + 1] = beta[i]
        T[i + 1][i] = beta[i]

    var eig = _jacobi_smallest_eigpair(T, tol)
    var E0 = eig[0]
    # `List` is not implicitly copyable, and `^` can't be applied to a temporary
    # like `eig[1]` (it has no origin). Just copy; this list is tiny (<= max_iter).
    var coeffs = eig[1].copy()

    # Reconstruct Ritz vector in the original (flattened) space
    var host_out = ctx.enqueue_create_host_buffer[dtype](dim)
    for j in range(dim):
        host_out[j] = Scalar[dtype](0.0)
    for i in range(nK):
        var ci = coeffs[i]
        var vi = krylov[i].copy()
        for j in range(dim):
            host_out[j] = host_out[j] + Scalar[dtype](ci) * vi[j]

    # Normalize host_out
    var out_norm: Float64 = 0.0
    for j in range(dim):
        var val = Float64(host_out[j])
        out_norm += val * val
    out_norm = sqrt(out_norm)
    if out_norm > 0.0:
        for j in range(dim):
            host_out[j] = host_out[j] / Scalar[dtype](out_norm)

    var theta_opt = create_dense_tensor[dtype](ctx, shape^, init_value=Scalar[dtype](0.0))
    ctx.enqueue_copy(theta_opt.storage, host_out)
    ctx.synchronize()

    return (E0, theta_opt^)


fn dmrg_local_update_two_site[dtype: DType](
    site_index: Int,
    mut mps: MatrixProductState[dtype],
    mpo: MatrixProductOperator[dtype],
    mut workspace: DMRGWorkspace[dtype],
    params: DMRGParams,
    ctx: DeviceContext,
    left_to_right: Bool,
) raises -> Float64:
    """Perform local two-site optimization at given position.
    
    1. Build two-site theta from MPS
    2. Optimize using Lanczos with effective Hamiltonian
    3. Split with SVD and truncation
    4. Update MPS and environments
    
    Args:
        site_index: Index i (optimizes sites i and i+1).
        mps: Matrix product state (modified in-place).
        mpo: Matrix product operator.
        workspace: DMRG workspace with environments.
        params: DMRG parameters.
        ctx: Device context.
        left_to_right: Sweep direction (affects canonicalization).
    
    Returns:
        Local ground state energy.
    """
    var i = site_index
    
    # Build two-site theta
    var theta = build_two_site_theta[dtype](mps.sites[i], mps.sites[i + 1], ctx)
    
    # Get environments
    var L_i = workspace.L_env[i]
    var R_ip2 = workspace.R_env[i + 2]
    var W_i = mpo.sites[i]
    var W_ip1 = mpo.sites[i + 1]
    
    # Optimize with specialized Lanczos (avoids closure/function pointer issues)
    var lanczos_result = lanczos_two_site_optimize[dtype](
        theta^,
        L_i,
        R_ip2,
        W_i,
        W_ip1,
        ctx,
        max_iter=params.max_krylov_iter,
        tol=params.krylov_tol,
        reorthogonalize=params.reorthogonalize,
    )
    
    var eigenvalue = lanczos_result[0]
    var theta_opt = lanczos_result[1]
    
    # Split with SVD
    var split_result = split_two_site_theta_svd[dtype](
        theta_opt^,
        params.chi_max,
        params.eps_trunc,
        ctx,
        left_to_right=left_to_right,
    )
    
    var A_i_new = split_result[0]
    var A_ip1_new = split_result[1]
    _ = split_result[2]
    
    # Update MPS
    mps.sites[i] = A_i_new
    mps.sites[i + 1] = A_ip1_new
    
    # Update bond dimensions
    var new_chi = A_i_new.right_bond_dim()
    mps.bond_dims[i + 1] = new_chi
    
    # Update environments
    if left_to_right:
        # Update left environment for next position
        var L_ip1 = update_left_environment[dtype](L_i, A_i_new, W_i, ctx)
        workspace.L_env[i + 1] = L_ip1^
    else:
        # Update right environment for next position
        var R_ip1 = update_right_environment[dtype](R_ip2, A_ip1_new, W_ip1, ctx)
        workspace.R_env[i + 1] = R_ip1^
    
    return eigenvalue
