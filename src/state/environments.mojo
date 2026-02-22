from collections.list import List
from gpu.host import DeviceContext
from src.m_tensor.dense_tensor import (
    DenseTensor,
    create_dense_tensor,
    dense_tensor_dot,
)
from src.state.mps_state import MPSSite, MatrixProductState
from src.state.mpo_state import MPOSite, MatrixProductOperator
from src.state.mpo_state import create_identity_mpo


fn update_left_environment[dtype: DType](
    L_prev: DenseTensor[dtype],
    A_site: MPSSite[dtype],
    W_site: MPOSite[dtype],
    ctx: DeviceContext,
) raises -> DenseTensor[dtype]:
    """Update left environment by contracting one site.
    
    Contracts the previous left environment with the current MPS site, MPO site,
    and the conjugate of the MPS site to propagate the environment rightward.
    
    Convention:
        L_prev[wL, Dl, Dl'] (MPO left bond, MPS bra left bond, MPS ket left bond)
        A[Dl, s, Dr]        (MPS site: left bond, physical, right bond)
        A*[Dl', s', Dr']    (MPS conjugate)
        W[wL, s, s', wR]    (MPO: left bond, phys_in, phys_out, right bond)
        
        Result: L_next[wR, Dr, Dr']
    
    Contraction order (optimized to minimize intermediate sizes):
        1. Contract L_prev with A:        L_prev[wL,Dl,Dl'] x A[Dl,s,Dr] → temp1[wL,Dl',s,Dr]
        2. Contract temp1 with W:         temp1[wL,Dl',s,Dr] x W[wL,s,s',wR] → temp2[Dl',Dr,s',wR]
        3. Contract temp2 with A*:        temp2[Dl',Dr,s',wR] x A*[Dl',s',Dr'] → L_next[Dr,wR,Dr']
        4. Permute to standard order:     L_next[wR,Dr,Dr']
    
    Args:
        L_prev: Previous left environment, shape [wL, Dl, Dl'].
        A_site: MPS site tensor, shape [Dl, s, Dr].
        W_site: MPO site tensor, shape [wL, s, s', wR].
        ctx: Device context for GPU operations.
    
    Returns:
        Updated left environment with shape [wR, Dr, Dr'].
    """
    var A = A_site.tensor
    var W = W_site.tensor
    
    # Step 1: L_prev[wL, Dl, Dl'] × A[Dl, s, Dr] -> temp1[wL, Dl', s, Dr]
    # Contract on Dl (axis 1 of L_prev with axis 0 of A)
    var L_shape = L_prev.shape.copy()
    var A_shape = A.shape.copy()
    var wL = L_shape[0]
    var Dl_prev = L_shape[1]
    var Dl_prime = L_shape[2]
    var s = A_shape[1]
    var Dr = A_shape[2]
    
    # L_prev[wL, Dl, Dl'], want to contract Dl with A's first axis
    # Transpose L to [Dl, wL, Dl'], contract with A[Dl, s, Dr]
    var L_trans = L_prev.transpose(List[Int](1, 0, 2), ctx)  # [Dl, wL, Dl']
    var L_flat = L_trans^.reshape(List[Int](Dl_prev, wL * Dl_prime))
    var A_flat = A.reshape(List[Int](Dl_prev, s * Dr))
    
    var temp1_contract = create_dense_tensor[dtype](ctx, List[Int](wL * Dl_prime, s * Dr), init_value=Scalar[dtype](0.0))
    # Contract leading axis of L_flat (Dl) with leading axis of A_flat (Dl).
    dense_tensor_dot(temp1_contract, L_flat^, A_flat^, ctx, ndim_mult=1, axrange_A=True, axrange_B=True)
    
    var temp1_full = temp1_contract^.reshape(List[Int](wL, Dl_prime, s, Dr))
    
    # Step 2: temp1[wL, Dl', s, Dr] × W[wL, s, s', wR] -> temp2[Dl', Dr, s', wR]
    # Contract on wL and s
    
    var W_shape = W.shape.copy()
    var s_out = W_shape[2]
    var wR = W_shape[3]
    
    # temp1[wL, Dl', s, Dr] -> transpose to [Dl', Dr, wL, s] -> reshape to [Dl'*Dr, wL*s]
    var temp1_perm = temp1_full^.transpose(List[Int](1, 3, 0, 2), ctx)  # [Dl', Dr, wL, s]
    var temp1_mat = temp1_perm^.reshape(List[Int](Dl_prime * Dr, wL * s))
    
    # W[wL, s, s', wR] -> reshape to [wL*s, s'*wR]
    var W_mat = W.reshape(List[Int](wL * s, s_out * wR))
    
    var temp2_mat = create_dense_tensor[dtype](ctx, List[Int](Dl_prime * Dr, s_out * wR), init_value=Scalar[dtype](0.0))
    dense_tensor_dot(temp2_mat, temp1_mat^, W_mat^, ctx)
    
    var temp2 = temp2_mat^.reshape(List[Int](Dl_prime, Dr, s_out, wR))
    
    # Step 3: temp2[Dl', Dr, s', wR] × A*[Dl', s', Dr'] -> L_next[Dr, wR, Dr']
    # Contract on Dl' (axis 0) and s' (axis 2)
    
    # Transpose temp2 to [Dr, wR, Dl', s']
    var temp2_perm = temp2^.transpose(List[Int](1, 3, 0, 2), ctx)  # [Dr, wR, Dl', s']
    var temp2_mat2 = temp2_perm^.reshape(List[Int](Dr * wR, Dl_prime * s_out))
    
    # A[Dl', s', Dr'] - transpose to [Dl', s', Dr']
    # Reshape A to [Dl'*s', Dr'] (this is natural order, effectively merging first two dims)
    # But wait, A is [Dl', s', Dr']. We need [Dl'*s', Dr']?
    # No, we contract Dl' and s' (cols of temp2_mat2) with rows of A.
    # temp2_mat2 is [Dr*wR, Dl'*s'].
    # So we need A as [Dl'*s', Dr'].
    # A is [Dl', s', Dr']. Reshape(Dl'*s', Dr'). Yes.
    var A_flat2 = A.reshape(List[Int](Dl_prime * s_out, Dr))
    
    var L_next_mat = create_dense_tensor[dtype](ctx, List[Int](Dr * wR, Dr), init_value=Scalar[dtype](0.0))
    dense_tensor_dot(L_next_mat, temp2_mat2^, A_flat2^, ctx)
    
    var L_next_temp = L_next_mat^.reshape(List[Int](Dr, wR, Dr))
    
    # Permute to standard order: [wR, Dr, Dr']
    var L_next = L_next_temp^.transpose(List[Int](1, 0, 2), ctx)
    
    return L_next^


fn update_right_environment[dtype: DType](
    R_next: DenseTensor[dtype],
    A_site: MPSSite[dtype],
    W_site: MPOSite[dtype],
    ctx: DeviceContext,
) raises -> DenseTensor[dtype]:
    """Update right environment by contracting one site (propagating leftward).
    
    Contracts the next right environment with the current MPS site, MPO site,
    and the conjugate of the MPS site to propagate the environment leftward.
    
    Convention:
        R_next[wR, Dr, Dr'] (MPO right bond, MPS bra right bond, MPS ket right bond)
        A[Dl, s, Dr]        (MPS site: left bond, physical, right bond)
        A*[Dl', s', Dr']    (MPS conjugate)
        W[wL, s, s', wR]    (MPO: left bond, phys_in, phys_out, right bond)
        
        Result: R_prev[wL, Dl, Dl']
    
    Similar contraction pattern to update_left, but in reverse direction.
    
    Args:
        R_next: Next right environment, shape [wR, Dr, Dr'].
        A_site: MPS site tensor, shape [Dl, s, Dr].
        W_site: MPO site tensor, shape [wL, s, s', wR].
        ctx: Device context for GPU operations.
    
    Returns:
        Updated right environment with shape [wL, Dl, Dl'].
    """
    var A = A_site.tensor
    var W = W_site.tensor
    
    # Contract R_next[wR, Dr, Dr'] × A[Dl, s, Dr] × W[wL, s, s', wR] × A*[Dl', s', Dr']
    # Result: R_prev[wL, Dl, Dl']
    
    var R_shape = R_next.shape.copy()
    var A_shape = A.shape.copy()
    var W_shape = W.shape.copy()
    
    var wR = R_shape[0]
    var Dr = R_shape[1]
    var Dr_prime = R_shape[2]
    var Dl = A_shape[0]
    var s = A_shape[1]
    var wL = W_shape[0]
    var s_out = W_shape[2]
    
    # Step 1: R_next[wR, Dr, Dr'] × A[Dl, s, Dr] -> temp1[wR, Dr', Dl, s]
    # Contract on Dr (axis 1 of R with axis 2 of A)
    
    # R[wR, Dr, Dr'] -> Transpose to [Dr, wR, Dr'] -> Reshape [Dr, wR*Dr']
    var R_trans = R_next.transpose(List[Int](1, 0, 2), ctx)
    var R_flat = R_trans^.reshape(List[Int](Dr, wR * Dr_prime))
    
    # A[Dl, s, Dr] -> Transpose to [Dr, Dl, s] -> Reshape [Dr, Dl*s]
    var A_trans = A.transpose(List[Int](2, 0, 1), ctx)
    var A_flat = A_trans^.reshape(List[Int](Dr, Dl * s))
    
    var temp1_mat = create_dense_tensor[dtype](ctx, List[Int](wR * Dr_prime, Dl * s), init_value=Scalar[dtype](0.0))
    # Contract leading axis of R_flat (Dr) with leading axis of A_flat (Dr).
    dense_tensor_dot(temp1_mat, R_flat^, A_flat^, ctx, ndim_mult=1, axrange_A=True, axrange_B=True)
    
    var temp1 = temp1_mat^.reshape(List[Int](wR, Dr_prime, Dl, s))
    
    # Step 2: temp1[wR, Dr', Dl, s] × W[wL, s, s', wR] -> temp2[Dr', Dl, wL, s']
    # Contract wR (axis 0 of temp1, axis 3 of W) and s (axis 3 of temp1, axis 1 of W)
    
    # Transpose temp1 to [Dr', Dl, wR, s] -> Reshape [Dr'*Dl, wR*s]
    var temp1_perm = temp1^.transpose(List[Int](1, 2, 0, 3), ctx)
    var temp1_mat2 = temp1_perm^.reshape(List[Int](Dr_prime * Dl, wR * s))
    
    # W[wL, s, s', wR] -> Transpose to [wL, s', wR, s] -> Reshape [wL*s', wR*s]
    var W_perm = W.transpose(List[Int](0, 2, 3, 1), ctx)
    var W_mat = W_perm^.reshape(List[Int](wL * s_out, wR * s))
    
    # Contract temp1_mat2 [M, K] with W_mat^T [N, K]^T -> [M, K] x [K, N] -> [M, N]
    # temp1_mat2 is [Dr'*Dl, wR*s]
    # W_mat is [wL*s', wR*s]
    # We want temp1_mat2 x W_mat.T -> [Dr'*Dl, wL*s']
    
    # Transpose W_mat for dot
    var W_mat_T = W_mat^.transpose(List[Int](1, 0), ctx) # [wR*s, wL*s']
    
    var temp2_mat = create_dense_tensor[dtype](ctx, List[Int](Dr_prime * Dl, wL * s_out), init_value=Scalar[dtype](0.0))
    dense_tensor_dot(temp2_mat, temp1_mat2^, W_mat_T^, ctx)
    
    var temp2 = temp2_mat^.reshape(List[Int](Dr_prime, Dl, wL, s_out))
    
    # Step 3: temp2[Dr', Dl, wL, s'] × A*[Dl', s', Dr'] -> R_prev[Dl, wL, Dl']
    # Contract Dr' and s'
    
    # Transpose temp2 to [Dl, wL, Dr', s'] -> Reshape [Dl*wL, Dr'*s']
    var temp2_perm = temp2^.transpose(List[Int](1, 2, 0, 3), ctx)
    var temp2_mat3 = temp2_perm^.reshape(List[Int](Dl * wL, Dr_prime * s_out))
    
    # A*[Dl', s', Dr'] -> Transpose to [Dr', s', Dl'] -> Reshape [Dr'*s', Dl']
    var A_trans2 = A.transpose(List[Int](2, 1, 0), ctx)
    var A_mat2 = A_trans2^.reshape(List[Int](Dr_prime * s_out, Dl))
    
    var R_prev_mat = create_dense_tensor[dtype](ctx, List[Int](Dl * wL, Dl), init_value=Scalar[dtype](0.0))
    dense_tensor_dot(R_prev_mat, temp2_mat3^, A_mat2^, ctx)
    
    var R_prev_temp = R_prev_mat^.reshape(List[Int](Dl, wL, Dl))
    
    # Permute to standard order: [wL, Dl, Dl']
    var R_prev = R_prev_temp^.transpose(List[Int](1, 0, 2), ctx)
    
    return R_prev^


fn build_right_environments[dtype: DType](
    mps: MatrixProductState[dtype],
    mpo: MatrixProductOperator[dtype],
    ctx: DeviceContext,
) raises -> List[DenseTensor[dtype]]:
    """Build all right environments from scratch (initialization for DMRG).
    
    Sweeps from right to left, building R[N], R[N-1], ..., R[0].
    
    Args:
        mps: Matrix product state.
        mpo: Matrix product operator (Hamiltonian).
        ctx: Device context.
    
    Returns:
        List of right environments, R[i] has shape [wR_i, Dr_i, Dr_i].
        Length is N+1 (includes boundary at position N).
    """
    var N = mps.num_sites()
    var envs = List[DenseTensor[dtype]](capacity=N + 1)
    
    # Boundary condition: R[N] is identity (trivial environment at the right edge)
    var wR_boundary = mpo.bond_dimension(N)
    var Dr_boundary = mps.bond_dimension(N)
    
    var R_boundary_shape = List[Int](wR_boundary, Dr_boundary, Dr_boundary)
    var R_boundary = create_dense_tensor[dtype](ctx, R_boundary_shape^, init_value=Scalar[dtype](0.0))
    
    # Set to identity in the (Dr, Dr) subspace
    var host_R = ctx.enqueue_create_host_buffer[dtype](wR_boundary * Dr_boundary * Dr_boundary)
    for w in range(wR_boundary):
        for d in range(Dr_boundary):
            var idx = w * (Dr_boundary * Dr_boundary) + d * Dr_boundary + d
            host_R[idx] = Scalar[dtype](1.0)
    ctx.enqueue_copy(R_boundary.storage, host_R)
    ctx.synchronize()
    
    # Initialize list with None placeholders, then fill
    for _ in range(N + 1):
        envs.append(R_boundary)  # Placeholder, will be overwritten
    
    envs[N] = R_boundary^
    
    # Build environments from right to left
    for i in range(N - 1, -1, -1):
        var R_next = envs[i + 1]
        var A_i = mps.sites[i]
        var W_i = mpo.sites[i]
        
        var R_i = update_right_environment[dtype](R_next, A_i, W_i, ctx)
        envs[i] = R_i^
    
    return envs^


fn expectation_value[dtype: DType](
    mps: MatrixProductState[dtype],
    mpo: MatrixProductOperator[dtype],
    ctx: DeviceContext,
) raises -> Float64:
    """Compute the scalar expectation value <mps|mpo|mps>.
    
    For real-valued tensors this reduces to contracting the MPS with itself.
    (For complex support we'd need conjugation on the bra.)
    
    Returns:
        Float64 energy (host scalar).
    """
    var N = mps.num_sites()
    if mpo.num_sites() != N:
        raise Error("MPS and MPO must have same length in expectation_value")

    # Left boundary environment: shape [wL0, Dl0, Dl0] with identity in the bond space.
    var wL0 = mpo.bond_dimension(0)
    var Dl0 = mps.bond_dimension(0)
    var L = create_dense_tensor[dtype](
        ctx,
        List[Int](wL0, Dl0, Dl0)^,
        init_value=Scalar[dtype](0.0),
    )
    var host_L = ctx.enqueue_create_host_buffer[dtype](wL0 * Dl0 * Dl0)
    for w in range(wL0):
        for d in range(Dl0):
            host_L[w * (Dl0 * Dl0) + d * Dl0 + d] = Scalar[dtype](1.0)
    ctx.enqueue_copy(L.storage, host_L)
    ctx.synchronize()

    for i in range(N):
        L = update_left_environment[dtype](L^, mps.sites[i], mpo.sites[i], ctx)

    # At the right boundary we should have shape [1, 1, 1]
    if L.size != 1:
        # Still return the (0) element if it exists; but flag shape issues loudly.
        raise Error("Unexpected final environment size in expectation_value: " + String(L.size))

    var host_out = ctx.enqueue_create_host_buffer[dtype](1)
    ctx.enqueue_copy(host_out, L.storage)
    ctx.synchronize()
    return Float64(host_out[0])


fn expectation_value_normalized[dtype: DType](
    mps: MatrixProductState[dtype],
    mpo: MatrixProductOperator[dtype],
    ctx: DeviceContext,
) raises -> Float64:
    """Compute <mps|mpo|mps> / <mps|mps>."""
    var num = expectation_value[dtype](mps, mpo, ctx)
    var id = create_identity_mpo[dtype](ctx, mps.num_sites(), mps.physical_dim)
    var denom = expectation_value[dtype](mps, id^, ctx)
    if abs(denom) < 1e-20:
        raise Error("MPS norm is ~0 in expectation_value_normalized")
    return num / denom


fn update_left_environment_two_mps[dtype: DType](
    L_prev: DenseTensor[dtype],
    A_bra: MPSSite[dtype],
    A_ket: MPSSite[dtype],
    W_site: MPOSite[dtype],
    ctx: DeviceContext,
) raises -> DenseTensor[dtype]:
    """Update left environment with different bra and ket MPS (for overlap, etc.).
    
    Same as update_left_environment but A_bra and A_ket can differ.
    L_prev[wL,Dl,Dl'] x A_ket[Dl,s,Dr] x W[wL,s,s',wR] x A_bra[Dl',s',Dr'] -> L_next[wR,Dr,Dr'].
    """
    var A_ket_t = A_ket.tensor
    var A_bra_t = A_bra.tensor
    var W = W_site.tensor

    var L_shape = L_prev.shape.copy()
    var wL = L_shape[0]
    var Dl_prev = L_shape[1]
    var Dl_prime = L_shape[2]
    var s = A_ket_t.shape[1]
    var Dr = A_ket_t.shape[2]

    var L_trans = L_prev.transpose(List[Int](1, 0, 2), ctx)
    var L_flat = L_trans^.reshape(List[Int](Dl_prev, wL * Dl_prime))
    var A_ket_flat = A_ket_t.reshape(List[Int](Dl_prev, s * Dr))

    var temp1_contract = create_dense_tensor[dtype](ctx, List[Int](wL * Dl_prime, s * Dr), init_value=Scalar[dtype](0.0))
    dense_tensor_dot(temp1_contract, L_flat^, A_ket_flat^, ctx, ndim_mult=1, axrange_A=True, axrange_B=True)

    var temp1_full = temp1_contract^.reshape(List[Int](wL, Dl_prime, s, Dr))

    var W_shape = W.shape.copy()
    var s_out = W_shape[2]
    var wR = W_shape[3]

    var temp1_perm = temp1_full^.transpose(List[Int](1, 3, 0, 2), ctx)
    var temp1_mat = temp1_perm^.reshape(List[Int](Dl_prime * Dr, wL * s))
    var W_mat = W.reshape(List[Int](wL * s, s_out * wR))

    var temp2_mat = create_dense_tensor[dtype](ctx, List[Int](Dl_prime * Dr, s_out * wR), init_value=Scalar[dtype](0.0))
    dense_tensor_dot(temp2_mat, temp1_mat^, W_mat^, ctx)

    var temp2 = temp2_mat^.reshape(List[Int](Dl_prime, Dr, s_out, wR))
    var temp2_perm = temp2^.transpose(List[Int](1, 3, 0, 2), ctx)
    var temp2_mat2 = temp2_perm^.reshape(List[Int](Dr * wR, Dl_prime * s_out))

    var A_bra_flat = A_bra_t.reshape(List[Int](Dl_prime * s_out, Dr))

    var L_next_mat = create_dense_tensor[dtype](ctx, List[Int](Dr * wR, Dr), init_value=Scalar[dtype](0.0))
    dense_tensor_dot(L_next_mat, temp2_mat2^, A_bra_flat^, ctx)

    var L_next_temp = L_next_mat^.reshape(List[Int](Dr, wR, Dr))
    var L_next = L_next_temp^.transpose(List[Int](1, 0, 2), ctx)
    return L_next^


fn expectation_value_two_mps[dtype: DType](
    mps_bra: MatrixProductState[dtype],
    mpo: MatrixProductOperator[dtype],
    mps_ket: MatrixProductState[dtype],
    ctx: DeviceContext,
) raises -> Float64:
    """Compute <mps_bra|mpo|mps_ket> with different bra and ket MPS."""
    var N = mps_bra.num_sites()
    if mpo.num_sites() != N or mps_ket.num_sites() != N:
        raise Error("expectation_value_two_mps: length mismatch")

    var wL0 = mpo.bond_dimension(0)
    var Dl0 = mps_bra.bond_dimension(0)
    var L = create_dense_tensor[dtype](ctx, List[Int](wL0, Dl0, Dl0)^, init_value=Scalar[dtype](0.0))
    var host_L = ctx.enqueue_create_host_buffer[dtype](wL0 * Dl0 * Dl0)
    for w in range(wL0):
        for d in range(Dl0):
            host_L[w * (Dl0 * Dl0) + d * Dl0 + d] = Scalar[dtype](1.0)
    ctx.enqueue_copy(L.storage, host_L)
    ctx.synchronize()

    for i in range(N):
        L = update_left_environment_two_mps[dtype](L^, mps_bra.sites[i], mps_ket.sites[i], mpo.sites[i], ctx)

    if L.size != 1:
        raise Error("Unexpected final environment size in expectation_value_two_mps: " + String(L.size))

    var host_out = ctx.enqueue_create_host_buffer[dtype](1)
    ctx.enqueue_copy(host_out, L.storage)
    ctx.synchronize()
    return Float64(host_out[0])


fn mps_overlap[dtype: DType](
    mps_bra: MatrixProductState[dtype],
    mps_ket: MatrixProductState[dtype],
    ctx: DeviceContext,
) raises -> Float64:
    """Compute <mps_bra|mps_ket> using identity MPO (gauge-invariant for fidelity)."""
    var id_mpo = create_identity_mpo[dtype](ctx, mps_bra.num_sites(), mps_bra.physical_dim)
    return expectation_value_two_mps[dtype](mps_bra, id_mpo, mps_ket, ctx)


fn variance[dtype: DType](
    mps: MatrixProductState[dtype],
    mpo: MatrixProductOperator[dtype],
    ctx: DeviceContext,
) raises -> Float64:
    """Compute var = <H^2> - <H>^2 via MPO composition.
    
    Builds H^2 MPO and computes <mps|H^2|mps> using dense_tensor_dot for composition.
    """
    var E = expectation_value_normalized[dtype](mps, mpo, ctx)
    var H2 = mpo_compose[dtype](mpo, mpo, ctx)
    var E2 = expectation_value_normalized[dtype](mps, H2^, ctx)
    return E2 - E * E


fn mpo_compose[dtype: DType](
    mpo_a: MatrixProductOperator[dtype],
    mpo_b: MatrixProductOperator[dtype],
    ctx: DeviceContext,
) raises -> MatrixProductOperator[dtype]:
    """Compose two MPOs: (A o B)|psi> = A(B|psi>).
    
    At each site: W_new = sum_d_mid W_a[:,:,d_mid,:] * W_b[:,d_mid,:,:]
    Flatten bonds: left (wLa,wLb), right (wRa,wRb).
    """
    var N = mpo_a.num_sites()
    if mpo_b.num_sites() != N:
        raise Error("MPO compose: length mismatch")

    var sites = List[MPOSite[dtype]](capacity=N)
    for i in range(N):
        var Wa = mpo_a.sites[i].tensor
        var Wb = mpo_b.sites[i].tensor
        var sa = Wa.shape.copy()
        var sb = Wb.shape.copy()
        var wLa = sa[0]
        var d_in = sa[1]
        var d_mid = sa[2]
        var wRa = sa[3]
        var wLb = sb[0]
        var d_out = sb[2]
        var wRb = sb[3]
        if sb[1] != d_mid:
            raise Error("MPO compose: physical dim mismatch at site " + String(i))

        var wL_new = wLa * wLb
        var wR_new = wRa * wRb

        var Wa_perm = Wa.transpose(List[Int](0, 1, 3, 2), ctx)
        var Wa_flat = Wa_perm^.reshape(List[Int](wLa * d_in * wRa, d_mid))
        var Wb_perm = Wb.transpose(List[Int](1, 0, 2, 3), ctx)
        var Wb_flat = Wb_perm^.reshape(List[Int](d_mid, wLb * d_out * wRb))

        var C_mat = create_dense_tensor[dtype](ctx, List[Int](wLa * d_in * wRa, wLb * d_out * wRb), init_value=Scalar[dtype](0.0))
        dense_tensor_dot(C_mat, Wa_flat^, Wb_flat^, ctx)

        var C = C_mat^.reshape(List[Int](wLa, d_in, wRa, wLb, d_out, wRb))
        var C_perm = C^.transpose(List[Int](0, 3, 1, 4, 2, 5), ctx)
        var site_tensor = C_perm^.reshape(List[Int](wL_new, d_in, d_out, wR_new))
        sites.append(MPOSite[dtype](site_tensor^))

    return MatrixProductOperator[dtype](sites^)


fn apply_mpo[dtype: DType](
    mpo: MatrixProductOperator[dtype],
    mps: MatrixProductState[dtype],
    ctx: DeviceContext,
) raises -> MatrixProductState[dtype]:
    """Apply MPO to MPS: |out> = O|mps>.

    At each site: contract MPS site A[Dl, d, Dr] with MPO site W[wL, d, d_out, wR]
    on the physical index d. Result site has shape [Dl*wL, d_out, Dr*wR].

    Matches C apply_mpo(op, psi, op_psi) for benchmarking.
    """
    var N = mps.num_sites()
    if mpo.num_sites() != N:
        raise Error("apply_mpo: MPS and MPO must have same number of sites")

    var out_sites = List[MPSSite[dtype]](capacity=N)
    for i in range(N):
        var A = mps.sites[i].tensor
        var W = mpo.sites[i].tensor
        var Ash = A.shape.copy()
        var Wsh = W.shape.copy()
        var Dl = Ash[0]
        var d = Ash[1]
        var Dr = Ash[2]
        var wL = Wsh[0]
        var d_in = Wsh[1]
        var d_out = Wsh[2]
        var wR = Wsh[3]
        if d_in != d:
            raise Error("apply_mpo: physical dim mismatch at site " + String(i))

        # Contract A[Dl, d, Dr] with W[wL, d, d_out, wR] on index d.
        # Reshape A to [Dl*Dr, d], W to [d, wL*d_out*wR]
        var A_flat = A.reshape(List[Int](Dl * Dr, d))
        var W_flat = W.reshape(List[Int](d, wL * d_out * wR))
        var C_mat = create_dense_tensor[dtype](
            ctx, List[Int](Dl * Dr, wL * d_out * wR), init_value=Scalar[dtype](0.0)
        )
        dense_tensor_dot(C_mat, A_flat^, W_flat^, ctx)
        # C_mat is [Dl*Dr, wL*d_out*wR]. Reshape to [Dl, Dr, wL, d_out, wR]
        var C = C_mat^.reshape(List[Int](Dl, Dr, wL, d_out, wR))
        # Permute to [Dl, wL, d_out, Dr, wR] then reshape to [Dl*wL, d_out, Dr*wR]
        var C_perm = C^.transpose(List[Int](0, 2, 3, 1, 4), ctx)
        var site_tensor = C_perm^.reshape(List[Int](Dl * wL, d_out, Dr * wR))
        out_sites.append(MPSSite[dtype](site_tensor^))

    return MatrixProductState[dtype](out_sites^)
