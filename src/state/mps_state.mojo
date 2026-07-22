from collections.list import List
from math import sqrt
from gpu.host import DeviceContext
from src.m_tensor.tensor_traits import TensorOps
from src.m_tensor.dense_tensor import (
    DenseTensor,
    create_dense_tensor,
    create_dense_tensor_from_data,
    dense_tensor_qr,
    dense_tensor_dot,
)
from src.m_tensor.block_sparse_tensor import (
    BlockSparseTensor,
    QNumber,
    block_sparse_to_dense,
    create_block_sparse_tensor,
)
from src.m_tensor.tensor_ops import (
    tensor_qr,
    tensor_dot,
    tensor_reshape,
    tensor_transpose,
    tensor_norm,
    tensor_scale_in_place,
    create_tensor,
    create_tensor_from_data,
)


# =============================================================================
# Helper: Convert DenseTensor to trivial BlockSparseTensor
# =============================================================================
# TODO: make this more optimized to figuring out the sparsity pattern instead of just using all-zero quantum numbers.
fn _dense_to_trivial_block_sparse[dtype: DType](
    var t: DenseTensor[dtype], ctx: DeviceContext
) raises -> BlockSparseTensor[dtype]:
    """Convert DenseTensor to BlockSparseTensor with trivial (all-zero) quantum numbers."""
    var sh = t.shape.copy()
    var ndim = len(sh)
    var qnums_per_leg = List[List[QNumber]](capacity=ndim)
    for i in range(ndim):
        var leg = List[QNumber](capacity=sh[i])
        for _ in range(sh[i]):
            leg.append(QNumber(0))
        qnums_per_leg.append(leg^)
    var out = create_block_sparse_tensor[dtype](ctx, sh^, qnums_per_leg^, init_value=Scalar[dtype](0.0))
    # Copy data from dense to block-sparse
    var host_t = ctx.enqueue_create_host_buffer[dtype](t.size)
    ctx.enqueue_copy(host_t, t.storage)
    ctx.synchronize()
    # For trivial qnums, there's only one block with all the data
    var blk = out.blocks_flat[0]
    ctx.enqueue_copy(blk.storage, host_t)
    ctx.synchronize()
    return out^


# =============================================================================
# Convenience Type Aliases for Common Tensor Backends
# =============================================================================

alias DenseMPSSite = MPSSite[DType.float32, DenseTensor[DType.float32]]
"""Dense MPS site with float32 data type."""

alias DenseMPS = MatrixProductState[DType.float32, DenseTensor[DType.float32]]
"""Dense MPS with float32 data type."""

alias BlockSparseMPSSite = MPSSite[DType.float32, BlockSparseTensor[DType.float32]]
"""Block-sparse MPS site with float32 data type."""

alias BlockSparseMPS = MatrixProductState[DType.float32, BlockSparseTensor[DType.float32]]
"""Block-sparse MPS with float32 data type."""


struct MPSSite[dtype: DType, T: TensorOps](Writable, Movable, ImplicitlyCopyable):
    """Single site tensor inside an MPS.

    Each site is stored as a rank-3 tensor with layout
    [left_bond, physical, right_bond].
    
    The tensor type T must implement the TensorOps trait (plus Movable and Copyable),
    allowing both DenseTensor and BlockSparseTensor to be used.
    
    Parameters:
        dtype: The data type of tensor elements.
        T: The tensor type (must implement TensorOps, Movable, Copyable).
    """
    var tensor: T

    fn __init__(out self, var tensor: T):
        """Initialize an MPS site with a tensor."""
        self.tensor = tensor^

    fn __copyinit__(out self, other: Self):
        """Copy an MPS site."""
        self.tensor = other.tensor.copy()
    
    fn __moveinit__(out self, deinit other: Self):
        """Move an MPS site."""
        self.tensor = other.tensor^

    fn rank(self) -> Int:
        """Get the rank (number of dimensions) of the site tensor."""
        return self.tensor.get_rank()

    fn shape(self) -> List[Int]:
        """Get the shape of the site tensor."""
        return self.tensor.get_shape()

    fn physical_dim(self) raises -> Int:
        """Get the physical dimension (middle index of rank-3 tensor)."""
        self._assert_rank3()
        return self.tensor.get_shape_at(1)

    fn left_bond_dim(self) raises -> Int:
        """Get the left bond dimension (first index of rank-3 tensor)."""
        self._assert_rank3()
        return self.tensor.get_shape_at(0)

    fn right_bond_dim(self) raises -> Int:
        """Get the right bond dimension (last index of rank-3 tensor)."""
        self._assert_rank3()
        return self.tensor.get_shape_at(2)

    fn _assert_rank3(self) raises -> None:
        """Validate that the tensor has rank 3."""
        var r = self.tensor.get_rank()
        if r != 3:
            raise Error(
                "MPSSite expects rank-3 tensors [bond_left, physical, bond_right], got rank "
                + String(r)
            )

    fn write_to[W: Writer](self, mut writer: W) -> None:
        """Write tensor info to a writer (for Writable trait)."""
        var s = self.tensor.get_shape()
        writer.write("MPSSite[")
        for i in range(len(s)):
            if i > 0:
                writer.write(", ")
            writer.write(s[i])
        writer.write("]")


struct MatrixProductState[dtype: DType, T: TensorOps](Writable, Movable, ImplicitlyCopyable):
    """Matrix Product State generic over tensor type.

    This MPS implementation works with any tensor type T that implements the
    TensorOps trait, allowing both DenseTensor and BlockSparseTensor backends.
    
    Parameters:
        dtype: The data type of tensor elements.
        T: The tensor type (must implement TensorOps).
    """
    var sites: List[MPSSite[dtype, T]]
    var physical_dim: Int
    var length: Int
    var bond_dims: List[Int]

    fn __init__(out self, var sites: List[MPSSite[dtype, T]]) raises:
        if len(sites) == 0:
            raise Error("MatrixProductState requires at least one site tensor")

        var first_site: MPSSite[dtype, T] = sites[0]
        var phys_dim = first_site.physical_dim()
        var bonds: List[Int] = List[Int](capacity=len(sites))
        bonds.append(first_site.left_bond_dim())

        for idx in range(len(sites)):
            var site = sites[idx]
            if site.physical_dim() != phys_dim:
                raise Error("All MPS sites must have the same physical dimension")

            if idx > 0:
                var expected = sites[idx - 1].right_bond_dim()
                if site.left_bond_dim() != expected:
                    raise Error(
                        "Bond mismatch between sites "
                        + String(idx - 1)
                        + " and "
                        + String(idx)
                    )

            bonds.append(site.right_bond_dim())

        self.sites = sites^
        self.physical_dim = phys_dim
        self.length = len(self.sites)
        self.bond_dims = bonds^

    fn __copyinit__(out self, existing: Self):
        self.sites = existing.sites.copy()
        self.physical_dim = existing.physical_dim
        self.length = existing.length
        self.bond_dims = existing.bond_dims.copy()

    fn num_sites(self) -> Int:
        return self.length

    fn bond_dimension(self, index: Int) -> Int:
        return self.bond_dims[index]

    fn site_shape(self, index: Int) -> List[Int]:
        return self.sites[index].shape()

    fn describe(self) -> None:
        print(
            "MatrixProductState(length=",
            self.length,
            ", physical_dim=",
            self.physical_dim,
            ")",
        )
        
        var bond_str = String("  bond dims = [")
        for i in range(len(self.bond_dims)):
            if i > 0:
                bond_str += ", "
            bond_str += String(self.bond_dims[i])
        bond_str += "]"
        print(bond_str)
        
        for idx in range(self.length):
            var shape = self.sites[idx].shape()
            print(
                "  site ",
                idx,
                ": [",
                shape[0],
                ", ",
                shape[1],
                ", ",
                shape[2],
                "]",
            )

    fn write_to[W: Writer](self, mut writer: W) -> None:
        writer.write("MatrixProductState(length=")
        writer.write(self.length)
        writer.write(", physical_dim=")
        writer.write(self.physical_dim)
        writer.write(", bond_dims=[")
        for i in range(len(self.bond_dims)):
            if i > 0:
                writer.write(", ")
            writer.write(self.bond_dims[i])
        writer.write("], sites=[")
        for idx in range(self.length):
            if idx > 0:
                writer.write(", ")
            var shape = self.sites[idx].shape()
            writer.write("(")
            writer.write(shape[0])
            writer.write(", ")
            writer.write(shape[1])
            writer.write(", ")
            writer.write(shape[2])
            writer.write(")")
        writer.write("])")


fn create_uniform_mps[dtype: DType = DType.float32](
    ctx: DeviceContext,
    num_sites: Int,
    physical_dim: Int,
    bond_dims: List[Int],
    init_value: Optional[Scalar[dtype]] = None,
) raises -> MatrixProductState[dtype, DenseTensor[dtype]]:
    """Allocate a DenseTensor-based MPS with constant entries in every site tensor."""
    if num_sites < 1:
        raise Error("num_sites must be >= 1")
    if physical_dim < 1:
        raise Error("physical_dim must be >= 1")
    if len(bond_dims) != num_sites + 1:
        raise Error("bond_dims must have length num_sites + 1")

    var sites = List[MPSSite[dtype, DenseTensor[dtype]]](capacity=num_sites)
    for i in range(num_sites):
        var left_dim = bond_dims[i]
        var right_dim = bond_dims[i + 1]
        if left_dim < 1 or right_dim < 1:
            raise Error("Bond dimensions must be >= 1")

        var shape = List[Int](left_dim, physical_dim, right_dim)
        var site_tensor: DenseTensor[dtype]
        if init_value is None:
            site_tensor = DenseTensor[dtype].random(ctx, shape^)
            var nelem = left_dim * physical_dim * right_dim
            var scale_val = 1.0 / sqrt(Float64(nelem))
            site_tensor.scale_in_place(Scalar[dtype](scale_val), ctx)
        else:
            site_tensor = create_dense_tensor[dtype](
                ctx, shape^, row_major=True, init_value=init_value.value()
            )
        sites.append(MPSSite[dtype, DenseTensor[dtype]](site_tensor^))
    return MatrixProductState[dtype, DenseTensor[dtype]](sites^)


fn create_product_mps[dtype: DType = DType.float32](
    ctx: DeviceContext,
    physical_dim: Int,
    basis: List[Int],
) raises -> MatrixProductState[dtype, DenseTensor[dtype]]:
    """Create a product-state MPS from a list of local basis choices.

    Args:
        ctx: GPU device context.
        physical_dim: Local Hilbert-space dimension (e.g., 2 for qubits).
        basis: List of integers (length = num_sites) specifying |basis[i]> at each site.

    Returns:
        DenseTensor-based MatrixProductState where all internal bonds are 1 (unentangled).
    """
    var num_sites = len(basis)
    if num_sites < 1:
        raise Error("basis must contain at least one element")
    if physical_dim < 1:
        raise Error("physical_dim must be >= 1")

    var sites = List[MPSSite[dtype, DenseTensor[dtype]]](capacity=num_sites)
    for i in range(num_sites):
        var choice = basis[i]
        if choice < 0 or choice >= physical_dim:
            raise Error(
                "Invalid basis index "
                + String(choice)
                + " at site "
                + String(i)
                + " (must be in [0, physical_dim))"
            )

        var shape = List[Int](1, physical_dim, 1)
        var total_size = physical_dim
        var data = List[Scalar[dtype]](capacity=total_size)

        for p in range(physical_dim):
            if p == choice:
                data.append(Scalar[dtype](1.0))
            else:
                data.append(Scalar[dtype](0.0))

        var site_tensor = create_dense_tensor_from_data[dtype](ctx, data, shape^)
        sites.append(MPSSite[dtype, DenseTensor[dtype]](site_tensor^))

    return MatrixProductState[dtype, DenseTensor[dtype]](sites^)


fn mps_local_orthonormalize_qr[dtype: DType = DType.float32](
    ctx: DeviceContext,
    var block: DenseTensor[dtype],
) raises -> Tuple[MPSSite[dtype, DenseTensor[dtype]], DenseTensor[dtype]]:
    """Left-orthonormalize a single site tensor and absorb R into the remainder.

    Mirrors the behavior of `mps_local_orthonormalize_qr` in the reference
    ChemTensor implementation.
    """
    var shape = block.shape.copy()
    if len(shape) < 3:
        raise Error("mps_local_orthonormalize_qr requires rank >= 3 (left, physical, remainder)")

    var left_dim = shape[0]
    var phys_dim = shape[1]
    var tail_shape = List[Int](capacity=len(shape) - 2)
    var right_dim = 1
    for idx in range(2, len(shape)):
        tail_shape.append(shape[idx])
        right_dim *= shape[idx]
    if right_dim < 1:
        raise Error("Right block dimension must be >= 1 in mps_local_orthonormalize_qr")

    var mat_shape = List[Int](left_dim * phys_dim, right_dim)
    var mat = block.reshape(mat_shape^)

    var qr_result = dense_tensor_qr[dtype](mat^, ctx)
    var Q_full = qr_result[0]
    var R_full = qr_result[1]

    var m = Q_full.shape[0]  # = left_dim * phys_dim
    var q_cols = Q_full.shape[1]
    var r_cols = R_full.shape[1]
    var reduced_cols = m
    if reduced_cols > r_cols:
        reduced_cols = r_cols

    # Copy Q data and keep only the first reduced_cols columns (reduced QR)
    var host_Q_full = ctx.enqueue_create_host_buffer[dtype](Q_full.size)
    ctx.enqueue_copy(host_Q_full, Q_full.storage)
    ctx.synchronize()

    var q_data = List[Scalar[dtype]](capacity=m * reduced_cols)
    for row in range(m):
        for col in range(reduced_cols):
            var idx_full = row * q_cols + col
            q_data.append(host_Q_full[idx_full])
    var reduced_Q = create_dense_tensor_from_data[dtype](
        ctx,
        q_data,
        List[Int](m, reduced_cols)
    )

    var host_R_full = ctx.enqueue_create_host_buffer[dtype](R_full.size)
    ctx.enqueue_copy(host_R_full, R_full.storage)
    ctx.synchronize()

    var r_data = List[Scalar[dtype]](capacity=reduced_cols * r_cols)
    for row in range(reduced_cols):
        for col in range(r_cols):
            var idx_full = row * r_cols + col
            r_data.append(host_R_full[idx_full])
    var reduced_R = create_dense_tensor_from_data[dtype](
        ctx,
        r_data,
        List[Int](reduced_cols, r_cols)
    )

    var site_shape = List[Int](left_dim, phys_dim, reduced_cols)
    var site_tensor = reduced_Q.reshape(site_shape^)
    var site = MPSSite[dtype, DenseTensor[dtype]](site_tensor^)

    var next_shape = List[Int](capacity=len(tail_shape) + 1)
    next_shape.append(reduced_cols)
    for idx in range(len(tail_shape)):
        next_shape.append(tail_shape[idx])
    var next_remainder = reduced_R.reshape(next_shape^)

    return (site, next_remainder)


fn mps_local_orthonormalize_qr_pair[dtype: DType = DType.float32](
    ctx: DeviceContext,
    var A_i: DenseTensor[dtype],
    var A_ip1: DenseTensor[dtype],
) raises -> Tuple[MPSSite[dtype, DenseTensor[dtype]], MPSSite[dtype, DenseTensor[dtype]]]:
    """Left-orthonormalize site i and absorb R into site i+1 (C `mps_local_orthonormalize_qr`)."""
    var shape_i = A_i.shape.copy()
    var Dl = shape_i[0]
    var d = shape_i[1]
    var Dm = shape_i[2]
    var mat = A_i.reshape(List[Int](Dl * d, Dm))

    var qr_result = dense_tensor_qr[dtype](mat^, ctx)
    var Q_full = qr_result[0]
    var R_full = qr_result[1]

    var m = Q_full.shape[0]
    var q_cols = Q_full.shape[1]
    var r_cols = R_full.shape[1]
    var reduced_cols = m
    if reduced_cols > r_cols:
        reduced_cols = r_cols

    var host_Q = ctx.enqueue_create_host_buffer[dtype](Q_full.size)
    ctx.enqueue_copy(host_Q, Q_full.storage)
    var host_R = ctx.enqueue_create_host_buffer[dtype](R_full.size)
    ctx.enqueue_copy(host_R, R_full.storage)
    ctx.synchronize()

    var q_data = List[Scalar[dtype]](capacity=m * reduced_cols)
    for row in range(m):
        for col in range(reduced_cols):
            q_data.append(host_Q[row * q_cols + col])
    var Q_red = create_dense_tensor_from_data[dtype](ctx, q_data, List[Int](m, reduced_cols))

    var r_data = List[Scalar[dtype]](capacity=reduced_cols * r_cols)
    for row in range(reduced_cols):
        for col in range(r_cols):
            r_data.append(host_R[row * r_cols + col])
    var R_red = create_dense_tensor_from_data[dtype](ctx, r_data, List[Int](reduced_cols, r_cols))

    var A_i_new = Q_red.reshape(List[Int](Dl, d, reduced_cols))

    var shape_ip1 = A_ip1.shape.copy()
    var d_ip1 = shape_ip1[1]
    var Dr = shape_ip1[2]
    var A_ip1_mat = A_ip1.reshape(List[Int](Dm, d_ip1 * Dr))
    var R_shape = List[Int](reduced_cols, Dm)
    var dot_shape = List[Int](reduced_cols, d_ip1 * Dr)
    var A_ip1_new_mat = create_dense_tensor[dtype](ctx, dot_shape^, init_value=Scalar[dtype](0.0))
    dense_tensor_dot(A_ip1_new_mat, R_red^, A_ip1_mat^, ctx)
    var A_ip1_new = A_ip1_new_mat^.reshape(List[Int](reduced_cols, d_ip1, Dr))

    return (MPSSite[dtype, DenseTensor[dtype]](A_i_new^), MPSSite[dtype, DenseTensor[dtype]](A_ip1_new^))


fn mps_local_orthonormalize_rq_pair[dtype: DType = DType.float32](
    ctx: DeviceContext,
    var A_i: DenseTensor[dtype],
    var A_im1: DenseTensor[dtype],
) raises -> Tuple[MPSSite[dtype, DenseTensor[dtype]], MPSSite[dtype, DenseTensor[dtype]]]:
    """Right-orthonormalize site i and absorb R into site i-1 (C `mps_local_orthonormalize_rq`)."""
    var shape_i = A_i.shape.copy()
    var Dl = shape_i[0]
    var d = shape_i[1]
    var Dr = shape_i[2]
    var mat = A_i.reshape(List[Int](Dl, d * Dr))

    var mat_T = mat^.transpose(List[Int](1, 0), ctx)
    var qr_result = dense_tensor_qr[dtype](mat_T^, ctx)
    var Q_T = qr_result[0]
    var R_T = qr_result[1]

    var n = Q_T.shape[0]
    var k = Q_T.shape[1]
    var host_QT = ctx.enqueue_create_host_buffer[dtype](Q_T.size)
    ctx.enqueue_copy(host_QT, Q_T.storage)
    var host_RT = ctx.enqueue_create_host_buffer[dtype](R_T.size)
    ctx.enqueue_copy(host_RT, R_T.storage)
    ctx.synchronize()

    var r_data = List[Scalar[dtype]](capacity=Dl * k)
    for i in range(Dl):
        for j in range(k):
            r_data.append(host_RT[j * Dl + i])
    var R_out = create_dense_tensor_from_data[dtype](ctx, r_data, List[Int](Dl, k))

    var q_data = List[Scalar[dtype]](capacity=k * n)
    for i in range(k):
        for j in range(n):
            q_data.append(host_QT[j * k + i])
    var Q_out = create_dense_tensor_from_data[dtype](ctx, q_data, List[Int](k, n))
    var A_i_new = Q_out^.reshape(List[Int](k, d, Dr))

    var shape_im1 = A_im1.shape.copy()
    var Dl_prev = shape_im1[0]
    var d_prev = shape_im1[1]
    var Dm = shape_im1[2]
    var A_im1_mat = A_im1.reshape(List[Int](Dl_prev * d_prev, Dm))
    var dot_shape = List[Int](Dl_prev * d_prev, k)
    var A_im1_new_mat = create_dense_tensor[dtype](ctx, dot_shape^, init_value=Scalar[dtype](0.0))
    dense_tensor_dot(A_im1_new_mat, A_im1_mat^, R_out^, ctx)
    var A_im1_new = A_im1_new_mat^.reshape(List[Int](Dl_prev, d_prev, k))

    return (MPSSite[dtype, DenseTensor[dtype]](A_i_new^), MPSSite[dtype, DenseTensor[dtype]](A_im1_new^))


fn mps_orthonormalize_right[dtype: DType = DType.float32](
    ctx: DeviceContext,
    mut mps: MatrixProductState[dtype, DenseTensor[dtype]],
) raises:
    """Right-canonicalize MPS."""
    var N = mps.num_sites()
    if N < 1:
        return
    
    if N == 1:
        var Dl0 = mps.sites[0].left_bond_dim()
        var tail_elems = Dl0 * Dl0
        var tail_data = List[Scalar[dtype]](capacity=tail_elems)
        for _ in range(tail_elems):
            tail_data.append(Scalar[dtype](0.0))
        for b in range(Dl0):
            tail_data[b * Dl0 + b] = Scalar[dtype](1.0)
        var tail = create_dense_tensor_from_data[dtype](
            ctx, tail_data^, List[Int](Dl0, 1, Dl0)
        )
        var ortho0 = mps_local_orthonormalize_rq_pair[dtype](
            ctx, mps.sites[0].tensor, tail^
        )
        mps.sites[0] = ortho0[0]
        return

    for i in range(N - 1, 0, -1):
        var ortho = mps_local_orthonormalize_rq_pair[dtype](
            ctx, mps.sites[i].tensor, mps.sites[i - 1].tensor
        )
        mps.sites[i] = ortho[0]
        mps.sites[i - 1] = ortho[1]
        mps.bond_dims[i] = mps.sites[i - 1].right_bond_dim()

    var Dl0 = mps.sites[0].left_bond_dim()
    var tail_elems = Dl0 * Dl0
    var tail_data = List[Scalar[dtype]](capacity=tail_elems)
    for _ in range(tail_elems):
        tail_data.append(Scalar[dtype](0.0))
    for b in range(Dl0):
        tail_data[b * Dl0 + b] = Scalar[dtype](1.0)
    var tail = create_dense_tensor_from_data[dtype](
        ctx, tail_data^, List[Int](Dl0, 1, Dl0)
    )
    var ortho0 = mps_local_orthonormalize_rq_pair[dtype](
        ctx, mps.sites[0].tensor, tail^
    )
    mps.sites[0] = ortho0[0]


fn mps_orthogonalize_qr[dtype: DType = DType.float32](
    ctx: DeviceContext,
    var full_state: DenseTensor[dtype],
) raises -> MatrixProductState[dtype, DenseTensor[dtype]]:
    """Decompose a dense rank-N tensor into an MPS via successive QR sweeps.
    
    We iteratively reshape the remaining tensor into a matrix, run a QR, keep the Q
    part as the current site, and absorb R into the remaining block.

    TODO: We can add RQ sweeps to implement the right-canonicalization.
    
    Args:
        ctx: GPU device context.
        full_state: Dense tensor of shape [d, d, ..., d] (rank = num_sites).
                    All physical dimensions must be identical because the current
                    MatrixProductState assumes uniform local Hilbert spaces.
    
    Returns:
        Left-canonical MPS representing the same many-body vector.
    """
    var dims = full_state.shape.copy()
    var num_sites = len(dims)
    if num_sites == 0:
        raise Error("full_state must have rank >= 1 to build an MPS")

    var physical_dim = dims[0]
    for idx in range(num_sites):
        if dims[idx] != physical_dim:
            raise Error(
                "All physical dimensions must match the first axis ("
                + String(physical_dim)
                + "), but axis "
                + String(idx)
                + " has size "
                + String(dims[idx])
            )

    var sites = List[MPSSite[dtype, DenseTensor[dtype]]](capacity=num_sites)

    var augmented_shape = List[Int](capacity=num_sites + 1)
    augmented_shape.append(1)
    for dim in dims:
        augmented_shape.append(dim)
    var remainder = full_state.reshape(augmented_shape^)

    for _ in range(num_sites - 1):
        var ortho = mps_local_orthonormalize_qr[dtype](ctx, remainder^)
        sites.append(ortho[0])
        remainder = ortho[1]

    if len(remainder.shape) != 2:
        raise Error(
            "Unexpected remainder rank "
            + String(len(remainder.shape))
            + " while constructing final MPS site"
        )

    var final_shape = List[Int](remainder.shape[0], remainder.shape[1], 1)
    var final_site = remainder.reshape(final_shape^)
    var final_norm = final_site.norm(ctx)
    if final_norm > 0:
        final_site.scale_in_place(Scalar[dtype](1.0 / final_norm), ctx)
    sites.append(MPSSite[dtype, DenseTensor[dtype]](final_site^))

    return MatrixProductState[dtype, DenseTensor[dtype]](sites^)


fn mps_to_statevector[dtype: DType = DType.float32](
    psi: MatrixProductState[dtype, DenseTensor[dtype]],
    ctx: DeviceContext,
) raises -> DenseTensor[dtype]:
    """Contract MPS to full state vector (size d^L) in row-major physical index order.

    Same convention as C mps_to_statevector: coefficient of |i0,i1,...,i_{L-1}>
    is at linear index i0 + i1*d + ... + i_{L-1}*d^{L-1}.

    Only feasible for small L due to exponential size.
    """
    var L = psi.num_sites()
    if L == 0:
        raise Error("mps_to_statevector requires at least one site")
    
    var result = psi.sites[0].tensor
    var shape0 = result.shape.copy()
    var left_dim = shape0[0] * shape0[1]
    var right_dim = shape0[2]
    result = result^.reshape(List[Int](left_dim, right_dim))
    
    for i in range(1, L):
        var site = psi.sites[i].tensor
        var site_shape = site.shape.copy()
        var D_i = site_shape[0]
        var d_i = site_shape[1]
        var D_next = site_shape[2]
        if right_dim != D_i:
            raise Error("Bond dimension mismatch in mps_to_statevector")
        var site_flat = site.reshape(List[Int](D_i, d_i * D_next))
        var dot_rows = left_dim
        var dot_cols = d_i * D_next
        var result_new = create_dense_tensor[dtype](
            ctx, List[Int](dot_rows, dot_cols), row_major=True, init_value=Scalar[dtype](0.0)
        )
        dense_tensor_dot(result_new, result^, site_flat^, ctx)
        result = result_new^.reshape(List[Int](left_dim * d_i, D_next))
        left_dim = left_dim * d_i
        right_dim = D_next
    
    if right_dim != 1:
        raise Error("Expected trailing bond dimension 1 in mps_to_statevector")
    var total = left_dim
    return result^.reshape(List[Int](total))


fn mps_norm[dtype: DType = DType.float32](
    psi: MatrixProductState[dtype, DenseTensor[dtype]],
    ctx: DeviceContext,
) raises -> Float64:
    """Compute Euclidean norm of the MPS (sqrt of inner product with itself).

    Implemented by contracting to state vector then computing norm, so only
    suitable for small systems.
    """
    var vec = mps_to_statevector[dtype](psi, ctx)
    return vec.norm(ctx)

