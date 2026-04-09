"""Block sparse tensor: ChemTensor-compatible layout (additive quantum numbers).

Mirrors ``struct block_sparse_tensor`` / ``allocate_block_sparse_tensor`` in
``chemtensor/src/tensor/block_sparse_tensor.c``: logical dimensions, per-axis
sector quantum numbers, axis directions, and a dense ``DenseTensor`` per
conserved sector (flat index in ``dim_blocks[0] × … × dim_blocks[ndim-1]``).

Use :func:`allocate_block_sparse_for_tensor_dot` to build a pre-allocated ``C``
matching :func:`dense_tensor_dot` / C ``block_sparse_tensor_dot`` conventions,
then call :func:`block_sparse_tensor_dot`.
"""

from collections.list import List
from math import sqrt
from gpu.host import DeviceContext
from src.m_tensor.tensor_traits import TensorOps, TensorBackend
from src.m_tensor.dense_tensor import (
    DenseTensor,
    create_dense_tensor,
    create_dense_tensor_uninitialized,
    compute_row_major_strides,
    dense_tensor_dot,
    dense_tensor_svd_trunc,
)


# =============================================================================
# Quantum numbers & block identifiers (user-facing)
# =============================================================================


@value
struct QNumber:
    """Conserved quantum number label (maps to C ``qnumber`` / int)."""

    var value: Int

    fn __init__(out self, value: Int = 0):
        self.value = value

    fn __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    fn __add__(self, other: Self) -> Self:
        return QNumber(self.value + other.value)

    fn __neg__(self) -> Self:
        return QNumber(-self.value)


@value
struct BlockIndex:
    """Sector quantum numbers: one conserved label per tensor leg (like C ``get_block``)."""

    var qnums: List[QNumber]

    fn __init__(out self, var qnums: List[QNumber]):
        self.qnums = qnums^

    fn rank(self) -> Int:
        return len(self.qnums)


@value
struct Block[dtype: DType]:
    """One dense sector block and its sector ``BlockIndex``."""

    var index: BlockIndex
    var data: DenseTensor[dtype]

    fn __init__(out self, index: BlockIndex, data: DenseTensor[dtype]):
        self.index = index
        self.data = data


# =============================================================================
# Layout helpers (C ``tensor_index_to_offset`` / ``offset_to_tensor_index``)
# =============================================================================


fn _dim_product(dims: List[Int]) -> Int:
    var p = 1
    for i in range(len(dims)):
        p *= dims[i]
    return p


fn _tensor_index_to_offset(ndim: Int, dims: List[Int], idx: List[Int]) -> Int:
    var offset = 0
    var fac = 1
    for i in range(ndim - 1, -1, -1):
        offset += fac * idx[i]
        fac *= dims[i]
    return offset


fn _offset_to_tensor_index(offset: Int, ndim: Int, dims: List[Int], mut idx: List[Int]) -> None:
    var n = offset
    for i in range(ndim - 1, -1, -1):
        idx[i] = n % dims[i]
        n = n // dims[i]


fn _next_tensor_index(ndim: Int, dims: List[Int], mut idx: List[Int]) -> Bool:
    """Lexicographic next; returns False when past end."""
    for i in range(ndim - 1, -1, -1):
        idx[i] += 1
        if idx[i] < dims[i]:
            return True
        idx[i] = 0
    return False


fn _leg_to_ints(var leg: List[QNumber]) -> List[Int]:
    var out = List[Int](capacity=len(leg))
    for q in leg:
        out.append(q.value)
    return out^


fn _sort_sector_pairs(inout qnums: List[Int], inout counts: List[Int]) -> None:
    """Sort ``qnums`` ascending; permute ``counts`` the same way (insertion sort)."""
    var n = len(qnums)
    for i in range(1, n):
        var qk = qnums[i]
        var ck = counts[i]
        var j = i - 1
        while j >= 0 and qnums[j] > qk:
            qnums[j + 1] = qnums[j]
            counts[j + 1] = counts[j]
            j -= 1
        qnums[j + 1] = qk
        counts[j + 1] = ck


@value
struct _SectorLists:
    var qnums: List[Int]
    var counts: List[Int]


fn _collect_sectors(leg: List[Int]) raises -> _SectorLists:
    """Distinct quantum numbers on one leg with multiplicities, sorted by qnum."""
    var uniq = List[Int]()
    var counts = List[Int]()
    for t in range(len(leg)):
        var q = leg[t]
        var found = False
        for k in range(len(uniq)):
            if uniq[k] == q:
                counts[k] += 1
                found = True
                break
        if not found:
            uniq.append(q)
            counts.append(1)
    _sort_sector_pairs(uniq, counts)
    return _SectorLists(uniq^, counts^)


fn _qnums_lists_from_per_leg(var per_leg: List[List[QNumber]]) raises -> List[List[Int]]:
    var out = List[List[Int]](capacity=len(per_leg))
    for leg in per_leg:
        var ints = List[Int](capacity=len(leg))
        for q in leg:
            ints.append(q.value)
        out.append(ints^)
    return out^


fn _deep_copy_int_matrix(src: List[List[Int]]) -> List[List[Int]]:
    var out = List[List[Int]](capacity=len(src))
    for i in range(len(src)):
        out.append(src[i].copy())
    return out^


# =============================================================================
# BlockSparseTensor
# =============================================================================


struct BlockSparseTensor[dtype: DType](Writable, Movable, TensorOps):
    """ChemTensor-style block-sparse tensor with GPU ``DenseTensor`` blocks."""

    var ndim: Int
    var dim_logical: List[Int]
    var dim_blocks: List[Int]
    var axis_dir: List[Int]
    var qnums_logical: List[List[Int]]
    var qnums_blocks: List[List[Int]]
    var sector_counts: List[List[Int]]
    var stride_logical: List[Int]
    var logical_size: Int
    var nblocks_total: Int
    var block_present: List[Bool]
    var blocks_flat: List[DenseTensor[dtype]]
    var _dummy_scalar: DenseTensor[dtype]

    fn __init__(
        out self,
        ndim: Int,
        var dim_logical: List[Int],
        var dim_blocks: List[Int],
        var axis_dir: List[Int],
        var qnums_logical: List[List[Int]],
        var qnums_blocks: List[List[Int]],
        var sector_counts: List[List[Int]],
        var stride_logical: List[Int],
        logical_size: Int,
        nblocks_total: Int,
        var block_present: List[Bool],
        var blocks_flat: List[DenseTensor[dtype]],
        dummy: DenseTensor[dtype],
    ):
        self.ndim = ndim
        self.dim_logical = dim_logical^
        self.dim_blocks = dim_blocks^
        self.axis_dir = axis_dir^
        self.qnums_logical = qnums_logical^
        self.qnums_blocks = qnums_blocks^
        self.sector_counts = sector_counts^
        self.stride_logical = stride_logical^
        self.logical_size = logical_size
        self.nblocks_total = nblocks_total
        self.block_present = block_present^
        self.blocks_flat = blocks_flat^
        self._dummy_scalar = dummy

    fn __copyinit__(out self, existing: Self):
        self.ndim = existing.ndim
        self.dim_logical = existing.dim_logical.copy()
        self.dim_blocks = existing.dim_blocks.copy()
        self.axis_dir = existing.axis_dir.copy()
        self.qnums_logical = _deep_copy_int_matrix(existing.qnums_logical)
        self.qnums_blocks = _deep_copy_int_matrix(existing.qnums_blocks)
        self.sector_counts = _deep_copy_int_matrix(existing.sector_counts)
        self.stride_logical = existing.stride_logical.copy()
        self.logical_size = existing.logical_size
        self.nblocks_total = existing.nblocks_total
        self.block_present = existing.block_present.copy()
        self.blocks_flat = existing.blocks_flat.copy()
        self._dummy_scalar = existing._dummy_scalar

    fn get_shape(self) -> List[Int]:
        return self.dim_logical.copy()

    fn get_stride(self) -> List[Int]:
        return self.stride_logical.copy()

    fn get_size(self) -> Int:
        return self.logical_size

    fn get_rank(self) -> Int:
        return self.ndim

    fn is_contiguous(self) -> Bool:
        return False

    fn get_flat_index(self, indices: List[Int]) -> Int:
        var flat_idx = 0
        for i in range(self.ndim):
            flat_idx += indices[i] * self.stride_logical[i]
        return flat_idx

    fn compute_norm_sq(self, ctx: DeviceContext) raises -> Float64:
        var s = 0.0
        for k in range(self.nblocks_total):
            if not self.block_present[k]:
                continue
            var ns = self.blocks_flat[k].compute_norm_sq(ctx)
            s += ns
        return s

    fn compute_norm(self, ctx: DeviceContext) raises -> Float64:
        return sqrt(self.compute_norm_sq(ctx))

    fn compute_dot_product(self, other: Self, ctx: DeviceContext) raises -> Float64:
        if self.logical_size != other.logical_size or self.ndim != other.ndim:
            raise Error("BlockSparseTensor.compute_dot_product: incompatible tensors")
        for i in range(self.ndim):
            if self.dim_logical[i] != other.dim_logical[i]:
                raise Error("BlockSparseTensor.compute_dot_product: shape mismatch")
            if self.dim_blocks[i] != other.dim_blocks[i]:
                raise Error("BlockSparseTensor.compute_dot_product: block layout mismatch")
            for j in range(self.dim_blocks[i]):
                if self.qnums_blocks[i][j] != other.qnums_blocks[i][j]:
                    raise Error("BlockSparseTensor.compute_dot_product: sector qnums mismatch")
            if self.axis_dir[i] != other.axis_dir[i]:
                raise Error("BlockSparseTensor.compute_dot_product: axis_dir mismatch")
        var acc = 0.0
        for k in range(self.nblocks_total):
            if not (self.block_present[k] and other.block_present[k]):
                continue
            acc += self.blocks_flat[k].compute_dot_product(other.blocks_flat[k], ctx)
        return acc

    fn print_contents(self, ctx: DeviceContext) raises -> None:
        print("BlockSparseTensor ndim=", self.ndim, " logical_shape=", end="")
        for i in range(self.ndim):
            print(" ", self.dim_logical[i], end="")
        print("  num_block_slots=", self.nblocks_total, " occupied=", end="")
        var occ = 0
        for k in range(self.nblocks_total):
            if self.block_present[k]:
                occ += 1
        print(occ, " sparsity_ratio=", self.sparsity_ratio())

    fn dot_product(self, other: Self, ctx: DeviceContext) raises -> Float64:
        return self.compute_dot_product(other, ctx)

    fn norm(self, ctx: DeviceContext) raises -> Float64:
        return self.compute_norm(ctx)

    fn norm_sq(self, ctx: DeviceContext) raises -> Float64:
        return self.compute_norm_sq(ctx)

    @staticmethod
    fn backend() -> TensorBackend:
        return TensorBackend(TensorBackend.BLOCK_SPARSE)

    fn write_to[W: Writer](self, mut writer: W) -> None:
        writer.write("BlockSparseTensor[dtype=")
        writer.write(Self.dtype)
        writer.write(", ndim=")
        writer.write(self.ndim)
        writer.write(", logical=(")
        for i in range(self.ndim):
            if i > 0:
                writer.write(", ")
            writer.write(self.dim_logical[i])
        writer.write("), slots=")
        writer.write(self.nblocks_total)
        writer.write("]")

    fn actual_nonzero_count(self) -> Int:
        var total = 0
        for k in range(self.nblocks_total):
            if self.block_present[k]:
                total += self.blocks_flat[k].size
        return total

    fn sparsity_ratio(self) -> Float64:
        if self.logical_size == 0:
            return 0.0
        return Float64(self.actual_nonzero_count()) / Float64(self.logical_size)

    fn get_block(self, index: BlockIndex) raises -> Block[dtype]:
        if len(index.qnums) != self.ndim:
            raise Error("get_block: BlockIndex rank does not match tensor ndim")
        var idx = List[Int](capacity=self.ndim)
        for i in range(self.ndim):
            var qv = index.qnums[i].value
            var found = False
            for j in range(self.dim_blocks[i]):
                if self.qnums_blocks[i][j] == qv:
                    idx.append(j)
                    found = True
                    break
            if not found:
                raise Error("get_block: quantum number not found on axis")
        var flat = _tensor_index_to_offset(self.ndim, self.dim_blocks, idx)
        if not self.block_present[flat]:
            raise Error("get_block: sector violates charge conservation (no block)")
        return Block[dtype](index, self.blocks_flat[flat])

    fn transpose(var self, perm: List[Int], ctx: DeviceContext) raises -> BlockSparseTensor[dtype]:
        if len(perm) != self.ndim:
            raise Error("transpose: perm length must equal ndim")
        var used = List[Bool](capacity=self.ndim)
        for _ in range(self.ndim):
            used.append(False)
        for i in range(self.ndim):
            var p = perm[i]
            if p < 0 or p >= self.ndim:
                raise Error("transpose: invalid permutation")
            if used[p]:
                raise Error("transpose: perm is not a permutation")
            used[p] = True

        var r_dim = List[Int](capacity=self.ndim)
        var r_db = List[Int](capacity=self.ndim)
        var r_ad = List[Int](capacity=self.ndim)
        var r_ql = List[List[Int]](capacity=self.ndim)
        var r_qb = List[List[Int]](capacity=self.ndim)
        var r_sc = List[List[Int]](capacity=self.ndim)
        for i in range(self.ndim):
            var ax = perm[i]
            r_dim.append(self.dim_logical[ax])
            r_db.append(self.dim_blocks[ax])
            r_ad.append(self.axis_dir[ax])
            r_ql.append(self.qnums_logical[ax].copy())
            r_qb.append(self.qnums_blocks[ax].copy())
            r_sc.append(self.sector_counts[ax].copy())

        var r_stride = compute_row_major_strides(r_dim, self.ndim)
        var r_logical = _dim_product(r_dim)
        var nblocks = _dim_product(r_db)
        var r_present = List[Bool](capacity=nblocks)
        var r_blocks = List[DenseTensor[dtype]](capacity=nblocks)
        for _ in range(nblocks):
            r_present.append(False)
            r_blocks.append(self._dummy_scalar)

        var idx_t = List[Int](capacity=self.ndim)
        for _ in range(self.ndim):
            idx_t.append(0)
        var idx_r = List[Int](capacity=self.ndim)
        for _ in range(self.ndim):
            idx_r.append(0)

        for k in range(self.nblocks_total):
            if not self.block_present[k]:
                continue
            _offset_to_tensor_index(k, self.ndim, self.dim_blocks, idx_t)
            for i in range(self.ndim):
                idx_r[i] = idx_t[perm[i]]
            var j = _tensor_index_to_offset(self.ndim, r_db, idx_r)
            var bt = self.blocks_flat[k]
            var perm_copy = perm.copy()
            var tnew = bt^.transpose(perm_copy^, ctx)
            r_blocks[j] = tnew^
            r_present[j] = True

        return BlockSparseTensor[dtype](
            self.ndim,
            r_dim^,
            r_db^,
            r_ad^,
            r_ql^,
            r_qb^,
            r_sc^,
            r_stride^,
            r_logical,
            nblocks,
            r_present^,
            r_blocks^,
            self._dummy_scalar,
        )

    fn reshape(var self, var new_shape: List[Int]) raises -> BlockSparseTensor[dtype]:
        _ = self^
        _ = new_shape^
        raise Error(
            "BlockSparseTensor.reshape: use block_sparse_to_dense / dense_to_block_sparse "
            + "with a DeviceContext when reshaping changes the sector structure"
        )

    fn flatten_dims(var self, start: Int, end: Int, ctx: DeviceContext) raises -> BlockSparseTensor[
        dtype
    ]:
        _ = self^
        _ = ctx
        raise Error(
            "BlockSparseTensor.flatten_axes: not implemented (see ChemTensor "
            + "block_sparse_tensor_flatten_axes)"
        )

    fn copy_to_contiguous(var self, ctx: DeviceContext) raises -> BlockSparseTensor[dtype]:
        var s = self^
        var r_present = List[Bool](capacity=s.nblocks_total)
        var r_blocks = List[DenseTensor[dtype]](capacity=s.nblocks_total)
        for k in range(s.nblocks_total):
            r_present.append(s.block_present[k])
            if s.block_present[k]:
                r_blocks.append(s.blocks_flat[k]^.copy_to_contiguous(ctx))
            else:
                r_blocks.append(s._dummy_scalar)
        return BlockSparseTensor[dtype](
            s.ndim,
            s.dim_logical.copy(),
            s.dim_blocks.copy(),
            s.axis_dir.copy(),
            _deep_copy_int_matrix(s.qnums_logical),
            _deep_copy_int_matrix(s.qnums_blocks),
            _deep_copy_int_matrix(s.sector_counts),
            s.stride_logical.copy(),
            s.logical_size,
            s.nblocks_total,
            r_present^,
            r_blocks^,
            s._dummy_scalar,
        )

    fn scale_in_place(mut self, scale: Scalar[dtype], ctx: DeviceContext) raises -> None:
        for k in range(self.nblocks_total):
            if self.block_present[k]:
                self.blocks_flat[k].scale_in_place(scale, ctx)

    fn axpy_in_place(mut self, alpha: Scalar[dtype], x: Self, ctx: DeviceContext) raises -> None:
        if self.nblocks_total != x.nblocks_total or self.ndim != x.ndim:
            raise Error("axpy_in_place: incompatible tensors")
        for k in range(self.nblocks_total):
            if not (self.block_present[k] and x.block_present[k]):
                if x.block_present[k] and not self.block_present[k]:
                    raise Error("axpy_in_place: output missing block present in x")
                continue
            self.blocks_flat[k].axpy_in_place(alpha, x.blocks_flat[k], ctx)


fn _allocate_block_sparse_tensor_inner[dtype: DType](
    ctx: DeviceContext,
    ndim: Int,
    var dim_logical: List[Int],
    var axis_dir: List[Int],
    var qnums_logical: List[List[Int]],
    init_value: Optional[Scalar[dtype]],
    zero_fill: Bool = False,
) raises -> BlockSparseTensor[dtype]:
    if ndim == 0:
        var dummy = create_dense_tensor[dtype](ctx, List[Int](1)^, init_value=Scalar[dtype](0.0))
        var zero_block = create_dense_tensor[dtype](ctx, List[Int]()^, init_value=Scalar[dtype](0.0))
        var bp = List[Bool]()
        bp.append(True)
        var bf = List[DenseTensor[dtype]]()
        bf.append(zero_block^)
        var empty = List[Int]()
        var empty_ql = List[List[Int]]()
        var empty_qb = List[List[Int]]()
        var empty_sc = List[List[Int]]()
        var st = List[Int]()
        return BlockSparseTensor[dtype](
            0,
            empty^,
            empty^,
            empty^,
            empty_ql^,
            empty_qb^,
            empty_sc^,
            st^,
            1,
            1,
            bp^,
            bf^,
            dummy^,
        )

    if len(axis_dir) != ndim or len(dim_logical) != ndim or len(qnums_logical) != ndim:
        raise Error("allocate_block_sparse_tensor: inconsistent ndim")

    var dim_blocks = List[Int](capacity=ndim)
    var qnums_blocks = List[List[Int]](capacity=ndim)
    var sector_counts = List[List[Int]](capacity=ndim)

    for i in range(ndim):
        if len(qnums_logical[i]) != dim_logical[i]:
            raise Error("allocate_block_sparse_tensor: qnums length must match dim_logical")
        var pair = _collect_sectors(qnums_logical[i])
        dim_blocks.append(len(pair.qnums))
        qnums_blocks.append(pair.qnums.copy())
        sector_counts.append(pair.counts.copy())

    var stride = compute_row_major_strides(dim_logical, ndim)
    var logical_size = _dim_product(dim_logical)
    var nblocks = _dim_product(dim_blocks)

    var dummy = create_dense_tensor[dtype](ctx, List[Int](1)^, init_value=Scalar[dtype](0.0))

    var block_present = List[Bool](capacity=nblocks)
    var blocks_flat = List[DenseTensor[dtype]](capacity=nblocks)

    var idx = List[Int](capacity=ndim)
    for _ in range(ndim):
        idx.append(0)

    for k in range(nblocks):
        _offset_to_tensor_index(k, ndim, dim_blocks, idx)
        var qsum = 0
        for i in range(ndim):
            qsum += axis_dir[i] * qnums_blocks[i][idx[i]]
        if qsum != 0:
            block_present.append(False)
            blocks_flat.append(dummy)
            continue
        var bshape = List[Int](capacity=ndim)
        for i in range(ndim):
            bshape.append(sector_counts[i][idx[i]])
        var bt: DenseTensor[dtype]
        if zero_fill:
            bt = create_dense_tensor[dtype](ctx, bshape^, init_value=Scalar[dtype](0.0))
        elif init_value is not None:
            bt = create_dense_tensor[dtype](ctx, bshape^, init_value=init_value.value())
        else:
            bt = create_dense_tensor_uninitialized[dtype](ctx, bshape^)
        block_present.append(True)
        blocks_flat.append(bt^)

    return BlockSparseTensor[dtype](
        ndim,
        dim_logical^,
        dim_blocks^,
        axis_dir^,
        qnums_logical^,
        qnums_blocks^,
        sector_counts^,
        stride^,
        logical_size,
        nblocks,
        block_present^,
        blocks_flat^,
        dummy^,
    )


fn create_block_sparse_tensor[dtype: DType](
    ctx: DeviceContext,
    var shape: List[Int],
    var qnums_per_leg: List[List[QNumber]],
    init_value: Optional[Scalar[dtype]] = None,
    axis_dir: Optional[List[Int]] = None,
) raises -> BlockSparseTensor[dtype]:
    """Allocate like C ``allocate_block_sparse_tensor`` (default ``axis_dir`` = +1 on every leg)."""
    var ndim = len(shape)
    var ad = List[Int](capacity=ndim)
    if axis_dir is None:
        for _ in range(ndim):
            ad.append(1)
    else:
        var ad_in = axis_dir.value()
        if len(ad_in) != ndim:
            raise Error("create_block_sparse_tensor: axis_dir length must match rank")
        for i in range(ndim):
            var v = ad_in[i]
            if v != 1 and v != -1:
                raise Error("create_block_sparse_tensor: axis_dir entries must be ±1")
            ad.append(v)
    var qnums_int = _qnums_lists_from_per_leg(qnums_per_leg^)
    return _allocate_block_sparse_tensor_inner[dtype](ctx, ndim, shape^, ad^, qnums_int^, init_value)


fn allocate_block_sparse_tensor_like[dtype: DType](
    ctx: DeviceContext, existing: BlockSparseTensor[dtype], init_value: Optional[Scalar[dtype]] = None
) raises -> BlockSparseTensor[dtype]:
    """Same sector layout as ``existing``; new GPU buffers."""
    return _allocate_block_sparse_tensor_inner[dtype](
        ctx,
        existing.ndim,
        existing.dim_logical.copy(),
        existing.axis_dir.copy(),
        _deep_copy_int_matrix(existing.qnums_logical),
        init_value,
    )


fn _effective_axrange_B_for_dot[dtype: DType](
    a: BlockSparseTensor[dtype], b: BlockSparseTensor[dtype], ndim_mult: Int, axrange_a: Bool, axrange_b: Bool
) raises -> Bool:
    """Match ``dense_tensor_dot`` inference when both B flags are false."""
    var rank_a = a.ndim
    var rank_b = b.ndim
    if not axrange_a and not axrange_b:
        var b_leading = True
        var b_trailing = True
        for i in range(ndim_mult):
            var a_sz = a.dim_logical[(0 if axrange_a else rank_a - ndim_mult) + i]
            if a_sz != b.dim_logical[i]:
                b_leading = False
            if a_sz != b.dim_logical[rank_b - ndim_mult + i]:
                b_trailing = False
        if b_leading:
            return True
        if b_trailing:
            return False
        return True
    return axrange_b


fn allocate_block_sparse_for_tensor_dot[dtype: DType](
    a: BlockSparseTensor[dtype],
    b: BlockSparseTensor[dtype],
    ctx: DeviceContext,
    ndim_mult: Int,
    axrange_a: Bool,
    axrange_b: Bool,
    init_value: Optional[Scalar[dtype]] = None,
) raises -> BlockSparseTensor[dtype]:
    """Build an output tensor ``C`` with the sector structure of ``A @ B`` (ChemTensor dot)."""
    var eff_b = _effective_axrange_B_for_dot[dtype](a, b, ndim_mult, axrange_a, axrange_b)
    var shift_a = 0 if axrange_a else a.ndim - ndim_mult
    var shift_b = 0 if eff_b else b.ndim - ndim_mult
    var offset_a = ndim_mult if axrange_a else 0
    var offset_b = ndim_mult if eff_b else 0

    if ndim_mult < 1 or a.ndim < ndim_mult or b.ndim < ndim_mult:
        raise Error("allocate_block_sparse_for_tensor_dot: invalid ndim_mult")

    for i in range(ndim_mult):
        if a.dim_logical[shift_a + i] != b.dim_logical[shift_b + i]:
            raise Error("allocate_block_sparse_for_tensor_dot: contracted dim mismatch")
        if a.dim_blocks[shift_a + i] != b.dim_blocks[shift_b + i]:
            raise Error("allocate_block_sparse_for_tensor_dot: contracted sector count mismatch")
        if a.axis_dir[shift_a + i] != -b.axis_dir[shift_b + i]:
            raise Error("allocate_block_sparse_for_tensor_dot: axis_dir must be opposite on contracted legs")
        for t in range(a.dim_logical[shift_a + i]):
            if a.qnums_logical[shift_a + i][t] != b.qnums_logical[shift_b + i][t]:
                raise Error("allocate_block_sparse_for_tensor_dot: logical qnums must match on contracted legs")
        for s in range(a.dim_blocks[shift_a + i]):
            if a.qnums_blocks[shift_a + i][s] != b.qnums_blocks[shift_b + i][s]:
                raise Error("allocate_block_sparse_for_tensor_dot: sector qnums mismatch")

    if a.ndim + b.ndim == 2 * ndim_mult:
        return _allocate_block_sparse_tensor_inner[dtype](
            ctx, 0, List[Int]()^, List[Int]()^, List[List[Int]]()^, init_value, zero_fill=True
        )

    var ndim_r = a.ndim + b.ndim - 2 * ndim_mult
    var r_dim = List[Int](capacity=ndim_r)
    var r_axis = List[Int](capacity=ndim_r)
    var r_ql = List[List[Int]](capacity=ndim_r)

    for i in range(a.ndim - ndim_mult):
        r_dim.append(a.dim_logical[offset_a + i])
        r_axis.append(a.axis_dir[offset_a + i])
        r_ql.append(a.qnums_logical[offset_a + i].copy())
    for i in range(b.ndim - ndim_mult):
        r_dim.append(b.dim_logical[offset_b + i])
        r_axis.append(b.axis_dir[offset_b + i])
        r_ql.append(b.qnums_logical[offset_b + i].copy())

    return _allocate_block_sparse_tensor_inner[dtype](
        ctx, ndim_r, r_dim^, r_axis^, r_ql^, init_value, zero_fill=True
    )


fn block_sparse_tensor_dot[dtype: DType](
    mut c: BlockSparseTensor[dtype],
    var a: BlockSparseTensor[dtype],
    var b: BlockSparseTensor[dtype],
    ctx: DeviceContext,
    ndim_mult: Int = 1,
    axrange_a: Bool = False,
    axrange_b: Bool = False,
) raises:
    """Blocked contraction; ``C`` must be pre-allocated (use :func:`allocate_block_sparse_for_tensor_dot`)."""
    var eff_b = _effective_axrange_B_for_dot[dtype](a, b, ndim_mult, axrange_a, axrange_b)
    var shift_a = 0 if axrange_a else a.ndim - ndim_mult
    var shift_b = 0 if eff_b else b.ndim - ndim_mult
    var offset_a = ndim_mult if axrange_a else 0
    var offset_b = ndim_mult if eff_b else 0

    if ndim_mult < 1 or a.ndim < ndim_mult or b.ndim < ndim_mult:
        raise Error("block_sparse_tensor_dot: invalid ndim_mult")

    for i in range(ndim_mult):
        if a.dim_logical[shift_a + i] != b.dim_logical[shift_b + i]:
            raise Error("block_sparse_tensor_dot: contracted dim mismatch")
        if a.dim_blocks[shift_a + i] != b.dim_blocks[shift_b + i]:
            raise Error("block_sparse_tensor_dot: sector layout mismatch on contraction")
        if a.axis_dir[shift_a + i] != -b.axis_dir[shift_b + i]:
            raise Error("block_sparse_tensor_dot: axis_dir mismatch")

    # Zero occupied output blocks (caller should use zero-initialized ``C`` from the allocator)
    for k in range(c.nblocks_total):
        if c.block_present[k]:
            c.blocks_flat[k].scale_in_place(Scalar[dtype](0.0), ctx)

    # Scalar (0-tensor) output
    if c.ndim == 0:
        if not (a.ndim == ndim_mult and b.ndim == ndim_mult):
            raise Error("block_sparse_tensor_dot: scalar output dimension mismatch")
        var ncontract = 1
        for i in range(ndim_mult):
            ncontract *= a.dim_blocks[i]
        var idx_c = List[Int](capacity=ndim_mult)
        for _ in range(ndim_mult):
            idx_c.append(0)
        var out0 = c.blocks_flat[0]
        var tmp_sc = create_dense_tensor_uninitialized[dtype](ctx, out0.shape.copy())
        var first_sc = True
        for m in range(ncontract):
            _offset_to_tensor_index(m, ndim_mult, a.dim_blocks, idx_c)
            var qsum = 0
            for i in range(a.ndim):
                qsum += a.axis_dir[i] * a.qnums_blocks[i][idx_c[i]]
            if qsum != 0:
                continue
            var ib = _tensor_index_to_offset(a.ndim, a.dim_blocks, idx_c)
            if not (a.block_present[ib] and b.block_present[ib]):
                continue
            if first_sc:
                dense_tensor_dot(
                    out0,
                    a.blocks_flat[ib]^,
                    b.blocks_flat[ib]^,
                    ctx,
                    ndim_mult,
                    axrange_a,
                    eff_b,
                )
                first_sc = False
            else:
                dense_tensor_dot(
                    tmp_sc,
                    a.blocks_flat[ib]^,
                    b.blocks_flat[ib]^,
                    ctx,
                    ndim_mult,
                    axrange_a,
                    eff_b,
                )
                out0.axpy_in_place(Scalar[dtype](1.0), tmp_sc, ctx)
        _ = tmp_sc^
        _ = a^
        _ = b^
        return

    if c.ndim != a.ndim + b.ndim - 2 * ndim_mult:
        raise Error("block_sparse_tensor_dot: C has wrong rank")

    var idx_r = List[Int](capacity=c.ndim)
    var idx_sa = List[Int](capacity=a.ndim)
    var idx_sb = List[Int](capacity=b.ndim)
    var idx_ct = List[Int](capacity=ndim_mult)
    for _ in range(c.ndim):
        idx_r.append(0)
    for _ in range(a.ndim):
        idx_sa.append(0)
    for _ in range(b.ndim):
        idx_sb.append(0)
    for _ in range(ndim_mult):
        idx_ct.append(0)

    var contract_dims = List[Int](capacity=ndim_mult)
    for i in range(ndim_mult):
        contract_dims.append(b.dim_blocks[shift_b + i])
    var ncontract = _dim_product(contract_dims)

    for k in range(c.nblocks_total):
        if not c.block_present[k]:
            continue
        _offset_to_tensor_index(k, c.ndim, c.dim_blocks, idx_r)
        var first = True
        var tmp = create_dense_tensor_uninitialized[dtype](ctx, c.blocks_flat[k].shape.copy())
        for m in range(ncontract):
            _offset_to_tensor_index(m, ndim_mult, contract_dims, idx_ct)
            for i in range(a.ndim - ndim_mult):
                idx_sa[offset_a + i] = idx_r[i]
            for i in range(ndim_mult):
                idx_sa[shift_a + i] = idx_ct[i]
            var qsum_a = 0
            for i in range(a.ndim):
                qsum_a += a.axis_dir[i] * a.qnums_blocks[i][idx_sa[i]]
            if qsum_a != 0:
                continue
            for i in range(b.ndim - ndim_mult):
                idx_sb[offset_b + i] = idx_r[(a.ndim - ndim_mult) + i]
            for i in range(ndim_mult):
                idx_sb[shift_b + i] = idx_ct[i]
            var oa = _tensor_index_to_offset(a.ndim, a.dim_blocks, idx_sa)
            var ob = _tensor_index_to_offset(b.ndim, b.dim_blocks, idx_sb)
            if not (a.block_present[oa] and b.block_present[ob]):
                continue
            if first:
                dense_tensor_dot(
                    c.blocks_flat[k],
                    a.blocks_flat[oa]^,
                    b.blocks_flat[ob]^,
                    ctx,
                    ndim_mult,
                    axrange_a,
                    eff_b,
                )
                first = False
            else:
                dense_tensor_dot(
                    tmp,
                    a.blocks_flat[oa]^,
                    b.blocks_flat[ob]^,
                    ctx,
                    ndim_mult,
                    axrange_a,
                    eff_b,
                )
                c.blocks_flat[k].axpy_in_place(Scalar[dtype](1.0), tmp, ctx)
        _ = tmp^

    _ = a^
    _ = b^


fn block_sparse_tensor_svd_trunc[dtype: DType](
    var tensor: BlockSparseTensor[dtype],
    ctx: DeviceContext,
    chi_max: Int,
    eps_trunc: Float64 = 1e-12,
) raises -> Tuple[BlockSparseTensor[dtype], BlockSparseTensor[dtype], BlockSparseTensor[dtype], Int]:
    """Expand to dense, run :func:`dense_tensor_svd_trunc`, wrap results as single-sector block tensors."""
    var dense = block_sparse_to_dense(tensor^, ctx)
    var uu_ss_vv = dense_tensor_svd_trunc[dtype](dense^, ctx, chi_max, eps_trunc)
    var uu = uu_ss_vv[0]
    var ss = uu_ss_vv[1]
    var vv = uu_ss_vv[2]
    var chi = uu_ss_vv[3]
    var u_bs = _dense_matrix_to_single_sector_block_sparse(ctx, uu^)
    var s_bs = _dense_vector_to_single_sector_block_sparse(ctx, ss^)
    var v_bs = _dense_matrix_to_single_sector_block_sparse(ctx, vv^)
    return (u_bs^, s_bs^, v_bs^, chi)


fn _dense_matrix_to_single_sector_block_sparse[dtype: DType](
    ctx: DeviceContext, var t: DenseTensor[dtype]
) raises -> BlockSparseTensor[dtype]:
    var sh = t.shape
    if len(sh) != 2:
        raise Error("_dense_matrix_to_single_sector_block_sparse: expected matrix")
    var qrows = List[QNumber](capacity=sh[0])
    for _ in range(sh[0]):
        qrows.append(QNumber(0))
    var qcols = List[QNumber](capacity=sh[1])
    for _ in range(sh[1]):
        qcols.append(QNumber(0))
    var legs = List[List[QNumber]]()
    legs.append(qrows^)
    legs.append(qcols^)
    var out = create_block_sparse_tensor[dtype](ctx, sh.copy(), legs^, init_value=Scalar[dtype](0.0))
    dense_to_block_sparse_entries(t^, out^, ctx)
    return out^


fn _dense_vector_to_single_sector_block_sparse[dtype: DType](
    ctx: DeviceContext, var t: DenseTensor[dtype]
) raises -> BlockSparseTensor[dtype]:
    var sh = t.shape
    if len(sh) != 1:
        raise Error("_dense_vector_to_single_sector_block_sparse: expected vector")
    var q0 = List[QNumber](capacity=sh[0])
    for _ in range(sh[0]):
        q0.append(QNumber(0))
    var legs = List[List[QNumber]]()
    legs.append(q0^)
    var out = create_block_sparse_tensor[dtype](ctx, sh.copy(), legs^, init_value=Scalar[dtype](0.0))
    dense_to_block_sparse_entries(t^, out^, ctx)
    return out^


fn dense_to_block_sparse[dtype: DType](
    var dense: DenseTensor[dtype],
    var qnums_per_leg: List[List[QNumber]],
    ctx: DeviceContext,
    tol: Float64 = 1e-14,
) raises -> BlockSparseTensor[dtype]:
    _ = tol
    var shape = dense.shape.copy()
    var ad = List[Int](capacity=len(shape))
    for _ in range(len(shape)):
        ad.append(1)
    var qnums_int = _qnums_lists_from_per_leg(qnums_per_leg^)
    var out = _allocate_block_sparse_tensor_inner[dtype](
        ctx, len(shape), shape^, ad^, qnums_int^, init_value=Scalar[dtype](0.0)
    )
    dense_to_block_sparse_entries(dense^, out^, ctx)
    return out^


fn dense_to_block_sparse_entries[dtype: DType](
    var dense: DenseTensor[dtype], mut target: BlockSparseTensor[dtype], ctx: DeviceContext
) raises:
    """Copy entries into an already allocated tensor (C ``dense_to_block_sparse_tensor_entries``)."""
    if len(dense.shape) != target.ndim:
        raise Error("dense_to_block_sparse_entries: ndim mismatch")
    for i in range(target.ndim):
        if dense.shape[i] != target.dim_logical[i]:
            raise Error("dense_to_block_sparse_entries: shape mismatch")

    var idx_block = List[Int](capacity=target.ndim)
    for _ in range(target.ndim):
        idx_block.append(0)
    var idx_b = List[Int](capacity=target.ndim)
    var idx_t = List[Int](capacity=target.ndim)
    for _ in range(target.ndim):
        idx_b.append(0)
        idx_t.append(0)

    for k in range(target.nblocks_total):
        if not target.block_present[k]:
            continue
        _offset_to_tensor_index(k, target.ndim, target.dim_blocks, idx_block)
        var index_map = List[List[Int]](capacity=target.ndim)
        for i in range(target.ndim):
            var qbi = target.qnums_blocks[i][idx_block[i]]
            var mult = target.sector_counts[i][idx_block[i]]
            var mp = List[Int](capacity=mult)
            var c = 0
            for j in range(target.dim_logical[i]):
                if target.qnums_logical[i][j] == qbi:
                    mp.append(j)
                    c += 1
            if c != mult:
                raise Error("dense_to_block_sparse_entries: internal sector multiplicity error")
            index_map.append(mp^)

        var blk = target.blocks_flat[k]
        var nelem = blk.size
        var host_d = ctx.enqueue_create_host_buffer[dtype](dense.size)
        ctx.enqueue_copy(host_d, dense.storage)
        var host_b = ctx.enqueue_create_host_buffer[dtype](nelem)
        ctx.synchronize()

        for j in range(nelem):
            if target.ndim > 0:
                _offset_to_tensor_index(j, target.ndim, blk.shape, idx_b)
            for i in range(target.ndim):
                idx_t[i] = index_map[i][idx_b[i]]
            var toff = 0
            if target.ndim > 0:
                toff = _tensor_index_to_offset(target.ndim, dense.shape, idx_t)
            host_b[j] = host_d[toff]

        ctx.enqueue_copy(blk.storage, host_b)
        ctx.synchronize()
    _ = dense^


fn block_sparse_to_dense[dtype: DType](var sparse: BlockSparseTensor[dtype], ctx: DeviceContext) raises -> DenseTensor[
    dtype
]:
    """Scatter blocks into a dense row-major tensor (C ``block_sparse_to_dense_tensor``)."""
    var out = create_dense_tensor[dtype](
        ctx, sparse.dim_logical.copy(), init_value=Scalar[dtype](0.0)
    )
    var idx_block = List[Int](capacity=sparse.ndim)
    for _ in range(sparse.ndim):
        idx_block.append(0)
    var idx_b = List[Int](capacity=sparse.ndim)
    var idx_t = List[Int](capacity=sparse.ndim)
    for _ in range(sparse.ndim):
        idx_b.append(0)
        idx_t.append(0)

    var host_out = ctx.enqueue_create_host_buffer[dtype](out.size)
    ctx.enqueue_copy(host_out, out.storage)
    ctx.synchronize()

    for k in range(sparse.nblocks_total):
        if not sparse.block_present[k]:
            continue
        _offset_to_tensor_index(k, sparse.ndim, sparse.dim_blocks, idx_block)
        var index_map = List[List[Int]](capacity=sparse.ndim)
        for i in range(sparse.ndim):
            var qbi = sparse.qnums_blocks[i][idx_block[i]]
            var mult = sparse.sector_counts[i][idx_block[i]]
            var mp = List[Int](capacity=mult)
            var c = 0
            for j in range(sparse.dim_logical[i]):
                if sparse.qnums_logical[i][j] == qbi:
                    mp.append(j)
                    c += 1
            index_map.append(mp^)

        var blk = sparse.blocks_flat[k]
        var nelem = blk.size
        var host_b = ctx.enqueue_create_host_buffer[dtype](nelem)
        ctx.enqueue_copy(host_b, blk.storage)
        ctx.synchronize()

        for j in range(nelem):
            _offset_to_tensor_index(j, sparse.ndim, blk.shape, idx_b)
            for i in range(sparse.ndim):
                idx_t[i] = index_map[i][idx_b[i]]
            var toff = _tensor_index_to_offset(sparse.ndim, out.shape, idx_t)
            host_out[toff] = host_b[j]

    ctx.enqueue_copy(out.storage, host_out)
    ctx.synchronize()
    _ = sparse^
    return out^
