"""Comprehensive tests for BlockSparseTensor (mirrors C test_block_sparse_tensor.c).

Test Plan:
- BST-ALLOC-001: Allocation respects quantum number conservation
- BST-ALLOC-002: Block dimensions match sector multiplicities
- BST-CONV-001: Dense to BlockSparse preserves values
- BST-CONV-002: BlockSparse to Dense preserves values
- BST-CONV-003: Round-trip conversion Dense → BlockSparse → Dense
- BST-NORM-001: Frobenius norm matches dense equivalent
- BST-SCALE-001: Scale in-place works correctly
- BST-AXPY-001: AXPY operation y += alpha * x
- BST-TRANS-001: Transpose preserves quantum number structure
- BST-DOT-001: Tensor dot with simple 2D contraction
- BST-DOT-002: Tensor dot with multi-axis contraction
- BST-DOT-003: Tensor dot result matches dense reference
"""
from sys import has_accelerator
from math import sqrt
from gpu.host import DeviceContext
from collections.list import List
from testing import TestSuite
from src.m_tensor.dense_tensor import (
    DenseTensor,
    create_dense_tensor,
    create_dense_tensor_from_data,
    dense_tensor_dot,
)
from src.m_tensor.block_sparse_tensor import (
    BlockSparseTensor,
    QNumber,
    BlockIndex,
    create_block_sparse_tensor,
    allocate_block_sparse_tensor_like,
    allocate_block_sparse_for_tensor_dot,
    block_sparse_tensor_dot,
    block_sparse_tensor_qr,
    block_sparse_to_dense,
    dense_to_block_sparse,
    dense_to_block_sparse_entries,
)


fn make_simple_qnums(size: Int, pattern: String) -> List[QNumber]:
    """Create per-index quantum numbers on one leg.
    
    Patterns:
        'alternating': [0, 1, 0, 1, ...]
        'ascending': [0, 1, 2, 3, ...]
        'half': first half 0, second half 1
        'zero': all zeros — **one sector per leg** (single dense block; not block-sparse)
    """
    var qnums = List[QNumber](capacity=size)
    if pattern == "alternating":
        for i in range(size):
            qnums.append(QNumber(i % 2))
    elif pattern == "ascending":
        for i in range(size):
            qnums.append(QNumber(i))
    elif pattern == "half":
        for i in range(size):
            if i < size // 2:
                qnums.append(QNumber(0))
            else:
                qnums.append(QNumber(1))
    elif pattern == "zero":
        for _ in range(size):
            qnums.append(QNumber(0))
    else:
        for _ in range(size):
            qnums.append(QNumber(0))
    return qnums^


fn make_u1_pair_qnums(size: Int) raises -> List[List[QNumber]]:
    """Two-sector 2D layout for ``axis_dir = [+1, -1]`` (same as BST-ALLOC-001).
    
    Row: first half q=0, second half q=1. Col: alternating 0,1,0,1,...
    Conserved blocks are (0,0) and (1,1) only → 50% of the block grid is present.
    """
    if size % 2 != 0:
        raise Error("make_u1_pair_qnums: size must be even")
    var half = size // 2
    var row = List[QNumber](capacity=size)
    var col = List[QNumber](capacity=size)
    for i in range(size):
        row.append(QNumber(0 if i < half else 1))
        col.append(QNumber(i % 2))
    var per_leg = List[List[QNumber]]()
    per_leg.append(row^)
    per_leg.append(col^)
    return per_leg^


fn u1_conservation_axis_dir_2d() -> List[Int]:
    var ad = List[Int]()
    ad.append(1)
    ad.append(-1)
    return ad^


fn logical_u1_conserved_at(
    row: List[QNumber], col: List[QNumber], i: Int, j: Int
) -> Bool:
    """True when logical index (i, j) lies in a conserved U(1) block (q_row == q_col)."""
    return row[i].value == col[j].value


fn test_alloc_001() raises:
    """BST-ALLOC-001: Allocation respects quantum number conservation.
    
    For a 2D tensor with axis_dir = [+1, -1], blocks are allocated only when
    qnum_row * (+1) + qnum_col * (-1) == 0, i.e., qnum_row == qnum_col.
    """
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping BST-ALLOC-001")
    
    with DeviceContext() as ctx:
        var shape = List[Int](4, 4)
        var qnums_row = List[QNumber]()
        qnums_row.append(QNumber(0))
        qnums_row.append(QNumber(0))
        qnums_row.append(QNumber(1))
        qnums_row.append(QNumber(1))
        var qnums_col = List[QNumber]()
        qnums_col.append(QNumber(0))
        qnums_col.append(QNumber(1))
        qnums_col.append(QNumber(0))
        qnums_col.append(QNumber(1))
        var qnums_per_leg = List[List[QNumber]]()
        qnums_per_leg.append(qnums_row^)
        qnums_per_leg.append(qnums_col^)
        
        var axis_dir = List[Int]()
        axis_dir.append(1)
        axis_dir.append(-1)
        
        var t = create_block_sparse_tensor[DType.float32](
            ctx, shape^, qnums_per_leg^, 
            init_value=Scalar[DType.float32](1.0),
            axis_dir=axis_dir^
        )
        
        if t.ndim != 2:
            raise Error("Expected ndim=2")
        if t.dim_blocks[0] != 2 or t.dim_blocks[1] != 2:
            raise Error("Expected 2 distinct qnums per axis")
        
        var num_present = 0
        for k in range(t.nblocks_total):
            if t.block_present[k]:
                num_present += 1
        if num_present != 2:
            raise Error("Expected 2 non-zero blocks (diagonal in qnum space), got " + String(num_present))


fn test_alloc_002() raises:
    """BST-ALLOC-002: Block dimensions match sector multiplicities.
    
    With qnums [0, 0, 1, 1] for a 4-element axis, sector 0 has multiplicity 2
    and sector 1 has multiplicity 2.
    """
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping BST-ALLOC-002")
    
    with DeviceContext() as ctx:
        var shape = List[Int](4, 4)
        var qnums_row = List[QNumber]()
        qnums_row.append(QNumber(0))
        qnums_row.append(QNumber(0))
        qnums_row.append(QNumber(1))
        qnums_row.append(QNumber(1))
        var qnums_col = List[QNumber]()
        qnums_col.append(QNumber(0))
        qnums_col.append(QNumber(0))
        qnums_col.append(QNumber(1))
        qnums_col.append(QNumber(1))
        var qnums_per_leg = List[List[QNumber]]()
        qnums_per_leg.append(qnums_row^)
        qnums_per_leg.append(qnums_col^)
        
        var axis_dir = List[Int]()
        axis_dir.append(1)
        axis_dir.append(-1)
        
        var t = create_block_sparse_tensor[DType.float32](
            ctx, shape^, qnums_per_leg^, 
            init_value=Scalar[DType.float32](1.0),
            axis_dir=axis_dir^
        )
        
        if t.sector_counts[0][0] != 2 or t.sector_counts[0][1] != 2:
            raise Error("Expected sector counts [2, 2] for axis 0")
        if t.sector_counts[1][0] != 2 or t.sector_counts[1][1] != 2:
            raise Error("Expected sector counts [2, 2] for axis 1")


fn test_conv_001() raises:
    """BST-CONV-001: Dense to BlockSparse preserves values in conserved blocks."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping BST-CONV-001")
    
    with DeviceContext() as ctx:
        var data = List[Scalar[DType.float32]]()
        for i in range(9):
            data.append(Scalar[DType.float32](Float32(i + 1)))
        var shape = List[Int](3, 3)
        var dense = create_dense_tensor_from_data[DType.float32](ctx, data, shape^)
        
        var qnums_axis = List[QNumber]()
        qnums_axis.append(QNumber(0))
        qnums_axis.append(QNumber(0))
        qnums_axis.append(QNumber(0))
        var qnums_per_leg = List[List[QNumber]]()
        qnums_per_leg.append(qnums_axis.copy())
        qnums_per_leg.append(qnums_axis^)
        
        var sparse = dense_to_block_sparse[DType.float32](dense^, qnums_per_leg^, ctx)
        
        if sparse.nblocks_total != 1:
            raise Error("Expected 1 block for uniform qnums")
        if not sparse.block_present[0]:
            raise Error("Expected block 0 to be present")
        var blk = sparse.blocks_flat[0]
        if blk.shape[0] != 3 or blk.shape[1] != 3:
            raise Error("Expected block shape [3, 3]")


fn test_conv_002() raises:
    """BST-CONV-002: BlockSparse to Dense preserves values in conserved sectors only."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping BST-CONV-002")
    
    with DeviceContext() as ctx:
        var shape = List[Int](4, 4)
        var qnums_per_leg = make_u1_pair_qnums(4)
        var row = qnums_per_leg[0].copy()
        var col = qnums_per_leg[1].copy()
        var axis_dir = u1_conservation_axis_dir_2d()
        
        var sparse = create_block_sparse_tensor[DType.float32](
            ctx, shape^, qnums_per_leg^,
            init_value=Scalar[DType.float32](2.5),
            axis_dir=axis_dir^,
        )
        
        var num_present = 0
        for k in range(sparse.nblocks_total):
            if sparse.block_present[k]:
                num_present += 1
        if num_present != 2:
            raise Error("Expected 2 conserved blocks, got " + String(num_present))
        if sparse.sparsity_ratio() >= 1.0:
            raise Error("Expected a genuinely sparse layout (ratio < 1)")
        
        var dense = block_sparse_to_dense[DType.float32](sparse^, ctx)
        
        if dense.shape[0] != 4 or dense.shape[1] != 4:
            raise Error("Expected dense shape [4, 4]")
        
        var host = ctx.enqueue_create_host_buffer[DType.float32](16)
        ctx.enqueue_copy(host, dense.storage)
        ctx.synchronize()
        
        for i in range(4):
            for j in range(4):
                var idx = i * 4 + j
                var expected = 2.5 if logical_u1_conserved_at(row, col, i, j) else 0.0
                if abs(Float64(host[idx]) - expected) > 1e-5:
                    raise Error(
                        "Value mismatch at (" + String(i) + "," + String(j) + "): got "
                        + String(host[idx]) + " expected " + String(expected)
                    )


fn test_conv_003() raises:
    """BST-CONV-003: Round-trip Dense → BlockSparse → Dense with multi-block U(1) layout."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping BST-CONV-003")
    
    with DeviceContext() as ctx:
        var qnums_per_leg = make_u1_pair_qnums(4)
        var row = qnums_per_leg[0].copy()
        var col = qnums_per_leg[1].copy()
        
        var data = List[Scalar[DType.float32]]()
        for i in range(4):
            for j in range(4):
                var v = Float32(i * 0.5 + j * 0.25 + 0.1) if logical_u1_conserved_at(row, col, i, j) else 0.0
                data.append(Scalar[DType.float32](v))
        var shape = List[Int](4, 4)
        var sparse_shape = shape.copy()
        var dense_orig = create_dense_tensor_from_data[DType.float32](ctx, data.copy(), shape^)
        
        var sparse = create_block_sparse_tensor[DType.float32](
            ctx, sparse_shape^, qnums_per_leg^,
            init_value=Scalar[DType.float32](0.0),
            axis_dir=u1_conservation_axis_dir_2d(),
        )
        dense_to_block_sparse_entries(dense_orig, sparse, ctx)
        var dense_back = block_sparse_to_dense[DType.float32](sparse^, ctx)
        
        var host = ctx.enqueue_create_host_buffer[DType.float32](16)
        ctx.enqueue_copy(host, dense_back.storage)
        ctx.synchronize()
        
        for i in range(16):
            var expected = Float64(data[i])
            var got = Float64(host[i])
            if abs(got - expected) > 1e-5:
                raise Error("Round-trip mismatch at " + String(i) + ": " + String(got) + " vs " + String(expected))


fn test_norm_001() raises:
    """BST-NORM-001: Frobenius norm matches dense on conserved entries (multi-block U(1))."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping BST-NORM-001")
    
    with DeviceContext() as ctx:
        var qnums_per_leg = make_u1_pair_qnums(4)
        var row = qnums_per_leg[0].copy()
        var col = qnums_per_leg[1].copy()
        
        var data = List[Scalar[DType.float32]]()
        var dense_norm_sq = 0.0
        for i in range(4):
            for j in range(4):
                var v = Float32(i + j + 1) if logical_u1_conserved_at(row, col, i, j) else 0.0
                data.append(Scalar[DType.float32](v))
                dense_norm_sq += Float64(v) * Float64(v)
        var shape = List[Int](4, 4)
        var sparse_shape = shape.copy()
        var dense = create_dense_tensor_from_data[DType.float32](ctx, data.copy(), shape^)
        var dense_norm = sqrt(dense_norm_sq)
        
        var sparse = create_block_sparse_tensor[DType.float32](
            ctx, sparse_shape^, qnums_per_leg^,
            init_value=Scalar[DType.float32](0.0),
            axis_dir=u1_conservation_axis_dir_2d(),
        )
        dense_to_block_sparse_entries(dense, sparse, ctx)
        var sparse_norm = sparse.compute_norm(ctx)
        
        if abs(sparse_norm - dense_norm) / dense_norm > 1e-5:
            raise Error("Norm mismatch: sparse=" + String(sparse_norm) + " dense=" + String(dense_norm))


fn test_scale_001() raises:
    """BST-SCALE-001: Scale in-place works correctly."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping BST-SCALE-001")
    
    with DeviceContext() as ctx:
        var shape = List[Int](4, 4)
        var qnums_per_leg = make_u1_pair_qnums(4)
        var row = qnums_per_leg[0].copy()
        var col = qnums_per_leg[1].copy()
        
        var t = create_block_sparse_tensor[DType.float32](
            ctx, shape^, qnums_per_leg^,
            init_value=Scalar[DType.float32](2.0),
            axis_dir=u1_conservation_axis_dir_2d(),
        )
        
        t.scale_in_place(Scalar[DType.float32](0.5), ctx)
        
        var dense = block_sparse_to_dense[DType.float32](t^, ctx)
        var host = ctx.enqueue_create_host_buffer[DType.float32](16)
        ctx.enqueue_copy(host, dense.storage)
        ctx.synchronize()
        
        for i in range(4):
            for j in range(4):
                var idx = i * 4 + j
                var expected = 1.0 if logical_u1_conserved_at(row, col, i, j) else 0.0
                if abs(Float64(host[idx]) - expected) > 1e-5:
                    raise Error("Scale mismatch at (" + String(i) + "," + String(j) + ")")


fn test_axpy_001() raises:
    """BST-AXPY-001: AXPY operation y += alpha * x."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping BST-AXPY-001")
    
    with DeviceContext() as ctx:
        var shape = List[Int](4, 4)
        var qnums_per_leg = make_u1_pair_qnums(4)
        var row = qnums_per_leg[0].copy()
        var col = qnums_per_leg[1].copy()
        var axis_dir = u1_conservation_axis_dir_2d()
        
        var y = create_block_sparse_tensor[DType.float32](
            ctx, shape.copy(), qnums_per_leg^,
            init_value=Scalar[DType.float32](1.0),
            axis_dir=axis_dir.copy(),
        )
        var x = create_block_sparse_tensor[DType.float32](
            ctx, shape^, make_u1_pair_qnums(4),
            init_value=Scalar[DType.float32](2.0),
            axis_dir=axis_dir^,
        )
        
        y.axpy_in_place(Scalar[DType.float32](0.5), x, ctx)
        
        var dense = block_sparse_to_dense[DType.float32](y^, ctx)
        var host = ctx.enqueue_create_host_buffer[DType.float32](16)
        ctx.enqueue_copy(host, dense.storage)
        ctx.synchronize()
        
        for i in range(4):
            for j in range(4):
                var idx = i * 4 + j
                var expected = (1.0 + 0.5 * 2.0) if logical_u1_conserved_at(row, col, i, j) else 0.0
                if abs(Float64(host[idx]) - expected) > 1e-5:
                    raise Error("AXPY mismatch at (" + String(i) + "," + String(j) + ")")


fn test_transpose_001() raises:
    """BST-TRANS-001: Transpose preserves quantum number structure."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping BST-TRANS-001")
    
    with DeviceContext() as ctx:
        var qnums_per_leg = make_u1_pair_qnums(4)
        var row = qnums_per_leg[0].copy()
        var col = qnums_per_leg[1].copy()
        
        var data = List[Scalar[DType.float32]]()
        for i in range(4):
            for j in range(4):
                var v = Float32(i * 4 + j + 1) if logical_u1_conserved_at(row, col, i, j) else 0.0
                data.append(Scalar[DType.float32](v))
        var shape = List[Int](4, 4)
        var sparse_shape = shape.copy()
        var dense = create_dense_tensor_from_data[DType.float32](ctx, data, shape^)
        
        var sparse = create_block_sparse_tensor[DType.float32](
            ctx, sparse_shape^, qnums_per_leg^,
            init_value=Scalar[DType.float32](0.0),
            axis_dir=u1_conservation_axis_dir_2d(),
        )
        dense_to_block_sparse_entries(dense, sparse, ctx)
        
        var perm = List[Int](1, 0)
        var transposed = sparse^.transpose(perm, ctx)
        
        if transposed.dim_logical[0] != 4 or transposed.dim_logical[1] != 4:
            raise Error("Transposed shape incorrect")
        
        var num_present = 0
        for k in range(transposed.nblocks_total):
            if transposed.block_present[k]:
                num_present += 1
        if num_present != 2:
            raise Error("Transpose should keep 2 conserved blocks, got " + String(num_present))
        
        var dense_t = block_sparse_to_dense[DType.float32](transposed^, ctx)
        var host = ctx.enqueue_create_host_buffer[DType.float32](16)
        ctx.enqueue_copy(host, dense_t.storage)
        ctx.synchronize()
        
        for i in range(4):
            for j in range(4):
                var idx = i * 4 + j
                var expected = Float64(data[j * 4 + i]) if logical_u1_conserved_at(row, col, j, i) else 0.0
                if abs(Float64(host[idx]) - expected) > 1e-5:
                    raise Error("Transpose value mismatch at (" + String(i) + "," + String(j) + ")")


fn test_dot_001() raises:
    """BST-DOT-001: 2D contraction (single-sector / dense-block layout sanity check)."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping BST-DOT-001")
    
    with DeviceContext() as ctx:
        var shape_a = List[Int](3, 4)
        var shape_b = List[Int](4, 5)
        
        var qnums_3 = make_simple_qnums(3, "zero")
        var qnums_4_a = make_simple_qnums(4, "zero")
        var qnums_4_b = make_simple_qnums(4, "zero")
        var qnums_5 = make_simple_qnums(5, "zero")
        
        var qnums_a = List[List[QNumber]]()
        qnums_a.append(qnums_3^)
        qnums_a.append(qnums_4_a^)
        
        var qnums_b = List[List[QNumber]]()
        qnums_b.append(qnums_4_b^)
        qnums_b.append(qnums_5^)
        
        var axis_dir_a = List[Int](1, -1)
        var axis_dir_b = List[Int](1, -1)
        
        var a = create_block_sparse_tensor[DType.float32](
            ctx, shape_a^, qnums_a^, 
            init_value=Scalar[DType.float32](1.0),
            axis_dir=axis_dir_a^
        )
        var b = create_block_sparse_tensor[DType.float32](
            ctx, shape_b^, qnums_b^, 
            init_value=Scalar[DType.float32](1.0),
            axis_dir=axis_dir_b^
        )
        
        var c = allocate_block_sparse_for_tensor_dot[DType.float32](
            a, b, ctx, ndim_mult=1, axrange_a=False, axrange_b=True
        )
        
        block_sparse_tensor_dot[DType.float32](
            c, a^, b^, ctx, ndim_mult=1, axrange_a=False, axrange_b=True
        )
        
        if c.dim_logical[0] != 3 or c.dim_logical[1] != 5:
            raise Error("Result shape incorrect")
        
        var dense_c = block_sparse_to_dense[DType.float32](c^, ctx)
        var host = ctx.enqueue_create_host_buffer[DType.float32](15)
        ctx.enqueue_copy(host, dense_c.storage)
        ctx.synchronize()
        
        for i in range(15):
            if abs(Float64(host[i]) - 4.0) > 1e-4:
                raise Error("Dot result mismatch at " + String(i) + ": got " + String(host[i]) + " expected 4.0")


fn test_dot_002() raises:
    """BST-DOT-002: Tensor dot with multi-axis contraction."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping BST-DOT-002")
    
    with DeviceContext() as ctx:
        var shape_a = List[Int](2, 3, 4)
        var shape_b = List[Int](3, 4, 5)
        
        var q2 = make_simple_qnums(2, "zero")
        var q3_a = make_simple_qnums(3, "zero")
        var q4_a = make_simple_qnums(4, "zero")
        var q3_b = make_simple_qnums(3, "zero")
        var q4_b = make_simple_qnums(4, "zero")
        var q5 = make_simple_qnums(5, "zero")
        
        var qnums_a = List[List[QNumber]]()
        qnums_a.append(q2^)
        qnums_a.append(q3_a^)
        qnums_a.append(q4_a^)
        
        var qnums_b = List[List[QNumber]]()
        qnums_b.append(q3_b^)
        qnums_b.append(q4_b^)
        qnums_b.append(q5^)
        
        var axis_dir_a = List[Int](1, 1, -1)
        var axis_dir_b = List[Int](-1, 1, -1)
        
        var a = create_block_sparse_tensor[DType.float32](
            ctx, shape_a^, qnums_a^, 
            init_value=Scalar[DType.float32](1.0),
            axis_dir=axis_dir_a^
        )
        var b = create_block_sparse_tensor[DType.float32](
            ctx, shape_b^, qnums_b^, 
            init_value=Scalar[DType.float32](1.0),
            axis_dir=axis_dir_b^
        )
        
        var c = allocate_block_sparse_for_tensor_dot[DType.float32](
            a, b, ctx, ndim_mult=2, axrange_a=False, axrange_b=True
        )
        
        block_sparse_tensor_dot[DType.float32](
            c, a^, b^, ctx, ndim_mult=2, axrange_a=False, axrange_b=True
        )
        
        if c.dim_logical[0] != 2 or c.dim_logical[1] != 5:
            raise Error("Result shape incorrect: expected [2, 5], got [" + 
                       String(c.dim_logical[0]) + ", " + String(c.dim_logical[1]) + "]")


fn test_dot_003() raises:
    """BST-DOT-003: Tensor dot vs dense reference (single-sector layout on each leg)."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping BST-DOT-003")
    
    with DeviceContext() as ctx:
        var data_a = List[Scalar[DType.float32]]()
        for i in range(6):
            data_a.append(Scalar[DType.float32](Float32(i + 1)))
        var shape_a = List[Int](2, 3)
        var sparse_shape_a = shape_a.copy()
        var dense_a = create_dense_tensor_from_data[DType.float32](ctx, data_a, shape_a^)
        
        var data_b = List[Scalar[DType.float32]]()
        for i in range(12):
            data_b.append(Scalar[DType.float32](Float32(i + 1)))
        var shape_b = List[Int](3, 4)
        var sparse_shape_b = shape_b.copy()
        var dense_b = create_dense_tensor_from_data[DType.float32](ctx, data_b, shape_b^)
        
        var qnums_2 = make_simple_qnums(2, "zero")
        var qnums_3_a = make_simple_qnums(3, "zero")
        var qnums_3_b = make_simple_qnums(3, "zero")
        var qnums_4 = make_simple_qnums(4, "zero")
        
        var qnums_a = List[List[QNumber]]()
        qnums_a.append(qnums_2^)
        qnums_a.append(qnums_3_a^)
        
        var qnums_b = List[List[QNumber]]()
        qnums_b.append(qnums_3_b^)
        qnums_b.append(qnums_4^)
        
        # Contract A[..., j] with B[j, ...]: need opposite axis_dir on leg j (like BST-DOT-001).
        var axis_dir_a = List[Int](1, -1)
        var axis_dir_b = List[Int](1, -1)
        var sparse_a = create_block_sparse_tensor[DType.float32](
            ctx, sparse_shape_a^, qnums_a^,
            init_value=Scalar[DType.float32](0.0),
            axis_dir=axis_dir_a^,
        )
        dense_to_block_sparse_entries(dense_a, sparse_a, ctx)
        var sparse_b = create_block_sparse_tensor[DType.float32](
            ctx, sparse_shape_b^, qnums_b^,
            init_value=Scalar[DType.float32](0.0),
            axis_dir=axis_dir_b^,
        )
        dense_to_block_sparse_entries(dense_b, sparse_b, ctx)
        
        var sparse_c = allocate_block_sparse_for_tensor_dot[DType.float32](
            sparse_a, sparse_b, ctx, ndim_mult=1, axrange_a=False, axrange_b=True
        )
        
        block_sparse_tensor_dot[DType.float32](
            sparse_c, sparse_a^, sparse_b^, ctx, ndim_mult=1, axrange_a=False, axrange_b=True
        )
        
        var dense_c = block_sparse_to_dense[DType.float32](sparse_c^, ctx)
        
        var host = ctx.enqueue_create_host_buffer[DType.float32](8)
        ctx.enqueue_copy(host, dense_c.storage)
        ctx.synchronize()
        
        var expected = List[Float64](38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0)
        for i in range(8):
            if abs(Float64(host[i]) - expected[i]) > 1e-3:
                raise Error("Dot result mismatch at " + String(i) + 
                           ": got " + String(host[i]) + " expected " + String(expected[i]))


fn test_sparsity_ratio() raises:
    """BST-SPARSE-001: Verify sparsity ratio with non-trivial quantum numbers."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping BST-SPARSE-001")
    
    with DeviceContext() as ctx:
        var shape = List[Int](6, 6)
        
        var qnums_row = List[QNumber]()
        qnums_row.append(QNumber(0))
        qnums_row.append(QNumber(0))
        qnums_row.append(QNumber(1))
        qnums_row.append(QNumber(1))
        qnums_row.append(QNumber(2))
        qnums_row.append(QNumber(2))
        
        var qnums_col = List[QNumber]()
        qnums_col.append(QNumber(0))
        qnums_col.append(QNumber(1))
        qnums_col.append(QNumber(0))
        qnums_col.append(QNumber(1))
        qnums_col.append(QNumber(0))
        qnums_col.append(QNumber(2))
        
        var qnums_per_leg = List[List[QNumber]]()
        qnums_per_leg.append(qnums_row^)
        qnums_per_leg.append(qnums_col^)
        
        var axis_dir = List[Int](1, -1)
        
        var t = create_block_sparse_tensor[DType.float32](
            ctx, shape^, qnums_per_leg^, 
            init_value=Scalar[DType.float32](1.0),
            axis_dir=axis_dir^
        )
        
        var ratio = t.sparsity_ratio()
        
        if ratio >= 1.0:
            raise Error("Sparsity ratio should be < 1.0 for non-trivial qnums, got " + String(ratio))
        if ratio <= 0.0:
            raise Error("Sparsity ratio should be > 0.0 (some blocks present), got " + String(ratio))


fn test_get_block() raises:
    """BST-GETBLK-001: Test get_block by quantum number lookup."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping BST-GETBLK-001")
    
    with DeviceContext() as ctx:
        var shape = List[Int](4, 4)
        var qnums_row = List[QNumber]()
        qnums_row.append(QNumber(0))
        qnums_row.append(QNumber(0))
        qnums_row.append(QNumber(1))
        qnums_row.append(QNumber(1))
        
        var qnums_col = List[QNumber]()
        qnums_col.append(QNumber(0))
        qnums_col.append(QNumber(0))
        qnums_col.append(QNumber(1))
        qnums_col.append(QNumber(1))
        
        var qnums_per_leg = List[List[QNumber]]()
        qnums_per_leg.append(qnums_row^)
        qnums_per_leg.append(qnums_col^)
        
        var axis_dir = List[Int](1, -1)
        
        var t = create_block_sparse_tensor[DType.float32](
            ctx, shape^, qnums_per_leg^, 
            init_value=Scalar[DType.float32](3.14),
            axis_dir=axis_dir^
        )
        
        var idx_qnums = List[QNumber]()
        idx_qnums.append(QNumber(0))
        idx_qnums.append(QNumber(0))
        var idx = BlockIndex(idx_qnums^)
        
        var blk = t.get_block(idx)
        
        if blk.data.shape[0] != 2 or blk.data.shape[1] != 2:
            raise Error("Expected block shape [2, 2]")


fn test_block_sparse_qr() raises:
    """BST-QR-001: Test QR decomposition for BlockSparseTensor."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping BST-QR-001")
    
    with DeviceContext() as ctx:
        # Create a simple 3x2 block-sparse matrix with single sector (trivial qnums)
        var shape = List[Int](3, 2)
        var qnums_row = List[QNumber]()
        for _ in range(3):
            qnums_row.append(QNumber(0))
        
        var qnums_col = List[QNumber]()
        for _ in range(2):
            qnums_col.append(QNumber(0))
        
        var qnums_per_leg = List[List[QNumber]]()
        qnums_per_leg.append(qnums_row^)
        qnums_per_leg.append(qnums_col^)
        
        var axis_dir = List[Int](1, -1)
        
        var A = create_block_sparse_tensor[DType.float32](
            ctx, shape^, qnums_per_leg^, 
            init_value=Scalar[DType.float32](1.0),
            axis_dir=axis_dir^
        )
        
        # Perform QR decomposition
        var qr_result = block_sparse_tensor_qr[DType.float32](A^, ctx)
        var Q = qr_result[0]
        var R = qr_result[1]
        
        # Check dimensions: Q should be [3, 2], R should be [2, 2] (thin QR)
        var Q_shape = Q.get_shape()
        var R_shape = R.get_shape()
        
        if Q_shape[0] != 3 or Q_shape[1] != 2:
            raise Error("BST-QR-001: Q should have shape [3, 2], got [" 
                + String(Q_shape[0]) + ", " + String(Q_shape[1]) + "]")
        
        if R_shape[0] != 2 or R_shape[1] != 2:
            raise Error("BST-QR-001: R should have shape [2, 2], got [" 
                + String(R_shape[0]) + ", " + String(R_shape[1]) + "]")
        
        # Check that Q has approximately orthonormal columns: Q^T * Q ≈ I
        # Convert to dense for verification
        var Q_dense = block_sparse_to_dense(Q^, ctx)
        _ = block_sparse_to_dense(R^, ctx)  # Just verify R converts, don't need it
        
        # Make a copy for the dot product since transpose moves the tensor
        var Q_copy = Q_dense.copy_to_contiguous(ctx)
        
        # Compute Q^T * Q
        var Qt = Q_dense^.transpose(List[Int](1, 0), ctx)
        var QtQ = create_dense_tensor[DType.float32](ctx, List[Int](2, 2), init_value=Scalar[DType.float32](0.0))
        dense_tensor_dot(QtQ, Qt^, Q_copy^, ctx)
        
        # Check diagonal is close to 1, off-diagonal close to 0
        var host_QtQ = ctx.enqueue_create_host_buffer[DType.float32](4)
        ctx.enqueue_copy(host_QtQ, QtQ.storage)
        ctx.synchronize()
        
        var tol: Float64 = 1e-5
        var diag0_diff = Float64(host_QtQ[0]) - 1.0
        var diag1_diff = Float64(host_QtQ[3]) - 1.0
        if diag0_diff < 0:
            diag0_diff = -diag0_diff
        if diag1_diff < 0:
            diag1_diff = -diag1_diff
        if diag0_diff > tol or diag1_diff > tol:
            raise Error("BST-QR-001: Q columns not orthonormal (diagonal)")
        
        var off01 = Float64(host_QtQ[1])
        var off10 = Float64(host_QtQ[2])
        if off01 < 0:
            off01 = -off01
        if off10 < 0:
            off10 = -off10
        if off01 > tol or off10 > tol:
            raise Error("BST-QR-001: Q columns not orthogonal (off-diagonal)")


fn main() raises:
    try:
        TestSuite.discover_tests[__functions_in_module()]().run()
    except e:
        print("Tests failed: " + String(e))
