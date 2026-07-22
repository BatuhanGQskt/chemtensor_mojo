"""Example: Block Sparse Tensor Contractions

This example demonstrates realistic tensor network contractions using BlockSparseTensor
with quantum number conservation. This is the core operation in DMRG and other
tensor network algorithms.

Key Concepts:
1. Quantum numbers enforce selection rules (e.g., particle number conservation)
2. Only blocks where quantum numbers sum to zero are non-zero
3. Contractions only involve matching quantum number sectors
4. This provides automatic sparsity with no numerical truncation error

Scenario: MPS-style tensor network
- MPS tensor A: [χ_L, d, χ_R] where χ are virtual bond dimensions, d is physical
- MPO tensor W: [χ'_L, d_out, d_in, χ'_R]
- We demonstrate contraction of A with W to produce an updated MPS tensor
"""

from sys import has_accelerator
from gpu.host import DeviceContext
from collections.list import List
from math import sqrt

from src.m_tensor.dense_tensor import (
    DenseTensor,
    create_dense_tensor,
)
from src.m_tensor.block_sparse_tensor import (
    BlockSparseTensor,
    QNumber,
    BlockIndex,
    create_block_sparse_tensor,
    allocate_block_sparse_for_tensor_dot,
    block_sparse_tensor_dot,
    block_sparse_to_dense,
)


fn print_separator(title: String):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


fn print_tensor_info[dtype: DType](name: String, t: BlockSparseTensor[dtype]):
    print("\n", name, ":")
    for dim_i_logical in t.dim_logical:
        print("  ", dim_i_logical)
    for dim_i_blocks in t.dim_blocks:
        print("  ", dim_i_blocks)
    for axis_dir_i in t.axis_dir:
        print("  ", axis_dir_i)
    print("  Total blocks:", t.nblocks_total)
    var present = 0
    for k in range(t.nblocks_total):
        if t.block_present[k]:
            present += 1
    print("  Non-zero blocks:", present)
    print("  Block dimensions:", t.dim_blocks)
    print("  Axis directions:", t.axis_dir)
    print("  Total blocks:", t.nblocks_total)
    var present = 0
    for k in range(t.nblocks_total):
        if t.block_present[k]:
            present += 1
    print("  Non-zero blocks:", present)
    print("  Sparsity ratio:", t.sparsity_ratio())


fn example_1_simple_matrix_contraction(ctx: DeviceContext) raises:
    """Simple 2D matrix contraction with quantum numbers.
    
    Demonstrates how quantum number conservation affects which blocks contribute
    to the contraction.
    """
    print_separator("Example 1: Simple Matrix Contraction with Quantum Numbers")
    
    print("\nSetup:")
    print("  Matrix A[4,4]: rows have qnums [0,0,1,1], cols have qnums [0,0,1,1]")
    print("  Matrix B[4,4]: same structure")
    print("  With axis_dir A=[+1,-1], B=[+1,-1]:")
    print("  A has blocks (0,0) and (1,1) where row_qnum == col_qnum")
    print("  B has blocks (0,0) and (1,1) similarly")
    
    var shape = List[Int](4, 4)
    
    var qnums_a_row = List[QNumber]()
    qnums_a_row.append(QNumber(0))
    qnums_a_row.append(QNumber(0))
    qnums_a_row.append(QNumber(1))
    qnums_a_row.append(QNumber(1))
    
    var qnums_a_col = List[QNumber]()
    qnums_a_col.append(QNumber(0))
    qnums_a_col.append(QNumber(0))
    qnums_a_col.append(QNumber(1))
    qnums_a_col.append(QNumber(1))
    
    var qnums_b_row = List[QNumber]()
    qnums_b_row.append(QNumber(0))
    qnums_b_row.append(QNumber(0))
    qnums_b_row.append(QNumber(1))
    qnums_b_row.append(QNumber(1))
    
    var qnums_b_col = List[QNumber]()
    qnums_b_col.append(QNumber(0))
    qnums_b_col.append(QNumber(0))
    qnums_b_col.append(QNumber(1))
    qnums_b_col.append(QNumber(1))
    
    var qnums_a = List[List[QNumber]]()
    qnums_a.append(qnums_a_row^)
    qnums_a.append(qnums_a_col^)
    
    var qnums_b = List[List[QNumber]]()
    qnums_b.append(qnums_b_row^)
    qnums_b.append(qnums_b_col^)
    
    var axis_dir_a = List[Int](1, -1)
    var axis_dir_b = List[Int](1, -1)
    
    var a = create_block_sparse_tensor[DType.float32](
        ctx, shape.copy(), qnums_a^, 
        init_value=Scalar[DType.float32](1.0),
        axis_dir=axis_dir_a
    )
    var b = create_block_sparse_tensor[DType.float32](
        ctx, shape^, qnums_b^, 
        init_value=Scalar[DType.float32](1.0),
        axis_dir=axis_dir_b
    )
    
    print_tensor_info("A", a)
    print_tensor_info("B", b)
    
    var c = allocate_block_sparse_for_tensor_dot[DType.float32](
        a, b, ctx, ndim_mult=1, axrange_a=False, axrange_b=True
    )
    
    block_sparse_tensor_dot[DType.float32](
        c, a^, b^, ctx, ndim_mult=1, axrange_a=False, axrange_b=True
    )
    
    print_tensor_info("C = A @ B", c)
    
    print("\nResult analysis:")
    print("  C[4,4] has the same block structure as A and B")
    print("  Block (0,0): 2x2, all elements = 2 (sum of 2 products)")
    print("  Block (1,1): 2x2, all elements = 2 (sum of 2 products)")
    
    var dense_c = block_sparse_to_dense[DType.float32](c^, ctx)
    var host = ctx.enqueue_create_host_buffer[DType.float32](16)
    ctx.enqueue_copy(host, dense_c.storage)
    ctx.synchronize()
    
    print("\nDense representation of C:")
    for i in range(4):
        var row = String("  [")
        for j in range(4):
            row += String(host[i * 4 + j])
            if j < 3:
                row += ", "
        row += "]"
        print(row)


fn example_2_three_particle_sectors(ctx: DeviceContext) raises:
    """Contraction with three quantum number sectors (particle numbers 0, 1, 2).
    
    This simulates a physical scenario where we have states with 0, 1, or 2 particles.
    """
    print_separator("Example 2: Three Particle Sectors (N=0, 1, 2)")
    
    print("\nSetup:")
    print("  We model a system where each index can have particle number 0, 1, or 2")
    print("  A[6,6]: qnums = [0,0, 1,1, 2,2] for both axes")
    print("  With axis_dir = [+1, -1], blocks exist where Δparticle = 0")
    
    var shape = List[Int](6, 6)
    
    var qnums = List[QNumber]()
    qnums.append(QNumber(0))
    qnums.append(QNumber(0))
    qnums.append(QNumber(1))
    qnums.append(QNumber(1))
    qnums.append(QNumber(2))
    qnums.append(QNumber(2))
    
    var qnums_a = List[List[QNumber]]()
    qnums_a.append(qnums.copy())
    qnums_a.append(qnums.copy())
    
    var qnums_b = List[List[QNumber]]()
    qnums_b.append(qnums.copy())
    qnums_b.append(qnums^)
    
    var axis_dir = List[Int](1, -1)
    
    var a = create_block_sparse_tensor[DType.float32](
        ctx, shape.copy(), qnums_a^, 
        init_value=Scalar[DType.float32](1.0),
        axis_dir=axis_dir.copy()
    )
    var b = create_block_sparse_tensor[DType.float32](
        ctx, shape^, qnums_b^, 
        init_value=Scalar[DType.float32](1.0),
        axis_dir=axis_dir
    )
    
    print_tensor_info("A", a)
    print_tensor_info("B", b)
    
    print("\nBlock structure analysis:")
    print("  dim_blocks = [3, 3] (three sectors: 0, 1, 2)")
    print("  Non-zero blocks: (0,0), (1,1), (2,2) - diagonal in qnum space")
    print("  Each block is 2x2 (two indices per sector)")
    
    var c = allocate_block_sparse_for_tensor_dot[DType.float32](
        a, b, ctx, ndim_mult=1, axrange_a=False, axrange_b=True
    )
    
    block_sparse_tensor_dot[DType.float32](
        c, a^, b^, ctx, ndim_mult=1, axrange_a=False, axrange_b=True
    )
    
    print_tensor_info("C = A @ B", c)
    
    var dense_c = block_sparse_to_dense[DType.float32](c^, ctx)
    print("\nC norm:", dense_c.norm(ctx))


fn example_3_mps_mpo_style_contraction(ctx: DeviceContext) raises:
    """MPS-MPO style contraction simulating DMRG tensor operations.
    
    MPS tensor: A[χ_L, d, χ_R] with shape [4, 2, 4]
    - χ_L, χ_R: bond dimensions with particle number qnums
    - d: physical dimension (spin up/down)
    
    We contract two MPS tensors to simulate a basic MPS-MPS overlap operation.
    """
    print_separator("Example 3: MPS-Style Tensor Contraction")
    
    print("\nSetup: MPS tensor A[χ_L, d, χ_R]")
    print("  χ_L, χ_R = 4 (bond dimension, qnums [0,0,1,1])")
    print("  d = 2 (physical dimension, spin up=0, spin down=1)")
    
    var shape_a = List[Int](4, 2, 4)
    var shape_b = List[Int](4, 2, 4)
    
    var qnums_bond = List[QNumber]()
    qnums_bond.append(QNumber(0))
    qnums_bond.append(QNumber(0))
    qnums_bond.append(QNumber(1))
    qnums_bond.append(QNumber(1))
    
    var qnums_phys = List[QNumber]()
    qnums_phys.append(QNumber(0))
    qnums_phys.append(QNumber(1))
    
    var qnums_a = List[List[QNumber]]()
    qnums_a.append(qnums_bond.copy())
    qnums_a.append(qnums_phys.copy())
    qnums_a.append(qnums_bond.copy())
    
    var qnums_b = List[List[QNumber]]()
    qnums_b.append(qnums_bond.copy())
    qnums_b.append(qnums_phys^)
    qnums_b.append(qnums_bond^)
    
    var axis_dir_a = List[Int](1, 1, -1)
    var axis_dir_b = List[Int](-1, -1, 1)
    
    var a = create_block_sparse_tensor[DType.float32](
        ctx, shape_a^, qnums_a^, 
        init_value=Scalar[DType.float32](0.5),
        axis_dir=axis_dir_a
    )
    var b = create_block_sparse_tensor[DType.float32](
        ctx, shape_b^, qnums_b^, 
        init_value=Scalar[DType.float32](0.5),
        axis_dir=axis_dir_b
    )
    
    print_tensor_info("MPS tensor A", a)
    print_tensor_info("MPS tensor B (conjugate)", b)
    
    print("\nContracting over right bond of A with left bond of B")
    print("  A[χ_L, d, χ_R] × B[χ_R, d', χ'_R] → C[χ_L, d, d', χ'_R]")
    
    var c = allocate_block_sparse_for_tensor_dot[DType.float32](
        a, b, ctx, ndim_mult=1, axrange_a=False, axrange_b=True
    )
    
    block_sparse_tensor_dot[DType.float32](
        c, a^, b^, ctx, ndim_mult=1, axrange_a=False, axrange_b=True
    )
    
    print_tensor_info("Result C[χ_L, d, d', χ'_R]", c)
    
    var dense_c = block_sparse_to_dense[DType.float32](c^, ctx)
    print("\nResult norm:", dense_c.norm(ctx))
    print("Sparsity provides automatic selection rule enforcement!")


fn example_4_higher_rank_contraction(ctx: DeviceContext) raises:
    """Higher-rank tensor contraction (4D tensors with 2 contraction axes).
    
    This demonstrates the ndim_mult > 1 case where we contract multiple axes.
    """
    print_separator("Example 4: Multi-Axis Contraction (ndim_mult=2)")
    
    print("\nSetup:")
    print("  Tensor A[2, 3, 4, 5] with trailing axes [4, 5] to be contracted")
    print("  Tensor B[4, 5, 6, 7] with leading axes [4, 5] to be contracted")
    print("  Result C[2, 3, 6, 7]")
    
    fn make_uniform_qnums(size: Int) -> List[QNumber]:
        var q = List[QNumber](capacity=size)
        for _ in range(size):
            q.append(QNumber(0))
        return q^
    
    var q2 = make_uniform_qnums(2)
    var q3 = make_uniform_qnums(3)
    var q4_a = make_uniform_qnums(4)
    var q5_a = make_uniform_qnums(5)
    var q4_b = make_uniform_qnums(4)
    var q5_b = make_uniform_qnums(5)
    var q6 = make_uniform_qnums(6)
    var q7 = make_uniform_qnums(7)
    
    var qnums_a = List[List[QNumber]]()
    qnums_a.append(q2^)
    qnums_a.append(q3^)
    qnums_a.append(q4_a^)
    qnums_a.append(q5_a^)
    
    var qnums_b = List[List[QNumber]]()
    qnums_b.append(q4_b^)
    qnums_b.append(q5_b^)
    qnums_b.append(q6^)
    qnums_b.append(q7^)
    
    var axis_dir_a = List[Int](1, 1, -1, -1)
    var axis_dir_b = List[Int](1, 1, -1, -1)
    
    var a = create_block_sparse_tensor[DType.float32](
        ctx, List[Int](2, 3, 4, 5)^, qnums_a^, 
        init_value=Scalar[DType.float32](1.0),
        axis_dir=axis_dir_a
    )
    var b = create_block_sparse_tensor[DType.float32](
        ctx, List[Int](4, 5, 6, 7)^, qnums_b^, 
        init_value=Scalar[DType.float32](1.0),
        axis_dir=axis_dir_b
    )
    
    print_tensor_info("A[2,3,4,5]", a)
    print_tensor_info("B[4,5,6,7]", b)
    
    var c = allocate_block_sparse_for_tensor_dot[DType.float32](
        a, b, ctx, ndim_mult=2, axrange_a=False, axrange_b=True
    )
    
    block_sparse_tensor_dot[DType.float32](
        c, a^, b^, ctx, ndim_mult=2, axrange_a=False, axrange_b=True
    )
    
    print_tensor_info("C[2,3,6,7]", c)
    
    var dense_c = block_sparse_to_dense[DType.float32](c^, ctx)
    var host = ctx.enqueue_create_host_buffer[DType.float32](2 * 3 * 6 * 7)
    ctx.enqueue_copy(host, dense_c.storage)
    ctx.synchronize()
    
    var expected_val = 4.0 * 5.0
    print("\nExpected element value (uniform case):", expected_val)
    print("Actual element [0,0,0,0]:", host[0])


fn example_5_sparsity_advantage(ctx: DeviceContext) raises:
    """Demonstrate the computational advantage of sparsity.
    
    Compare the number of non-zero elements in block-sparse vs dense representation.
    """
    print_separator("Example 5: Sparsity Advantage Analysis")
    
    print("\nSetup: Large tensor with structured quantum numbers")
    print("  Shape: [20, 20] with 5 quantum number sectors")
    
    var shape = List[Int](20, 20)
    
    var qnums = List[QNumber]()
    for sector in range(5):
        for _ in range(4):
            qnums.append(QNumber(sector))
    
    var qnums_per_leg = List[List[QNumber]]()
    qnums_per_leg.append(qnums.copy())
    qnums_per_leg.append(qnums^)
    
    var axis_dir = List[Int](1, -1)
    
    var t = create_block_sparse_tensor[DType.float32](
        ctx, shape^, qnums_per_leg^, 
        init_value=Scalar[DType.float32](1.0),
        axis_dir=axis_dir
    )
    
    var logical_size = t.logical_size
    var actual_nonzero = t.actual_nonzero_count()
    var sparsity = t.sparsity_ratio()
    
    print("\nResults:")
    print("  Logical size (dense would store):", logical_size)
    print("  Actual non-zero elements:", actual_nonzero)
    print("  Sparsity ratio:", sparsity)
    print("  Memory savings:", (1.0 - sparsity) * 100.0, "%")
    
    print("\nBlock structure:")
    print("  dim_blocks:", t.dim_blocks)
    print("  Each sector has 4 indices → 4×4 = 16 elements per block")
    print("  5 diagonal blocks → 5 × 16 = 80 elements")
    print("  (Matches actual_nonzero above)")


fn main() raises:
    @parameter
    if not has_accelerator():
        print("No compatible GPU found - examples require GPU")
        return
    
    print("=" * 70)
    print("Block Sparse Tensor Contraction Examples")
    print("Demonstrating quantum number conservation in tensor networks")
    print("=" * 70)
    
    with DeviceContext() as ctx:
        example_1_simple_matrix_contraction(ctx)
        example_2_three_particle_sectors(ctx)
        example_3_mps_mpo_style_contraction(ctx)
        example_4_higher_rank_contraction(ctx)
        example_5_sparsity_advantage(ctx) 
    
    print("\n" + "=" * 70)
    print("All contraction examples completed successfully!")
    print("=" * 70)
