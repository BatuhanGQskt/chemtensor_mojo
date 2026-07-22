"""Example: DenseTensor and BlockSparseTensor Interchangeability

This example demonstrates that DenseTensor and BlockSparseTensor can be used
interchangeably through the generic tensor_ops interface. Both tensor types:

1. Implement the TensorOps trait
2. Support the same operations (dot, norm, transpose, etc.)
3. Can be used in generic algorithms via function overloading

The key insight is that tensor_ops.mojo provides overloaded functions that
dispatch to the correct implementation at compile time based on argument types.
"""

from sys import has_accelerator
from gpu.host import DeviceContext
from collections.list import List
from math import sqrt

from src.m_tensor.dense_tensor import (
    DenseTensor,
    create_dense_tensor,
    create_dense_tensor_from_data,
    dense_tensor_dot,
)
from src.m_tensor.block_sparse_tensor import (
    BlockSparseTensor,
    QNumber,
    create_block_sparse_tensor,
    allocate_block_sparse_for_tensor_dot,
    block_sparse_tensor_dot,
    block_sparse_to_dense,
    dense_to_block_sparse,
)
from src.m_tensor.tensor_ops import (
    tensor_dot,
    tensor_norm,
    tensor_transpose,
    tensor_scale_in_place,
    tensor_axpy_in_place,
    tensor_dot_product,
    Tensor,
    TensorBackendType,
)
from src.m_tensor.tensor_traits import TensorOps, TensorBackend


fn print_separator(title: String):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


fn example_1_trait_methods(ctx: DeviceContext) raises:
    """Demonstrate that both tensor types implement TensorOps trait methods."""
    print_separator("Example 1: TensorOps Trait Methods")
    
    var data = List[Scalar[DType.float32]]()
    for i in range(12):
        data.append(Scalar[DType.float32](Float32(i + 1)))
    var shape = List[Int](3, 4)
    
    print("\nCreating DenseTensor [3,4] with values 1..12")
    var dense = create_dense_tensor_from_data[DType.float32](ctx, data.copy(), shape.copy())
    
    print("  get_shape():", dense.get_shape())
    print("  get_stride():", dense.get_stride())
    print("  get_size():", dense.get_size())
    print("  get_rank():", dense.get_rank())
    print("  is_contiguous():", dense.is_contiguous())
    print("  compute_norm():", dense.compute_norm(ctx))
    
    print("\nConverting to BlockSparseTensor with uniform quantum numbers")
    var qnums_row = List[QNumber](capacity=3)
    for _ in range(3):
        qnums_row.append(QNumber(0))
    var qnums_col = List[QNumber](capacity=4)
    for _ in range(4):
        qnums_col.append(QNumber(0))
    var qnums_per_leg = List[List[QNumber]]()
    qnums_per_leg.append(qnums_row^)
    qnums_per_leg.append(qnums_col^)
    
    var sparse = dense_to_block_sparse[DType.float32](dense^, qnums_per_leg^, ctx)
    
    print("  get_shape():", sparse.get_shape())
    print("  get_stride():", sparse.get_stride())
    print("  get_size():", sparse.get_size())
    print("  get_rank():", sparse.get_rank())
    print("  is_contiguous():", sparse.is_contiguous())
    print("  compute_norm():", sparse.compute_norm(ctx))
    print("  sparsity_ratio():", sparse.sparsity_ratio())


fn example_2_generic_operations(ctx: DeviceContext) raises:
    """Demonstrate using generic tensor_ops functions with both types."""
    print_separator("Example 2: Generic tensor_ops Functions")
    
    print("\n--- DenseTensor Operations ---")
    
    var dense_a = create_dense_tensor[DType.float32](
        ctx, List[Int](3, 4)^, init_value=Scalar[DType.float32](2.0)
    )
    var dense_b = create_dense_tensor[DType.float32](
        ctx, List[Int](4, 5)^, init_value=Scalar[DType.float32](3.0)
    )
    var dense_c = create_dense_tensor[DType.float32](
        ctx, List[Int](3, 5)^, init_value=Scalar[DType.float32](0.0)
    )
    
    print("A[3,4] filled with 2.0, B[4,5] filled with 3.0")
    
    tensor_dot[DType.float32](dense_c, dense_a^, dense_b^, ctx)
    
    var c_shape = dense_c.get_shape()
    print("C = A @ B shape:", c_shape)
    print("C norm:", tensor_norm[DType.float32](dense_c, ctx))
    
    var dense_t = tensor_transpose[DType.float32](dense_c^, List[Int](1, 0), ctx)
    var t_shape = dense_t.get_shape()
    print("C^T shape:", t_shape)
    
    print("\n--- BlockSparseTensor Operations ---")
    
    fn make_qnums(size: Int) -> List[QNumber]:
        var q = List[QNumber](capacity=size)
        for _ in range(size):
            q.append(QNumber(0))
        return q^
    
    var q3_a = make_qnums(3)
    var q4_a = make_qnums(4)
    var qnums_a = List[List[QNumber]]()
    qnums_a.append(q3_a^)
    qnums_a.append(q4_a^)
    
    var q4_b = make_qnums(4)
    var q5_b = make_qnums(5)
    var qnums_b = List[List[QNumber]]()
    qnums_b.append(q4_b^)
    qnums_b.append(q5_b^)
    
    var sparse_a = create_block_sparse_tensor[DType.float32](
        ctx, List[Int](3, 4)^, qnums_a^, init_value=Scalar[DType.float32](2.0)
    )
    var sparse_b = create_block_sparse_tensor[DType.float32](
        ctx, List[Int](4, 5)^, qnums_b^, init_value=Scalar[DType.float32](3.0)
    )
    var sparse_c = allocate_block_sparse_for_tensor_dot[DType.float32](
        sparse_a, sparse_b, ctx, ndim_mult=1, axrange_a=False, axrange_b=True
    )
    
    print("A[3,4] filled with 2.0, B[4,5] filled with 3.0")
    
    tensor_dot[DType.float32](sparse_c, sparse_a^, sparse_b^, ctx)
    
    var sc_shape = sparse_c.get_shape()
    print("C = A @ B shape:", sc_shape)
    print("C norm:", tensor_norm[DType.float32](sparse_c, ctx))
    
    var sparse_t = tensor_transpose[DType.float32](sparse_c^, List[Int](1, 0), ctx)
    var st_shape = sparse_t.get_shape()
    print("C^T shape:", st_shape)


fn example_3_in_place_operations(ctx: DeviceContext) raises:
    """Demonstrate in-place operations on both tensor types."""
    print_separator("Example 3: In-Place Operations")
    
    print("\n--- DenseTensor scale and axpy ---")
    
    var dense_x = create_dense_tensor[DType.float32](
        ctx, List[Int](4, 4)^, init_value=Scalar[DType.float32](2.0)
    )
    var dense_y = create_dense_tensor[DType.float32](
        ctx, List[Int](4, 4)^, init_value=Scalar[DType.float32](1.0)
    )
    
    print("Initial y norm:", tensor_norm[DType.float32](dense_y, ctx))
    
    tensor_scale_in_place[DType.float32](dense_y, Scalar[DType.float32](0.5), ctx)
    print("After y *= 0.5, norm:", tensor_norm[DType.float32](dense_y, ctx))
    
    tensor_axpy_in_place[DType.float32](dense_y, Scalar[DType.float32](2.0), dense_x, ctx)
    print("After y += 2.0 * x, norm:", tensor_norm[DType.float32](dense_y, ctx))
    
    print("\n--- BlockSparseTensor scale and axpy ---")
    
    fn make_qnums_4() -> List[QNumber]:
        var q = List[QNumber](capacity=4)
        for _ in range(4):
            q.append(QNumber(0))
        return q^
    
    var qx1 = make_qnums_4()
    var qx2 = make_qnums_4()
    var qnums_x = List[List[QNumber]]()
    qnums_x.append(qx1^)
    qnums_x.append(qx2^)
    
    var qy1 = make_qnums_4()
    var qy2 = make_qnums_4()
    var qnums_y = List[List[QNumber]]()
    qnums_y.append(qy1^)
    qnums_y.append(qy2^)
    
    var sparse_x = create_block_sparse_tensor[DType.float32](
        ctx, List[Int](4, 4)^, qnums_x^, init_value=Scalar[DType.float32](2.0)
    )
    var sparse_y = create_block_sparse_tensor[DType.float32](
        ctx, List[Int](4, 4)^, qnums_y^, init_value=Scalar[DType.float32](1.0)
    )
    
    print("Initial y norm:", tensor_norm[DType.float32](sparse_y, ctx))
    
    tensor_scale_in_place[DType.float32](sparse_y, Scalar[DType.float32](0.5), ctx)
    print("After y *= 0.5, norm:", tensor_norm[DType.float32](sparse_y, ctx))
    
    tensor_axpy_in_place[DType.float32](sparse_y, Scalar[DType.float32](2.0), sparse_x, ctx)
    print("After y += 2.0 * x, norm:", tensor_norm[DType.float32](sparse_y, ctx))


fn example_4_dot_product_comparison(ctx: DeviceContext) raises:
    """Compare dot product results between dense and sparse representations."""
    print_separator("Example 4: Dot Product Comparison")
    
    var data = List[Scalar[DType.float32]]()
    for i in range(16):
        data.append(Scalar[DType.float32](Float32(i + 1)))
    
    var dense_a = create_dense_tensor_from_data[DType.float32](
        ctx, data.copy(), List[Int](4, 4)^
    )
    var dense_b = create_dense_tensor_from_data[DType.float32](
        ctx, data.copy(), List[Int](4, 4)^
    )
    
    var dense_dot = tensor_dot_product[DType.float32](dense_a, dense_b, ctx)
    print("DenseTensor <a, b>:", dense_dot)
    
    fn make_qnums_4() -> List[QNumber]:
        var q = List[QNumber](capacity=4)
        for _ in range(4):
            q.append(QNumber(0))
        return q^
    
    var qa1 = make_qnums_4()
    var qa2 = make_qnums_4()
    var qnums_a = List[List[QNumber]]()
    qnums_a.append(qa1^)
    qnums_a.append(qa2^)
    
    var qb1 = make_qnums_4()
    var qb2 = make_qnums_4()
    var qnums_b = List[List[QNumber]]()
    qnums_b.append(qb1^)
    qnums_b.append(qb2^)
    
    var sparse_a = dense_to_block_sparse[DType.float32](dense_a^, qnums_a^, ctx)
    var sparse_b = dense_to_block_sparse[DType.float32](dense_b^, qnums_b^, ctx)
    
    var sparse_dot = tensor_dot_product[DType.float32](sparse_a, sparse_b, ctx)
    print("BlockSparseTensor <a, b>:", sparse_dot)
    
    var diff = abs(dense_dot - sparse_dot)
    print("Difference:", diff)
    if diff < 1e-4:
        print("PASS: Results match within tolerance")
    else:
        print("FAIL: Results differ significantly")


fn example_5_backend_selection(ctx: DeviceContext) raises:
    """Demonstrate compile-time backend selection via TensorBackendType."""
    print_separator("Example 5: Compile-Time Backend Selection")
    
    print("\nCurrent backend type constant:", TensorBackendType)
    print("Backend name:", TensorBackend.name(TensorBackendType))
    
    print("\nThe `Tensor` type alias resolves at compile time:")
    print("  - If TensorBackendType == DENSE: Tensor = DenseTensor")
    print("  - If TensorBackendType == BLOCK_SPARSE: Tensor = BlockSparseTensor")
    
    print("\nThis allows algorithms to be written generically:")
    print("  var mps_tensor: Tensor[DType.float32] = ...")
    print("And the concrete type is determined by changing TensorBackendType.")


fn main() raises:
    @parameter
    if not has_accelerator():
        print("No compatible GPU found - examples require GPU")
        return
    
    print("=" * 60)
    print("DenseTensor / BlockSparseTensor Interchangeability Demo")
    print("=" * 60)
    
    with DeviceContext() as ctx:
        example_1_trait_methods(ctx)
        example_2_generic_operations(ctx)
        example_3_in_place_operations(ctx)
        example_4_dot_product_comparison(ctx)
        example_5_backend_selection(ctx)
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
