"""Generic tensor operations with compile-time dispatch via function overloading.

This module provides generic tensor operations that work with BOTH DenseTensor
and BlockSparseTensor through function overloading. The compiler automatically
selects the correct implementation based on the argument types at compile time.

Usage:
    from src.m_tensor.tensor_ops import tensor_dot, create_tensor, Tensor
    
    # The SAME function name works for both backends:
    var A_dense: DenseTensor[DType.float32] = ...
    var B_dense: DenseTensor[DType.float32] = ...
    tensor_dot[DType.float32](C_dense, A_dense^, B_dense^, ctx)  # Uses dense implementation
    
    var A_sparse: BlockSparseTensor[DType.float32] = ...
    var B_sparse: BlockSparseTensor[DType.float32] = ...
    tensor_dot[DType.float32](C_sparse, A_sparse^, B_sparse^, ctx)  # Uses sparse implementation

Compile-Time Selection:
    The compiler selects the correct overload based on argument types.
    No runtime dispatch overhead - the decision is made at compile time.

Global Backend Selection:
    Use the `Tensor` type alias for algorithms that should use a single backend:
    
    alias TensorBackendType: Int = TensorBackend.DENSE  # Change to BLOCK_SPARSE to switch
    
    # Tensor resolves to DenseTensor or BlockSparseTensor at compile time
    var A: Tensor[DType.float32] = create_tensor[DType.float32](ctx, shape^)
"""

from collections.list import List
from gpu.host import DeviceContext

from src.m_tensor.tensor_traits import TensorOps, TensorBackend

# Import DenseTensor and its operations
from src.m_tensor.dense_tensor import (
    DenseTensor,
    dense_tensor_dot,
    dense_tensor_qr,
    dense_tensor_svd_trunc,
    create_dense_tensor,
    create_dense_tensor_uninitialized,
    create_dense_tensor_from_data,
)

# Import BlockSparseTensor and its operations
from src.m_tensor.block_sparse_tensor import (
    BlockSparseTensor,
    block_sparse_tensor_dot,
    block_sparse_tensor_svd_trunc,
    create_block_sparse_tensor,
    QNumber,
)


# =============================================================================
# COMPILE-TIME BACKEND SELECTION (for Tensor type alias)
# =============================================================================

alias TensorBackendType: Int = TensorBackend.DENSE
"""Compile-time backend selector for the `Tensor` type alias.

Change this to switch the default tensor type globally:
    alias TensorBackendType: Int = TensorBackend.DENSE        # Dense tensors
    alias TensorBackendType: Int = TensorBackend.BLOCK_SPARSE # Block sparse tensors
"""

# Type alias that resolves to the correct tensor type
@parameter
if TensorBackendType == TensorBackend.DENSE:
    alias Tensor = DenseTensor
else:
    alias Tensor = BlockSparseTensor


# =============================================================================
# TENSOR CONTRACTION - Overloaded for both backends
# =============================================================================

fn tensor_dot[dtype: DType](
    C: DenseTensor[dtype],
    var A: DenseTensor[dtype],
    var B: DenseTensor[dtype],
    ctx: DeviceContext,
    ndim_mult: Int = 1,
    axrange_A: Bool = False,
    axrange_B: Bool = False,
) raises:
    """Tensor contraction for DenseTensor.
    
    Contracts A and B, storing result in C.
    
    Args:
        C: Output tensor (pre-allocated).
        A: First input tensor (ownership transferred).
        B: Second input tensor (ownership transferred).
        ctx: Device context.
        ndim_mult: Number of axes to contract.
        axrange_A: Contract leading (True) or trailing (False) axes of A.
        axrange_B: Contract leading (True) or trailing (False) axes of B.
    """
    dense_tensor_dot[dtype](C, A^, B^, ctx, ndim_mult, axrange_A, axrange_B)


fn tensor_dot[dtype: DType](
    mut C: BlockSparseTensor[dtype],
    var A: BlockSparseTensor[dtype],
    var B: BlockSparseTensor[dtype],
    ctx: DeviceContext,
    ndim_mult: Int = 1,
    axrange_A: Bool = False,
    axrange_B: Bool = False,
) raises:
    """Tensor contraction for BlockSparseTensor.
    
    Contracts A and B with symmetry preservation, storing result in C.
    
    Args:
        C: Output tensor (pre-allocated with correct block structure; use
           :func:`allocate_block_sparse_for_tensor_dot` and zero-filled blocks).
        A: First input tensor (ownership transferred).
        B: Second input tensor (ownership transferred).
        ctx: Device context.
        ndim_mult: Number of axes to contract.
        axrange_A: Contract leading (True) or trailing (False) axes of A.
        axrange_B: Contract leading (True) or trailing (False) axes of B.
    """
    block_sparse_tensor_dot[dtype](C, A^, B^, ctx, ndim_mult, axrange_A, axrange_B)


# =============================================================================
# QR DECOMPOSITION - Overloaded for both backends
# =============================================================================

fn tensor_qr[dtype: DType](
    var tensor: DenseTensor[dtype],
    ctx: DeviceContext,
) raises -> Tuple[DenseTensor[dtype], DenseTensor[dtype]]:
    """QR decomposition for DenseTensor.
    
    Args:
        tensor: 2D matrix to decompose (ownership transferred).
        ctx: Device context.
    
    Returns:
        Tuple of (Q, R) where Q is orthogonal and R is upper triangular.
    """
    return dense_tensor_qr[dtype](tensor^, ctx)


# Note: BlockSparseTensor QR requires block-wise QR with quantum number handling.
# Add overload here when implemented:
# fn tensor_qr[dtype: DType](
#     var tensor: BlockSparseTensor[dtype],
#     ctx: DeviceContext,
# ) raises -> Tuple[BlockSparseTensor[dtype], BlockSparseTensor[dtype]]:
#     return block_sparse_tensor_qr[dtype](tensor^, ctx)


# =============================================================================
# TRUNCATED SVD - Overloaded for both backends
# =============================================================================

fn tensor_svd_trunc[dtype: DType](
    var tensor: DenseTensor[dtype],
    ctx: DeviceContext,
    chi_max: Int,
    eps_trunc: Float64 = 1e-12,
) raises -> Tuple[DenseTensor[dtype], DenseTensor[dtype], DenseTensor[dtype], Int]:
    """Truncated SVD for DenseTensor.
    
    Args:
        tensor: 2D matrix to decompose (ownership transferred).
        ctx: Device context.
        chi_max: Maximum number of singular values to keep.
        eps_trunc: Truncation threshold for discarded weight.
    
    Returns:
        Tuple of (U, S, Vt, chi_kept).
    """
    return dense_tensor_svd_trunc[dtype](tensor^, ctx, chi_max, eps_trunc)


fn tensor_svd_trunc[dtype: DType](
    var tensor: BlockSparseTensor[dtype],
    ctx: DeviceContext,
    chi_max: Int,
    eps_trunc: Float64 = 1e-12,
) raises -> Tuple[BlockSparseTensor[dtype], BlockSparseTensor[dtype], BlockSparseTensor[dtype], Int]:
    """Truncated SVD for BlockSparseTensor.
    
    Performs block-wise SVD with global truncation across all blocks.
    
    Args:
        tensor: 2D block-sparse matrix to decompose (ownership transferred).
        ctx: Device context.
        chi_max: Maximum number of singular values to keep.
        eps_trunc: Truncation threshold for discarded weight.
    
    Returns:
        Tuple of (U, S, Vt, chi_kept) as block-sparse tensors.
    """
    return block_sparse_tensor_svd_trunc[dtype](tensor^, ctx, chi_max, eps_trunc)


# =============================================================================
# TENSOR CREATION - Overloaded factory functions
# =============================================================================

fn create_tensor[dtype: DType, _: DenseTensor[dtype] = DenseTensor[dtype]](
    ctx: DeviceContext,
    var shape: List[Int],
    row_major: Bool = True,
    init_value: Optional[Scalar[dtype]] = None,
) raises -> DenseTensor[dtype]:
    """Create a DenseTensor.
    
    Args:
        ctx: Device context.
        shape: Shape of the tensor.
        row_major: Use row-major layout (default True).
        init_value: Initial value (None for random initialization).
    
    Returns:
        New DenseTensor.
    
    Example:
        var tensor = create_tensor[DType.float32](ctx, List[Int](3, 4)^)
    """
    return create_dense_tensor[dtype](ctx, shape^, row_major, init_value)


fn create_tensor[dtype: DType, _: BlockSparseTensor[dtype] = BlockSparseTensor[dtype]](
    ctx: DeviceContext,
    var shape: List[Int],
    var qnums_per_leg: List[List[QNumber]],
    init_value: Optional[Scalar[dtype]] = None,
) raises -> BlockSparseTensor[dtype]:
    """Create a BlockSparseTensor with quantum number structure.
    
    Args:
        ctx: Device context.
        shape: Logical shape of the tensor.
        qnums_per_leg: Quantum numbers for each index value on each leg.
        init_value: Initial value for non-zero blocks.
    
    Returns:
        New BlockSparseTensor.
    """
    return create_block_sparse_tensor[dtype](ctx, shape^, qnums_per_leg^, init_value)


fn create_tensor_uninitialized[dtype: DType, _: DenseTensor[dtype] = DenseTensor[dtype]](
    ctx: DeviceContext,
    var shape: List[Int],
    row_major: Bool = True,
) raises -> DenseTensor[dtype]:
    """Create an uninitialized DenseTensor.
    
    Use for output buffers that will be immediately overwritten.
    
    Args:
        ctx: Device context.
        shape: Shape of the tensor.
        row_major: Use row-major layout.
    
    Returns:
        New uninitialized DenseTensor.
    """
    return create_dense_tensor_uninitialized[dtype](ctx, shape^, row_major)


fn create_tensor_from_data[dtype: DType, _: DenseTensor[dtype] = DenseTensor[dtype]](
    ctx: DeviceContext,
    data: List[Scalar[dtype]],
    var shape: List[Int],
    row_major: Bool = True,
) raises -> DenseTensor[dtype]:
    """Create a DenseTensor from host data.
    
    Args:
        ctx: Device context.
        data: Flat list of values.
        shape: Shape of the tensor.
        row_major: Use row-major layout.
    
    Returns:
        New DenseTensor with the provided data.
    """
    return create_dense_tensor_from_data[dtype](ctx, data, shape^, row_major)


# =============================================================================
# TRANSPOSE - Overloaded for both backends
# =============================================================================

fn tensor_transpose[dtype: DType](
    var tensor: DenseTensor[dtype],
    perm: List[Int],
    ctx: DeviceContext,
) raises -> DenseTensor[dtype]:
    """Transpose a DenseTensor.
    
    Args:
        tensor: Tensor to transpose (ownership transferred).
        perm: Permutation of dimensions.
        ctx: Device context.
    
    Returns:
        Transposed tensor (contiguous).
    """
    return tensor^.transpose(perm, ctx)


fn tensor_transpose[dtype: DType](
    var tensor: BlockSparseTensor[dtype],
    perm: List[Int],
    ctx: DeviceContext,
) raises -> BlockSparseTensor[dtype]:
    """Transpose a BlockSparseTensor.
    
    Args:
        tensor: Tensor to transpose (ownership transferred).
        perm: Permutation of dimensions.
        ctx: Device context.
    
    Returns:
        Transposed tensor with reordered block structure.
    """
    return tensor^.transpose(perm, ctx)


# =============================================================================
# RESHAPE - Overloaded for both backends
# =============================================================================

fn tensor_reshape[dtype: DType](
    var tensor: DenseTensor[dtype],
    var new_shape: List[Int],
) raises -> DenseTensor[dtype]:
    """Reshape a DenseTensor.
    
    Args:
        tensor: Tensor to reshape (ownership transferred).
        new_shape: New shape (must have same total size).
    
    Returns:
        Reshaped tensor (view, no data copy).
    """
    return tensor^.reshape(new_shape^)


fn tensor_reshape[dtype: DType](
    var tensor: BlockSparseTensor[dtype],
    var new_shape: List[Int],
) raises -> BlockSparseTensor[dtype]:
    """Reshape a BlockSparseTensor.
    
    Note: May require block restructuring.
    
    Args:
        tensor: Tensor to reshape (ownership transferred).
        new_shape: New shape (must have same total size).
    
    Returns:
        Reshaped tensor.
    """
    return tensor^.reshape(new_shape^)


# =============================================================================
# FLATTEN DIMENSIONS - Overloaded for both backends
# =============================================================================

fn tensor_flatten_dims[dtype: DType](
    var tensor: DenseTensor[dtype],
    start: Int,
    end: Int,
    ctx: DeviceContext,
) raises -> DenseTensor[dtype]:
    """Flatten a range of dimensions in a DenseTensor.
    
    Args:
        tensor: Tensor to flatten (ownership transferred).
        start: Starting dimension (inclusive).
        end: Ending dimension (exclusive).
        ctx: Device context.
    
    Returns:
        Tensor with flattened dimensions.
    """
    return tensor^.flatten_dims(start, end, ctx)


fn tensor_flatten_dims[dtype: DType](
    var tensor: BlockSparseTensor[dtype],
    start: Int,
    end: Int,
    ctx: DeviceContext,
) raises -> BlockSparseTensor[dtype]:
    """Flatten a range of dimensions in a BlockSparseTensor.
    
    Args:
        tensor: Tensor to flatten (ownership transferred).
        start: Starting dimension (inclusive).
        end: Ending dimension (exclusive).
        ctx: Device context.
    
    Returns:
        Tensor with flattened dimensions.
    """
    return tensor^.flatten_dims(start, end, ctx)


# =============================================================================
# COPY TO CONTIGUOUS - Overloaded for both backends
# =============================================================================

fn tensor_copy_to_contiguous[dtype: DType](
    var tensor: DenseTensor[dtype],
    ctx: DeviceContext,
) raises -> DenseTensor[dtype]:
    """Make a DenseTensor contiguous in memory.
    
    Args:
        tensor: Tensor (ownership transferred).
        ctx: Device context.
    
    Returns:
        Contiguous tensor (may be same tensor if already contiguous).
    """
    return tensor^.copy_to_contiguous(ctx)


fn tensor_copy_to_contiguous[dtype: DType](
    var tensor: BlockSparseTensor[dtype],
    ctx: DeviceContext,
) raises -> BlockSparseTensor[dtype]:
    """Optimize block layout in a BlockSparseTensor.
    
    Args:
        tensor: Tensor (ownership transferred).
        ctx: Device context.
    
    Returns:
        Tensor with optimized block layout.
    """
    return tensor^.copy_to_contiguous(ctx)


# =============================================================================
# SCALE IN-PLACE - Overloaded for both backends
# =============================================================================

fn tensor_scale_in_place[dtype: DType](
    var tensor: DenseTensor[dtype],
    scale: Scalar[dtype],
    ctx: DeviceContext,
) raises -> None:
    """Scale a DenseTensor in-place.
    
    Args:
        tensor: Tensor to scale (modified in-place).
        scale: Scalar multiplier.
        ctx: Device context.
    """
    tensor.scale_in_place(scale, ctx)


fn tensor_scale_in_place[dtype: DType](
    var tensor: BlockSparseTensor[dtype],
    scale: Scalar[dtype],
    ctx: DeviceContext,
) raises -> None:
    """Scale a BlockSparseTensor in-place.
    
    Args:
        tensor: Tensor to scale (modified in-place).
        scale: Scalar multiplier.
        ctx: Device context.
    """
    tensor.scale_in_place(scale, ctx)


# =============================================================================
# AXPY IN-PLACE - Overloaded for both backends
# =============================================================================

fn tensor_axpy_in_place[dtype: DType](
    var y: DenseTensor[dtype],
    alpha: Scalar[dtype],
    x: DenseTensor[dtype],
    ctx: DeviceContext,
) raises -> None:
    """Perform y += alpha * x for DenseTensor.
    
    Args:
        y: Target tensor (modified in-place).
        alpha: Scalar coefficient.
        x: Source tensor.
        ctx: Device context.
    """
    y.axpy_in_place(alpha, x, ctx)


fn tensor_axpy_in_place[dtype: DType](
    var y: BlockSparseTensor[dtype],
    alpha: Scalar[dtype],
    x: BlockSparseTensor[dtype],
    ctx: DeviceContext,
) raises -> None:
    """Perform y += alpha * x for BlockSparseTensor.
    
    Args:
        y: Target tensor (modified in-place).
        alpha: Scalar coefficient.
        x: Source tensor (must have compatible block structure).
        ctx: Device context.
    """
    y.axpy_in_place(alpha, x, ctx)


# =============================================================================
# DOT PRODUCT - Overloaded for both backends
# =============================================================================

fn tensor_dot_product[dtype: DType](
    a: DenseTensor[dtype],
    b: DenseTensor[dtype],
    ctx: DeviceContext,
) raises -> Float64:
    """Compute inner product <a, b> for DenseTensor.
    
    Args:
        a: First tensor.
        b: Second tensor (must have same size).
        ctx: Device context.
    
    Returns:
        Inner product.
    """
    return a.dot_product(b, ctx)


fn tensor_dot_product[dtype: DType](
    a: BlockSparseTensor[dtype],
    b: BlockSparseTensor[dtype],
    ctx: DeviceContext,
) raises -> Float64:
    """Compute inner product <a, b> for BlockSparseTensor.
    
    Args:
        a: First tensor.
        b: Second tensor (must have compatible block structure).
        ctx: Device context.
    
    Returns:
        Inner product.
    """
    return a.dot_product(b, ctx)


# =============================================================================
# NORM - Overloaded for both backends
# =============================================================================

fn tensor_norm[dtype: DType](
    tensor: DenseTensor[dtype],
    ctx: DeviceContext,
) raises -> Float64:
    """Compute Frobenius norm of a DenseTensor.
    
    Args:
        tensor: Input tensor.
        ctx: Device context.
    
    Returns:
        Frobenius norm.
    """
    return tensor.norm(ctx)


fn tensor_norm[dtype: DType](
    tensor: BlockSparseTensor[dtype],
    ctx: DeviceContext,
) raises -> Float64:
    """Compute Frobenius norm of a BlockSparseTensor.
    
    Args:
        tensor: Input tensor.
        ctx: Device context.
    
    Returns:
        Frobenius norm.
    """
    return tensor.norm(ctx)


# =============================================================================
# Runtime Configuration
# =============================================================================

@value
struct TensorConfig:
    """Runtime configuration for tensor operations.
    
    Note: Backend selection is compile-time (via TensorBackendType and overloading).
    This struct is for runtime behavior like debugging and profiling.
    """
    var use_gpu: Bool
    var verbose: Bool
    var check_contiguity: Bool
    
    fn __init__(
        out self,
        use_gpu: Bool = True,
        verbose: Bool = False,
        check_contiguity: Bool = False,
    ):
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.check_contiguity = check_contiguity


var _tensor_config = TensorConfig()


fn get_tensor_config() -> TensorConfig:
    """Get the current tensor configuration."""
    return _tensor_config


fn set_tensor_config(config: TensorConfig):
    """Set the tensor configuration."""
    _tensor_config = config


# =============================================================================
# Backend Query Functions
# =============================================================================

fn is_using_dense_backend() -> Bool:
    """Returns True if Tensor alias resolves to DenseTensor."""
    @parameter
    if TensorBackendType == TensorBackend.DENSE:
        return True
    else:
        return False


fn is_using_sparse_backend() -> Bool:
    """Returns True if Tensor alias resolves to BlockSparseTensor."""
    @parameter
    if TensorBackendType == TensorBackend.BLOCK_SPARSE:
        return True
    else:
        return False
