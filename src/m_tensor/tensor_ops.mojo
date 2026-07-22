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
    block_sparse_tensor_qr,
    block_sparse_tensor_svd_trunc,
    create_block_sparse_tensor,
    QNumber,
)


# =============================================================================
# COMPILE-TIME BACKEND SELECTION (for Tensor type alias)
# =============================================================================
comptime TensorBackendType: Int = TensorBackend.DENSE

# =============================================================================
# TENSOR CONTRACTION - Overloaded for both backends
# =============================================================================

def tensor_dot[dtype: DType](
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


def tensor_dot[dtype: DType](
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

def tensor_qr[dtype: DType](
    var tensor: DenseTensor[dtype] | BlockSparseTensor[dtype],
    ctx: DeviceContext,
) raises -> Tuple[DenseTensor[dtype] | BlockSparseTensor[dtype], DenseTensor[dtype] | BlockSparseTensor[dtype]]:
    """QR decomposition for DenseTensor.
    
    Args:
        tensor: 2D matrix to decompose (ownership transferred).
        ctx: Device context.
    
    Returns:
        Tuple of (Q, R) where Q is orthogonal and R is upper triangular.
    """
    return dense_tensor_qr[dtype](tensor^, ctx)


# =============================================================================
# TRUNCATED SVD - Overloaded for both backends
# =============================================================================

def tensor_svd_trunc[dtype: DType](
    var tensor: DenseTensor[dtype]  | BlockSparseTensor[dtype],
    ctx: DeviceContext,
    chi_max: Int,
    eps_trunc: Float64 = 1e-12,
) raises -> Tuple[DenseTensor[dtype] | BlockSparseTensor[dtype], DenseTensor[dtype] | BlockSparseTensor[dtype], DenseTensor[dtype] | BlockSparseTensor[dtype], Int]:
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

# =============================================================================
# TENSOR CREATION - Overloaded factory functions
# =============================================================================

def create_tensor[dtype: DType](
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
    """
    return create_dense_tensor[dtype](ctx, shape^, row_major, init_value)


def create_tensor[dtype: DType](
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


def create_tensor_uninitialized[dtype: DType](
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


def create_tensor_from_data[dtype: DType](
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

def tensor_transpose[dtype: DType](
    var tensor: DenseTensor[dtype] | BlockSparseTensor[dtype],
    perm: List[Int],
    ctx: DeviceContext,
) raises -> DenseTensor[dtype] | BlockSparseTensor[dtype]:
    """Transpose a DenseTensor.
    
    Args:
        tensor: Tensor to transpose (ownership transferred).
        perm: Permutation of dimensions.
        ctx: Device context.
    
    Returns:
        Transposed tensor (contiguous).
    """
    return tensor^.transpose(perm, ctx)


# =============================================================================
# RESHAPE - Overloaded for both backends
# =============================================================================

def tensor_reshape[dtype: DType](
    var tensor: DenseTensor[dtype] | BlockSparseTensor[dtype],
    var new_shape: List[Int],
) raises -> DenseTensor[dtype] | BlockSparseTensor[dtype]:
    """Reshape a DenseTensor.
    
    Args:
        tensor: Tensor to reshape (ownership transferred).
        new_shape: New shape (must have same total size).
    
    Returns:
        Reshaped tensor (view, no data copy).
    """
    return tensor^.reshape(new_shape^)

# =============================================================================
# FLATTEN DIMENSIONS - Overloaded for both backends
# =============================================================================

def tensor_flatten_dims[dtype: DType](
    var tensor: DenseTensor[dtype] | BlockSparseTensor[dtype],
    start: Int,
    end: Int,
    ctx: DeviceContext,
) raises -> DenseTensor[dtype] | BlockSparseTensor[dtype]:
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


# =============================================================================
# COPY TO CONTIGUOUS - Overloaded for both backends
# =============================================================================

def tensor_copy_to_contiguous[dtype: DType](
    var tensor: DenseTensor[dtype] | BlockSparseTensor[dtype],
    ctx: DeviceContext,
) raises -> DenseTensor[dtype] | BlockSparseTensor[dtype]:
    """Make a DenseTensor contiguous in memory.
    
    Args:
        tensor: Tensor (ownership transferred).
        ctx: Device context.
    
    Returns:
        Contiguous tensor (may be same tensor if already contiguous).
    """
    return tensor^.copy_to_contiguous(ctx)


# =============================================================================
# SCALE IN-PLACE - Overloaded for both backends
# =============================================================================

def tensor_scale_in_place[dtype: DType](
    var tensor: DenseTensor[dtype] | BlockSparseTensor[dtype],
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

# =============================================================================
# AXPY IN-PLACE - Overloaded for both backends
# =============================================================================

def tensor_axpy_in_place[dtype: DType](
    var y: DenseTensor[dtype] | BlockSparseTensor[dtype],
    alpha: Scalar[dtype],
    x: DenseTensor[dtype] | BlockSparseTensor[dtype],
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

# =============================================================================
# DOT PRODUCT - Overloaded for both backends
# =============================================================================

def tensor_dot_product[dtype: DType](
    a: DenseTensor[dtype] | BlockSparseTensor[dtype],
    b: DenseTensor[dtype] | BlockSparseTensor[dtype],
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


# =============================================================================
# NORM - Overloaded for both backends
# =============================================================================

def tensor_norm[dtype: DType](
    tensor: DenseTensor[dtype] | BlockSparseTensor[dtype],
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


# =============================================================================
# Runtime Configuration
# =============================================================================

@fieldwise_init
struct TensorConfig(ImplicitlyCopyable):
    """Runtime configuration for tensor operations.
    
    Note: Backend selection is compile-time (via TensorBackendType and overloading).
    This struct is for runtime behavior like debugging and profiling.
    """
    var use_gpu: Bool
    var verbose: Bool
    var check_contiguity: Bool

comptime _tensor_config: TensorConfig = TensorConfig(use_gpu=True, verbose=False, check_contiguity=False)

def get_tensor_config() -> TensorConfig:
    """Get the current tensor configuration."""
    return _tensor_config.copy()


def set_tensor_config(config: TensorConfig):
    """Set the tensor configuration."""
    _tensor_config = config.copy()


# =============================================================================
# Backend Query Functions
# =============================================================================

def is_using_dense_backend() -> Bool:
    """Returns True if Tensor alias resolves to DenseTensor."""
    comptime if TensorBackendType == TensorBackend.DENSE:
        return True
    else:
        return False


def is_using_sparse_backend() -> Bool:
    """Returns True if Tensor alias resolves to BlockSparseTensor."""
    comptime if TensorBackendType == TensorBackend.BLOCK_SPARSE:
        return True
    else:
        return False
