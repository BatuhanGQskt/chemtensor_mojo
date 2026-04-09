"""Tensor abstraction layer for chemtensor.

This module defines the TensorOps trait that provides a complete interface
for tensor operations. Both DenseTensor and BlockSparseTensor implement
this trait, allowing them to be used interchangeably in MPS, MPO, DMRG,
and other algorithms.

Usage:
    # Import the trait and tensor types
    from src.m_tensor.tensor_traits import TensorOps, TensorBackend
    from src.m_tensor.tensor_ops import Tensor  # Alias based on compile-time selection
    
    # Write backend-agnostic algorithms:
    fn my_algorithm[T: TensorOps](tensor: T, ctx: DeviceContext) raises:
        var norm = tensor.compute_norm(ctx)
        var shape = tensor.get_shape()
        # ... algorithm implementation ...

Compile-Time Backend Selection:
    Change TensorBackendType in tensor_ops.mojo to switch backends globally:
        alias TensorBackendType: Int = TensorBackend.DENSE        # Dense tensors
        alias TensorBackendType: Int = TensorBackend.BLOCK_SPARSE # Block sparse

Note on Trait Limitations:
    Mojo traits cannot have methods that return Self or take `var self`.
    Such methods (transpose, reshape, etc.) are implemented as:
    1. Regular methods on each tensor type
    2. Generic free functions in tensor_ops.mojo for dispatch
"""

from collections.list import List
from gpu.host import DeviceContext, DeviceBuffer


trait TensorOps:
    """Complete interface for tensor operations.
    
    All tensor implementations (DenseTensor, BlockSparseTensor) must implement
    this trait to be usable in chemtensor algorithms.
    
    The trait is organized into categories:
    - Property Access: get_shape, get_stride, get_size, get_rank
    - Memory Layout: is_contiguous
    - Index Operations: get_flat_index
    - Numerical Operations: compute_norm, compute_norm_sq, compute_dot_product
    - Debug/Utility: print_contents
    
    Methods that return Self (transpose, reshape, flatten_dims, etc.) cannot
    be part of the trait due to Mojo limitations. They are provided as:
    - Instance methods on each tensor type
    - Generic dispatch functions in tensor_ops.mojo
    """
    
    # =========================================================================
    # Property Access
    # =========================================================================
    
    fn get_shape(self) -> List[Int]:
        """Get the shape of the tensor as a list of dimensions.
        
        Returns:
            Copy of the shape list, length equals tensor rank.
        
        Example:
            ```mojo
            var tensor = create_tensor(ctx, List[Int](3, 4, 5)^)
            var shape = tensor.get_shape()  # [3, 4, 5]
            ```
        """
        ...
    
    fn get_stride(self) -> List[Int]:
        """Get the stride of the tensor for each dimension.
        
        For row-major tensors, stride[i] is the number of elements to skip
        to move one step in dimension i.
        
        Returns:
            Copy of the stride list.
        
        Example:
            ```mojo
            var tensor = create_tensor(ctx, List[Int](3, 4, 5)^)
            var stride = tensor.get_stride()  # [20, 5, 1] for row-major
            ```
        """
        ...
    
    fn get_size(self) -> Int:
        """Get the total number of elements in the tensor.
        
        Returns:
            Product of all dimensions.
        
        Example:
            ```mojo
            var tensor = create_tensor(ctx, List[Int](3, 4, 5)^)
            var size = tensor.get_size()  # 60
            ```
        """
        ...
    
    fn get_rank(self) -> Int:
        """Get the number of dimensions (rank) of the tensor.
        
        Returns:
            Number of dimensions.
        
        Example:
            ```mojo
            var tensor = create_tensor(ctx, List[Int](3, 4, 5)^)
            var rank = tensor.get_rank()  # 3
            ```
        """
        ...
    
    # =========================================================================
    # Memory Layout
    # =========================================================================
    
    fn is_contiguous(self) -> Bool:
        """Check if tensor memory layout is contiguous in row-major order.
        
        A contiguous tensor has elements stored sequentially in memory.
        This is important for GPU efficiency and for operations like
        reshape that require contiguous data.
        
        Returns:
            True if contiguous in row-major order, False otherwise.
        
        Note:
            For BlockSparseTensor, this typically returns False since
            blocks are stored separately.
        """
        ...
    
    # =========================================================================
    # Index Operations
    # =========================================================================
    
    fn get_flat_index(self, indices: List[Int]) -> Int:
        """Compute flat linear index from multi-dimensional indices.
        
        Uses the tensor's stride to compute the linear offset.
        
        Args:
            indices: Multi-dimensional indices (length must equal rank).
        
        Returns:
            Linear index into the underlying storage.
        
        Example:
            ```mojo
            var tensor = create_tensor(ctx, List[Int](3, 4)^)  # stride [4, 1]
            var idx = tensor.get_flat_index(List[Int](1, 2))  # 1*4 + 2*1 = 6
            ```
        """
        ...
    
    # =========================================================================
    # Numerical Operations (Read-Only)
    # =========================================================================
    
    fn compute_norm(self, ctx: DeviceContext) raises -> Float64:
        """Compute the Frobenius norm of the tensor.
        
        The Frobenius norm is sqrt(sum of squared elements).
        Computation is done on GPU with only the scalar result transferred.
        
        Args:
            ctx: Device context for GPU operations.
        
        Returns:
            Frobenius norm ||T||_F = sqrt(sum_i |T_i|^2).
        """
        ...
    
    fn compute_norm_sq(self, ctx: DeviceContext) raises -> Float64:
        """Compute the squared Frobenius norm of the tensor.
        
        Avoids the sqrt for cases where only the squared norm is needed.
        
        Args:
            ctx: Device context for GPU operations.
        
        Returns:
            Squared Frobenius norm ||T||_F^2 = sum_i |T_i|^2.
        """
        ...
    
    fn compute_dot_product(self, other: Self, ctx: DeviceContext) raises -> Float64:
        """Compute the inner product <self, other>.
        
        The inner product is sum_i self_i * other_i (for real tensors).
        Both tensors must have the same size.
        
        Args:
            other: Another tensor of the same size.
            ctx: Device context for GPU operations.
        
        Returns:
            Inner product <self, other>.
        
        Raises:
            Error: If tensors have different sizes.
        """
        ...
    
    # =========================================================================
    # Debug/Utility
    # =========================================================================
    
    fn print_contents(self, ctx: DeviceContext) raises -> None:
        """Print the tensor contents for debugging.
        
        Transfers data from GPU to host for printing.
        For large tensors, only a subset may be printed.
        
        Args:
            ctx: Device context for GPU operations.
        """
        ...


# =============================================================================
# Methods NOT in Trait (Return Self - Mojo Limitation)
# =============================================================================
# The following methods are implemented on each tensor type but cannot be
# in the trait because they return Self or take `var self`:
#
# Shape Manipulation (return new tensor):
#   - transpose(var self, perm: List[Int], ctx: DeviceContext) -> Self
#   - reshape(var self, new_shape: List[Int]) -> Self
#   - flatten_dims(var self, start: Int, end: Int, ctx: DeviceContext) -> Self
#   - copy_to_contiguous(var self, ctx: DeviceContext) -> Self
#
# In-Place Operations (mutate self):
#   - scale_in_place(var self, scale: Scalar[dtype], ctx: DeviceContext) -> None
#   - axpy_in_place(var self, alpha: Scalar[dtype], x: Self, ctx: DeviceContext) -> None
#
# Static Factory Methods:
#   - random(ctx: DeviceContext, shape: List[Int], row_major: Bool) -> Self
#   - backend() -> TensorBackend
#
# These are accessed via:
# 1. Direct method calls on the tensor type
# 2. Generic dispatch functions in tensor_ops.mojo


# =============================================================================
# Tensor Type Selection
# =============================================================================
# Use these type aliases to easily switch between tensor implementations
# throughout your codebase. Change the alias definition to switch backends.

# Default tensor type - change this to switch implementations globally
# alias DefaultTensor = DenseTensor  # Uncomment when using dense tensors
# alias DefaultTensor = BlockSparseTensor  # Uncomment when using block sparse

# =============================================================================
# Compile-Time Backend Selection Constants
# =============================================================================
# These are compile-time constants used for @parameter if dispatch.
# They allow selecting the tensor backend at compile time with zero overhead.

struct TensorBackend:
    """Compile-time constants for tensor backend selection.
    
    Use these constants with @parameter if for compile-time dispatch:
    
    Example:
        ```mojo
        alias MyBackend: Int = TensorBackend.DENSE
        
        @parameter
        if MyBackend == TensorBackend.DENSE:
            # Dense tensor code path (compiled only if DENSE is selected)
            pass
        elif MyBackend == TensorBackend.BLOCK_SPARSE:
            # Block sparse code path (compiled only if BLOCK_SPARSE is selected)
            pass
        ```
    """
    # Compile-time constants for backend selection
    alias DENSE: Int = 0
    alias BLOCK_SPARSE: Int = 1
    alias COMPLEX_DENSE: Int = 2
    
    # Runtime value (for cases where runtime checks are needed)
    var value: Int
    
    fn __init__(out self, value: Int):
        self.value = value
    
    fn is_dense(self) -> Bool:
        """Runtime check if this is a dense backend."""
        return self.value == Self.DENSE
    
    fn is_block_sparse(self) -> Bool:
        """Runtime check if this is a block-sparse backend."""
        return self.value == Self.BLOCK_SPARSE
    
    fn is_complex(self) -> Bool:
        """Runtime check if this is a complex dense backend."""
        return self.value == Self.COMPLEX_DENSE
    
    @staticmethod
    fn name(backend_type: Int) -> String:
        """Get the name of a backend type."""
        if backend_type == Self.DENSE:
            return "DenseTensor"
        elif backend_type == Self.BLOCK_SPARSE:
            return "BlockSparseTensor"
        elif backend_type == Self.COMPLEX_DENSE:
            return "ComplexDenseTensor"
        else:
            return "Unknown"


# =============================================================================
# Generic Operation Signatures
# =============================================================================
# These are type signatures that generic tensor operations should follow.
# Actual implementations are in tensor_ops.mojo

# Tensor contraction signature:
# fn tensor_dot[T: TensorOps, dtype: DType](
#     C: T, A: T, B: T, ctx: DeviceContext, 
#     ndim_mult: Int, axrange_A: Bool, axrange_B: Bool
# ) raises

# Tensor decomposition signatures:
# fn tensor_qr[T: TensorOps, dtype: DType](
#     tensor: T, ctx: DeviceContext
# ) raises -> Tuple[T, T]

# fn tensor_svd_trunc[T: TensorOps, dtype: DType](
#     tensor: T, ctx: DeviceContext, chi_max: Int, eps_trunc: Float64
# ) raises -> Tuple[T, T, T, Int]
