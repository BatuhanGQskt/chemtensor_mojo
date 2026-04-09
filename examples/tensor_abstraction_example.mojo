"""Example: Using the Generic Tensor Operations

This example demonstrates how to write backend-agnostic tensor code
using function overloading. The SAME function names work with both
DenseTensor and BlockSparseTensor - the compiler selects the correct
implementation based on argument types at compile time.

Key Concepts:
1. Function overloading: tensor_dot, tensor_transpose, etc. work with any tensor type
2. Tensor type alias: `Tensor` resolves to DenseTensor or BlockSparseTensor at compile time
3. Zero runtime overhead: All dispatch decisions are made at compile time
"""

from gpu.host import DeviceContext
from collections.list import List

# Import generic operations - these work with BOTH backends!
from src.m_tensor import (
    # Type aliases
    Tensor,            # Resolves to DenseTensor or BlockSparseTensor
    TensorBackendType, # Current backend selection
    TensorBackend,
    
    # Generic operations (overloaded for both backends)
    tensor_dot,
    tensor_transpose,
    tensor_reshape,
    tensor_flatten_dims,
    tensor_svd_trunc,
    tensor_norm,
    tensor_dot_product,
    tensor_scale_in_place,
    create_tensor,
    create_tensor_uninitialized,
    
    # Backend query
    is_using_dense_backend,
    is_using_sparse_backend,
)

# Can also import specific types for explicit control
from src.m_tensor import DenseTensor, BlockSparseTensor


# =============================================================================
# Example 1: Using the Tensor Type Alias
# =============================================================================

fn example_with_tensor_alias() raises:
    """Use the Tensor alias for backend-agnostic code.
    
    The Tensor alias resolves at compile time based on TensorBackendType.
    Change TensorBackendType in tensor_ops.mojo to switch backends.
    """
    print("\n=== Example 1: Using Tensor Type Alias ===")
    print("Current backend:", TensorBackend.name(TensorBackendType))
    
    with DeviceContext() as ctx:
        # Create tensors using the Tensor alias
        # This automatically uses the selected backend
        var A: Tensor[DType.float32] = create_tensor[DType.float32](
            ctx, List[Int](3, 4)^, init_value=Scalar[DType.float32](1.0)
        )
        var B: Tensor[DType.float32] = create_tensor[DType.float32](
            ctx, List[Int](4, 5)^, init_value=Scalar[DType.float32](2.0)
        )
        var C: Tensor[DType.float32] = create_tensor_uninitialized[DType.float32](
            ctx, List[Int](3, 5)^
        )
        
        # Generic operations work with the Tensor type
        tensor_dot[DType.float32](C, A^, B^, ctx)
        
        print("  Created tensors and performed contraction")
        print("  Result shape:", C.get_shape()[0], "x", C.get_shape()[1])
        print("  Result norm:", tensor_norm[DType.float32](C, ctx))


# =============================================================================
# Example 2: Explicit DenseTensor Usage
# =============================================================================

fn example_explicit_dense() raises:
    """Explicitly use DenseTensor regardless of TensorBackendType.
    
    The same generic functions work - compiler selects based on argument types.
    """
    print("\n=== Example 2: Explicit DenseTensor Usage ===")
    
    with DeviceContext() as ctx:
        # Explicitly create DenseTensor
        var A: DenseTensor[DType.float32] = create_tensor[DType.float32](
            ctx, List[Int](4, 4)^, init_value=Scalar[DType.float32](0.5)
        )
        
        # Same generic functions work!
        var norm = tensor_norm[DType.float32](A, ctx)
        print("  DenseTensor norm:", norm)
        
        # Transpose using generic function
        var perm = List[Int](1, 0)
        var A_t = tensor_transpose[DType.float32](A^, perm, ctx)
        print("  Transposed shape:", A_t.get_shape()[0], "x", A_t.get_shape()[1])
        
        # SVD using generic function
        var A_mat: DenseTensor[DType.float32] = create_tensor[DType.float32](
            ctx, List[Int](5, 3)^, init_value=Scalar[DType.float32](1.0)
        )
        var svd_result = tensor_svd_trunc[DType.float32](A_mat^, ctx, chi_max=2)
        var U = svd_result[0]
        var S = svd_result[1]
        var Vt = svd_result[2]
        var chi_kept = svd_result[3]
        print("  SVD kept", chi_kept, "singular values")
        print("  U shape:", U.get_shape()[0], "x", U.get_shape()[1])


# =============================================================================
# Example 3: Writing Generic Algorithms
# =============================================================================

fn normalize_tensor[dtype: DType](
    var tensor: DenseTensor[dtype],
    ctx: DeviceContext,
) raises -> DenseTensor[dtype]:
    """Normalize a tensor to unit Frobenius norm.
    
    This function works with DenseTensor. An overload for BlockSparseTensor
    would have the same implementation pattern.
    """
    var norm = tensor_norm[dtype](tensor, ctx)
    if norm < 1e-14:
        raise Error("Cannot normalize tensor with zero norm")
    
    # Scale in-place
    var scale = Scalar[dtype](1.0 / norm)
    tensor_scale_in_place[dtype](tensor, scale, ctx)
    
    return tensor^


fn example_generic_algorithm() raises:
    """Example of writing algorithms using generic operations."""
    print("\n=== Example 3: Generic Algorithm ===")
    
    with DeviceContext() as ctx:
        # Create a tensor with random values
        var tensor: DenseTensor[DType.float32] = create_tensor[DType.float32](
            ctx, List[Int](10, 10)^
        )
        
        print("  Original norm:", tensor_norm[DType.float32](tensor, ctx))
        
        # Normalize it
        var normalized = normalize_tensor[DType.float32](tensor^, ctx)
        
        print("  Normalized norm:", tensor_norm[DType.float32](normalized, ctx))
        print("  (Should be ~1.0)")


# =============================================================================
# Example 4: Mixed Operations
# =============================================================================

fn example_mixed_operations() raises:
    """Demonstrate various tensor operations."""
    print("\n=== Example 4: Mixed Operations ===")
    
    with DeviceContext() as ctx:
        # Create a 3D tensor
        var tensor_3d: DenseTensor[DType.float32] = create_tensor[DType.float32](
            ctx, List[Int](2, 3, 4)^, init_value=Scalar[DType.float32](1.0)
        )
        print("  Original shape: 2 x 3 x 4")
        
        # Flatten dimensions 1 and 2
        var flattened = tensor_flatten_dims[DType.float32](tensor_3d^, 1, 3, ctx)
        print("  After flatten_dims(1, 3): ", flattened.get_shape()[0], "x", flattened.get_shape()[1])
        
        # Reshape to 2D
        var reshaped = tensor_reshape[DType.float32](flattened^, List[Int](2, 12)^)
        print("  After reshape to [2, 12]: ", reshaped.get_shape()[0], "x", reshaped.get_shape()[1])
        
        # Compute dot product with itself
        var tensor_copy = reshaped  # Shallow copy
        var dot_prod = tensor_dot_product[DType.float32](reshaped, tensor_copy, ctx)
        print("  Dot product with self:", dot_prod)


# =============================================================================
# Example 5: Showing Backend Selection
# =============================================================================

fn example_backend_info() raises:
    """Show how to query the current backend."""
    print("\n=== Example 5: Backend Information ===")
    
    print("  TensorBackendType:", TensorBackendType)
    print("  Backend name:", TensorBackend.name(TensorBackendType))
    print("  is_using_dense_backend():", is_using_dense_backend())
    print("  is_using_sparse_backend():", is_using_sparse_backend())
    
    # The Tensor type alias resolves at compile time
    @parameter
    if TensorBackendType == TensorBackend.DENSE:
        print("  Tensor alias -> DenseTensor")
    elif TensorBackendType == TensorBackend.BLOCK_SPARSE:
        print("  Tensor alias -> BlockSparseTensor")


# =============================================================================
# Main Entry Point
# =============================================================================

fn main() raises:
    """Run all examples."""
    print("=" * 60)
    print("Generic Tensor Operations Examples")
    print("=" * 60)
    
    example_backend_info()
    example_with_tensor_alias()
    example_explicit_dense()
    example_generic_algorithm()
    example_mixed_operations()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print("\nTo switch backends globally, change TensorBackendType in tensor_ops.mojo:")
    print("  alias TensorBackendType: Int = TensorBackend.DENSE        # Current")
    print("  alias TensorBackendType: Int = TensorBackend.BLOCK_SPARSE # Alternative")
