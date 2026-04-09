"""Test suite for the tensor abstraction layer.

This file demonstrates how to use the tensor abstraction layer to write
backend-agnostic code that works with both DenseTensor and BlockSparseTensor.

To switch backends globally, modify TensorBackendType in tensor_ops.mojo:
    alias TensorBackendType: Int = TensorBackend.DENSE        # For dense
    alias TensorBackendType: Int = TensorBackend.BLOCK_SPARSE # For block sparse
"""

from gpu.host import DeviceContext
from collections.list import List
from testing import assert_true, assert_equal

# Import the abstraction layer
from src.m_tensor.tensor_ops import (
    # Compile-time backend selection
    TensorBackendType,
    Tensor,  # Type alias resolved at compile time based on TensorBackendType
    
    # Convenience aliases (dispatch based on TensorBackendType)
    tensor_dot,
    tensor_qr,
    tensor_svd_trunc,
    create_tensor,
    create_tensor_uninitialized,
    create_tensor_from_data,
    
    # Explicit backend-specific functions
    tensor_dot_dense,
    tensor_dot_sparse,
    create_tensor_dense,
    
    # Backend query
    is_using_dense_backend,
    is_using_sparse_backend,
)

# Direct imports for explicit backend usage
from src.m_tensor.dense_tensor import DenseTensor
from src.m_tensor.block_sparse_tensor import BlockSparseTensor
from src.m_tensor.tensor_traits import TensorOps, TensorBackend


# =============================================================================
# Test: Trait Method Verification
# =============================================================================

fn test_tensor_trait_methods() raises:
    """Test that DenseTensor implements all TensorOps trait methods."""
    print("Testing TensorOps trait methods...")
    
    with DeviceContext() as ctx:
        # Create a test tensor
        var tensor = create_tensor[DType.float32](
            ctx, List[Int](3, 4, 5)^, init_value=Scalar[DType.float32](1.0)
        )
        
        # Test get_shape
        var shape = tensor.get_shape()
        assert_equal(len(shape), 3)
        assert_equal(shape[0], 3)
        assert_equal(shape[1], 4)
        assert_equal(shape[2], 5)
        
        # Test get_stride
        var stride = tensor.get_stride()
        assert_equal(len(stride), 3)
        assert_equal(stride[0], 20)  # 4 * 5
        assert_equal(stride[1], 5)   # 5
        assert_equal(stride[2], 1)   # 1
        
        # Test get_size
        assert_equal(tensor.get_size(), 60)  # 3 * 4 * 5
        
        # Test get_rank
        assert_equal(tensor.get_rank(), 3)
        
        # Test is_contiguous
        assert_true(tensor.is_contiguous())
        
        # Test compute_norm (60 ones should have norm = sqrt(60))
        var norm = tensor.compute_norm(ctx)
        var expected_norm = 7.745966692414834  # sqrt(60)
        assert_true(abs(norm - expected_norm) < 0.01)
        
        # Test compute_norm_sq
        var norm_sq = tensor.compute_norm_sq(ctx)
        assert_true(abs(norm_sq - 60.0) < 0.01)
        
        print("  All trait methods passed!")


# =============================================================================
# Test: Backend Identification
# =============================================================================

fn test_backend_identification() raises:
    """Test that backend type can be identified."""
    print("Testing backend identification...")
    
    # Test runtime backend identification
    var backend = DenseTensor[DType.float32].backend()
    assert_true(backend.is_dense())
    assert_true(not backend.is_block_sparse())
    assert_true(not backend.is_complex())
    
    # Test compile-time backend selection
    print("  Current backend (compile-time):", TensorBackend.name(TensorBackendType))
    print("  is_using_dense_backend():", is_using_dense_backend())
    print("  is_using_sparse_backend():", is_using_sparse_backend())
    
    # Verify consistency
    @parameter
    if TensorBackendType == TensorBackend.DENSE:
        assert_true(is_using_dense_backend())
        assert_true(not is_using_sparse_backend())
    elif TensorBackendType == TensorBackend.BLOCK_SPARSE:
        assert_true(not is_using_dense_backend())
        assert_true(is_using_sparse_backend())
    
    print("  Backend identification passed!")


# =============================================================================
# Test: Generic Tensor Operations
# =============================================================================

fn test_generic_tensor_dot() raises:
    """Test generic tensor_dot function."""
    print("Testing generic tensor_dot...")
    
    with DeviceContext() as ctx:
        # Create input tensors
        var A = create_tensor[DType.float32](
            ctx, List[Int](3, 4)^, init_value=Scalar[DType.float32](2.0)
        )
        var B = create_tensor[DType.float32](
            ctx, List[Int](4, 5)^, init_value=Scalar[DType.float32](3.0)
        )
        var C = create_tensor_uninitialized[DType.float32](
            ctx, List[Int](3, 5)^
        )
        
        # Perform contraction
        tensor_dot[DType.float32](C, A^, B^, ctx)
        
        # Verify result (each element should be 2 * 3 * 4 = 24)
        var host_out = ctx.enqueue_create_host_buffer[DType.float32](15)
        ctx.enqueue_copy(host_out, C.storage)
        ctx.synchronize()
        
        for i in range(15):
            assert_true(abs(Float64(host_out[i]) - 24.0) < 0.01)
        
        print("  Generic tensor_dot passed!")


fn test_generic_tensor_qr() raises:
    """Test generic tensor_qr function."""
    print("Testing generic tensor_qr...")
    
    with DeviceContext() as ctx:
        # Create a simple matrix
        var data = List[Scalar[DType.float32]]()
        data.append(1.0)
        data.append(2.0)
        data.append(3.0)
        data.append(4.0)
        data.append(5.0)
        data.append(6.0)
        
        var matrix = create_tensor_from_data[DType.float32](
            ctx, data, List[Int](2, 3)^
        )
        
        # Perform QR decomposition
        var qr_result = tensor_qr[DType.float32](matrix^, ctx)
        var Q = qr_result[0]
        var R = qr_result[1]
        
        # Check dimensions
        assert_equal(Q.get_shape()[0], 2)  # m
        assert_equal(Q.get_shape()[1], 2)  # min(m, n)
        assert_equal(R.get_shape()[0], 2)  # min(m, n)
        assert_equal(R.get_shape()[1], 3)  # n
        
        print("  Generic tensor_qr passed!")


# =============================================================================
# Test: Type Alias Usage
# =============================================================================

fn test_type_alias_usage() raises:
    """Test that Tensor type alias works correctly."""
    print("Testing type alias usage...")
    
    with DeviceContext() as ctx:
        # Using the Tensor alias - resolves to DenseTensor or BlockSparseTensor
        # based on TensorBackendType at compile time
        @parameter
        if TensorBackendType == TensorBackend.DENSE:
            var tensor: Tensor[DType.float32] = create_tensor[DType.float32](
                ctx, List[Int](2, 3)^, init_value=Scalar[DType.float32](0.5)
            )
            
            # Should work with all standard operations via trait methods
            var shape = tensor.get_shape()
            assert_equal(shape[0], 2)
            assert_equal(shape[1], 3)
            
            var norm = tensor.compute_norm(ctx)
            assert_true(norm > 0)
            
            print("  Type alias resolved to: DenseTensor")
        else:
            # BlockSparseTensor would need quantum number structure
            print("  Type alias resolved to: BlockSparseTensor (skipping creation test)")
        
        print("  Type alias usage passed!")


# =============================================================================
# Test: Writing Backend-Agnostic Algorithms
# =============================================================================

fn example_backend_agnostic_function[dtype: DType](
    var tensor: DenseTensor[dtype],
    ctx: DeviceContext,
) raises -> DenseTensor[dtype]:
    """Example of a backend-agnostic function using traits.
    
    This function demonstrates how to write code that works with any
    tensor type implementing TensorOps.
    
    When BlockSparseTensor is implemented, this function can be made
    fully generic by changing the type parameter to use a trait bound.
    """
    # Use trait methods for portable operations
    var shape = tensor.get_shape()
    var rank = tensor.get_rank()
    
    # Verify it's a matrix
    if rank != 2:
        raise Error("Expected a matrix, got rank " + String(rank))
    
    # Compute and print norm
    var norm = tensor.compute_norm(ctx)
    print("  Input tensor norm:", norm)
    
    # Return transposed tensor
    var perm = List[Int](1, 0)
    return tensor^.transpose(perm, ctx)


fn test_backend_agnostic_algorithm() raises:
    """Test writing backend-agnostic algorithms."""
    print("Testing backend-agnostic algorithm...")
    
    with DeviceContext() as ctx:
        var input_tensor = create_tensor[DType.float32](
            ctx, List[Int](3, 4)^, init_value=Scalar[DType.float32](1.0)
        )
        
        var result = example_backend_agnostic_function[DType.float32](
            input_tensor^, ctx
        )
        
        # Verify transpose worked
        var result_shape = result.get_shape()
        assert_equal(result_shape[0], 4)
        assert_equal(result_shape[1], 3)
        
        print("  Backend-agnostic algorithm passed!")


# =============================================================================
# Main Test Runner
# =============================================================================

fn main() raises:
    """Run all tensor abstraction tests."""
    print("=" * 60)
    print("Tensor Abstraction Layer Tests")
    print("=" * 60)
    
    test_tensor_trait_methods()
    test_backend_identification()
    test_generic_tensor_dot()
    test_generic_tensor_qr()
    test_type_alias_usage()
    test_backend_agnostic_algorithm()
    
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
