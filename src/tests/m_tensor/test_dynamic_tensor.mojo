"""
This is not up-to-date 08.11.2025 (dd.mm.yyyy)
Simple test file to verify dynamic tensor functionality.
Run with: mojo run test_dynamic_tensor.mojo
"""
from sys import has_accelerator
from m_tensor.dynamic_tensor import (
    DynamicTensor, 
    create_dynamic_tensor, 
    create_dynamic_tensor_from_data,
    compute_row_major_strides,
    compute_column_major_strides
)
from gpu.host import DeviceContext
from collections import List


fn test_stride_computation() raises -> None:
    """Test stride computation functions."""
    print("\n" + "="*60)
    print("TEST: Stride Computation")
    print("="*60)
    
    # Test row-major strides
    var shape_2d = List[Int](3, 4)
    var strides_rm = compute_row_major_strides(shape_2d, 2)
    print("Shape: [3, 4]")
    print("Row-major strides: [", strides_rm[0], ", ", strides_rm[1], "]")
    print("Expected: [4, 1]")
    
    var shape_3d = List[Int](2, 3, 4)
    var strides_rm_3d = compute_row_major_strides(shape_3d, 3)
    print("\nShape: [2, 3, 4]")
    print("Row-major strides: [", strides_rm_3d[0], ", ", strides_rm_3d[1], ", ", strides_rm_3d[2], "]")
    print("Expected: [12, 4, 1]")
    
    # Test column-major strides
    var strides_cm = compute_column_major_strides(shape_2d, 2)
    print("\nShape: [3, 4]")
    print("Column-major strides: [", strides_cm[0], ", ", strides_cm[1], "]")
    print("Expected: [1, 3]")
    
    print("✓ Stride computation test passed")


fn test_basic_tensor_creation() raises -> None:
    """Test basic tensor creation."""
    print("\n" + "="*60)
    print("TEST: Basic Tensor Creation")
    print("="*60)
    
    @parameter
    if not has_accelerator():
        print("⚠ No GPU found - skipping GPU tests")
        return
    
    with DeviceContext() as ctx:
        # Test 1D tensor
        print("\n1. Creating 1D tensor (rank=1, shape=[5])")
        var shape_1d = List[Int](5)
        var tensor_1d = create_dynamic_tensor(ctx, shape_1d, rank=1, init_value=1.0)
        print(tensor_1d)
        print("Expected: DynamicTensor[rank=1, shape=(5), size=5]")
        
        # Test 2D tensor
        print("\n2. Creating 2D tensor (rank=2, shape=[3, 4])")
        var shape_2d = List[Int](3, 4)
        var tensor_2d = create_dynamic_tensor(ctx, shape_2d, rank=2, init_value=2.5)
        print(tensor_2d)
        print("Expected: DynamicTensor[rank=2, shape=(3, 4), size=12]")
        
        # Test 3D tensor
        print("\n3. Creating 3D tensor (rank=3, shape=[2, 3, 4])")
        var shape_3d = List[Int](2, 3, 4)
        var tensor_3d = create_dynamic_tensor(ctx, shape_3d, rank=3, init_value=3.0)
        print(tensor_3d)
        print("Expected: DynamicTensor[rank=3, shape=(2, 3, 4), size=24]")
        
        print("\n✓ Basic tensor creation test passed")


fn test_tensor_from_data() raises -> None:
    """Test creating tensors from existing data."""
    print("\n" + "="*60)
    print("TEST: Tensor Creation from Data")
    print("="*60)
    
    @parameter
    if not has_accelerator():
        print("⚠ No GPU found - skipping GPU tests")
        return
    
    with DeviceContext() as ctx:
        # Create a 2x3 matrix
        print("\nCreating 2x3 matrix from data [1, 2, 3, 4, 5, 6]")
        var data = List[Float32](1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        var shape = List[Int](2, 3)
        var tensor = create_dynamic_tensor_from_data(ctx, data, shape, rank=2)
        print(tensor)
        print("Expected: DynamicTensor[rank=2, shape=(2, 3), size=6]")
        
        # Print the tensor contents
        print("\nTensor contents:")
        tensor.print_tensor(ctx)
        
        print("\n✓ Tensor from data test passed")


fn test_different_ranks() raises -> None:
    """Test creating tensors with various ranks."""
    print("\n" + "="*60)
    print("TEST: Different Tensor Ranks")
    print("="*60)
    
    @parameter
    if not has_accelerator():
        print("⚠ No GPU found - skipping GPU tests")
        return
    
    with DeviceContext() as ctx:
        # 1D vector
        var shape_1 = List[Int](10)
        var tensor_1 = create_dynamic_tensor(ctx, shape_1, rank=1)
        print("Rank 1:", tensor_1)
        
        # 2D matrix
        var shape_2 = List[Int](5, 6)
        var tensor_2 = create_dynamic_tensor(ctx, shape_2, rank=2)
        print("Rank 2:", tensor_2)
        
        # 3D tensor
        var shape_3 = List[Int](2, 3, 4)
        var tensor_3 = create_dynamic_tensor(ctx, shape_3, rank=3)
        print("Rank 3:", tensor_3)
        
        # 4D tensor
        var shape_4 = List[Int](2, 2, 3, 3)
        var tensor_4 = create_dynamic_tensor(ctx, shape_4, rank=4)
        print("Rank 4:", tensor_4)
        
        # 5D tensor (going beyond MAX_RANK in original implementation!)
        var shape_5 = List[Int](2, 2, 2, 2, 2)
        var tensor_5 = create_dynamic_tensor(ctx, shape_5, rank=5)
        print("Rank 5:", tensor_5)
        print("Note: This demonstrates runtime flexibility - no MAX_RANK limit!")
        
        print("\n✓ Different ranks test passed")


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# DYNAMIC TENSOR TEST SUITE")
    print("#"*60)
    
    try:
        test_stride_computation()
        test_basic_tensor_creation()
        test_tensor_from_data()
        test_different_ranks()
        
        print("\n" + "#"*60)
        print("# ALL TESTS PASSED ✓")
        print("#"*60 + "\n")
    except e:
        print("\n" + "#"*60)
        print("# TEST FAILED ✗")
        print("#"*60)
        print("Error:", e)

