"""
Replication of the C dense_tensor_dot test cases.

This test validates:
1. Matrix-Matrix Multiplication
2. Matrix-Vector Multiplication
3. Square Matrix Multiplication
4. Small 2x2 @ 2x2
5. Vector Inner Product
"""

from sys import has_accelerator
from collections.list import List
from gpu.host import DeviceContext
from src.m_tensor.dense_tensor import create_dense_tensor_from_data, create_dense_tensor, dense_tensor_dot

fn test_dense_tensor_dot() raises:
    print("\n\n")
    print("=========================================")
    print("Testing Tensor Dot Products")
    print("=========================================")
    
    @parameter
    if not has_accelerator():
        print("No compatible GPU found - skipping dense tensor dot tests")
        return

    with DeviceContext() as ctx:
        # Test 1: Simple Matrix-Matrix Multiplication (2x3) @ (3x2)
        print("\n--- Test 1: Matrix-Matrix Multiplication ---")
        
        # A = [[1, 2, 3],
        #      [4, 5, 6]]  (2x3)
        var A_data = List[Float32](1, 2, 3, 4, 5, 6)
        var A = create_dense_tensor_from_data[DType.float32](ctx, A_data, List[Int](2, 3)^)
        print("A:")
        A.print_tensor(ctx)
        
        # B = [[7, 8],
        #      [9, 10],
        #      [11, 12]]  (3x2)
        var B_data = List[Float32](7, 8, 9, 10, 11, 12)
        var B = create_dense_tensor_from_data[DType.float32](ctx, B_data, List[Int](3, 2)^)
        print("B:")
        B.print_tensor(ctx)
        
        # C = A @ B (should be 2x2)
        # Expected: [[58, 64], [139, 154]]
        var C = create_dense_tensor[DType.float32](ctx, List[Int](2, 2)^, init_value=Scalar[DType.float32](0.0))
        dense_tensor_dot[DType.float32](C, A^, B^, ctx, ndim_mult=1)
        
        print("\nResult C = A @ B:")
        C.print_tensor(ctx)
        print("Expected: [[58.0, 64.0], [139.0, 154.0]]")
        
        
        # Test 2: Matrix-Vector Multiplication
        print("\n--- Test 2: Matrix-Vector Multiplication ---")
        
        # M = [[1, 2, 3],
        #      [4, 5, 6]]  (2x3)
        var M_data = List[Float32](1, 2, 3, 4, 5, 6)
        var M = create_dense_tensor_from_data[DType.float32](ctx, M_data, List[Int](2, 3)^)
        print("M:")
        M.print_tensor(ctx)
        
        # v = [10, 20, 30]  (3,)
        var v_data = List[Float32](10, 20, 30)
        var v = create_dense_tensor_from_data[DType.float32](ctx, v_data, List[Int](3)^)
        print("v:")
        v.print_tensor(ctx)
        
        # result = M @ v (should be 2,)
        # Expected: [140, 320]
        var result = create_dense_tensor[DType.float32](ctx, List[Int](2)^, init_value=Scalar[DType.float32](0.0))
        print("result:")
        dense_tensor_dot[DType.float32](result, M^, v^, ctx, ndim_mult=1)
        
        print("\nResult = M @ v:")
        result.print_tensor(ctx)
        print("Expected: [140.0, 320.0]")
        
        
        # Test 3: Square Matrix Multiplication
        print("\n--- Test 3: Square Matrix (3x3) @ (3x3) ---")
        
        # X = [[1, 0, 0],
        #      [0, 2, 0],
        #      [0, 0, 3]]  (3x3) - diagonal
        var X_data = List[Float32](1, 0, 0, 0, 2, 0, 0, 0, 3)
        var X = create_dense_tensor_from_data[DType.float32](ctx, X_data, List[Int](3, 3)^)
        print("X:")
        X.print_tensor(ctx)
        
        # Y = [[1, 2, 3],
        #      [4, 5, 6],
        #      [7, 8, 9]]  (3x3)
        var Y_data = List[Float32](1, 2, 3, 4, 5, 6, 7, 8, 9)
        var Y = create_dense_tensor_from_data[DType.float32](ctx, Y_data, List[Int](3, 3)^)
        print("Y:")
        Y.print_tensor(ctx)
        
        # Z = X @ Y (should be 3x3)
        # Expected: [[1, 2, 3], [8, 10, 12], [21, 24, 27]]
        var Z = create_dense_tensor[DType.float32](ctx, List[Int](3, 3)^, init_value=Scalar[DType.float32](0.0))
        dense_tensor_dot[DType.float32](Z, X^, Y^, ctx, ndim_mult=1)
        
        print("\nResult Z = X @ Y:")
        Z.print_tensor(ctx)
        print("Expected: [[1.0, 2.0, 3.0], [8.0, 10.0, 12.0], [21.0, 24.0, 27.0]]")
        
        
        # Test 4: Small example for manual verification
        print("\n--- Test 4: Small 2x2 @ 2x2 ---")
        
        # P = [[1, 2],
        #      [3, 4]]  (2x2)
        var P_data = List[Float32](1, 2, 3, 4)
        var P = create_dense_tensor_from_data[DType.float32](ctx, P_data, List[Int](2, 2)^)
        print("P:")
        P.print_tensor(ctx)
        
        # Q = [[5, 6],
        #      [7, 8]]  (2x2)
        var Q_data = List[Float32](5, 6, 7, 8)
        var Q = create_dense_tensor_from_data[DType.float32](ctx, Q_data, List[Int](2, 2)^)
        print("Q:")
        Q.print_tensor(ctx)
        
        # R = P @ Q (should be 2x2)
        # Manual: [1*5+2*7, 1*6+2*8] = [19, 22]
        #         [3*5+4*7, 3*6+4*8] = [43, 50]
        var R = create_dense_tensor[DType.float32](ctx, List[Int](2, 2)^, init_value=Scalar[DType.float32](0.0))
        dense_tensor_dot[DType.float32](R, P^, Q^, ctx, ndim_mult=1)
        
        print("\nResult R = P @ Q:")
        R.print_tensor(ctx)
        print("Expected: [[19.0, 22.0], [43.0, 50.0]]")
        
        
        # Test 5: Vector dot product (inner product)
        print("\n--- Test 5: Vector Inner Product ---")
        
        # u = [1, 2, 3, 4]
        var u_data = List[Float32](1, 2, 3, 4)
        var u = create_dense_tensor_from_data[DType.float32](ctx, u_data, List[Int](4)^)
        print("u:")
        u.print_tensor(ctx)
        
        # w = [5, 6, 7, 8]
        var w_data = List[Float32](5, 6, 7, 8)
        var w = create_dense_tensor_from_data[DType.float32](ctx, w_data, List[Int](4)^)
        print("w:")
        w.print_tensor(ctx)
        
        # dot_result = u · w (should be scalar, represented as 0-d tensor)
        # Expected: 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        var dot_result = create_dense_tensor[DType.float32](ctx, List[Int](1)^, init_value=Scalar[DType.float32](0.0))
        dense_tensor_dot[DType.float32](dot_result, u^, w^, ctx, ndim_mult=1)
        
        print("\nResult = u · w:")
        dot_result.print_tensor(ctx)
        print("Expected: 70.0")
        
    print("\n=== All dot product tests completed ===\n")

fn main():
    try:
        test_dense_tensor_dot()
    except e:
        print("Test failed: " + String(e))
