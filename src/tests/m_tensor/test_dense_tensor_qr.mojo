"""
QR property tests: full coverage of dense_tensor_qr.

- DT-QR-001, 002: tall and square matrices, Q^T Q ≈ I, Q @ R ≈ A, R upper triangular.
- DT-QR-003: non-2D input raises.
- DT-QR-004: wide matrix (m < n): Q [m,k], R [k,n], k = min(m,n).
- DT-QR-005: square matrix full reconstruction.
- DT-QR-006: small 2×2 edge case.
"""
from sys import has_accelerator
from gpu.host import DeviceContext
from collections.list import List

from src.m_tensor.dense_tensor import (
    create_dense_tensor,
    create_dense_tensor_from_data,
    dense_tensor_dot,
    dense_tensor_qr,
    DenseTensor,
)
from src.tests.test_utils import (
    make_host_data_f32,
    tensor_from_host,
    tensor_to_host,
    assert_allclose,
    assert_upper_triangular,
    assert_orthonormal,
)
from testing import TestSuite

fn test_qr_001() raises:
    """DT-QR-001: A [6,4]; (Q,R)=qr(A); Q [m,k], R [k,n]; Q^T Q ≈ I; Q @ R ≈ A."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping DT-QR-001 test")
        return
    with DeviceContext() as ctx:
        var data = make_host_data_f32(6 * 4, 100)
        var shape = List[Int](6, 4)
        var A = tensor_from_host(ctx, data, shape)
        var (Q, R) = dense_tensor_qr[A.dtype](A^, ctx)
        if len(Q.shape) != 2 or Q.shape[0] != 6 or Q.shape[1] != 4:
            raise Error("Q expected shape [6, 4]")
        if len(R.shape) != 2 or R.shape[0] != 4 or R.shape[1] != 4:
            raise Error("R expected shape [4, 4]")
        var host_Q = tensor_to_host(ctx, Q)
        var host_R = tensor_to_host(ctx, R)
        assert_orthonormal(host_Q, 6, 4, 1e-3)
        assert_upper_triangular(host_R, 4, 4, 1e-3)
        var Q2 = tensor_from_host(ctx, host_Q, List[Int](6, 4))
        var R2 = tensor_from_host(ctx, host_R, List[Int](4, 4))
        var QR = create_dense_tensor[DType.float32](ctx, List[Int](6, 4)^, init_value=Float32(0.0))
        dense_tensor_dot(QR, Q2^, R2^, ctx)
        var host_QR = tensor_to_host(ctx, QR)
        assert_allclose(host_QR, data, 1e-3, 1e-3)

fn test_qr_002() raises:
    """DT-QR-002: R is upper triangular (R[i,j] ~ 0 for i > j)."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping DT-QR-002 test")
        return
    with DeviceContext() as ctx:
        var data = make_host_data_f32(5 * 5, 101)
        var shape = List[Int](5, 5)
        var A = tensor_from_host(ctx, data, shape)
        var (_, R) = dense_tensor_qr[A.dtype](A^, ctx)
        var host_R = tensor_to_host(ctx, R)
        assert_upper_triangular(host_R, 5, 5, 1e-3)


fn test_qr_003_non_2d_raises() raises:
    """DT-QR-003: dense_tensor_qr raises for non-2D input (rank check)."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping DT-QR-003 test")
        return
    with DeviceContext() as ctx:
        var data = make_host_data_f32(6, 102)
        var shape_1d = List[Int](6)
        var A = tensor_from_host(ctx, data, shape_1d)
        var raised = False
        var err_msg = ""
        try:
            var (_q, _r) = dense_tensor_qr[A.dtype](A^, ctx)
        except e:
            raised = True
            err_msg = String(e)
        if not raised:
            raise Error("Expected dense_tensor_qr to raise for 1D input")
        if "2D matrix" not in err_msg and "rank" not in err_msg:
            raise Error("Expected QR to complain about 2D/rank, got: " + err_msg)


fn test_qr_004_wide_matrix() raises:
    """DT-QR-004: Wide matrix (m < n): A [4,6] -> Q [4,4], R [4,6]; Q^T Q ≈ I, R upper, Q@R ≈ A."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping DT-QR-004 test")
        return
    with DeviceContext() as ctx:
        var m = 4
        var n = 6
        var k = 4  # min(m,n)
        var data = make_host_data_f32(m * n, 103)
        var shape = List[Int](m, n)
        var A = tensor_from_host(ctx, data, shape)
        var (Q, R) = dense_tensor_qr[A.dtype](A^, ctx)
        if len(Q.shape) != 2 or Q.shape[0] != m or Q.shape[1] != k:
            raise Error("Q expected shape [4, 4], got [" + String(Q.shape[0]) + ", " + String(Q.shape[1]) + "]")
        if len(R.shape) != 2 or R.shape[0] != k or R.shape[1] != n:
            raise Error("R expected shape [4, 6], got [" + String(R.shape[0]) + ", " + String(R.shape[1]) + "]")
        var host_Q = tensor_to_host(ctx, Q)
        var host_R = tensor_to_host(ctx, R)
        assert_orthonormal(host_Q, m, k, 1e-3)
        assert_upper_triangular(host_R, k, n, 1e-3)
        var Q2 = tensor_from_host(ctx, host_Q, List[Int](m, k))
        var R2 = tensor_from_host(ctx, host_R, List[Int](k, n))
        var QR = create_dense_tensor[DType.float32](ctx, List[Int](m, n)^, init_value=Float32(0.0))
        dense_tensor_dot(QR, Q2^, R2^, ctx)
        var host_QR = tensor_to_host(ctx, QR)
        assert_allclose(host_QR, data, 1e-3, 1e-3)


fn test_qr_005_square_full() raises:
    """DT-QR-005: Square 5×5: full reconstruction Q@R ≈ A and orthonormal Q."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping DT-QR-005 test")
        return
    with DeviceContext() as ctx:
        var data = make_host_data_f32(5 * 5, 104)
        var shape = List[Int](5, 5)
        var A = tensor_from_host(ctx, data, shape)
        var (Q, R) = dense_tensor_qr[A.dtype](A^, ctx)
        var host_Q = tensor_to_host(ctx, Q)
        var host_R = tensor_to_host(ctx, R)
        assert_orthonormal(host_Q, 5, 5, 1e-3)
        assert_upper_triangular(host_R, 5, 5, 1e-3)
        var Q2 = tensor_from_host(ctx, host_Q, List[Int](5, 5))
        var R2 = tensor_from_host(ctx, host_R, List[Int](5, 5))
        var QR = create_dense_tensor[DType.float32](ctx, List[Int](5, 5)^, init_value=Float32(0.0))
        dense_tensor_dot(QR, Q2^, R2^, ctx)
        var host_QR = tensor_to_host(ctx, QR)
        assert_allclose(host_QR, data, 1e-3, 1e-3)


fn test_qr_006_small_2x2() raises:
    """DT-QR-006: Small 2×2 matrix; k=2, Q [2,2], R [2,2]; full properties."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping DT-QR-006 test")
        return
    with DeviceContext() as ctx:
        var data = make_host_data_f32(4, 105)
        var shape = List[Int](2, 2)
        var A = tensor_from_host(ctx, data, shape)
        var (Q, R) = dense_tensor_qr[A.dtype](A^, ctx)
        if Q.shape[0] != 2 or Q.shape[1] != 2 or R.shape[0] != 2 or R.shape[1] != 2:
            raise Error("Shapes expected [2,2] for Q and R")
        var host_Q = tensor_to_host(ctx, Q)
        var host_R = tensor_to_host(ctx, R)
        assert_orthonormal(host_Q, 2, 2, 1e-3)
        assert_upper_triangular(host_R, 2, 2, 1e-3)
        var Q2 = tensor_from_host(ctx, host_Q, List[Int](2, 2))
        var R2 = tensor_from_host(ctx, host_R, List[Int](2, 2))
        var QR = create_dense_tensor[DType.float32](ctx, List[Int](2, 2)^, init_value=Float32(0.0))
        dense_tensor_dot(QR, Q2^, R2^, ctx)
        var host_QR = tensor_to_host(ctx, QR)
        assert_allclose(host_QR, data, 1e-3, 1e-3)


fn main() raises:
    try:
        TestSuite.discover_tests[__functions_in_module()]().run()
    except e:
        print("Tests failed: " + String(e))
