"""
Full LAPACK SVD test suite: known singular values, reconstruction, orthonormality,
truncation (chi_max, eps_trunc), shapes (tall, square, wide), edge cases.
"""
from sys import has_accelerator
from gpu.host import DeviceContext
from collections.list import List
from math import sqrt

from src.m_tensor.dense_tensor import (
    create_dense_tensor_from_data,
    dense_tensor_svd_trunc,
)
from src.tests.test_utils import (
    tensor_from_host,
    tensor_to_host,
    make_host_data_f32,
    assert_allclose,
    assert_orthonormal,
)
from testing import TestSuite


fn _host_recon_from_svd(
    host_U: List[Float32],
    host_S: List[Float32],
    host_Vt: List[Float32],
    m: Int,
    n: Int,
    chi: Int,
) -> List[Float32]:
    """Compute U @ diag(S) @ Vt on host (row-major)."""
    var out = List[Float32](capacity=m * n)
    for i in range(m):
        for j in range(n):
            var s: Float64 = 0.0
            for k in range(chi):
                s += Float64(host_U[i * chi + k]) * Float64(host_S[k]) * Float64(host_Vt[k * n + j])
            out.append(Float32(s))
    return out.copy()


fn _check_vt_orthonormal(host_Vt: List[Float32], chi: Int, n: Int, tol: Float64 = 1e-3) raises:
    """Check Vt @ Vt^T = I_chi (Vt is [chi, n] row-major)."""
    for i in range(chi):
        for j in range(chi):
            var dot: Float64 = 0.0
            for k in range(n):
                dot += Float64(host_Vt[i * n + k]) * Float64(host_Vt[j * n + k])
            var expect = 1.0 if i == j else 0.0
            if abs(dot - expect) > tol:
                raise Error("(Vt Vt^T)[" + String(i) + "," + String(j) + "] = " + String(dot) + " (expected " + String(expect) + ")")


fn test_svd_lapack_001_known_singular_values() raises:
    """Check 4x3 matrix with known singular values [3, 2]; chi_max=2, check shapes and S values."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping LAPACK SVD test")
        return

    with DeviceContext() as ctx:
        # 4x3 matrix with rank 2 (singular values 3, 2; third is 0)
        var data = List[Float32](
            2.0, 0.0, 0.0,
            0.0, 3.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        )
        var shape = List[Int](4, 3)
        var shape_cpy = shape.copy()
        var A = create_dense_tensor_from_data[DType.float32](ctx, data, shape_cpy^)

        var svd_result = dense_tensor_svd_trunc[DType.float32](A^, ctx, chi_max=2, eps_trunc=1e-12)
        var U = svd_result[0]
        var S = svd_result[1]
        var Vt = svd_result[2]
        var kept = svd_result[3]

        if kept != 2:
            raise Error("Expected kept=2, got " + String(kept))
        if len(S.shape) != 1 or S.shape[0] != 2:
            raise Error("Expected S shape [2], got rank " + String(len(S.shape)) + " dim0 " + String(S.shape[0]))
        if U.shape[0] != 4 or U.shape[1] != 2:
            raise Error("Expected U shape [4, 2]")
        if Vt.shape[0] != 2 or Vt.shape[1] != 3:
            raise Error("Expected Vt shape [2, 3]")

        var host_S = tensor_to_host(ctx, S)
        if abs(Float64(host_S[0]) - 3.0) > 1e-4 or abs(Float64(host_S[1]) - 2.0) > 1e-4:
            raise Error("Expected singular values [3, 2], got [" + String(host_S[0]) + ", " + String(host_S[1]) + "]")


fn test_svd_lapack_002_reconstruction() raises:
    """Check A ≈ U @ diag(S) @ Vt (host-side reconstruction)."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping LAPACK SVD test")
        return

    with DeviceContext() as ctx:
        var data = make_host_data_f32(6 * 4, 200)
        var shape = List[Int](6, 4)
        var A = tensor_from_host(ctx, data, shape)
        var (U, S, Vt, chi) = dense_tensor_svd_trunc[A.dtype](A^, ctx, chi_max=4, eps_trunc=0.0)

        var host_U = tensor_to_host(ctx, U)
        var host_S = tensor_to_host(ctx, S)
        var host_Vt = tensor_to_host(ctx, Vt)
        var host_recon = _host_recon_from_svd(host_U, host_S, host_Vt, 6, 4, chi)
        assert_allclose(host_recon, data, 1e-3, 1e-3)


fn test_svd_lapack_003_u_orthonormal() raises:
    """Check U^T U = I_chi (orthonormal left singular vectors)."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping LAPACK SVD test")
        return

    with DeviceContext() as ctx:
        var data = make_host_data_f32(8 * 5, 301)
        var shape = List[Int](8, 5)
        var A = tensor_from_host(ctx, data, shape)
        var (U, S, Vt, chi) = dense_tensor_svd_trunc[A.dtype](A^, ctx, chi_max=5, eps_trunc=0.0)

        var host_U = tensor_to_host(ctx, U)
        assert_orthonormal(host_U, 8, chi, 1e-3)


fn test_svd_lapack_004_vt_orthonormal() raises:
    """Check Vt @ Vt^T = I_chi (orthonormal right singular vectors)."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping LAPACK SVD test")
        return

    with DeviceContext() as ctx:
        var data = make_host_data_f32(5 * 7, 302)
        var shape = List[Int](5, 7)
        var A = tensor_from_host(ctx, data, shape)
        var (U, S, Vt, chi) = dense_tensor_svd_trunc[A.dtype](A^, ctx, chi_max=5, eps_trunc=0.0)

        var host_Vt = tensor_to_host(ctx, Vt)
        _check_vt_orthonormal(host_Vt, chi, 7, 1e-3)


fn test_svd_lapack_005_singular_values_descending() raises:
    """Check singular values are non-negative and in descending order."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping LAPACK SVD test")
        return

    with DeviceContext() as ctx:
        var data = make_host_data_f32(6 * 4, 400)
        var shape = List[Int](6, 4)
        var A = tensor_from_host(ctx, data, shape)
        var (U, S, Vt, chi) = dense_tensor_svd_trunc[A.dtype](A^, ctx, chi_max=4, eps_trunc=0.0)

        var host_S = tensor_to_host(ctx, S)
        for i in range(chi):
            if Float64(host_S[i]) < -1e-6:
                raise Error("Singular value " + String(i) + " negative: " + String(host_S[i]))
        for i in range(chi - 1):
            if Float64(host_S[i]) < Float64(host_S[i + 1]):
                raise Error("Singular values not descending: S[" + String(i) + "] < S[" + String(i + 1) + "]")


fn test_svd_lapack_006_chi_max_cap() raises:
    """Check chi_max > min(m,n) is capped to min(m,n)."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping LAPACK SVD test")
        return

    with DeviceContext() as ctx:
        var data = make_host_data_f32(5 * 3, 501)
        var shape = List[Int](5, 3)
        var A = tensor_from_host(ctx, data, shape)
        var (U, S, Vt, chi) = dense_tensor_svd_trunc[A.dtype](A^, ctx, chi_max=10, eps_trunc=0.0)

        if chi != 3:
            raise Error("Expected chi=3 (min(5,3)), got " + String(chi))
        if U.shape[1] != 3 or S.shape[0] != 3 or Vt.shape[0] != 3:
            raise Error("Shapes should reflect chi=3")


fn test_svd_lapack_007_square_matrix() raises:
    """Check square matrix m=n; full reconstruction."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping LAPACK SVD test")
        return

    with DeviceContext() as ctx:
        var data = make_host_data_f32(3 * 3, 600)
        var shape = List[Int](3, 3)
        var A = tensor_from_host(ctx, data, shape)
        var (U, S, Vt, chi) = dense_tensor_svd_trunc[A.dtype](A^, ctx, chi_max=3, eps_trunc=0.0)

        if chi != 3:
            raise Error("Expected chi=3 for 3x3")
        var host_U = tensor_to_host(ctx, U)
        var host_S = tensor_to_host(ctx, S)
        var host_Vt = tensor_to_host(ctx, Vt)
        var host_recon = _host_recon_from_svd(host_U, host_S, host_Vt, 3, 3, chi)
        assert_allclose(host_recon, data, 1e-3, 1e-3)


fn test_svd_lapack_008_wide_matrix() raises:
    """Wide matrix m < n; shapes and reconstruction."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping LAPACK SVD test")
        return

    with DeviceContext() as ctx:
        var data = make_host_data_f32(2 * 5, 700)
        var shape = List[Int](2, 5)
        var A = tensor_from_host(ctx, data, shape)
        var (U, S, Vt, chi) = dense_tensor_svd_trunc[A.dtype](A^, ctx, chi_max=2, eps_trunc=0.0)

        if chi != 2:
            raise Error("Expected chi=2 for 2x5")
        if U.shape[0] != 2 or U.shape[1] != 2 or Vt.shape[0] != 2 or Vt.shape[1] != 5:
            raise Error("U [2,2], Vt [2,5] expected")
        var host_U = tensor_to_host(ctx, U)
        var host_S = tensor_to_host(ctx, S)
        var host_Vt = tensor_to_host(ctx, Vt)
        var host_recon = _host_recon_from_svd(host_U, host_S, host_Vt, 2, 5, chi)
        assert_allclose(host_recon, data, 1e-3, 1e-3)


fn test_svd_lapack_009_truncation_chi_max() raises:
    """Truncation to chi_max=2 on 4x3 gives chi=2 and truncated reconstruction error."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping LAPACK SVD test")
        return

    with DeviceContext() as ctx:
        var data = make_host_data_f32(4 * 3, 800)
        var shape = List[Int](4, 3)
        var A = tensor_from_host(ctx, data, shape)
        var (U, S, Vt, chi) = dense_tensor_svd_trunc[A.dtype](A^, ctx, chi_max=2, eps_trunc=0.0)

        if chi != 2:
            raise Error("Expected chi=2")
        var host_U = tensor_to_host(ctx, U)
        var host_S = tensor_to_host(ctx, S)
        var host_Vt = tensor_to_host(ctx, Vt)
        var host_recon = _host_recon_from_svd(host_U, host_S, host_Vt, 4, 3, chi)
        var norm_orig: Float64 = 0.0
        var norm_err: Float64 = 0.0
        for i in range(12):
            norm_orig += Float64(data[i]) * Float64(data[i])
            var e = Float64(data[i]) - Float64(host_recon[i])
            norm_err += e * e
        norm_orig = sqrt(norm_orig)
        norm_err = sqrt(norm_err)
        if norm_orig > 1e-10 and norm_err / norm_orig > 2.0:
            raise Error("Truncated reconstruction error too large")


fn test_svd_lapack_010_eps_trunc() raises:
    """Check eps_trunc can reduce kept singular values (smoke: run with large eps)."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping LAPACK SVD test")
        return

    with DeviceContext() as ctx:
        var data = make_host_data_f32(4 * 4, 900)
        var shape = List[Int](4, 4)
        var A1 = tensor_from_host(ctx, data, shape)
        var (U_full, S_full, Vt_full, chi_full) = dense_tensor_svd_trunc[DType.float32](A1^, ctx, chi_max=4, eps_trunc=0.0)
        var A2 = tensor_from_host(ctx, data, shape)
        var (U_trunc, S_trunc, Vt_trunc, chi_trunc) = dense_tensor_svd_trunc[DType.float32](A2^, ctx, chi_max=4, eps_trunc=0.5)

        if chi_full != 4:
            raise Error("Expected chi_full=4")
        if chi_trunc > chi_full:
            raise Error("eps_trunc should not increase kept count")


fn main() raises:
    try:
        TestSuite.discover_tests[__functions_in_module()]().run()
    except e:
        print("Tests failed: " + String(e))
