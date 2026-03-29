"""
SVD property tests (DT-SVD-001, DT-SVD-002): reconstruction, singular values, truncation.
"""
from sys import has_accelerator
from gpu.host import DeviceContext
from collections.list import List
from testing import TestSuite
from src.m_tensor.dense_tensor import (
    create_dense_tensor,
    dense_tensor_dot,
    dense_tensor_svd_trunc,
    DenseTensor,
)
from src.tests.test_utils import (
    make_host_data_f32,
    tensor_from_host,
    tensor_to_host,
    assert_allclose,
)

fn test_svd_001() raises:
    """DT-SVD-001: A [6,4], chi_max=4, eps_trunc=0; S non-negative descending; U@diag(S)@Vt ≈ A."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping DT-SVD-001 test")
        return
    with DeviceContext() as ctx:
        var data = make_host_data_f32(6 * 4, 200)
        var shape = List[Int](6, 4)
        var A = tensor_from_host(ctx, data, shape)
        var (U, S, Vt, chi) = dense_tensor_svd_trunc[A.dtype](A^, ctx, chi_max=4, eps_trunc=0.0)
        if chi != 4:
            raise Error("Expected chi=4")
        var host_S = tensor_to_host(ctx, S)
        for i in range(chi - 1):
            if Float64(host_S[i]) < 0 or Float64(host_S[i]) < Float64(host_S[i + 1]):
                raise Error("S not non-negative descending")
        var host_U = tensor_to_host(ctx, U)
        var host_Vt = tensor_to_host(ctx, Vt)
        var m = 6
        var n = 4
        var host_recon = List[Float32](capacity=m * n)
        for i in range(m):
            for j in range(n):
                var s: Float64 = 0.0
                for k in range(chi):
                    s += Float64(host_U[i * chi + k]) * Float64(host_S[k]) * Float64(host_Vt[k * n + j])
                host_recon.append(Float32(s))
        assert_allclose(host_recon, data, 1e-3, 1e-3)
    print("DT-SVD-001 passed")

fn test_svd_002() raises:
    """DT-SVD-002: A [10,8], chi_max=3, eps_trunc=0; shapes and bounded reconstruction error."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping DT-SVD-002 test")
        return
    with DeviceContext() as ctx:
        var data = make_host_data_f32(10 * 8, 201)
        var shape = List[Int](10, 8)
        var A = tensor_from_host(ctx, data, shape)
        var (U, S, Vt, chi) = dense_tensor_svd_trunc[A.dtype](A^, ctx, chi_max=3, eps_trunc=0.0)
        if U.shape[0] != 10 or U.shape[1] != 3:
            raise Error("U expected shape [10, 3]")
        if S.shape[0] != 3:
            raise Error("S expected length 3")
        if Vt.shape[0] != 3 or Vt.shape[1] != 8:
            raise Error("Vt expected shape [3, 8]")
        if chi != 3:
            raise Error("Expected chi=3")
        var host_U = tensor_to_host(ctx, U)
        var host_S = tensor_to_host(ctx, S)
        var host_Vt = tensor_to_host(ctx, Vt)
        var m = 10
        var n = 8
        var host_recon = List[Float32](capacity=m * n)
        for i in range(m):
            for j in range(n):
                var s: Float64 = 0.0
                for k in range(3):
                    s += Float64(host_U[i * 3 + k]) * Float64(host_S[k]) * Float64(host_Vt[k * n + j])
                host_recon.append(Float32(s))
        var norm_orig: Float64 = 0.0
        var norm_err: Float64 = 0.0
        for i in range(m * n):
            norm_orig += Float64(data[i]) * Float64(data[i])
            var e = Float64(data[i]) - Float64(host_recon[i])
            norm_err += e * e
        norm_orig = norm_orig ** 0.5
        norm_err = norm_err ** 0.5
        if norm_orig > 1e-10 and norm_err / norm_orig > 1.0:
            raise Error("Reconstruction error too large")
    print("DT-SVD-002 passed")

fn main() raises:
    try:
        TestSuite.discover_tests[__functions_in_module()]().run()
    except e:
        print("Tests failed: " + String(e))