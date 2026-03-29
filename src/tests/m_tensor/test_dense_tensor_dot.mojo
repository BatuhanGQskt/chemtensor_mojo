"""
Contract A tests for dense_tensor_dot (DT-DOT-001..006). No axis reversal; C-compatible.
"""
from sys import has_accelerator
from gpu.host import DeviceContext
from collections.list import List

from src.m_tensor.dense_tensor import (
    create_dense_tensor,
    dense_tensor_dot,
    DenseTensor,
)
from src.tests.test_utils import (
    make_host_data_f32,
    tensor_from_host,
    tensor_to_host,
    row_major_strides,
    assert_allclose,
    assert_shape_equal,
    linear_to_multi,
    multi_to_linear,
)
from testing import TestSuite

fn _host_dot_contract_a(
    host_A: List[Float32],
    shape_A: List[Int],
    host_B: List[Float32],
    shape_B: List[Int],
    ndim_mult: Int,
    axrange_A: Bool,
    axrange_B: Bool,
) -> List[Float32]:
    """Host reference: Contract A (aligned blocks, no reversal). Returns flat output row-major."""
    var rA = len(shape_A)
    var rB = len(shape_B)
    var off_A = 0 if axrange_A else rA - ndim_mult
    var off_B = 0 if axrange_B else rB - ndim_mult

    var out_shape = List[Int]()
    if axrange_A:
        for i in range(ndim_mult, rA):
            out_shape.append(shape_A[i])
    else:
        for i in range(rA - ndim_mult):
            out_shape.append(shape_A[i])
    if axrange_B:
        for i in range(ndim_mult, rB):
            out_shape.append(shape_B[i])
    else:
        for i in range(rB - ndim_mult):
            out_shape.append(shape_B[i])

    var strides_A = row_major_strides(shape_A)
    var strides_B = row_major_strides(shape_B)
    var out_size = 1
    for d in out_shape:
        out_size *= d
    var out = List[Float32](capacity=out_size)
    for _ in range(out_size):
        out.append(0.0)

    var n_contract = 1
    for i in range(ndim_mult):
        n_contract *= shape_A[off_A + i]

    var nA_noncon = rA - ndim_mult
    var nB_noncon = rB - ndim_mult

    for out_linear in range(out_size):
        var out_multi = linear_to_multi(out_linear, out_shape)
        var sum_val: Float64 = 0.0
        var c_multi = List[Int](capacity=ndim_mult)
        for _ in range(ndim_mult):
            c_multi.append(0)
        for c_linear in range(n_contract):
            var rem = c_linear
            for i in range(ndim_mult - 1, -1, -1):
                c_multi[i] = rem % shape_A[off_A + i]
                rem = rem // shape_A[off_A + i]

            var idx_A = List[Int](capacity=rA)
            for _ in range(rA):
                idx_A.append(0)
            for i in range(nA_noncon):
                if axrange_A:
                    idx_A[ndim_mult + i] = out_multi[i]
                else:
                    idx_A[i] = out_multi[i]
            for i in range(ndim_mult):
                idx_A[off_A + i] = c_multi[i]

            var idx_B = List[Int](capacity=rB)
            for _ in range(rB):
                idx_B.append(0)
            for i in range(nB_noncon):
                if axrange_B:
                    idx_B[ndim_mult + i] = out_multi[nA_noncon + i]
                else:
                    idx_B[i] = out_multi[nA_noncon + i]
            for i in range(ndim_mult):
                idx_B[off_B + i] = c_multi[i]

            sum_val += Float64(host_A[multi_to_linear(idx_A, strides_A)]) * Float64(host_B[multi_to_linear(idx_B, strides_B)])
        out[out_linear] = Float32(sum_val)
    return out.copy()

fn _host_dot_contract_a_simple_2d(host_A: List[Float32], m: Int, k: Int, host_B: List[Float32], n: Int) -> List[Float32]:
    """Simple 2D matmul: A [m,k] @ B [k,n] = C [m,n] (row-major)."""
    var out = List[Float32](capacity=m * n)
    for i in range(m):
        for j in range(n):
            var s: Float64 = 0.0
            for kk in range(k):
                s += Float64(host_A[i * k + kk]) * Float64(host_B[kk * n + j])
            out.append(Float32(s))
    return out.copy()

fn test_dot_001() raises:
    """DT-DOT-001: A [3,4], B [4,5], ndim_mult=1, axrange_A=False, axrange_B=True -> C [3,5]."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping DT-DOT-001 test")
        return
    with DeviceContext() as ctx:
        var data_A = make_host_data_f32(3 * 4, 10)
        var data_B = make_host_data_f32(4 * 5, 11)
        var A = tensor_from_host(ctx, data_A, List[Int](3, 4))
        var B = tensor_from_host(ctx, data_B, List[Int](4, 5))
        var C = create_dense_tensor[DType.float32](ctx, List[Int](3, 5)^, init_value=Float32(0.0))
        dense_tensor_dot(C, A^, B^, ctx, ndim_mult=1, axrange_A=False, axrange_B=True)
        var host_C = tensor_to_host(ctx, C)
        var expected = _host_dot_contract_a_simple_2d(data_A, 3, 4, data_B, 5)
        assert_allclose(host_C, expected)

fn test_dot_002() raises:
    """DT-DOT-002: Default (axrange_A=False, axrange_B=False) behaves as A TRAILING × B LEADING."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping DT-DOT-002 test")
        return
    with DeviceContext() as ctx:
        var data_A = make_host_data_f32(3 * 4, 20)
        var data_B = make_host_data_f32(4 * 5, 21)
        var A = tensor_from_host(ctx, data_A, List[Int](3, 4))
        var B = tensor_from_host(ctx, data_B, List[Int](4, 5))
        var C = create_dense_tensor[DType.float32](ctx, List[Int](3, 5)^, init_value=Float32(0.0))
        dense_tensor_dot(C, A^, B^, ctx)
        var host_C = tensor_to_host(ctx, C)
        var expected = _host_dot_contract_a_simple_2d(data_A, 3, 4, data_B, 5)
        assert_allclose(host_C, expected)

fn test_dot_003() raises:
    """DT-DOT-003: A [2,3,4], B [4,5,6], A TRAILING, B LEADING, ndim_mult=1 -> C [2,3,5,6]."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping DT-DOT-003 test")
        return
    with DeviceContext() as ctx:
        var data_A = make_host_data_f32(2 * 3 * 4, 30)
        var data_B = make_host_data_f32(4 * 5 * 6, 31)
        var A = tensor_from_host(ctx, data_A, List[Int](2, 3, 4))
        var B = tensor_from_host(ctx, data_B, List[Int](4, 5, 6))
        var C = create_dense_tensor[DType.float32](ctx, List[Int](2, 3, 5, 6)^, init_value=Float32(0.0))
        dense_tensor_dot(C, A^, B^, ctx, ndim_mult=1, axrange_A=False, axrange_B=True)
        var host_C = tensor_to_host(ctx, C)
        var expected = _host_dot_contract_a(data_A, List[Int](2, 3, 4), data_B, List[Int](4, 5, 6), 1, False, True)
        assert_allclose(host_C, expected)

fn test_dot_004() raises:
    """DT-DOT-004: A [2,3,4,5], B [4,5,6,7], ndim_mult=2, A TRAILING, B LEADING -> C [2,3,6,7]."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping DT-DOT-004 test")
        return
    with DeviceContext() as ctx:
        var data_A = make_host_data_f32(2 * 3 * 4 * 5, 40)
        var data_B = make_host_data_f32(4 * 5 * 6 * 7, 41)
        var A = tensor_from_host(ctx, data_A, List[Int](2, 3, 4, 5))
        var B = tensor_from_host(ctx, data_B, List[Int](4, 5, 6, 7))
        var C = create_dense_tensor[DType.float32](ctx, List[Int](2, 3, 6, 7)^, init_value=Float32(0.0))
        dense_tensor_dot(C, A^, B^, ctx, ndim_mult=2, axrange_A=False, axrange_B=True)
        var host_C = tensor_to_host(ctx, C)
        var expected = _host_dot_contract_a(
            data_A, List[Int](2, 3, 4, 5), data_B, List[Int](4, 5, 6, 7), 2, False, True
        )
        assert_allclose(host_C, expected)

fn test_dot_005a() raises:
    """DT-DOT-005A: A LEADING, B LEADING. A [4,3], B [4,5], ndim_mult=1 -> C [3,5]."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping DT-DOT-005A test")
        return
    with DeviceContext() as ctx:
        var data_A = make_host_data_f32(4 * 3, 50)
        var data_B = make_host_data_f32(4 * 5, 51)
        var A = tensor_from_host(ctx, data_A, List[Int](4, 3))
        var B = tensor_from_host(ctx, data_B, List[Int](4, 5))
        var C = create_dense_tensor[DType.float32](ctx, List[Int](3, 5)^, init_value=Float32(0.0))
        dense_tensor_dot(C, A^, B^, ctx, ndim_mult=1, axrange_A=True, axrange_B=True)
        var host_C = tensor_to_host(ctx, C)
        var expected = _host_dot_contract_a(data_A, List[Int](4, 3), data_B, List[Int](4, 5), 1, True, True)
        assert_allclose(host_C, expected)

fn test_dot_005b() raises:
    """DT-DOT-005B: A LEADING, B TRAILING. A [4,3], B [5,4], ndim_mult=1 -> C [3,5]."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping DT-DOT-005B test")
        return
    with DeviceContext() as ctx:
        var data_A = make_host_data_f32(4 * 3, 52)
        var data_B = make_host_data_f32(5 * 4, 53)
        var A = tensor_from_host(ctx, data_A, List[Int](4, 3))
        var B = tensor_from_host(ctx, data_B, List[Int](5, 4))
        var C = create_dense_tensor[DType.float32](ctx, List[Int](3, 5)^, init_value=Float32(0.0))
        dense_tensor_dot(C, A^, B^, ctx, ndim_mult=1, axrange_A=True, axrange_B=False)
        var host_C = tensor_to_host(ctx, C)
        var expected = _host_dot_contract_a(data_A, List[Int](4, 3), data_B, List[Int](5, 4), 1, True, False)
        assert_allclose(host_C, expected)

fn test_dot_005c() raises:
    """DT-DOT-005C: A TRAILING, B TRAILING. A [3,4], B [5,4], ndim_mult=1 -> C [3,5]."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping DT-DOT-005C test")
        return
    with DeviceContext() as ctx:
        var data_A = make_host_data_f32(3 * 4, 54)
        var data_B = make_host_data_f32(5 * 4, 55)
        var A = tensor_from_host(ctx, data_A, List[Int](3, 4))
        var B = tensor_from_host(ctx, data_B, List[Int](5, 4))
        var C = create_dense_tensor[DType.float32](ctx, List[Int](3, 5)^, init_value=Float32(0.0))
        dense_tensor_dot(C, A^, B^, ctx, ndim_mult=1, axrange_A=False, axrange_B=False)
        var host_C = tensor_to_host(ctx, C)
        var expected = _host_dot_contract_a(data_A, List[Int](3, 4), data_B, List[Int](5, 4), 1, False, False)
        assert_allclose(host_C, expected)

fn test_dot_007() raises:
    """DT-DOT-007: ndim_mult=2, A trailing × B leading (inferred). A [2,3,4,5], B [4,5,6,7], no axrange args -> C [2,3,6,7]."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping DT-DOT-007 test")
        return
    with DeviceContext() as ctx:
        var data_A = make_host_data_f32(2 * 3 * 4 * 5, 42)
        var data_B = make_host_data_f32(4 * 5 * 6 * 7, 43)
        var A = tensor_from_host(ctx, data_A, List[Int](2, 3, 4, 5))
        var B = tensor_from_host(ctx, data_B, List[Int](4, 5, 6, 7))
        var C = create_dense_tensor[DType.float32](ctx, List[Int](2, 3, 6, 7)^, init_value=Float32(0.0))
        dense_tensor_dot(C, A^, B^, ctx, ndim_mult=2)
        var host_C = tensor_to_host(ctx, C)
        var expected = _host_dot_contract_a(
            data_A, List[Int](2, 3, 4, 5), data_B, List[Int](4, 5, 6, 7), 2, False, True
        )
        assert_allclose(host_C, expected)

fn test_dot_008() raises:
    """DT-DOT-008: ndim_mult=2, A trailing × B trailing (inferred). A [2,3,4,5], B [6,7,4,5], axrange_A=False, axrange_B=False -> C [2,3,6,7]."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping DT-DOT-008 test")
        return
    with DeviceContext() as ctx:
        var data_A = make_host_data_f32(2 * 3 * 4 * 5, 44)
        var data_B = make_host_data_f32(6 * 7 * 4 * 5, 45)
        var A = tensor_from_host(ctx, data_A, List[Int](2, 3, 4, 5))
        var B = tensor_from_host(ctx, data_B, List[Int](6, 7, 4, 5))
        var C = create_dense_tensor[DType.float32](ctx, List[Int](2, 3, 6, 7)^, init_value=Float32(0.0))
        dense_tensor_dot(C, A^, B^, ctx, ndim_mult=2, axrange_A=False, axrange_B=False)
        var host_C = tensor_to_host(ctx, C)
        var expected = _host_dot_contract_a(
            data_A, List[Int](2, 3, 4, 5), data_B, List[Int](6, 7, 4, 5), 2, False, False
        )
        assert_allclose(host_C, expected)

fn test_dot_006() raises:
    """DT-DOT-006: Scalar result. A [7], B [7], ndim_mult=1 -> scalar; accept shape [1]."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping DT-DOT-006 test")
        return
    with DeviceContext() as ctx:
        var data_A = make_host_data_f32(7, 60)
        var data_B = make_host_data_f32(7, 61)
        var A = tensor_from_host(ctx, data_A, List[Int](7))
        var B = tensor_from_host(ctx, data_B, List[Int](7))
        var C = create_dense_tensor[DType.float32](ctx, List[Int](1)^, init_value=Float32(0.0))
        dense_tensor_dot(C, A^, B^, ctx, ndim_mult=1, axrange_A=False, axrange_B=True)
        var host_C = tensor_to_host(ctx, C)
        var expected_val: Float64 = 0.0
        for i in range(7):
            expected_val += Float64(data_A[i]) * Float64(data_B[i])
        if abs(Float64(host_C[0]) - expected_val) > 1e-4:
            raise Error("Scalar dot mismatch")

fn main() raises:
    try:
        TestSuite.discover_tests[__functions_in_module()]().run()
    except e:
        print("Tests failed: " + String(e))