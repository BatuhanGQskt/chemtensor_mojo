"""
DT-STRIDE, DT-RESHAPE, DT-FLATTEN, DT-TRANSPOSE, DT-NORM, DT-SCALE (TEST_PLAN_DENSE_TENSOR).
"""
from sys import has_accelerator
from gpu.host import DeviceContext
from collections.list import List
from testing import TestSuite
from src.m_tensor.dense_tensor import (
    DenseTensor,
    create_dense_tensor,
    create_dense_tensor_from_data,
    compute_row_major_strides,
)
from src.tests.test_utils import (
    make_host_data_f32,
    tensor_from_host,
    tensor_to_host,
    row_major_strides,
    assert_allclose,
    assert_shape_equal,
)

fn test_stride_001() raises:
    """DT-STRIDE-001: shape [3,4] -> stride [4,1], [2,3,4] -> [12,4,1]; create_dense_tensor(row_major=True) contiguous."""
    var s24 = List[Int](3, 4)
    var strides_24 = row_major_strides(s24)
    if strides_24[0] != 4 or strides_24[1] != 1:
        raise Error("Stride [3,4] expected [4,1]")

    var s234 = List[Int](2, 3, 4)
    var strides_234 = row_major_strides(s234)
    if strides_234[0] != 12 or strides_234[1] != 4 or strides_234[2] != 1:
        raise Error("Stride [2,3,4] expected [12,4,1]")

    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping DT-STRIDE-001 test")
        return
    with DeviceContext() as ctx:
        var t = create_dense_tensor[DType.float32](ctx, s234^, row_major=True)
        if not t.is_contiguous():
            raise Error("Expected contiguous tensor")

fn test_stride_002() raises:
    """DT-STRIDE-002: After transpose(perm), result contiguous with correct row-major strides for new shape."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping DT-STRIDE-002 test")
        return
    with DeviceContext() as ctx:
        var data = make_host_data_f32(2 * 3 * 4, 42)
        var shape = List[Int](2, 3, 4)
        var t = tensor_from_host(ctx, data, shape)
        var perm = List[Int](0, 2, 1)
        var out = t.transpose(perm, ctx)
        if not out.is_contiguous():
            raise Error("Expected contiguous after transpose")
        var expected_shape = List[Int](2, 4, 3)
        assert_shape_equal(out.shape, expected_shape)
        var expected_strides = row_major_strides(expected_shape)
        assert_shape_equal(out.stride, expected_strides)

fn test_reshape_001() raises:
    """DT-RESHAPE-001: Contiguous [2,3,4] reshape to [6,4] preserves element order."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping DT-RESHAPE-001 test")
        return
    with DeviceContext() as ctx:
        var data = make_host_data_f32(2 * 3 * 4, 1)
        var shape = List[Int](2, 3, 4)
        var t = tensor_from_host(ctx, data, shape)
        var new_shape = List[Int](6, 4)
        var new_shape_cpy = new_shape.copy()
        var r = t.reshape(new_shape_cpy^)
        var host_r = tensor_to_host(ctx, r)
        assert_allclose(host_r, data)

fn test_flatten_001() raises:
    """DT-FLATTEN-001: Contiguous [2,3,4,5] flatten_dims(1,3) -> [2,12,5], element order preserved."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping DT-FLATTEN-001 test")
        return
    with DeviceContext() as ctx:
        var data = make_host_data_f32(2 * 3 * 4 * 5, 2)
        var shape = List[Int](2, 3, 4, 5)
        var t = tensor_from_host(ctx, data, shape)
        var flat = t.flatten_dims(1, 3, ctx)
        assert_shape_equal(flat.shape, List[Int](2, 12, 5))
        var host_flat = tensor_to_host(ctx, flat)
        assert_allclose(host_flat, data)

fn test_transpose_001() raises:
    """DT-TRANSPOSE-001: 2D shape [2,3], perm [1,0], compare to host transpose."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping DT-TRANSPOSE-001 test")
        return
    with DeviceContext() as ctx:
        var data = make_host_data_f32(6, 3)
        var shape = List[Int](2, 3)
        var t = tensor_from_host(ctx, data, shape)
        var perm = List[Int](1, 0)
        var out = t.transpose(perm, ctx)
        var host_out = tensor_to_host(ctx, out)
        for i in range(2):
            for j in range(3):
                if host_out[j * 2 + i] != data[i * 3 + j]:
                    raise Error("Transpose value mismatch")

fn test_transpose_002() raises:
    """DT-TRANSPOSE-002: Rank-4 [2,2,3,4], perm [0,2,1,3], compare to host reference."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping DT-TRANSPOSE-002 test")
        return
    with DeviceContext() as ctx:
        var n = 2 * 2 * 3 * 4
        var data = make_host_data_f32(n, 4)
        var shape = List[Int](2, 2, 3, 4)
        var t = tensor_from_host(ctx, data, shape)
        var perm = List[Int](0, 2, 1, 3)
        var out = t.transpose(perm, ctx)
        var strides = row_major_strides(shape)
        var new_shape = List[Int](2, 3, 2, 4)
        var new_strides = row_major_strides(new_shape)
        var host_out = tensor_to_host(ctx, out)
        for i in range(n):
            var multi = List[Int](capacity=4)
            var rem = i
            for d in range(4):
                multi.append(rem // new_strides[d])
                rem = rem % new_strides[d]
            var src_i = multi[0] * strides[0] + multi[2] * strides[1] + multi[1] * strides[2] + multi[3] * strides[3]
            if abs(Float64(host_out[i] - data[src_i])) >= 1e-5:
                raise Error("Transpose value mismatch at " + String(i))

fn test_norm_001() raises:
    """DT-NORM-001: Frobenius norm vs host-computed."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping DT-NORM-001 test")
        return
    with DeviceContext() as ctx:
        var data = make_host_data_f32(12, 5)
        var shape = List[Int](3, 4)
        var t = tensor_from_host(ctx, data, shape)
        var norm_mojo = t.norm(ctx)
        var norm_host: Float64 = 0.0
        for i in range(12):
            var x = Float64(data[i])
            norm_host += x * x
        norm_host = norm_host ** 0.5
        if abs(norm_mojo - norm_host) > 1e-4:
            raise Error("Norm mismatch: " + String(norm_mojo) + " vs " + String(norm_host))

fn test_scale_001() raises:
    """DT-SCALE-001: Scale by 0.25, compare to host."""
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found - skipping DT-SCALE-001 test")
        return
    with DeviceContext() as ctx:
        var data = make_host_data_f32(8, 6)
        var shape = List[Int](2, 4)
        var t = tensor_from_host(ctx, data, shape)
        t.scale_in_place(Float32(0.25), ctx)
        var host = tensor_to_host(ctx, t)
        for i in range(8):
            if abs(Float64(host[i]) - Float64(data[i]) * 0.25) > 1e-4:
                raise Error("Scale mismatch at " + String(i))

fn main() raises:
    try:
        TestSuite.discover_tests[__functions_in_module()]().run()
    except e:
        print("Tests failed: " + String(e))
