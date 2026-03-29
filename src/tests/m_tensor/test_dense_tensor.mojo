"""
Minimal DenseTensor smoke tests (stride + creation).
Full conformance tests: test_dense_tensor_basic, test_dense_tensor_dot_contract_a,
test_dense_tensor_qr, test_dense_tensor_svd (see TEST_PLAN_DENSE_TENSOR.md).
"""
from sys import has_accelerator
from testing import TestSuite
from src.m_tensor.dense_tensor import (
    DenseTensor,
    create_dense_tensor,
    create_dense_tensor_from_data,
    compute_row_major_strides,
)
from gpu.host import DeviceContext
from collections.list import List


fn test_stride_computation() raises:
    """DT-STRIDE-001 style: row-major strides for [3,4] and [2,3,4]."""
    var shape_2d = List[Int](3, 4)
    var strides_rm = compute_row_major_strides(shape_2d, 2)
    if strides_rm[0] != 4 or strides_rm[1] != 1:
        raise Error("Stride [3,4] expected [4,1]")

    var shape_3d = List[Int](2, 3, 4)
    var strides_rm_3d = compute_row_major_strides(shape_3d, 3)
    if strides_rm_3d[0] != 12 or strides_rm_3d[1] != 4 or strides_rm_3d[2] != 1:
        raise Error("Stride [2,3,4] expected [12,4,1]")


fn test_basic_creation() raises:
    """Smoke test: create contiguous tensors (GPU if available)."""
    @parameter
    if not has_accelerator():
        return
    with DeviceContext() as ctx:
        var shape_2d = List[Int](3, 4)
        var tensor = create_dense_tensor[DType.float32](ctx, shape_2d^, row_major=True)
        if not tensor.is_contiguous():
            raise Error("Expected contiguous tensor")
        if len(tensor.shape) != 2 or tensor.shape[0] != 3 or tensor.shape[1] != 4:
            raise Error("Expected shape [3, 4]")


fn main() raises:
    try:
        TestSuite.discover_tests[__functions_in_module()]().run()
    except e:
        print("Tests failed: " + String(e))
