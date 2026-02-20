"""
Test utilities for DenseTensor conformance tests (C-baseline).
Deterministic RNG, host/device helpers, row-major indexing, assertions.
"""
from collections.list import List
from gpu.host import DeviceContext
from src.m_tensor.dense_tensor import (
    create_dense_tensor_from_data,
    compute_row_major_strides,
    DenseTensor,
)

fn make_host_data_f32(nelem: Int, seed: Int) -> List[Float32]:
    """Deterministic host data in [-0.5, 0.5]. LCG with given seed (no global state)."""
    var state = seed
    var out = List[Float32](capacity=nelem)
    for _ in range(nelem):
        state = (1664525 * state + 1013904223) % (1 << 32)
        var u = Float64(state) / Float64(1 << 32)
        out.append(Float32(u - 0.5))
    return out.copy()

fn tensor_from_host(
    ctx: DeviceContext,
    host_data: List[Float32],
    shape: List[Int],
) raises -> DenseTensor[DType.float32]:
    """Build DenseTensor[float32] from host list and shape (row-major)."""
    var shape_copy = shape.copy()
    return create_dense_tensor_from_data[DType.float32](ctx, host_data, shape_copy^)

fn tensor_to_host(ctx: DeviceContext, tensor: DenseTensor[DType.float32]) raises -> List[Float32]:
    """Copy tensor to host list (row-major order)."""
    var out = List[Float32](capacity=tensor.size)
    var host = ctx.enqueue_create_host_buffer[DType.float32](tensor.size)
    ctx.enqueue_copy(host, tensor.storage)
    ctx.synchronize()
    for i in range(tensor.size):
        out.append(host[i])
    return out.copy()

fn row_major_strides(shape: List[Int]) -> List[Int]:
    """Row-major strides for shape. stride[i] = prod(shape[i+1:])."""
    return compute_row_major_strides(shape, len(shape))

fn linear_to_multi(idx: Int, shape: List[Int]) -> List[Int]:
    """Convert linear index to multi-index (row-major)."""
    var strides = row_major_strides(shape)
    var multi = List[Int](capacity=len(shape))
    var rem = idx
    for i in range(len(shape)):
        multi.append(rem // strides[i])
        rem = rem % strides[i]
    return multi.copy()

fn multi_to_linear(multi: List[Int], strides: List[Int]) -> Int:
    """Convert multi-index to linear index."""
    var idx: Int = 0
    for i in range(len(multi)):
        idx += multi[i] * strides[i]
    return idx

fn assert_allclose(
    host_a: List[Float32],
    host_b: List[Float32],
    atol: Float64 = 1e-4,
    rtol: Float64 = 1e-4,
) raises:
    """Raise if any pair differs beyond atol or rtol."""
    if len(host_a) != len(host_b):
        raise Error("Length mismatch in assert_allclose")
    for i in range(len(host_a)):
        var a = Float64(host_a[i])
        var b = Float64(host_b[i])
        var diff = abs(a - b)
        var tol = atol + rtol * max(abs(a), abs(b))
        if diff > tol:
            raise Error("Mismatch at index " + String(i) + ": " + String(a) + " vs " + String(b))

fn assert_shape_equal(shape_a: List[Int], shape_b: List[Int]) raises:
    """Raise if shapes differ."""
    if len(shape_a) != len(shape_b):
        raise Error("Rank mismatch: " + String(len(shape_a)) + " vs " + String(len(shape_b)))
    for i in range(len(shape_a)):
        if shape_a[i] != shape_b[i]:
            raise Error("Shape dim " + String(i) + " mismatch: " + String(shape_a[i]) + " vs " + String(shape_b[i]))

fn assert_upper_triangular(host_R: List[Float32], rows: Int, cols: Int, tol: Float64 = 1e-4) raises:
    """Raise if R is not upper triangular (R[i,j] ~ 0 for i > j)."""
    for i in range(rows):
        for j in range(cols):
            if i > j:
                var v = abs(Float64(host_R[i * cols + j]))
                if v > tol:
                    raise Error("R[" + String(i) + "," + String(j) + "] = " + String(v) + " (expected ~0)")

fn assert_orthonormal(host_Q: List[Float32], m: Int, n: Int, tol: Float64 = 1e-3) raises:
    """Check Q^T Q ≈ I (n×n). Q is m×n stored row-major."""
    for i in range(n):
        for j in range(n):
            var dot: Float64 = 0.0
            for k in range(m):
                dot += Float64(host_Q[k * n + i]) * Float64(host_Q[k * n + j])
            var expect = 1.0 if i == j else 0.0
            if abs(dot - expect) > tol:
                raise Error("(Q^T Q)[" + String(i) + "," + String(j) + "] = " + String(dot) + " (expected " + String(expect) + ")")


# ============================================================================
# MPO Testing Utilities
# ============================================================================

fn assert_close(
    actual: Float64,
    expected: Float64,
    rtol: Float64 = 1e-6,
    atol: Float64 = 1e-8,
    label: String = "value"
) raises:
    """Assert that two scalar values are close within relative and absolute tolerance.
    
    Args:
        actual: The actual value to test.
        expected: The expected reference value.
        rtol: Relative tolerance (default: 1e-6).
        atol: Absolute tolerance (default: 1e-8).
        label: Description of the value being tested (for error messages).
    
    Raises:
        Error if values differ beyond tolerance.
    """
    var diff = abs(actual - expected)
    var tol = atol + rtol * max(abs(actual), abs(expected))
    if diff > tol:
        raise Error(
            label + " mismatch: actual=" + String(actual) + 
            ", expected=" + String(expected) + 
            ", diff=" + String(diff) + 
            " (tol=" + String(tol) + ")"
        )


fn compare_tensors[
    dtype: DType
](
    t1: DenseTensor[dtype],
    t2: DenseTensor[dtype],
    ctx: DeviceContext,
    rtol: Float64 = 1e-5,
    atol: Float64 = 1e-7,
    site_index: Int = -1,
    max_errors_to_print: Int = 10
) raises:
    """Compare two tensors element-wise with detailed error reporting.
    
    Args:
        t1: First tensor to compare.
        t2: Second tensor to compare.
        ctx: Device context.
        rtol: Relative tolerance (default: 1e-5).
        atol: Absolute tolerance (default: 1e-7).
        site_index: Optional site index for error messages (default: -1 = no site).
        max_errors_to_print: Maximum number of mismatches to print before stopping.
    
    Raises:
        Error if tensors have different shapes or if any element differs beyond tolerance.
    """
    # Check shapes
    if len(t1.shape) != len(t2.shape):
        raise Error(
            "Tensor rank mismatch: " + String(len(t1.shape)) + 
            " vs " + String(len(t2.shape))
        )
    
    for i in range(len(t1.shape)):
        if t1.shape[i] != t2.shape[i]:
            raise Error(
                "Shape mismatch at dimension " + String(i) + ": " + 
                String(t1.shape[i]) + " vs " + String(t2.shape[i])
            )
    
    # Compare elements
    var host1 = ctx.enqueue_create_host_buffer[dtype](t1.size)
    var host2 = ctx.enqueue_create_host_buffer[dtype](t2.size)
    ctx.enqueue_copy(host1, t1.storage)
    ctx.enqueue_copy(host2, t2.storage)
    ctx.synchronize()
    
    var num_errors = 0
    var max_abs_error: Float64 = 0.0
    var max_rel_error: Float64 = 0.0
    
    for i in range(t1.size):
        var v1 = Float64(host1[i])
        var v2 = Float64(host2[i])
        var diff = abs(v1 - v2)
        var tol = atol + rtol * max(abs(v1), abs(v2))
        
        if diff > tol:
            num_errors += 1
            if diff > max_abs_error:
                max_abs_error = diff
            
            var rel_error = diff / max(abs(v2), 1e-10)
            if rel_error > max_rel_error:
                max_rel_error = rel_error
            
            if num_errors <= max_errors_to_print:
                var multi = linear_to_multi(i, t1.shape)
                var idx_str = String("[")
                for j in range(len(multi)):
                    if j > 0:
                        idx_str += ","
                    idx_str += String(multi[j])
                idx_str += "]"
                
                var site_str = ""
                if site_index >= 0:
                    site_str = "Site " + String(site_index) + " "
                
                print(
                    "  Mismatch " + String(num_errors) + ": " + site_str + 
                    idx_str + " -> " + String(v1) + " vs " + String(v2) + 
                    " (diff=" + String(diff) + ", tol=" + String(tol) + ")"
                )
    
    if num_errors > 0:
        var site_str = ""
        if site_index >= 0:
            site_str = " at site " + String(site_index)
        
        raise Error(
            "Tensor comparison failed" + site_str + ": " + 
            String(num_errors) + " elements differ. " +
            "Max abs error=" + String(max_abs_error) + 
            ", max rel error=" + String(max_rel_error)
        )


fn assert_equal(actual: Int, expected: Int, label: String = "value") raises:
    """Assert that two integers are equal.
    
    Args:
        actual: The actual value to test.
        expected: The expected reference value.
        label: Description of the value being tested (for error messages).
    
    Raises:
        Error if values are not equal.
    """
    if actual != expected:
        raise Error(
            label + " mismatch: actual=" + String(actual) + 
            ", expected=" + String(expected)
        )
