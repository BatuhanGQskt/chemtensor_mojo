# Tensor Operations & Matrix Multiplication Update

## Overview

The `dense_tensor.mojo` module provides a fully dynamic tensor implementation with GPU acceleration for arbitrary-rank tensor operations. Unlike compile-time tensor libraries, this allows complete flexibility at runtime for shape, rank, and stride operations.

## Core Features

### 1. DenseTensor Struct

A runtime-flexible tensor that supports:
- **Arbitrary rank** (1D, 2D, 3D, 4D, ..., ND)
- **GPU storage** using `DeviceBuffer`
- **Runtime shape/stride** flexibility
- **Row-major and column-major** layouts
- **Contiguity checking** for memory optimization
- **View operations** (transpose, flatten) without data copying

### 2. Generalized Tensor Dot Product (`dense_tensor_dot`)

**File**: `src/m_tensor/dense_tensor.mojo`

#### Key Capabilities:
- **ND × ND tensor contractions** (not limited to 2D!)
- **Flexible axis contraction** via `ndim_mult` parameter
- **Configurable contraction order** (leading vs trailing axes)
- **Automatic flattening** to 2D for GPU matmul
- **Automatic reshaping** back to ND result tensor
- **GPU-accelerated** using `linalg.matmul`

#### Algorithm Overview:
```mojo
fn dense_tensor_dot(
    C: DenseTensor,           # Output (pre-allocated)
    var A: DenseTensor,        # First input
    var B: DenseTensor,        # Second input
    ctx: DeviceContext,          # GPU context
    ndim_mult: Int = 1,          # Number of axes to contract
    axrange_A: Bool = False,     # False=trailing, True=leading
    axrange_B: Bool = False      # False=trailing, True=leading (auto-adjusted)
) raises:
    # 1. Validate dimensions and contraction compatibility
    # 2. Determine which axes to contract
    # 3. Ensure tensors are contiguous (copy if needed)
    # 4. Flatten non-contracted dimensions → intermediate shape
    # 5. Flatten contracted dimensions → final 2D matrices
    # 6. Transpose if using leading axes
    # 7. Perform 2D GPU matmul on flattened tensors
    # 8. Result automatically appears in C with correct ND shape
```

#### Convention: Trailing × Leading (Default)
By default, `dense_tensor_dot` follows **standard linear algebra convention**:
- Contracts **A's trailing axes** with **B's leading axes**
- `A[..., k] @ B[k, ...] = C[...]`
- Automatically sets `effective_axrange_B = True` when both are False

For **tensor network conventions** (leading × trailing), explicitly set:
- `axrange_A=True, axrange_B=False` → `A[k, ...] @ B[..., k]`

## Usage Examples

### Example 1: Standard 2D Matrix Multiplication
```mojo
with DeviceContext() as ctx:
    # A[3, 4] @ B[4, 5] = C[3, 5]
    var A = create_dense_tensor(ctx, List[Int](3, 4)^, init_value=2.0)
    var B = create_dense_tensor(ctx, List[Int](4, 5)^, init_value=3.0)
    var C = create_dense_tensor(ctx, List[Int](3, 5)^, init_value=0.0)
    
    dense_tensor_dot(C, A^, B^, ctx)
    # Result: Each C[i,j] = sum_k A[i,k] * B[k,j] = 4 * (2.0 * 3.0) = 24.0
```

### Example 2: 3D Tensor Contraction (Single Axis)
```mojo
with DeviceContext() as ctx:
    # A[2, 3, 4] @ B[4, 5, 6] = C[2, 3, 5, 6]
    # Contracts A's last axis (4) with B's first axis (4)
    var A = create_dense_tensor(ctx, List[Int](2, 3, 4)^, init_value=1.5)
    var B = create_dense_tensor(ctx, List[Int](4, 5, 6)^, init_value=2.0)
    var C = create_dense_tensor(ctx, List[Int](2, 3, 5, 6)^, init_value=0.0)
    
    dense_tensor_dot(C, A^, B^, ctx, ndim_mult=1)
    # Result shape: (2, 3, 5, 6) with 360 elements
    # Each element: sum over 4 values = 4 * (1.5 * 2.0) = 12.0
```

### Example 3: 4D Tensor Contraction (Multiple Axes)
```mojo
with DeviceContext() as ctx:
    # A[4, 3, 2, 1] @ B[1, 2, 5, 7] = C[4, 3, 5, 7]
    # Contracts A's last 2 axes (2,1) with B's first 2 axes (1,2)
    var A = create_dense_tensor(ctx, List[Int](4, 3, 2, 1)^, init_value=2.0)
    var B = create_dense_tensor(ctx, List[Int](1, 2, 5, 7)^, init_value=3.0)
    var C = create_dense_tensor(ctx, List[Int](4, 3, 5, 7)^, init_value=0.0)
    
    dense_tensor_dot(C, A^, B^, ctx, ndim_mult=2)
    # Contracts: A.shape[2,3]=(2,1) with B.shape[0,1]=(1,2) [reversed order!]
    # Result: Each element = sum over 2 values = 2 * (2.0 * 3.0) = 12.0
```

### Example 4: Tensor Network Style (Leading × Trailing)
```mojo
with DeviceContext() as ctx:
    # A[k, m, n] @ B[p, q, k] = C[m, n, p, q]
    # Contract first axis of A with last axis of B
    var A = create_dense_tensor(ctx, List[Int](3, 4, 5)^)
    var B = create_dense_tensor(ctx, List[Int](6, 7, 3)^)
    var C = create_dense_tensor(ctx, List[Int](4, 5, 6, 7)^)
    
    dense_tensor_dot(C, A^, B^, ctx, ndim_mult=1, 
                   axrange_A=True,   # Contract A's leading axis
                   axrange_B=False)  # Contract B's trailing axis
```

## How It Works Internally

### Step-by-Step for A(4,3,2,1) @ B(1,2,5,7) with ndim_mult=2

1. **Identify Axes**:
   - A's trailing 2 axes to contract: `[2, 1]` (indices 2, 3)
   - B's leading 2 axes to contract: `[2, 1]` (indices 1, 0) 
   - Matches: A[2]=2 ↔ B[1]=2 ✓, A[3]=1 ↔ B[0]=1 ✓

2. **Save Non-Contracted Shapes**:
   - A's non-contracted: `[4, 3]` (indices 0,1)
   - B's non-contracted: `[5, 7]` (indices 2,3)
   - Expected C shape: `[4, 3] + [5, 7] = [4, 3, 5, 7]`

3. **First Flatten (Non-Contracted Dims)**:
   - A: `(4,3,2,1)` → flatten `[0,2)` → `(12, 2, 1)`
   - B: `(1,2,5,7)` → flatten `[2,4)` → `(1, 2, 35)`

4. **Second Flatten (Contracted Dims)**:
   - A: `(12, 2, 1)` → flatten `[1,3)` → `(12, 2)` ✓
   - B: `(1, 2, 35)` → flatten `[0,2)` → `(2, 35)` ✓

5. **GPU Matrix Multiply**:
   - `(12, 2) @ (2, 35) = (12, 35)` = 420 elements

6. **Reshape to ND**:
   - Create 2D view of C's storage: shape `(12, 35)`
   - C still has original shape: `(4, 3, 5, 7)`
   - Since 12 = 4×3 and 35 = 5×7, storage layout matches!
   - Result automatically appears in correct ND shape

## Helper Functions & Features

### Tensor Creation

```mojo
# Create tensor with automatic stride calculation
fn create_dense_tensor(
    ctx: DeviceContext, 
    var shape: List[Int], 
    row_major: Bool = True,
    init_value: Float32 = 0.0
) raises -> DenseTensor

# Create from existing data
fn create_dense_tensor_from_data(
    ctx: DeviceContext, 
    data: List[Float32],
    var shape: List[Int], 
    row_major: Bool = True
) raises -> DenseTensor
```

**Row-major** (default) means the last dimension changes fastest:
- Shape `[2, 3, 4]` → Strides `[12, 4, 1]`

**Column-major** means the first dimension changes fastest:
- Shape `[2, 3, 4]` → Strides `[1, 2, 6]`

### Tensor Operations

#### Transpose
```mojo
fn transpose(var self, perm: List[Int], ctx: DeviceContext) raises -> DenseTensor
```
- For 2D tensors: Performs physical transpose (copies data)
- For ND tensors: Creates stride-based view (no copy)

#### Flatten Dimensions
```mojo
fn flatten_dims(var self, start: Int, end: Int, ctx: DeviceContext) raises -> DenseTensor
```
- Combines consecutive dimensions: `[2, 3, 4, 5]` → flatten `[1,3)` → `[2, 12, 5]`
- Creates a view (no data copy)
- Used internally by `dense_tensor_dot`

#### Check Contiguity
```mojo
fn is_contiguous(self: DenseTensor) -> Bool
```
- Verifies row-major contiguous layout
- Important for GPU performance

#### Make Contiguous
```mojo
fn copy_to_contiguous(var self, ctx: DeviceContext) raises -> DenseTensor
```
- Returns self if already contiguous (no copy)
- Otherwise allocates and copies to contiguous layout

## Testing

### Run ND Tensor Test
```bash
# Activate environment
pixi shell

# Run test
mojo src/main.mojo
```

The test function `test_nd_tensor_dot()` validates:
- **Test 1**: `A(4,3,2,1) @ B(1,2,5,7)` with `ndim_mult=2` → `C(4,3,5,7)`
- **Test 2**: `A(2,3,4) @ B(4,5,6)` with `ndim_mult=1` → `C(2,3,5,6)`

Expected output for Test 1:
```
Test 1: A(4,3,2,1) @ B(1,2,5,7) with ndim_mult=2
  Expected output shape: (4,3,5,7)
  Contracting A's last 2 dims (2,1) with B's first 2 dims (1,2)
  ...
  Expected value per element: 2.0 * 3.0 * 2 = 12.0
  Result: All 420 elements = 12.0 ✓
```

## Technical Details

### Memory Management (RAII-Compliant)

The implementation carefully manages GPU memory ownership:

```mojo
# 1. Original tensors own their DeviceBuffer storage
var A = create_dense_tensor(ctx, shape_A^)  # A owns storage

# 2. Get non-owning pointer for NDBuffer creation
var ptr_A = A_flat.storage.unsafe_ptr()  # Non-owning pointer

# 3. Create NDBuffer as a view (doesn't own memory)
var shape_Af = IndexList[2](A_flat.shape[0], A_flat.shape[1])
var ndbuf_A = NDBuffer[dtype, 2, MutableAnyOrigin](ptr_A, shape_Af)

# 4. NDBuffer is dropped after matmul, but storage remains in DenseTensor
# 5. DenseTensor cleans up GPU memory when it goes out of scope
```

**Key Points**:
- Use `unsafe_ptr()` for non-owning pointers (not `take_ptr()`)
- `NDBuffer` is a view that doesn't manage memory lifetime
- Original `DenseTensor` retains ownership via RAII
- No use-after-free or double-free issues

### Two-Stage Flattening Algorithm

Why two flatten operations?

```mojo
# Problem: A(4, 3, 2, 1) needs to become (12, 2) for matmul
# Single flatten would only flatten consecutive ranges

# Solution: Two-stage process
# Stage 1: Flatten non-contracted dims
A(4, 3, 2, 1) → flatten_dims(0, 2) → A(12, 2, 1)

# Stage 2: Flatten contracted dims  
A(12, 2, 1) → flatten_dims(1, 3) → A(12, 2) ✓

# This ensures proper alignment for matrix multiplication!
```

### Axis Matching with Reversal

When contracting A's trailing with B's leading, dimensions match in **reverse order**:

```python
# A[4, 3, 2, 1] @ B[1, 2, 5, 7] with ndim_mult=2
# 
# Contraction pairing:
#   A's dim 2 (size=2) ↔ B's dim 1 (size=2) ✓
#   A's dim 3 (size=1) ↔ B's dim 0 (size=1) ✓
#
# General rule for trailing×leading:
#   A.shape[rank_A - ndim_mult + i] ↔ B.shape[ndim_mult - 1 - i]
```

### NDBuffer Parameters
- **dtype**: Data type (`DType.float32`)
- **rank**: Number of dimensions (2 after flattening)
- **origin**: Memory origin (`MutableAnyOrigin` for GPU)
- **mut**: Mutability flag (`True` for output tensor C)

## Performance Characteristics

- **GPU Acceleration**: Uses `linalg.matmul` with tiled shared memory kernels
- **Optimal Tile Sizes**: 16-32 for float32 (matches GPU warp size)
- **Memory Efficiency**: 
  - Views avoid copying when possible
  - Contiguity check before operations
  - Only copies if non-contiguous
- **Zero-Copy Reshaping**: Result appears in ND tensor without explicit reshape

## Use Cases

This implementation is ideal for:

1. **Tensor Networks**: Quantum chemistry, physics simulations
2. **Deep Learning**: Custom tensor contractions beyond standard matmul
3. **Scientific Computing**: Multi-dimensional array operations
4. **Dynamic Shapes**: Runtime-determined tensor sizes and ranks
5. **GPU Acceleration**: Leverage GPU for arbitrary tensor contractions

## Next Steps & Future Enhancements

Potential improvements:
- ✅ **ND tensor support** (implemented!)
- ✅ **Flexible axis contraction** (implemented!)
- ✅ **Automatic dimension handling** (implemented!)
- 🔄 **Batched operations** (could optimize for batch dims)
- 🔄 **GPU kernel for non-contiguous copy** (currently uses CPU for transpose and contiguous checks)
- 🔄 **Sparse tensor support**

## References
- [Mojo NDBuffer Documentation](https://docs.modular.com/mojo/stdlib/buffer/buffer/NDBuffer/)
- [Mojo GPU Programming Guide](https://docs.modular.com/mojo/manual/gpu/)
### Can be changed to einsum method for more flexability
- [Einstein Summation Convention](https://en.wikipedia.org/wiki/Einstein_notation)