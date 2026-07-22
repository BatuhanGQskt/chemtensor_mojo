# BlockSparseTensor: C vs Mojo Implementation Comparison

This document provides a comprehensive comparison between the ChemTensor C implementation
and the Mojo port, verifying structural and functional equivalence.

## 1. Data Structure Comparison

### C Implementation (`block_sparse_tensor.h`)

```c
struct block_sparse_tensor {
    struct dense_tensor** blocks;           // Dense blocks (NULL for non-conserved qnums)
    long* dim_blocks;                       // Number of distinct quantum numbers per axis
    long* dim_logical;                      // Logical dimensions of the overall tensor
    enum tensor_axis_direction* axis_dir;   // Axis directions (+1 = OUT, -1 = IN)
    qnumber** qnums_blocks;                 // Distinct quantum numbers per axis (sorted)
    qnumber** qnums_logical;                // Quantum number for each logical index
    enum numeric_type dtype;                // Data type
    int ndim;                               // Number of dimensions
};
```

### Mojo Implementation (`block_sparse_tensor.mojo`)

```mojo
struct BlockSparseTensor[dtype: DType]:
    var ndim: Int
    var dim_logical: List[Int]             # Logical dimensions
    var dim_blocks: List[Int]              # Number of distinct qnums per axis
    var axis_dir: List[Int]                # Axis directions (+1 or -1)
    var qnums_logical: List[List[Int]]     # Quantum number for each logical index
    var qnums_blocks: List[List[Int]]      # Distinct quantum numbers per axis (sorted)
    var sector_counts: List[List[Int]]     # Multiplicities (not in C - derived)
    var stride_logical: List[Int]          # Row-major strides (not in C - computed)
    var logical_size: Int                  # Total logical elements
    var nblocks_total: Int                 # dim_blocks[0] × ... × dim_blocks[ndim-1]
    var block_present: List[Bool]          # Which blocks are non-zero
    var blocks_flat: List[DenseTensor[dtype]]  # Dense blocks (flat array)
    var _dummy_scalar: DenseTensor[dtype]  # Placeholder for empty blocks
```

## 2. Field Mapping

| C Field | Mojo Field | Notes |
|---------|------------|-------|
| `blocks` | `blocks_flat` + `block_present` | Mojo uses flat List + presence flags instead of NULL pointers |
| `dim_blocks` | `dim_blocks` | Direct mapping |
| `dim_logical` | `dim_logical` | Direct mapping |
| `axis_dir` | `axis_dir` | C uses enum {-1, +1}, Mojo uses Int |
| `qnums_blocks` | `qnums_blocks` | Direct mapping |
| `qnums_logical` | `qnums_logical` | Direct mapping |
| `dtype` | `dtype` (type parameter) | Compile-time in Mojo |
| `ndim` | `ndim` | Direct mapping |
| - | `sector_counts` | Mojo pre-computes multiplicities |
| - | `stride_logical` | Mojo pre-computes strides |
| - | `logical_size` | Mojo pre-computes total size |
| - | `nblocks_total` | Mojo pre-computes block count |

## 3. Function Mapping

### Allocation/Deallocation

| C Function | Mojo Function | Status |
|------------|---------------|--------|
| `allocate_block_sparse_tensor` | `create_block_sparse_tensor` | ✅ Implemented |
| `allocate_block_sparse_tensor_like` | `allocate_block_sparse_tensor_like` | ✅ Implemented |
| `delete_block_sparse_tensor` | (automatic via `__del__`) | ✅ Automatic |
| `copy_block_sparse_tensor` | `__copyinit__` | ✅ Implemented |

### Block Access

| C Function | Mojo Function | Status |
|------------|---------------|--------|
| `block_sparse_tensor_get_block` | `get_block` | ✅ Implemented |

### Norms and Scaling

| C Function | Mojo Function | Status |
|------------|---------------|--------|
| `block_sparse_tensor_norm2` | `compute_norm` | ✅ Implemented |
| `scale_block_sparse_tensor` | `scale_in_place` | ✅ Implemented |
| `block_sparse_tensor_scalar_multiply_add` | `axpy_in_place` | ✅ Implemented |

### Conversions

| C Function | Mojo Function | Status |
|------------|---------------|--------|
| `block_sparse_to_dense_tensor` | `block_sparse_to_dense` | ✅ Implemented |
| `dense_to_block_sparse_tensor` | `dense_to_block_sparse` | ✅ Implemented |
| `dense_to_block_sparse_tensor_entries` | `dense_to_block_sparse_entries` | ✅ Implemented |

### Transpose

| C Function | Mojo Function | Status |
|------------|---------------|--------|
| `transpose_block_sparse_tensor` | `transpose` | ✅ Implemented |
| `conjugate_transpose_block_sparse_tensor` | - | ❌ Not implemented |

### Contraction

| C Function | Mojo Function | Status |
|------------|---------------|--------|
| `block_sparse_tensor_dot` | `block_sparse_tensor_dot` | ✅ Implemented |
| `block_sparse_tensor_multiply_axis` | - | ❌ Not implemented |

### Reshape Operations

| C Function | Mojo Function | Status |
|------------|---------------|--------|
| `block_sparse_tensor_flatten_axes` | `flatten_dims` (partial) | ⚠️ Raises NotImplemented |
| `block_sparse_tensor_split_axis` | - | ❌ Not implemented |
| `block_sparse_tensor_matricize_axis` | - | ❌ Not implemented |
| `block_sparse_tensor_dematricize_axis` | - | ❌ Not implemented |

### Decompositions

| C Function | Mojo Function | Status |
|------------|---------------|--------|
| `block_sparse_tensor_qr` | - | ❌ Not implemented (falls back to dense) |
| `block_sparse_tensor_rq` | - | ❌ Not implemented |
| `block_sparse_tensor_svd` | `block_sparse_tensor_svd_trunc` | ⚠️ Falls back to dense |

### Other Operations

| C Function | Mojo Function | Status |
|------------|---------------|--------|
| `block_sparse_tensor_slice` | - | ❌ Not implemented |
| `block_sparse_tensor_concatenate` | - | ❌ Not implemented |
| `block_sparse_tensor_block_diag` | - | ❌ Not implemented |
| `block_sparse_tensor_cyclic_partial_trace` | - | ❌ Not implemented |
| `block_sparse_tensor_is_identity` | - | ❌ Not implemented |
| `block_sparse_tensor_is_isometry` | - | ❌ Not implemented |
| `block_sparse_tensor_allclose` | - | ❌ Not implemented |

## 4. Quantum Number Conservation

Both implementations enforce the **additive quantum number conservation** rule:

```
A block at index [i_0, i_1, ..., i_{ndim-1}] is non-zero if and only if:
    sum(axis_dir[k] * qnums_blocks[k][i_k] for k in 0..ndim) == 0
```

### C Implementation (allocation)
```c
for (long k = 0; k < nblocks; k++) {
    qnumber qsum = 0;
    for (int i = 0; i < ndim; i++) {
        qsum += axis_dir[i] * t->qnums_blocks[i][index_block[i]];
    }
    if (qsum != 0) {
        continue;  // Block is NULL
    }
    // Allocate block...
}
```

### Mojo Implementation (allocation)
```mojo
for k in range(nblocks):
    _offset_to_tensor_index(k, ndim, dim_blocks, idx)
    var qsum = 0
    for i in range(ndim):
        qsum += axis_dir[i] * qnums_blocks[i][idx[i]]
    if qsum != 0:
        block_present.append(False)
        blocks_flat.append(dummy)
        continue
    # Allocate block...
```

## 5. Index Mapping

Both implementations use row-major flat indexing for blocks:

```
offset = sum(index_block[i] * product(dim_blocks[i+1:]) for i in 0..ndim)
```

C: `tensor_index_to_offset` / `offset_to_tensor_index`
Mojo: `_tensor_index_to_offset` / `_offset_to_tensor_index`

## 6. Contraction (tensor_dot)

Both implementations follow the same algorithm:

1. Validate contracted dimensions match
2. Check axis directions are opposite on contracted legs
3. For each output block:
   - Sum over all valid contraction quantum number combinations
   - Only include blocks where qnums sum to zero
   - Call dense tensor dot for each contributing block pair
   - Accumulate results

## 7. Key Differences

1. **Memory Management**: C uses explicit malloc/free, Mojo uses automatic memory management
2. **GPU Support**: Mojo implementation uses GPU via `DeviceContext`, C uses CPU (with OpenMP)
3. **Type Safety**: Mojo uses compile-time dtype parameter, C uses runtime enum
4. **Null Handling**: C uses NULL pointers for missing blocks, Mojo uses `block_present` flags
5. **Pre-computation**: Mojo pre-computes `sector_counts`, `stride_logical`, `logical_size`

## 8. Recommendations

### Features to Implement in Mojo

1. **High Priority** (used in DMRG):
   - `block_sparse_tensor_qr` (block-wise, not fallback to dense)
   - `block_sparse_tensor_rq`
   - `block_sparse_tensor_flatten_axes`
   - `block_sparse_tensor_split_axis`

2. **Medium Priority**:
   - `block_sparse_tensor_allclose` (for testing)
   - `block_sparse_tensor_is_isometry` (for validation)
   - `block_sparse_tensor_multiply_axis`

3. **Lower Priority**:
   - `block_sparse_tensor_slice`
   - `block_sparse_tensor_concatenate`
   - `block_sparse_tensor_cyclic_partial_trace`

## 9. Testing Strategy

Tests should verify:
1. Block allocation respects quantum number conservation
2. Dense ↔ BlockSparse conversion round-trips correctly
3. Tensor dot produces correct results vs dense reference
4. Transpose preserves quantum number structure
5. Norm computation matches dense equivalent
