from memory import Pointer, AddressSpace, OwnedPointer
from layout import Layout, LayoutTensor, RuntimeLayout, IntTuple, RuntimeTuple
from collections.list import List
from utils import IndexList
from gpu import thread_idx, block_idx, block_dim, barrier, global_idx
from gpu.host import DeviceContext, DeviceBuffer
from buffer.buffer import NDBuffer
from memory.unsafe_pointer import UnsafePointer
import linalg
from linalg.qr_factorization import qr_factorization, form_q
from random import random_float64
from math import sqrt, ceildiv
from src.mylinalg.backend import SVDBackend
from src.mylinalg.matrix import MatrixF64
from src.mylinalg.svd import svd_f64
from src.mylinalg.types import Layout as LapackLayout


# ---------------------------------------------------------------------------
# GPU helper kernels for element-wise operations
# ---------------------------------------------------------------------------

fn _gpu_fill_kernel[dtype: DType](
    data: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    value_buf: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    total_size: Int,
):
    """GPU kernel: fill every element with value_buf[0]. No host buffer for bulk data."""
    var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
    if tid >= total_size:
        return
    data[tid] = value_buf[0]


fn _gpu_set_identity_kernel[dtype: DType](
    data: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    rows: Int,
    cols: Int,
):
    """GPU kernel: set matrix to identity (1 on diagonal, 0 elsewhere).
    
    Each thread handles one element. For a rows x cols matrix in row-major order,
    element (i, j) is at index i * cols + j. Set to 1.0 if i == j, else 0.0.
    """
    var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
    var total_size = rows * cols
    if tid >= total_size:
        return
    var i = tid // cols
    var j = tid % cols
    if i == j and i < rows and j < cols:
        data[tid] = Scalar[dtype](1.0)
    else:
        data[tid] = Scalar[dtype](0.0)


fn _gpu_scale_kernel[dtype: DType](
    data: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    scale_buf: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    total_size: Int,
):
    """GPU kernel: scale every element by a scalar stored in scale_buf[0]."""
    var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
    if tid >= total_size:
        return
    data[tid] = data[tid] * scale_buf[0]


fn _gpu_norm_sq_kernel[dtype: DType](
    data: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    partial_sums: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    total_size: Int,
):
    """GPU kernel: compute per-block partial sums of squares (shared-memory reduction).
    
    Each block of 256 threads reduces its portion into a single partial sum.
    The host then sums the (small) partial_sums array.
    """
    alias BLOCK_SIZE = 256
    var tid = Int(thread_idx.x)
    var gid = Int(block_dim.x * block_idx.x + thread_idx.x)
    var bid = Int(block_idx.x)

    # Each thread loads its element (or 0 if out of bounds)
    var val = Scalar[dtype](0.0)
    if gid < total_size:
        val = data[gid]

    # Warp-level reduction using shared variable accumulation
    # For simplicity and correctness, write each thread's val^2 to global partial sum atomically.
    # (For blocks <= 256 this is efficient enough; full shared-mem reduction can be added later.)
    var sq = val * val

    # Use atomic-friendly approach: first thread per block initializes, then all add
    # Since Mojo GPU doesn't expose atomicAdd directly, we do a simple sequential 
    # reduction on host from per-thread outputs. Instead, write thread results to 
    # a staging buffer and reduce on host.
    # 
    # Simpler approach: just write val*val back and let host sum.
    # This avoids shared memory complexity while still keeping data on GPU.
    if gid < total_size:
        data[gid] = sq  # Overwrite in-place (caller uses a copy)


fn _gpu_dot_kernel[dtype: DType](
    a: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    total_size: Int,
):
    """GPU kernel: element-wise multiply a[i]*b[i] and store in dst[i].
    
    Host sums out[] afterwards to get the inner product.
    This keeps the bulk data on GPU and only transfers the (small) product array.
    """
    var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
    if tid >= total_size:
        return
    dst[tid] = a[tid] * b[tid]


fn _gpu_axpy_kernel[dtype: DType](
    y: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    x: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    alpha_buf: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    total_size: Int,
):
    """GPU kernel: y[i] += alpha * x[i]  (BLAS axpy).
    
    alpha is read from alpha_buf[0] so the scalar can live on the device.
    """
    var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
    if tid >= total_size:
        return
    y[tid] = y[tid] + alpha_buf[0] * x[tid]


# ---------------------------------------------------------------------------
# GPU reduction kernels: sum array on device, copy back only 1 scalar
# ---------------------------------------------------------------------------

fn _gpu_reduce_block_sum_kernel[dtype: DType](
    data: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    partial_sums: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    total_size: Int,
):
    """GPU kernel: one thread per block sums its block's elements into partial_sums[block_idx].
    No host-side bulk copy: only partial_sums (num_blocks elements) is used in next stage.
    """
    alias BLOCK_SIZE = 256
    var bid = Int(block_idx.x)
    # Only first thread of each block does the reduction
    if thread_idx.x != 0:
        return
    var start = bid * BLOCK_SIZE
    var end = start + BLOCK_SIZE
    if end > total_size:
        end = total_size
    var s = Scalar[dtype](0.0)
    for i in range(start, end):
        s += data[i]
    partial_sums[bid] = s


fn _gpu_reduce_final_kernel[dtype: DType](
    partial_sums: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    out_single: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    n_blocks: Int,
):
    """GPU kernel: single thread sums partial_sums[0..n_blocks) into out_single[0].
    Enables copying only 1 element back to host.
    """
    if block_idx.x != 0 or thread_idx.x != 0:
        return
    var s = Scalar[dtype](0.0)
    for i in range(n_blocks):
        s += partial_sums[i]
    out_single[0] = s


fn _device_reduce_sum_to_scalar[dtype: DType](
    ctx: DeviceContext,
    data: DeviceBuffer[dtype],
    total_size: Int,
) raises -> Float64:
    """Reduce device buffer to a single sum; copy only 1 element to host. No O(n) transfer."""
    if total_size == 0:
        return 0.0
    alias BLOCK_SIZE = 256
    var num_blocks = ceildiv(total_size, BLOCK_SIZE)
    var partial_sums = ctx.enqueue_create_buffer[dtype](num_blocks)
    var single_out = ctx.enqueue_create_buffer[dtype](1)
    ctx.enqueue_function[_gpu_reduce_block_sum_kernel[dtype]](
        data.unsafe_ptr(),
        partial_sums.unsafe_ptr(),
        total_size,
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE,
    )
    ctx.enqueue_function[_gpu_reduce_final_kernel[dtype]](
        partial_sums.unsafe_ptr(),
        single_out.unsafe_ptr(),
        num_blocks,
        grid_dim=1,
        block_dim=1,
    )
    var host_single = ctx.enqueue_create_host_buffer[dtype](1)
    ctx.enqueue_copy(host_single, single_out)
    ctx.synchronize()
    return Float64(host_single[0])


fn _gpu_strided_copy_2d_kernel[dtype: DType](
    src: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    m: Int,
    n: Int,
    stride0: Int,
    stride1: Int,
):
    """GPU kernel: copy 2D tensor from strided layout to row-major contiguous. No host transfer."""
    var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
    var total = m * n
    if tid >= total:
        return
    var i = tid // n
    var j = tid % n
    var src_idx = i * stride0 + j * stride1
    dst[tid] = src[src_idx]


fn _gpu_transpose_kernel[dtype: DType](
    src: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    meta: UnsafePointer[Scalar[DType.int32], origin=MutAnyOrigin],
    total_size: Int,
    rank: Int,
):
    """GPU kernel: each thread transposes one element.

    For destination element at linear index `tid`, decomposes into
    multi-index in the new (transposed) shape, then maps back to
    the source linear index using precomputed stride mapping.

    meta layout: [mapped_stride[0..rank-1], new_shape[0..rank-1]]
    where mapped_stride[i] = old_stride[perm[i]].
    """
    var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
    if tid >= total_size:
        return

    var rem = tid
    var src_linear: Int = 0
    for i in range(rank - 1, -1, -1):
        var new_dim = Int(meta[rank + i])
        var dest_idx = rem % new_dim
        rem = rem // new_dim
        src_linear += dest_idx * Int(meta[i])

    dst[tid] = src[src_linear]


## Fully dense tensor with runtime-determined rank, shape, and stride
@fieldwise_init
struct DenseTensor[dtype: DType](Writable, Movable, ImplicitlyCopyable):
    """A tensor where rank, shape, and stride are all determined at runtime.
    
    Unlike DenseTensor which requires compile-time Layout parameter, this tensor
    allows complete flexibility at runtime.
    
    Parameters:
        dtype: The data type of the tensor elements (e.g., DType.float32, DType.float32).
    """
    var storage: DeviceBuffer[dtype]  # GPU storage for tensor data
    var shape: List[Int]  # Runtime shape
    var stride: List[Int]  # Runtime stride
    var size: Int  # Total number of elements

    fn __init__(out self, storage: DeviceBuffer[dtype], var shape: List[Int], var stride: List[Int]):
        """Initialize a dense tensor with runtime parameters.
        
        Args:
            storage: Device buffer containing the tensor data.
            shape: List of dimensions (length = rank).
            stride: List of strides (length = rank).
            rank: Number of dimensions.
        """
        self.storage = storage
        self.shape = shape^
        self.stride = stride^
        
        var total_size = 1
        for elem in self.shape:
            total_size *= elem
        self.size = total_size

    fn __copyinit__(out self, existing: Self):
        """Copy constructor for DenseTensor.
        
        Creates a new tensor that shares the same GPU storage but has independent
        shape, stride, and size metadata. This is a shallow copy - the underlying
        GPU buffer is shared between copies.
        
        Args:
            existing: The tensor to copy from.
        """
        self.storage = existing.storage
        self.shape = existing.shape.copy()
        self.stride = existing.stride.copy()
        self.size = existing.size

    fn write_to[W: Writer](self, mut writer: W) -> None:
        """Write tensor information to a writer."""
        var rank = len(self.shape)
        writer.write("DenseTensor[rank=")
        writer.write(rank)
        writer.write(", shape=(")
        for i in range(rank):
            if i > 0:
                writer.write(", ")
            writer.write(self.shape[i])
        writer.write("), size=")
        writer.write(self.size)
        writer.write("]")

    fn print_tensor(self, ctx: DeviceContext) raises -> None:
        """Print the entire tensor contents (works for any rank)."""
        var host_out = ctx.enqueue_create_host_buffer[Self.dtype](self.size)
        ctx.enqueue_copy(host_out, self.storage)
        ctx.synchronize()

        var rank = len(self.shape)
        print("DenseTensor with rank ", rank, " and shape ", end="")
        print("(", end="")
        for i, elem in enumerate(self.shape):
            if i > 0:
                print(", ", end="")
            print(elem, end="")
        print("):")
        
        # For simplicity, print as flat array for rank > 2
        if rank == 1:
            # Vector
            for i in range(self.shape[0]):
                print("[", i, "] = ", host_out[i])
        elif rank == 2:
            # Matrix
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    var idx = i * self.stride[0] + j * self.stride[1]
                    print("[", i, ",", j, "] = ", host_out[idx])
        else:
            # Higher rank - print with multi-indices
            print("(Printing as flat array with indices):")
            for i in range(min(self.size, 100)):  # Limit output to first 100 elements
                print("[flat_idx ", i, "] = ", host_out[i])
            if self.size > 100:
                print("... (", self.size - 100, " more elements)")

    fn get_flat_index(self, indices: List[Int]) -> Int:
        """Compute flat index from multi-dimensional indices using stride.
        
        Args:
            indices: Multi-dimensional indices (length must equal rank).
        
        Returns:
            Flat index into the storage buffer.
        """
        var flat_idx = 0
        for i in range(len(self.shape)):
            flat_idx += indices[i] * self.stride[i]
        return flat_idx

    fn is_contiguous(self: DenseTensor) -> Bool:
        """Check if tensor memory layout is contiguous in row-major order.
        
        A contiguous tensor means elements are stored sequentially in memory without gaps.
        This is important for GPU efficiency as contiguous memory access patterns are faster.
        
        Row-major order means the last dimension changes fastest:
        - For shape [3, 4], row-major strides are [4, 1]
        - For shape [2, 3, 4], row-major strides are [12, 4, 1]
        
        Args:
            self: The tensor to check.
        
        Returns:
            True if the tensor is contiguous in row-major order, False otherwise.
        
        Example:
            ```
            
            mojo
            # Contiguous tensor
            var shape = List[Int](3, 4)
            var tensor = create_dense_tensor(ctx, shape^)
            print(tensor.is_contiguous())  # True, strides are [4, 1]
            
            # Non-contiguous after slicing or view operations
            var transposed = tensor^.transpose(List[Int](1, 0), ctx)
            print(transposed.is_contiguous())  # Might be False
            
        """
        var expected_stride = 1
        for i in range(len(self.shape) - 1, -1, -1):
            if self.stride[i] != expected_stride:
                return False
            expected_stride *= self.shape[i]
        return True

    fn copy_to_contiguous(var self, ctx: DeviceContext) raises -> DenseTensor[dtype]:
        """Create a contiguous copy of the tensor in row-major order.
        
        If the tensor is already contiguous, transfers ownership without copying.
        Otherwise, allocates new GPU memory and copies data to a contiguous layout.
        
        This operation is useful when:
        - Working with sliced or transposed tensors that are not contiguous
        - Preparing tensors for operations that require contiguous memory
        - Optimizing memory access patterns for GPU kernels
        
        Args:
            self: The tensor to make contiguous (ownership transferred).
            ctx: Device context for GPU memory operations.
        
        Returns:
            A new DenseTensor with contiguous row-major layout.
        
        Raises:
            Error: If GPU memory allocation or copy fails.
        
        Note:
            This is a simplified implementation. For true non-contiguous tensors
            (e.g., after advanced slicing), a more complex reorganization kernel
            would be needed.
        
        Example:mojo
            # Create a tensor and transpose it (may become non-contiguous)
            var tensor = create_dense_tensor(ctx, List[Int](4, 3)^)
            var perm = List[Int](1, 0)
            var transposed = tensor^.transpose(perm, ctx)
            
            # Make it contiguous for efficient operations
            var contiguous = transposed^.copy_to_contiguous(ctx)
            print(contiguous.is_contiguous())  # True
        """
        if self.is_contiguous():
            return self^  # No copy needed, transfer ownership

        var total_size = self.size
        var new_shape = self.shape.copy()
        var new_strides = compute_row_major_strides(new_shape, len(new_shape))
        var new_storage = ctx.enqueue_create_buffer[Self.dtype](total_size)
        
        # Simple copy - this works for contiguous data
        # For true non-contiguous support, need more complex kernel
        ctx.enqueue_copy(new_storage, self.storage)
        return DenseTensor[dtype](new_storage, new_shape^, new_strides^)

    fn transpose(var self, perm: List[Int], ctx: DeviceContext) raises -> DenseTensor[dtype]:
        """Transpose tensor dimensions according to a permutation.
        
        Reorders the dimensions of the tensor according to the permutation list.
        Creates a physically transposed copy in memory so the result is contiguous
        and compatible with reshape and other operations that assume row-major layout.
        
        Runs entirely on GPU via `_gpu_transpose_kernel` — no bulk host-device
        data transfers. Only a tiny metadata buffer (~tens of bytes) is copied
        to the device for the permutation map.
        
        Args:
            self: The tensor to transpose (ownership transferred).
            perm: Permutation list specifying the new axis order.
                  perm[i] = j means new dimension i comes from original dimension j.
                  Must have length equal to tensor rank.
            ctx: Device context for GPU operations.
        
        Returns:
            A new DenseTensor with physically transposed data (contiguous row-major).
        
        Raises:
            Error: If perm length doesn't match tensor rank.
        
        Example:mojo
            # 2D matrix transpose (standard transpose)
            var matrix = create_dense_tensor_from_data(
                ctx,
                List[Float32](1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
                List[Int](2, 3)^  # Shape: [2, 3]
            )
            var perm = List[Int](1, 0)  # Swap rows and columns
            var transposed = matrix^.transpose(perm, ctx)
            # Result shape: [3, 2], contiguous in memory
            
            # 6D tensor: [Wl0, d_in0, d_out0, d_in1, d_out1, Wr1] -> [Wl0, d_in0, d_in1, d_out0, d_out1, Wr1]
            var perm6 = List[Int](0, 1, 3, 2, 4, 5)
            var transposed_6d = tensor_6d^.transpose(perm6, ctx)
        """
        var rank = len(self.shape)
        if len(perm) != rank:
            raise Error("Perm length mismatch")

        # Build new shape: new_shape[k] = old_shape[perm[k]]
        var new_shape = List[Int](capacity=rank)
        for k in range(rank):
            new_shape.append(self.shape[perm[k]])

        var new_strides = compute_row_major_strides(new_shape, rank)
        var total_size = self.size

        if total_size == 0:
            var empty_storage = ctx.enqueue_create_buffer[Self.dtype](1)
            return DenseTensor[dtype](empty_storage, new_shape^, new_strides^)

        var new_storage = ctx.enqueue_create_buffer[Self.dtype](total_size)

        # Pack small metadata buffer for the GPU kernel:
        #   [mapped_stride[0..rank-1], new_shape[0..rank-1]]
        # where mapped_stride[i] = old_stride[perm[i]]
        # This is at most ~12 int32 values (rank <= 6), so the H->D copy is negligible.
        var meta_size = 2 * rank
        var host_meta = ctx.enqueue_create_host_buffer[DType.int32](meta_size)
        for i in range(rank):
            host_meta[i] = Scalar[DType.int32](self.stride[perm[i]])
        for i in range(rank):
            host_meta[rank + i] = Scalar[DType.int32](new_shape[i])

        var dev_meta = ctx.enqueue_create_buffer[DType.int32](meta_size)
        ctx.enqueue_copy(dev_meta, host_meta)

        # Launch GPU kernel — all tensor data stays on device
        alias BLOCK_SIZE = 256
        var grid_size = ceildiv(total_size, BLOCK_SIZE)

        ctx.enqueue_function[_gpu_transpose_kernel[Self.dtype]](
            self.storage.unsafe_ptr(),
            new_storage.unsafe_ptr(),
            dev_meta.unsafe_ptr(),
            total_size,
            rank,
            grid_dim=grid_size,
            block_dim=BLOCK_SIZE,
        )

        return DenseTensor[dtype](new_storage, new_shape^, new_strides^)

    fn flatten_dims(var self, start: Int, end: Int, ctx: DeviceContext) raises -> DenseTensor[dtype]:
        """Flatten a contiguous range of dimensions into a single dimension.
        
        Combines multiple consecutive dimensions into one by multiplying their sizes.
        This is useful for reshaping tensors before operations like matrix multiplication,
        or when converting from higher-dimensional tensors to matrices.
        
        The flattening occurs on dimensions in the range [start, end) (end is exclusive).
        Dimensions before start and after end remain unchanged.
        
        Args:
            self: The tensor to flatten (ownership transferred).
            start: Starting dimension index (inclusive, 0-based).
            end: Ending dimension index (exclusive).
            ctx: Device context (currently unused but kept for API consistency).
        
        Returns:
            A new DenseTensor with the specified dimensions flattened.
        
        Raises:
            Error: If start < 0, end > rank, or start >= end.
        
        Note:
            This creates a view with recomputed strides. No data is copied.
            The underlying storage is shared with the original tensor.
        
        Example:mojo
            # Flatten middle dimensions of a 4D tensor
            var tensor_4d = create_dense_tensor(ctx, List[Int](2, 3, 4, 5)^)
            # Original shape: [2, 3, 4, 5]
            
            # Flatten dimensions 1 and 2 (indices 1 and 2, i.e., 3 and 4)
            var flattened = tensor_4d^.flatten_dims(1, 3, ctx)
            # Result shape: [2, 12, 5]  (3*4=12)
            
            # Flatten all dimensions (convert to 1D vector)
            var tensor_3d = create_dense_tensor(ctx, List[Int](2, 3, 4)^)
            var vector = tensor_3d^.flatten_dims(0, 3, ctx)
            # Original shape: [2, 3, 4]
            # Result shape:   [24]  (2*3*4=24)
            
            # Flatten last two dimensions (useful for batched matrix ops)
            var batched = create_dense_tensor(ctx, List[Int](8, 5, 10)^)
            var flat_batch = batched^.flatten_dims(1, 3, ctx)
            # Original shape: [8, 5, 10]  (batch of 8 matrices)
            # Result shape:   [8, 50]     (batch of 8 vectors)
        """
        if start < 0 or end > len(self.shape) or start >= end:
            raise Error("Invalid flatten range")
        
        var flat_dim = 1
        for i in range(start, end):
            flat_dim *= self.shape[i]
        
        var new_shape = List[Int]()
        for i in range(start):
            new_shape.append(self.shape[i])
        new_shape.append(flat_dim)
        for i in range(end, len(self.shape)):
            new_shape.append(self.shape[i])
        
        # Compute new strides for the flattened shape
        var new_strides = compute_row_major_strides(new_shape, len(new_shape))
        
        # Return a view with updated shape/stride (no data copy)
        return DenseTensor[dtype](self.storage^, new_shape^, new_strides^)

    fn reshape(var self, var new_shape: List[Int]) raises -> DenseTensor[dtype]:
        """Return a tensor view with a different shape but identical storage."""
        if len(new_shape) == 0:
            raise Error("Reshape requires rank >= 1")

        var total = 1
        for dim in new_shape:
            if dim < 1:
                raise Error("Reshape dimensions must be positive, got " + String(dim))
            total *= dim

        if total != self.size:
            raise Error(
                "Reshape size mismatch: original "
                + String(self.size)
                + " vs new "
                + String(total)
            )

        var rank = len(new_shape)
        var new_strides = compute_row_major_strides(new_shape, rank)
        return DenseTensor[dtype](self.storage^, new_shape^, new_strides^)

    fn norm(self, ctx: DeviceContext) raises -> Float64:
        """Compute the Frobenius norm using GPU element-wise square + GPU reduction.
        
        No O(n) device→host copy: reduction is done on GPU, only 1 scalar is copied back.
        """
        if self.size == 0:
            return 0.0

        var scratch = ctx.enqueue_create_buffer[Self.dtype](self.size)
        ctx.enqueue_copy(scratch, self.storage)

        alias BLOCK_SIZE = 256
        var grid_size = ceildiv(self.size, BLOCK_SIZE)
        ctx.enqueue_function[_gpu_norm_sq_kernel[Self.dtype]](
            scratch.unsafe_ptr(),
            scratch.unsafe_ptr(),
            self.size,
            grid_dim=grid_size,
            block_dim=BLOCK_SIZE,
        )

        var sum_sq = _device_reduce_sum_to_scalar[Self.dtype](ctx, scratch, self.size)
        return sqrt(sum_sq)

    fn norm_sq(self, ctx: DeviceContext) raises -> Float64:
        """Compute the squared Frobenius norm (no sqrt).
        
        Returns ||self||_F^2 = sum_i |x_i|^2. GPU reduction; only 1 scalar copied to host.
        """
        if self.size == 0:
            return 0.0

        var scratch = ctx.enqueue_create_buffer[Self.dtype](self.size)
        ctx.enqueue_copy(scratch, self.storage)

        alias BLOCK_SIZE = 256
        var grid_size = ceildiv(self.size, BLOCK_SIZE)
        ctx.enqueue_function[_gpu_norm_sq_kernel[Self.dtype]](
            scratch.unsafe_ptr(),
            scratch.unsafe_ptr(),
            self.size,
            grid_dim=grid_size,
            block_dim=BLOCK_SIZE,
        )

        return _device_reduce_sum_to_scalar[Self.dtype](ctx, scratch, self.size)

    fn scale_in_place(var self, scale: Scalar[dtype], ctx: DeviceContext) raises -> None:
        """Scale tensor entries by a scalar factor in-place (GPU kernel).
        No synchronize: work is enqueued only; caller syncs when a host value is needed.
        """
        if self.size == 0:
            return

        var host_scale = ctx.enqueue_create_host_buffer[Self.dtype](1)
        host_scale[0] = scale
        var dev_scale = ctx.enqueue_create_buffer[Self.dtype](1)
        ctx.enqueue_copy(dev_scale, host_scale)

        alias BLOCK_SIZE = 256
        var grid_size = ceildiv(self.size, BLOCK_SIZE)
        ctx.enqueue_function[_gpu_scale_kernel[Self.dtype]](
            self.storage.unsafe_ptr(),
            dev_scale.unsafe_ptr(),
            self.size,
            grid_dim=grid_size,
            block_dim=BLOCK_SIZE,
        )

    fn dot_product(self, other: Self, ctx: DeviceContext) raises -> Float64:
        """Compute inner product <self, other> using GPU element-wise multiply + GPU reduction.
        
        No O(n) device→host: only 1 scalar is copied back after reduction on GPU.
        """
        if self.size != other.size:
            raise Error("dot_product size mismatch: " + String(self.size) + " vs " + String(other.size))
        if self.size == 0:
            return 0.0

        var scratch = ctx.enqueue_create_buffer[Self.dtype](self.size)

        alias BLOCK_SIZE = 256
        var grid_size = ceildiv(self.size, BLOCK_SIZE)
        ctx.enqueue_function[_gpu_dot_kernel[Self.dtype]](
            self.storage.unsafe_ptr(),
            other.storage.unsafe_ptr(),
            scratch.unsafe_ptr(),
            self.size,
            grid_dim=grid_size,
            block_dim=BLOCK_SIZE,
        )

        return _device_reduce_sum_to_scalar[Self.dtype](ctx, scratch, self.size)

    fn axpy_in_place(var self, alpha: Scalar[dtype], x: Self, ctx: DeviceContext) raises -> None:
        """self += alpha * x   (BLAS-style axpy, GPU kernel).
        No synchronize: work is enqueued only; caller syncs when a host value is needed.
        """
        if self.size != x.size:
            raise Error("axpy size mismatch: " + String(self.size) + " vs " + String(x.size))
        if self.size == 0:
            return

        var host_alpha = ctx.enqueue_create_host_buffer[Self.dtype](1)
        host_alpha[0] = alpha
        var dev_alpha = ctx.enqueue_create_buffer[Self.dtype](1)
        ctx.enqueue_copy(dev_alpha, host_alpha)

        alias BLOCK_SIZE = 256
        var grid_size = ceildiv(self.size, BLOCK_SIZE)
        ctx.enqueue_function[_gpu_axpy_kernel[Self.dtype]](
            self.storage.unsafe_ptr(),
            x.storage.unsafe_ptr(),
            dev_alpha.unsafe_ptr(),
            self.size,
            grid_dim=grid_size,
            block_dim=BLOCK_SIZE,
        )

    @staticmethod
    fn random(
        ctx: DeviceContext,
        var shape: List[Int],
        row_major: Bool = True
    ) raises -> DenseTensor[dtype]:
        """Create a tensor filled with random values in [0, 1)."""
        return create_dense_tensor[dtype](ctx, shape^, row_major=row_major)



fn compute_row_major_strides(shape: List[Int], rank: Int) -> List[Int]:
    """Compute row-major strides for given shape.
    
    Row-major means the last dimension changes fastest.
    For shape [2, 3, 4], strides are [12, 4, 1].
    
    Args:
        shape: Dimensions of the tensor.
        rank: Number of dimensions.
    
    Returns:
        List of strides in row-major order.
    """
    var strides = List[Int]()
    var stride = 1
    
    # Compute strides in reverse order (row-major)
    for i in range(rank - 1, -1, -1):
        strides.insert(0, stride) # 24, 4, 1
        stride *= shape[i] # sh: 4, st:4 | sh:3, st: 12 | sh: 2, st: 24 
    
    return strides.copy()


fn compute_column_major_strides(shape: List[Int], rank: Int) -> List[Int]:
    """Compute column-major strides for given shape.
    
    Column-major means the first dimension changes fastest.
    For shape [2, 3, 4], strides are [1, 2, 6].
    
    Args:
        shape: Dimensions of the tensor.
        rank: Number of dimensions.
    
    Returns:
        List of strides in column-major order.
    """
    var strides = List[Int]()
    var stride = 1
    
    # Compute strides in forward order (column-major)
    for i in range(rank):
        strides.append(stride)
        stride *= shape[i]
    
    return strides.copy()


fn create_dense_tensor[dtype: DType = DType.float32](
    ctx: DeviceContext, 
    var shape: List[Int], 
    row_major: Bool = True,
    init_value: Optional[Scalar[dtype]] = None  # None for random init
) raises -> DenseTensor[dtype]:
    """Create a dense tensor with runtime-determined rank, shape, and stride.
    
    When init_value is set, storage is filled on device via a GPU kernel (no host buffer).
    When init_value is None (random init), data is filled on host then copied once to device.
    
    Args:
        ctx: Device context for GPU operations.
        shape: List of dimensions (length should be rank).
        row_major: If True, use row-major stride; otherwise column-major.
        init_value: Initial value to fill the tensor. If None, initialize with random values in [0,1).
    
    Returns:
        DenseTensor with allocated GPU storage.
    """
    var total_size = 1
    var rank = len(shape)
    for i in range(rank):
        total_size *= shape[i]
    
    var strides: List[Int]
    if row_major:
        strides = compute_row_major_strides(shape, rank)
    else:
        strides = compute_column_major_strides(shape, rank)
    
    if init_value is not None:
        # Device-only path: allocate on GPU, fill with GPU kernel. No host buffer for bulk data.
        var device_storage = ctx.enqueue_create_buffer[dtype](total_size)
        var value = init_value.value()
        var host_val = ctx.enqueue_create_host_buffer[dtype](1)
        host_val[0] = value
        var dev_val = ctx.enqueue_create_buffer[dtype](1)
        ctx.enqueue_copy(dev_val, host_val)
        alias BLOCK_SIZE = 256
        var grid_size = ceildiv(total_size, BLOCK_SIZE)
        ctx.enqueue_function[_gpu_fill_kernel[dtype]](
            device_storage.unsafe_ptr(),
            dev_val.unsafe_ptr(),
            total_size,
            grid_dim=grid_size,
            block_dim=BLOCK_SIZE,
        )
        return DenseTensor[dtype](device_storage, shape^, strides^)
    
    # Random init: host buffer then single copy to device (only path that uses host for bulk data)
    var host_storage = ctx.enqueue_create_host_buffer[dtype](total_size)
    for i in range(total_size):
        host_storage[i] = Scalar[dtype](random_float64())
    var device_storage = ctx.enqueue_create_buffer[dtype](total_size)
    ctx.enqueue_copy(device_storage, host_storage)
    return DenseTensor[dtype](device_storage, shape^, strides^)


fn create_dense_tensor_uninitialized[dtype: DType = DType.float32](
    ctx: DeviceContext,
    var shape: List[Int],
    row_major: Bool = True,
) raises -> DenseTensor[dtype]:
    """Create a dense tensor WITHOUT initialization - just allocate GPU memory.
    
    Use this for output tensors that will be immediately overwritten by matmul,
    contraction, or other operations. Avoids unnecessary overhead of:
    - Host buffer allocation for the scalar
    - Device buffer allocation for the scalar
    - Host-to-device copy of the scalar
    - GPU fill kernel launch
    
    Args:
        ctx: Device context for GPU operations.
        shape: List of dimensions.
        row_major: If True, use row-major stride; otherwise column-major.
    
    Returns:
        DenseTensor with allocated but uninitialized GPU storage.
    """
    var total_size = 1
    var rank = len(shape)
    for i in range(rank):
        total_size *= shape[i]
    
    var strides: List[Int]
    if row_major:
        strides = compute_row_major_strides(shape, rank)
    else:
        strides = compute_column_major_strides(shape, rank)
    
    var device_storage = ctx.enqueue_create_buffer[dtype](total_size)
    return DenseTensor[dtype](device_storage, shape^, strides^)


fn create_dense_tensor_from_data[dtype: DType = DType.float32](
    ctx: DeviceContext, 
    data: List[Scalar[dtype]],
    var shape: List[Int], 
    row_major: Bool = True
) raises -> DenseTensor[dtype]:
    """Create a dense tensor from existing host data.
    
    **Host boundary**: This is the only creation API that copies bulk data from host to device.
    Use only at boundaries (e.g. loading initial state); avoid in hot loops.
    
    Args:
        ctx: Device context for GPU operations.
        data: Flat list of data values.
        shape: List of dimensions (length should be rank).
        row_major: If True, use row-major stride; otherwise column-major.
    
    Returns:
        DenseTensor with data copied to GPU.
    
    Example:
        var data = List[Float32](1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        var shape = List[Int](2, 3)  # 2x3 matrix
        var tensor = create_dense_tensor_from_data(ctx, data, shape, 2).
    """
    # Verify data size matches shape
    var expected_size = 1
    var rank = len(shape)
    for i in range(rank):
        expected_size *= shape[i]
    
    var data_size = len(data)
    if data_size != expected_size:
        raise Error("Data size mismatch: expected " + String(expected_size) + " but got " + String(data_size))
    
    # Compute strides
    var strides: List[Int]
    if row_major:
        strides = compute_row_major_strides(shape, rank)
    else:
        strides = compute_column_major_strides(shape, rank)
    
    # Copy data to GPU
    var host_storage = ctx.enqueue_create_host_buffer[dtype](data_size)
    for i in range(data_size):
        host_storage[i] = data[i]
    
    var device_storage = ctx.enqueue_create_buffer[dtype](data_size)
    ctx.enqueue_copy(device_storage, host_storage)
    
    return DenseTensor[dtype](device_storage, shape^, strides^)

fn dense_tensor_dot[dtype: DType = DType.float32](C: DenseTensor[dtype], var A: DenseTensor[dtype], var B: DenseTensor[dtype], ctx: DeviceContext, ndim_mult: Int = 1, axrange_A: Bool = False, axrange_B: Bool = False) raises:  # axrange False=trailing, True=leading
    """Perform generalized tensor dot product (contraction) on GPU.
    
    This function implements Einstein summation-style tensor contraction by:
    1. Identifying which axes to contract (sum over)
    2. Flattening non-contracted and contracted dimensions
    3. Performing matrix multiplication on the flattened tensors
    4. Storing the result in the pre-allocated output tensor C
    
    For standard 2D matrix multiplication A[m,k] @ B[k,n] = C[m,n], simply use
    default parameters. The function automatically adjusts axis contraction.
    
    **IMPORTANT - Automatic Axis Inference**: 
    When BOTH axrange_A=False AND axrange_B=False (the defaults), the function
    automatically infers which axes of B to contract by checking dimensions:
    
    - If A's trailing ndim_mult axes match B's LEADING ndim_mult axes → uses B leading
      (standard linear algebra: A[..., k] @ B[k, ...])
    - If A's trailing ndim_mult axes match B's TRAILING ndim_mult axes → uses B trailing
      (tensor network style: A[..., k] @ B[..., k])
    - Prefers B leading if both match (e.g., square tensors)
    
    To explicitly control which axes to contract without inference, specify axrange_B:
    - For A[k, ...] @ B[..., k]: set `axrange_A=True, axrange_B=False`
    - For standard matmul without inference: set `axrange_A=False, axrange_B=True`
    
    Algorithm:
    - Contracts ndim_mult axes between A and B
    - axrange_A/B control whether to contract leading (True) or trailing (False) axes
    - Flattens remaining dimensions and performs GPU matrix multiplication
    - Handles non-contiguous tensors by copying to contiguous layout
    
    Args:
        C: Output tensor (must be pre-allocated with correct shape).
           For standard matmul: shape = [A.shape[0], ..., A.shape[rank_A-ndim_mult-1],
                                         B.shape[ndim_mult], ..., B.shape[rank_B-1]]
        A: First input tensor (ownership transferred for internal operations).
        B: Second input tensor (ownership transferred for internal operations).
        ctx: Device context for GPU operations and memory management.
        ndim_mult: Number of axes to contract (default=1 for matrix multiplication).
        axrange_A: If False (default), contract A's trailing axes.
                   If True, contract A's leading axes.
        axrange_B: If False (default), contract B's trailing axes.
                   If True, contract B's leading axes.
                   
                   **IMPORTANT**: When both axrange_A and axrange_B are False (defaults),
                   the function automatically infers which B axes to use by checking if
                   A's trailing dims match B's leading or trailing dims. This inference
                   is skipped if you explicitly pass axrange_B=True or axrange_A=True.
                   See "IMPORTANT - Automatic Axis Inference" section above for details.
    
    Raises:
        Error: If ndim_mult < 1, or tensors don't have enough dimensions.
        Error: If contracted dimensions don't match in size.
        Error: If C doesn't have the correct shape for the result.
    
    Examples:mojo
        # Example 1: Standard 2D matrix multiplication
        # A[3, 4] @ B[4, 5] = C[3, 5]
        with DeviceContext() as ctx:
            var A = create_dense_tensor(ctx, List[Int](3, 4)^, init_value=2.0)
            var B = create_dense_tensor(ctx, List[Int](4, 5)^, init_value=3.0)
            var C = create_dense_tensor(ctx, List[Int](3, 5)^, init_value=0.0)
            
            dense_tensor_dot(C, A^, B^, ctx)
            # C now contains the result of A @ B
            # Each element C[i,j] = sum over k of A[i,k] * B[k,j]
        
        # Example 2: 3D tensor contraction
        # A[2, 3, 4] @ B[4, 5, 6] = C[2, 3, 5, 6]
        # Contracts A's last axis (4) with B's first axis (4)
        with DeviceContext() as ctx:
            var A_3d = create_dense_tensor(ctx, List[Int](2, 3, 4)^, init_value=1.5)
            var B_3d = create_dense_tensor(ctx, List[Int](4, 5, 6)^, init_value=2.0)
            var C_4d = create_dense_tensor(ctx, List[Int](2, 3, 5, 6)^, init_value=0.0)
            
            dense_tensor_dot(C_4d, A_3d^, B_3d^, ctx, ndim_mult=1)
            # C now contains the 4D result
            # Non-contracted dims: [2,3] from A + [5,6] from B = [2,3,5,6]
        
        # Example 3: Contracting multiple axes
        # A[2, 3, 4, 5] with shape breakdown: [batch(2,3), contract(4,5)]
        # B[4, 5, 6, 7] with shape breakdown: [contract(4,5), result(6,7)]
        # Result C[2, 3, 6, 7]
        # This contracts 2 dimensions (4 and 5) between A and B
        with DeviceContext() as ctx:
            var A_4d = create_dense_tensor(ctx, List[Int](2, 3, 4, 5)^)
            var B_4d = create_dense_tensor(ctx, List[Int](4, 5, 6, 7)^)
            var C_4d = create_dense_tensor(ctx, List[Int](2, 3, 6, 7)^)
            
            # Contract last 2 dims of A with first 2 dims of B
            dense_tensor_dot(C_4d, A_4d^, B_4d^, ctx, ndim_mult=2)
            # Contracts over axes 2,3 of A with axes 0,1 of B
        
        # Example 4: Tensor Network Convention
        # Contract leading axes of A with trailing axes of B (common in quantum chemistry)
        with DeviceContext() as ctx:
            # A[k, m, n] @ B[p, q, k] = C[m, n, p, q]
            var A_tn = create_dense_tensor(ctx, List[Int](3, 4, 5)^)
            var B_tn = create_dense_tensor(ctx, List[Int](6, 7, 3)^)
            var C_tn = create_dense_tensor(ctx, List[Int](4, 5, 6, 7)^)
            
            # Contract first axis of A (size 3) with last axis of B (size 3)
            dense_tensor_dot(C_tn, A_tn^, B_tn^, ctx, ndim_mult=1, 
                           axrange_A=True,   # Contract A's leading axis
                           axrange_B=False)  # Contract B's trailing axis
        
        # Example 5: Standard vs Tensor Network notation comparison
        with DeviceContext() as ctx:
            # Standard linear algebra: A[m,k] @ B[k,n] (default)
            var A_std = create_dense_tensor(ctx, List[Int](3, 4)^)
            var B_std = create_dense_tensor(ctx, List[Int](4, 5)^)
            var C_std = create_dense_tensor(ctx, List[Int](3, 5)^)
            dense_tensor_dot(C_std, A_std^, B_std^, ctx)  # Uses defaults
            
            # Tensor network style: A[k,m] @ B[n,k] 
            # Need to explicitly set parameters:
            var A_tn2 = create_dense_tensor(ctx, List[Int](4, 3)^)
            var B_tn2 = create_dense_tensor(ctx, List[Int](5, 4)^)
            var C_tn2 = create_dense_tensor(ctx, List[Int](3, 5)^)
            dense_tensor_dot(C_tn2, A_tn2^, B_tn2^, ctx, ndim_mult=1,
                           axrange_A=True, axrange_B=False)
        
        # Example 6: Automatic Axis Inference (both defaults)
        with DeviceContext() as ctx:
            # Case 1: A trailing matches B leading → automatically infers B leading
            # A[3, 4], B[4, 5] - default call infers A[3,4] @ B[4,5] (standard matmul)
            var A1 = create_dense_tensor(ctx, List[Int](3, 4)^)
            var B1 = create_dense_tensor(ctx, List[Int](4, 5)^)
            var C1 = create_dense_tensor(ctx, List[Int](3, 5)^)
            dense_tensor_dot(C1, A1^, B1^, ctx)  # Infers: contract A's [4] with B's leading [4]
            
            # Case 2: A trailing matches B trailing → automatically infers B trailing
            # A[3, 4], B[5, 4] - default call infers A[3,4] @ B[5,4] with B trailing
            var A2 = create_dense_tensor(ctx, List[Int](3, 4)^)
            var B2 = create_dense_tensor(ctx, List[Int](5, 4)^)
            var C2 = create_dense_tensor(ctx, List[Int](3, 5)^)
            dense_tensor_dot(C2, A2^, B2^, ctx)  # Infers: contract A's [4] with B's trailing [4]
            
            # Case 3: Multi-dimensional inference (ndim_mult=2)
            # A[2,3,4,5], B[4,5,6,7] - infers B leading since [4,5] matches B's first 2 dims
            var A3 = create_dense_tensor(ctx, List[Int](2, 3, 4, 5)^)
            var B3 = create_dense_tensor(ctx, List[Int](4, 5, 6, 7)^)
            var C3 = create_dense_tensor(ctx, List[Int](2, 3, 6, 7)^)
            dense_tensor_dot(C3, A3^, B3^, ctx, ndim_mult=2)  # Infers B leading
    
    Performance Notes:
        - float32: Uses `linalg.matmul` exclusively. When a logical transpose is required
          (leading axes of A or trailing axes of B), physically transposes first via
          `DenseTensor.transpose`, then calls `linalg.matmul`.
        - Other dtypes: not supported on GPU in this build (raises after shape checks).
    
    Implementation Details:
        The function internally:
        1. Validates dimensions and contraction axes
        2. Saves shape information before transferring ownership
        3. Ensures tensors are contiguous (copies if needed) (TODO: NOT SURE ABOUT THIS COPYING MIGHT BE COSTLY)
        4. Flattens tensors to 2D matrices (non-contracted dims × contracted dims)
        5. Physically transposes A/B when needed for row-major GEMM layout
        6. Calls `linalg.matmul` and writes result directly to output tensor C
    """
    var rank_A = len(A.shape)
    var rank_B = len(B.shape)
    if ndim_mult < 1 or rank_A < ndim_mult or rank_B < ndim_mult:
        raise Error("Invalid ndim_mult")

    # ============================================================================
    # AUTOMATIC AXIS INFERENCE (when both axrange_A=False and axrange_B=False)
    # ============================================================================
    # When both flags are False (the defaults), we automatically determine which
    # axes of B to contract by checking if A's trailing axes match B's leading
    # or trailing axes. This allows the same default call to work for both:
    #   - Standard matmul: A[m,k] @ B[k,n] (A trailing matches B leading)
    #   - Tensor network: A[m,k] @ B[n,k] (A trailing matches B trailing)
    # 
    # To bypass this inference and explicitly control axes, pass axrange_B=True
    # (for B leading) or ensure you're not using both defaults.
    # ============================================================================
    var effective_axrange_B = axrange_B
    if not axrange_A and not axrange_B:
        # INFERENCE: Check which B axes (leading or trailing) match A's trailing axes
        var b_leading_matches = True
        var b_trailing_matches = True
        for i in range(ndim_mult):
            var a_sz = A.shape[rank_A - ndim_mult + i]
            if a_sz != B.shape[i]:
                b_leading_matches = False
            if a_sz != B.shape[rank_B - ndim_mult + i]:
                b_trailing_matches = False
        
        # Set effective_axrange_B based on which dimensions match
        if b_leading_matches:
            effective_axrange_B = True   # Inferred: A trailing × B leading (standard)
        elif b_trailing_matches:
            effective_axrange_B = False  # Inferred: A trailing × B trailing
        else:
            effective_axrange_B = True   # Neither matches; will raise "Dim mismatch" below

    # Check contracting dims match (Contract A: NO reversal, aligned blocks — matches C)
    # C: s->dim[(axrange_s ? 0 : s->ndim-ndim_mult)+i] == t->dim[(axrange_t ? 0 : t->ndim-ndim_mult)+i]
    for i in range(ndim_mult):
        var ax_A = (0 if axrange_A else rank_A - ndim_mult) + i
        var ax_B = (0 if effective_axrange_B else rank_B - ndim_mult) + i
        if A.shape[ax_A] != B.shape[ax_B]:
            raise Error("Dim mismatch")

    # Save shape information before moving ownership
    var A_noncont_start = ndim_mult if axrange_A else 0
    var A_noncont_end = rank_A if axrange_A else rank_A - ndim_mult
    var B_noncont_start = ndim_mult if effective_axrange_B else 0
    var B_noncont_end = rank_B if effective_axrange_B else rank_B - ndim_mult
    
    # Save the shapes we'll need later for C_shape computation
    var A_shape_for_C = List[Int]()
    for i in range(A_noncont_start, A_noncont_end):
        A_shape_for_C.append(A.shape[i])
    
    var B_shape_for_C = List[Int]()
    for i in range(B_noncont_start, B_noncont_end):
        B_shape_for_C.append(B.shape[i])

    # Ensure tensors are contiguous (copy_to_contiguous handles ownership transfer efficiently)
    var A_contig = A^ # .copy_to_contiguous(ctx)
    var B_contig = B^ # .copy_to_contiguous(ctx)

    # Flatten non-contracted dimensions first (skip if range is empty)
    var A_flat_step1: DenseTensor[dtype]
    if A_noncont_start < A_noncont_end:
        A_flat_step1 = A_contig^.flatten_dims(A_noncont_start, A_noncont_end, ctx)
    else:
        # No non-contracted dims (all dims are contracted) - use as-is
        A_flat_step1 = A_contig^
    
    var B_flat_step1: DenseTensor[dtype]
    if B_noncont_start < B_noncont_end:
        B_flat_step1 = B_contig^.flatten_dims(B_noncont_start, B_noncont_end, ctx)
    else:
        # No non-contracted dims (all dims are contracted) - use as-is
        B_flat_step1 = B_contig^
    
    # Now flatten the contracted dimensions
    # For A: if axrange_A is False (trailing), non-contracted are [0, rank-ndim], contracted are [rank-ndim, rank]
    #        After flattening non-contracted [0, rank-ndim), contracted dims are now at [1, 1+ndim_mult)
    # For A: if axrange_A is True (leading), non-contracted are [ndim, rank], contracted are [0, ndim]
    #        After flattening non-contracted [ndim, rank), contracted dims are still at [0, ndim_mult)
    var A_contract_start: Int
    var A_contract_end: Int
    if A_noncont_start < A_noncont_end:
        # Had non-contracted dims, so contracted dims shifted
        A_contract_start = 1 if not axrange_A else 0
        A_contract_end = A_contract_start + ndim_mult
    else:
        # All dims are contracted, they're at [0, ndim_mult)
        A_contract_start = 0
        A_contract_end = ndim_mult
    
    # For B: if effective_axrange_B is True (leading), contracted are [0, ndim], non-contracted are [ndim, rank]
    #        After flattening non-contracted [ndim, rank), contracted dims are still at [0, ndim_mult)
    # For B: if effective_axrange_B is False (trailing), non-contracted are [0, rank-ndim], contracted are [rank-ndim, rank]
    #        After flattening non-contracted [0, rank-ndim), contracted dims are now at [1, 1+ndim_mult)
    var B_contract_start: Int
    var B_contract_end: Int
    if B_noncont_start < B_noncont_end:
        # Had non-contracted dims
        B_contract_start = 0 if effective_axrange_B else 1
        B_contract_end = B_contract_start + ndim_mult
    else:
        # All dims are contracted (e.g., 1D vector), they're at [0, ndim_mult)
        B_contract_start = 0
        B_contract_end = ndim_mult
    
    var A_flat = A_flat_step1^.flatten_dims(A_contract_start, A_contract_end, ctx)  # Now 2D: m x k or k x m
    var B_flat = B_flat_step1^.flatten_dims(B_contract_start, B_contract_end, ctx)  # Now 2D: k x n or n x k
    
    # Handle 1D tensors: if B_flat is still 1D after flattening, reshape to 2D column vector
    if len(B_flat.shape) == 1:
        # Reshape (k,) to (k, 1) for matrix multiplication
        var B_flat_2d = B_flat^.reshape(List[Int](B_flat.shape[0], 1))
        B_flat = B_flat_2d^
    
    # Handle 1D tensors: if A_flat is still 1D after flattening, reshape to 2D row vector
    if len(A_flat.shape) == 1:
        # Reshape (m,) to (1, m) for matrix multiplication
        var A_flat_2d = A_flat^.reshape(List[Int](1, A_flat.shape[0]))
        A_flat = A_flat_2d^

    # Align to row-major GEMM layout (m×k) @ (k×n) for `linalg.matmul`.
    var trans_A = axrange_A
    var trans_B = not effective_axrange_B  # Flip for B if trailing
    if trans_A:
        var perm_A = List[Int](1, 0)  # Swap for 2D
        A_flat = A_flat^.transpose(perm_A, ctx)
    if trans_B:
        var perm_B = List[Int](1, 0)
        B_flat = B_flat^.transpose(perm_B, ctx)

    # Now A_flat: m x k, B_flat: k x n, both contiguous row-major
    # Compute expected C shape: non-cont A + non-cont B using saved shape info
    var C_shape_expected = List[Int]()
    for i in range(len(A_shape_for_C)):
        C_shape_expected.append(A_shape_for_C[i])
    for i in range(len(B_shape_for_C)):
        C_shape_expected.append(B_shape_for_C[i])
    
    # Special case: scalar result (rank 0) can be represented as (1,) in Mojo
    # This happens when both A and B are 1D and we contract their only axes
    var is_scalar_result = len(C_shape_expected) == 0
    var C_shape_actual = C.shape.copy()
    
    # Helper to format shape list as string
    fn format_shape(shape: List[Int]) -> String:
        var s = String("[")
        for i in range(len(shape)):
            if i > 0:
                s += ", "
            s += String(shape[i])
        s += "]"
        return s
    
    # Verify C has the correct shape
    if is_scalar_result:
        # Accept either [] (rank 0) or [1] (rank 1) as scalar representation
        if len(C_shape_actual) == 0:
            # True scalar - reshape C to (1,) for matmul compatibility
            C_shape_actual = List[Int](1)
        elif len(C_shape_actual) == 1 and C_shape_actual[0] == 1:
            # Already (1,) - this is fine
            pass
        else:
            raise Error("Output tensor C for scalar result must be [] or [1], got " + format_shape(C_shape_actual))
    else:
        # Non-scalar result: shapes must match exactly
        if len(C_shape_actual) != len(C_shape_expected):
            raise Error("Output tensor C has wrong rank: expected " + String(len(C_shape_expected)) + " got " + String(len(C_shape_actual)))
        for i in range(len(C_shape_expected)):
            if C_shape_actual[i] != C_shape_expected[i]:
                raise Error("Output tensor C has wrong shape: expected " + format_shape(C_shape_expected) + " got " + format_shape(C_shape_actual))

    # 2D output view of C: row-major (m, n)
    var m = A_flat.shape[0]
    var n = B_flat.shape[1]
    var C_flat_shape = List[Int](m, n)
    var C_flat_stride = List[Int](n, 1)
    var C_flat = DenseTensor[dtype](
        storage=C.storage,
        shape=C_flat_shape^,
        stride=C_flat_stride^
    )

    # Create NDBuffers (2D contiguous row-major)
    var shape_Af = IndexList[2](A_flat.shape[0], A_flat.shape[1])
    var shape_Bf = IndexList[2](B_flat.shape[0], B_flat.shape[1])
    var shape_Cf = IndexList[2](C_flat.shape[0], C_flat.shape[1])
    var ndbuf_A = NDBuffer[dtype, 2, MutAnyOrigin](A_flat.storage.unsafe_ptr(), shape_Af)
    var ndbuf_B = NDBuffer[dtype, 2, MutAnyOrigin](B_flat.storage.unsafe_ptr(), shape_Bf)
    var ndbuf_C = NDBuffer[mut=True, dtype, 2, MutAnyOrigin](C_flat.storage.unsafe_ptr(), shape_Cf)

    @parameter
    if dtype == DType.float32:
        linalg.matmul.matmul[target="gpu"](ndbuf_C, ndbuf_A, ndbuf_B, Optional(ctx))
    else:
        raise Error(
            "dense_tensor_dot GPU matmul supports only DType.float32 on this setup. "
            + "Use float32 for GPU execution (DMRG) or update MAX to a build that supports "
            + "your dtype."
        )

    # Grid: 2D (ceildiv(n, tile), ceildiv(m, tile)); Blocks: 2D (tile, tile) threads; Warps: tile/32 per dim, load tiles to shared, accumulate; Threads: each owns C element, loops over k/tilesize.
    # Tile typically 16/32 for float32.
    
fn dense_tensor_qr[dtype: DType = DType.float32](
        var tensor: DenseTensor[dtype],
        ctx: DeviceContext
    ) raises -> Tuple[DenseTensor[dtype], DenseTensor[dtype]]:
    """QR decomposition (economical, C-compatible): Q [m,k], R [k,n], k = min(m,n).
    
    **Host boundary**: Full matrix is copied device→host for CPU QR, then results host→device.
    This is the only remaining bulk host round-trip until GPU QR is available.
    """
    var rank = len(tensor.shape)
    if rank != 2:
        raise Error("QR decomposition requires a 2D matrix, got rank " + String(rank))

    var m = tensor.shape[0]
    var n = tensor.shape[1]
    var k = min(m, n)

    var tensor_shape_copy = tensor.shape.copy()
    var A_factorized = create_dense_tensor[dtype](ctx, tensor_shape_copy^, init_value=Scalar[dtype](0.0))
    ctx.enqueue_copy(A_factorized.storage, tensor.storage)
    ctx.synchronize()

    var host_A_factorized = ctx.enqueue_create_host_buffer[dtype](m * n)
    ctx.enqueue_copy(host_A_factorized, A_factorized.storage)
    ctx.synchronize()

    var sigma_shape = List[Int](k)
    var sigma = create_dense_tensor[dtype](ctx, sigma_shape^, init_value=Scalar[dtype](0.0))
    var host_sigma = ctx.enqueue_create_host_buffer[dtype](k)
    for i in range(k):
        host_sigma[i] = Scalar[dtype](0.0)

    alias matrix_layout = Layout.row_major(1, 1)
    alias sigma_layout = Layout.row_major(1)

    var sigma_shape_rt = RuntimeTuple[sigma_layout.shape](k)
    var sigma_stride_rt = RuntimeTuple[sigma_layout.stride](sigma.stride[0])
    var sigma_runtime_layout = RuntimeLayout[sigma_layout](
        shape=sigma_shape_rt,
        stride=sigma_stride_rt
    )
    var sigma_tensor = LayoutTensor[
        mut=True,
        dtype,
        sigma_layout,
        MutAnyOrigin
    ](
        host_sigma,
        runtime_layout=sigma_runtime_layout
    )

    var A_shape_rt = RuntimeTuple[matrix_layout.shape](m, n)
    var A_stride_rt = RuntimeTuple[matrix_layout.stride](A_factorized.stride[0], A_factorized.stride[1])
    var A_layout = RuntimeLayout[matrix_layout, linear_idx_type=_](shape=A_shape_rt, stride=A_stride_rt)
    var A_tensor = LayoutTensor[
        mut=True,
        dtype,
        matrix_layout,
        MutAnyOrigin
    ](
        host_A_factorized,
        runtime_layout=A_layout
    )

    qr_factorization(sigma_tensor, A_tensor)
    ctx.synchronize()
    ctx.enqueue_copy(sigma.storage, host_sigma)
    ctx.enqueue_copy(A_factorized.storage, host_A_factorized)
    ctx.synchronize()

    # R: economical [k, n] (C-compatible), upper triangular
    var R = create_dense_tensor[dtype](ctx, List[Int](k, n), init_value=Scalar[dtype](0.0))
    var host_A = host_A_factorized
    var host_R = ctx.enqueue_create_host_buffer[dtype](k * n)
    for i in range(k):
        for j in range(n):
            if j >= i:
                host_R[i * n + j] = host_A[i * n + j]
            else:
                host_R[i * n + j] = Scalar[dtype](0.0)
    ctx.enqueue_copy(R.storage, host_R)
    ctx.synchronize()

    # Form full Q [m,m] then take first k columns -> Q_out [m,k] (economical, C-compatible)
    var host_Q_full = ctx.enqueue_create_host_buffer[dtype](m * m)
    for i in range(m * m):
        host_Q_full[i] = Scalar[dtype](0.0)

    var Q_full_shape_rt = RuntimeTuple[matrix_layout.shape](m, m)
    var Q_full_stride_rt = RuntimeTuple[matrix_layout.stride](m, 1)
    var Q_full_layout = RuntimeLayout[matrix_layout](shape=Q_full_shape_rt, stride=Q_full_stride_rt)
    var Q_full_tensor = LayoutTensor[
        mut=True,
        dtype,
        matrix_layout,
        MutAnyOrigin
    ](
        host_Q_full,
        runtime_layout=Q_full_layout
    )
    form_q(sigma_tensor, A_tensor, Q_full_tensor)
    ctx.synchronize()

    var Q = create_dense_tensor[dtype](ctx, List[Int](m, k), init_value=Scalar[dtype](0.0))
    var host_Q = ctx.enqueue_create_host_buffer[dtype](m * k)
    for i in range(m):
        for j in range(k):
            host_Q[i * k + j] = host_Q_full[i * m + j]
    ctx.enqueue_copy(Q.storage, host_Q)
    ctx.synchronize()

    return (Q, R)

fn ensure_contiguous_2d[dtype: DType](
    var tensor: DenseTensor[dtype], 
    ctx: DeviceContext
) raises -> DenseTensor[dtype]:
    """Ensure tensor is contiguous 2D matrix with row-major layout.
    
    Uses a GPU strided-copy kernel only; no host buffer or device↔host transfer.
    """
    if len(tensor.shape) != 2:
        raise Error("Expected 2D tensor, got rank " + String(len(tensor.shape)))
    
    var m = tensor.shape[0]
    var n = tensor.shape[1]
    
    if tensor.is_contiguous() and tensor.stride[0] == n and tensor.stride[1] == 1:
        return tensor^  # Already contiguous row-major
    
    var shape_copy = tensor.shape.copy()
    var contiguous = create_dense_tensor[dtype](ctx, shape_copy^, init_value=Scalar[dtype](0.0))
    
    # GPU kernel: copy with stride; no host buffers
    var total = m * n
    alias BLOCK_SIZE = 256
    var grid_size = ceildiv(total, BLOCK_SIZE)
    ctx.enqueue_function[_gpu_strided_copy_2d_kernel[dtype]](
        tensor.storage.unsafe_ptr(),
        contiguous.storage.unsafe_ptr(),
        m,
        n,
        tensor.stride[0],
        tensor.stride[1],
        grid_dim=grid_size,
        block_dim=BLOCK_SIZE,
    )
    
    return contiguous^

fn dense_tensor_svd_trunc_lapack_f64[dtype: DType](
    var tensor: DenseTensor[dtype],
    ctx: DeviceContext,
    chi_max: Int,
    eps_trunc: Float64 = 1e-12,
    so_path: String = "native/libsvd_shim.so",
) raises -> Tuple[
    DenseTensor[dtype],
    DenseTensor[dtype],
    DenseTensor[dtype],
    Int,
]:
    """Truncated SVD using LAPACK (dgesdd) for Float64 tensors.

    **Host boundary**: Full matrix is copied device→host for LAPACK SVD, then results host→device.
    This is the only remaining bulk host round-trip until GPU SVD is available.
    """
    @parameter
    if dtype != DType.float32:
        raise Error("dense_tensor_svd_trunc_lapack_f64 only supports DType.float32")

    var A = ensure_contiguous_2d[dtype](tensor^, ctx)

    var m = A.shape[0]
    var n = A.shape[1]
    if m == 0 or n == 0:
        var U_empty = create_dense_tensor[dtype](ctx, List[Int](m, 0)^, init_value=Scalar[dtype](0.0))
        var S_empty = create_dense_tensor[dtype](ctx, List[Int](0)^, init_value=Scalar[dtype](0.0))
        var Vt_empty = create_dense_tensor[dtype](ctx, List[Int](0, n)^, init_value=Scalar[dtype](0.0))
        return (U_empty^, S_empty^, Vt_empty^, 0)

    # Copy A to host (row-major contiguous)
    var host_A = ctx.enqueue_create_host_buffer[dtype](m * n)
    ctx.enqueue_copy(host_A, A.storage)
    ctx.synchronize()

    # Build LAPACK input matrix (ROW_MAJOR) and copy data over
    var A_mat = MatrixF64(Int32(m), Int32(n), LapackLayout.ROW_MAJOR())
    for idx in range(m * n):
        A_mat.data[idx] = Float64(host_A[idx])

    var backend = SVDBackend(so_path)
    var jobz = Int8(ord('S'))  # thin U, thin VT
    var svd_res = svd_f64(backend, A_mat^, jobz)

    var k = Int(svd_res.k)  # min(m, n)
    var chi_kept = min(k, chi_max)
    if eps_trunc > 0.0 and k > 0:
        var total_norm_sq: Float64 = 0.0
        for i in range(k):
            var s = Float64(svd_res.S[i])
            total_norm_sq += s * s
        if total_norm_sq <= 0.0:
            chi_kept = 0
        else:
            # Find smallest chi (<= chi_max) such that discarded_weight <= eps_trunc
            var kept_norm_sq: Float64 = 0.0
            var limit = min(k, chi_max)
            chi_kept = limit  # default fallback
            for trial_chi in range(1, limit + 1):
                var s = Float64(svd_res.S[trial_chi - 1])
                kept_norm_sq += s * s
                var discarded_weight = (total_norm_sq - kept_norm_sq) / total_norm_sq
                if discarded_weight <= eps_trunc:
                    chi_kept = trial_chi
                    break

    if chi_kept == 0:
        var U_zero = create_dense_tensor[dtype](ctx, List[Int](m, 0)^, init_value=Scalar[dtype](0.0))
        var S_zero = create_dense_tensor[dtype](ctx, List[Int](0)^, init_value=Scalar[dtype](0.0))
        var Vt_zero = create_dense_tensor[dtype](ctx, List[Int](0, n)^, init_value=Scalar[dtype](0.0))
        return (U_zero^, S_zero^, Vt_zero^, 0)

    # Allocate output tensors
    var U_out = create_dense_tensor[dtype](
        ctx, List[Int](m, chi_kept)^, init_value=Scalar[dtype](0.0)
    )
    var S_out = create_dense_tensor[dtype](
        ctx, List[Int](chi_kept)^, init_value=Scalar[dtype](0.0)
    )
    var Vt_out = create_dense_tensor[dtype](
        ctx, List[Int](chi_kept, n)^, init_value=Scalar[dtype](0.0)
    )

    # Fill host buffers from LAPACK output
    var host_U = ctx.enqueue_create_host_buffer[dtype](m * chi_kept)
    var host_S = ctx.enqueue_create_host_buffer[dtype](chi_kept)
    var host_Vt = ctx.enqueue_create_host_buffer[dtype](chi_kept * n)

    for i in range(chi_kept):
        host_S[i] = Scalar[dtype](Float64(svd_res.S[i]))

    var u_cols_full = Int(svd_res.u_cols)  # = k for jobz='S'
    var vt_cols_full = Int(svd_res.vt_cols)  # = n for jobz='S'

    for i in range(m):
        for j in range(chi_kept):
            host_U[i * chi_kept + j] = Scalar[dtype](Float64(svd_res.U[i * u_cols_full + j]))

    for i in range(chi_kept):
        for j in range(n):
            host_Vt[i * n + j] = Scalar[dtype](Float64(svd_res.VT[i * vt_cols_full + j]))

    # Copy back to device
    ctx.enqueue_copy(U_out.storage, host_U)
    ctx.enqueue_copy(S_out.storage, host_S)
    ctx.enqueue_copy(Vt_out.storage, host_Vt)
    ctx.synchronize()

    return (U_out^, S_out^, Vt_out^, chi_kept)


fn dense_tensor_svd_trunc[dtype: DType](
    var tensor: DenseTensor[dtype],
    ctx: DeviceContext,
    chi_max: Int,
    eps_trunc: Float64 = 1e-12,
) raises -> Tuple[
    DenseTensor[dtype],
    DenseTensor[dtype],
    DenseTensor[dtype],
    Int,
]:
    """Truncated SVD helper used across the codebase.
    
    Backends:
    - Float64: LAPACK via `dense_tensor_svd_trunc_lapack_f64`
    - Float32: LAPACK in Float64 internally, then cast back to Float32
    """
    @parameter
    if dtype == DType.float32:
        return dense_tensor_svd_trunc_lapack_f64[dtype](
            tensor^,
            ctx,
            chi_max=chi_max,
            eps_trunc=eps_trunc,
        )
    else:
        raise Error(
            "dense_tensor_svd_trunc supports only DType.float32 currently."
        )