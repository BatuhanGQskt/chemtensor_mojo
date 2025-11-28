from memory import Pointer, AddressSpace, OwnedPointer
from layout import Layout, LayoutTensor, RuntimeLayout, IntTuple, RuntimeTuple
from collections.list import List
from utils import IndexList
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, DeviceBuffer
from layout.layout import DimList
from buffer.buffer import NDBuffer
from memory.unsafe_pointer import UnsafePointer
import linalg
from linalg.qr_factorization import qr_factorization, form_q

## Fully dynamic tensor with runtime-determined rank, shape, and stride
@fieldwise_init
struct DynamicTensor[dtype: DType](Writable, Movable, ImplicitlyCopyable):
    """A tensor where rank, shape, and stride are all determined at runtime.
    
    Unlike DenseTensor which requires compile-time Layout parameter, this tensor
    allows complete flexibility at runtime.
    
    Parameters:
        dtype: The data type of the tensor elements (e.g., DType.float32, DType.float64).
    """
    var storage: DeviceBuffer[dtype]  # GPU storage for tensor data
    var shape: List[Int]  # Runtime shape
    var stride: List[Int]  # Runtime stride
    var size: Int  # Total number of elements

    fn __init__(out self, storage: DeviceBuffer[dtype], var shape: List[Int], var stride: List[Int]):
        """Initialize a dynamic tensor with runtime parameters.
        
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
        """Copy constructor for DynamicTensor.
        
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
        writer.write("DynamicTensor[rank=")
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
        print("DynamicTensor with rank ", rank, " and shape ", end="")
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

    fn is_contiguous(self: DynamicTensor) -> Bool:
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
            ```mojo
            # Contiguous tensor
            var shape = List[Int](3, 4)
            var tensor = create_dynamic_tensor(ctx, shape^)
            print(tensor.is_contiguous())  # True, strides are [4, 1]
            
            # Non-contiguous after slicing or view operations
            var transposed = tensor^.transpose(List[Int](1, 0), ctx)
            print(transposed.is_contiguous())  # Might be False
            ```
        """
        var expected_stride = 1
        for i in range(len(self.shape) - 1, -1, -1):
            if self.stride[i] != expected_stride:
                return False
            expected_stride *= self.shape[i]
        return True

    fn copy_to_contiguous(var self, ctx: DeviceContext) raises -> DynamicTensor[dtype]:
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
            A new DynamicTensor with contiguous row-major layout.
        
        Raises:
            Error: If GPU memory allocation or copy fails.
        
        Note:
            This is a simplified implementation. For true non-contiguous tensors
            (e.g., after advanced slicing), a more complex reorganization kernel
            would be needed.
        
        Example:
            ```mojo
            # Create a tensor and transpose it (may become non-contiguous)
            var tensor = create_dynamic_tensor(ctx, List[Int](4, 3)^)
            var perm = List[Int](1, 0)
            var transposed = tensor^.transpose(perm, ctx)
            
            # Make it contiguous for efficient operations
            var contiguous = transposed^.copy_to_contiguous(ctx)
            print(contiguous.is_contiguous())  # True
            ```
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
        ctx.synchronize()

        return DynamicTensor[dtype](new_storage, new_shape^, new_strides^)

    fn transpose(var self, perm: List[Int], ctx: DeviceContext) raises -> DynamicTensor[dtype]:
        """Transpose tensor dimensions according to a permutation.
        
        Reorders the dimensions of the tensor according to the permutation list.
        For 2D tensors with permutation [1, 0], this performs a standard matrix transpose.
        
        Implementation details:
        - For 2D transpose [1,0]: Creates a physically transposed copy in memory
        - For other permutations: Creates a view with reordered shape/stride (no copy)
        
        Args:
            self: The tensor to transpose (ownership transferred).
            perm: Permutation list specifying the new axis order.
                  Must have length equal to tensor rank.
                  Each element is an index into the original dimensions.
            ctx: Device context for GPU operations (used for 2D transpose copy).
        
        Returns:
            A new DynamicTensor with transposed dimensions.
        
        Raises:
            Error: If perm length doesn't match tensor rank.
        
        Example:
            ```mojo
            # 2D matrix transpose (standard transpose)
            var matrix = create_dynamic_tensor_from_data(
                ctx,
                List[Float32](1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
                List[Int](2, 3)^  # Shape: [2, 3]
            )
            # Original: [[1, 2, 3],
            #            [4, 5, 6]]
            
            var perm = List[Int](1, 0)  # Swap rows and columns
            var transposed = matrix^.transpose(perm, ctx)
            # Result shape: [3, 2]
            # Result: [[1, 4],
            #          [2, 5],
            #          [3, 6]]
            
            # 3D tensor transpose
            var tensor_3d = create_dynamic_tensor(ctx, List[Int](2, 3, 4)^)
            var perm_3d = List[Int](2, 0, 1)  # Move last dim to front
            var transposed_3d = tensor_3d^.transpose(perm_3d, ctx)
            # Original shape: [2, 3, 4]
            # Result shape:   [4, 2, 3]
            ```
        """
        var rank = len(self.shape)
        if len(perm) != rank:
            raise Error("Perm length mismatch")

        # For simple 2D transpose on contiguous row-major
        if rank == 2 and perm[0] == 1 and perm[1] == 0:
            # Copy data to new transposed layout
            var new_shape = List[Int]()
            new_shape.append(self.shape[1])
            new_shape.append(self.shape[0])
            
            var new_strides = compute_row_major_strides(new_shape, rank)
            var total_size = self.size
            var new_storage = ctx.enqueue_create_buffer[Self.dtype](total_size)
            
            # Copy transposed data using host memory (simple but works)
            var host_src = ctx.enqueue_create_host_buffer[Self.dtype](total_size)
            ctx.enqueue_copy(host_src, self.storage)
            ctx.synchronize()
            
            var host_dst = ctx.enqueue_create_host_buffer[Self.dtype](total_size)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    var src_idx = i * self.shape[1] + j
                    var dst_idx = j * self.shape[0] + i
                    host_dst[dst_idx] = host_src[src_idx]
            
            ctx.enqueue_copy(new_storage, host_dst)
            ctx.synchronize()
            
            return DynamicTensor[dtype](new_storage, new_shape^, new_strides^)
        else:
            # For other permutations, create a view (stride-based)
            # This doesn't copy data, just reinterprets layout
            var new_shape = List[Int](capacity=rank)
            var new_stride = List[Int](capacity=rank)
            for i in range(rank):
                new_shape.append(self.shape[perm[i]])
                new_stride.append(self.stride[perm[i]])
            
            return DynamicTensor[dtype](self.storage^, new_shape^, new_stride^)

    fn flatten_dims(var self, start: Int, end: Int, ctx: DeviceContext) raises -> DynamicTensor[dtype]:
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
            A new DynamicTensor with the specified dimensions flattened.
        
        Raises:
            Error: If start < 0, end > rank, or start >= end.
        
        Note:
            This creates a view with recomputed strides. No data is copied.
            The underlying storage is shared with the original tensor.
        
        Example:
            ```mojo
            # Flatten middle dimensions of a 4D tensor
            var tensor_4d = create_dynamic_tensor(ctx, List[Int](2, 3, 4, 5)^)
            # Original shape: [2, 3, 4, 5]
            
            # Flatten dimensions 1 and 2 (indices 1 and 2, i.e., 3 and 4)
            var flattened = tensor_4d^.flatten_dims(1, 3, ctx)
            # Result shape: [2, 12, 5]  (3*4=12)
            
            # Flatten all dimensions (convert to 1D vector)
            var tensor_3d = create_dynamic_tensor(ctx, List[Int](2, 3, 4)^)
            var vector = tensor_3d^.flatten_dims(0, 3, ctx)
            # Original shape: [2, 3, 4]
            # Result shape:   [24]  (2*3*4=24)
            
            # Flatten last two dimensions (useful for batched matrix ops)
            var batched = create_dynamic_tensor(ctx, List[Int](8, 5, 10)^)
            var flat_batch = batched^.flatten_dims(1, 3, ctx)
            # Original shape: [8, 5, 10]  (batch of 8 matrices)
            # Result shape:   [8, 50]     (batch of 8 vectors)
            ```
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
        return DynamicTensor[dtype](self.storage^, new_shape^, new_strides^)



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
        strides.insert(0, stride)
        stride *= shape[i]
    
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


fn create_dynamic_tensor[dtype: DType = DType.float32](
    ctx: DeviceContext, 
    var shape: List[Int], 
    row_major: Bool = True,
    init_value: Scalar[dtype] = 0.0
) raises -> DynamicTensor[dtype]:
    """Create a dynamic tensor with runtime-determined rank, shape, and stride.
    
    Args:
        ctx: Device context for GPU operations.
        shape: List of dimensions (length should be rank).
        rank: Number of dimensions.
        row_major: If True, use row-major stride; otherwise column-major.
        init_value: Initial value to fill the tensor.
    
    Returns:
        DynamicTensor with allocated GPU storage.
    
    Example:
        var shape = List[Int](3, 4, 5)  # 3D tensor
        var tensor = create_dynamic_tensor(ctx, shape, 3)
        # Creates a 3x4x5 tensor with row-major layout.
    """
    # Compute total size
    var total_size = 1
    var rank = len(shape)
    for i in range(rank):
        total_size *= shape[i]
    
    # Compute strides based on layout preference
    var strides: List[Int]
    if row_major:
        strides = compute_row_major_strides(shape, rank)
    else:
        strides = compute_column_major_strides(shape, rank)
    
    # Allocate and initialize host buffer
    var host_storage = ctx.enqueue_create_host_buffer[dtype](total_size)
    for i in range(total_size):
        host_storage[i] = init_value
    
    # Copy to device
    var device_storage = ctx.enqueue_create_buffer[dtype](total_size)
    ctx.enqueue_copy(device_storage, host_storage)
    
    return DynamicTensor[dtype](device_storage, shape^, strides^)


fn create_dynamic_tensor_from_data[dtype: DType = DType.float32](
    ctx: DeviceContext, 
    data: List[Scalar[dtype]],
    var shape: List[Int], 
    row_major: Bool = True
) raises -> DynamicTensor[dtype]:
    """Create a dynamic tensor from existing data.
    
    Args:
        ctx: Device context for GPU operations.
        data: Flat list of data values.
        shape: List of dimensions (length should be rank).
        rank: Number of dimensions.
        row_major: If True, use row-major stride; otherwise column-major.
    
    Returns:
        DynamicTensor with data copied to GPU.
    
    Example:
        var data = List[Float32](1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        var shape = List[Int](2, 3)  # 2x3 matrix
        var tensor = create_dynamic_tensor_from_data(ctx, data, shape, 2).
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
    
    return DynamicTensor[dtype](device_storage, shape^, strides^)

# fn dense_tensor_dot(C: DynamicTensor, A: DynamicTensor, B: DynamicTensor, ctx: Optional[DeviceContext]) raises:
#     """Perform matrix multiplication C = A @ B using GPU acceleration.
    
#     Args:
#         C: Output tensor (must be properly sized for result)
#         A: Left input tensor 
#         B: Right input tensor
#         ctx: Device context for GPU operations
        
#     Note: Currently only supports rank-2 (matrix) operations.
#     """
#     if not ctx:
#         raise Error("GPU context required for dense_tensor_dot")
    
#     var rank_A = len(A.shape)
#     var rank_B = len(B.shape)
#     var rank_C = len(C.shape)
    
#     # For now, only support 2D matrix multiplication
#     if rank_A != 2 or rank_B != 2 or rank_C != 2:
#         raise Error("Only 2D matrix multiplication is currently supported")
    
#     # Create NDBuffers from DynamicTensor storage
#     # NDBuffer needs: pointer, dynamic_shape
#     var shape_A = IndexList[2](A.shape[0], A.shape[1])
#     var shape_B = IndexList[2](B.shape[0], B.shape[1])
#     var shape_C = IndexList[2](C.shape[0], C.shape[1])
    
#     # Get non-owning pointers from DeviceBuffer using unsafe_ptr()
#     # This follows RAII principles - the original DynamicTensor retains ownership
#     var ptr_A = A.storage.unsafe_ptr()
#     var ptr_B = B.storage.unsafe_ptr()
#     var ptr_C = C.storage.unsafe_ptr()
    
#     # Create NDBuffers with rank=2
#     # These NDBuffers are non-owning views - they don't free memory when dropped
#     var ndbuf_A = NDBuffer[dtype, 2, MutAnyOrigin](ptr_A, shape_A)
#     var ndbuf_B = NDBuffer[dtype, 2, MutAnyOrigin](ptr_B, shape_B)
#     var ndbuf_C = NDBuffer[mut=True, dtype, 2, MutAnyOrigin](ptr_C, shape_C)
    
#     # Call matmul with NDBuffers
#     # The inplace operation modifies C's underlying storage through the non-owning pointer
#     linalg.matmul.matmul[target="gpu"](ndbuf_C, ndbuf_A, ndbuf_B, ctx)



fn dense_tensor_dot[dtype: DType = DType.float32](C: DynamicTensor[dtype], var A: DynamicTensor[dtype], var B: DynamicTensor[dtype], ctx: DeviceContext, ndim_mult: Int = 1, axrange_A: Bool = False, axrange_B: Bool = False) raises:  # axrange False=trailing, True=leading
    """Perform generalized tensor dot product (contraction) on GPU.
    
    This function implements Einstein summation-style tensor contraction by:
    1. Identifying which axes to contract (sum over)
    2. Flattening non-contracted and contracted dimensions
    3. Performing matrix multiplication on the flattened tensors
    4. Storing the result in the pre-allocated output tensor C
    
    For standard 2D matrix multiplication A[m,k] @ B[k,n] = C[m,n], simply use
    default parameters. The function automatically adjusts axis contraction.
    
    **Convention Note**: 
    Current implementation follows standard linear algebra convention where default
    parameters contract A's trailing axes with B's leading axes: A[..., k] @ B[k, ...]
    
    For tensor network applications where you need A[k, ...] @ B[..., k], explicitly set:
    `axrange_A=True, axrange_B=False`
    
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
                   Note: Automatically set to True for standard 2D matmul.
    
    Raises:
        Error: If ndim_mult < 1, or tensors don't have enough dimensions.
        Error: If contracted dimensions don't match in size.
        Error: If C doesn't have the correct shape for the result.
    
    Examples:
        ```mojo
        # Example 1: Standard 2D matrix multiplication
        # A[3, 4] @ B[4, 5] = C[3, 5]
        with DeviceContext() as ctx:
            var A = create_dynamic_tensor(ctx, List[Int](3, 4)^, init_value=2.0)
            var B = create_dynamic_tensor(ctx, List[Int](4, 5)^, init_value=3.0)
            var C = create_dynamic_tensor(ctx, List[Int](3, 5)^, init_value=0.0)
            
            dense_tensor_dot(C, A^, B^, ctx)
            # C now contains the result of A @ B
            # Each element C[i,j] = sum over k of A[i,k] * B[k,j]
        
        # Example 2: 3D tensor contraction
        # A[2, 3, 4] @ B[4, 5, 6] = C[2, 3, 5, 6]
        # Contracts A's last axis (4) with B's first axis (4)
        with DeviceContext() as ctx:
            var A_3d = create_dynamic_tensor(ctx, List[Int](2, 3, 4)^, init_value=1.5)
            var B_3d = create_dynamic_tensor(ctx, List[Int](4, 5, 6)^, init_value=2.0)
            var C_4d = create_dynamic_tensor(ctx, List[Int](2, 3, 5, 6)^, init_value=0.0)
            
            dense_tensor_dot(C_4d, A_3d^, B_3d^, ctx, ndim_mult=1)
            # C now contains the 4D result
            # Non-contracted dims: [2,3] from A + [5,6] from B = [2,3,5,6]
        
        # Example 3: Contracting multiple axes
        # A[2, 3, 4, 5] with shape breakdown: [batch(2,3), contract(4,5)]
        # B[4, 5, 6, 7] with shape breakdown: [contract(4,5), result(6,7)]
        # Result C[2, 3, 6, 7]
        # This contracts 2 dimensions (4 and 5) between A and B
        with DeviceContext() as ctx:
            var A_4d = create_dynamic_tensor(ctx, List[Int](2, 3, 4, 5)^)
            var B_4d = create_dynamic_tensor(ctx, List[Int](4, 5, 6, 7)^)
            var C_4d = create_dynamic_tensor(ctx, List[Int](2, 3, 6, 7)^)
            
            # Contract last 2 dims of A with first 2 dims of B
            dense_tensor_dot(C_4d, A_4d^, B_4d^, ctx, ndim_mult=2)
            # Contracts over axes 2,3 of A with axes 0,1 of B
        
        # Example 4: Tensor Network Convention
        # Contract leading axes of A with trailing axes of B (common in quantum chemistry)
        with DeviceContext() as ctx:
            # A[k, m, n] @ B[p, q, k] = C[m, n, p, q]
            var A_tn = create_dynamic_tensor(ctx, List[Int](3, 4, 5)^)
            var B_tn = create_dynamic_tensor(ctx, List[Int](6, 7, 3)^)
            var C_tn = create_dynamic_tensor(ctx, List[Int](4, 5, 6, 7)^)
            
            # Contract first axis of A (size 3) with last axis of B (size 3)
            dense_tensor_dot(C_tn, A_tn^, B_tn^, ctx, ndim_mult=1, 
                           axrange_A=True,   # Contract A's leading axis
                           axrange_B=False)  # Contract B's trailing axis
        
        # Example 5: Standard vs Tensor Network notation comparison
        with DeviceContext() as ctx:
            # Standard linear algebra: A[m,k] @ B[k,n] (default)
            var A_std = create_dynamic_tensor(ctx, List[Int](3, 4)^)
            var B_std = create_dynamic_tensor(ctx, List[Int](4, 5)^)
            var C_std = create_dynamic_tensor(ctx, List[Int](3, 5)^)
            dense_tensor_dot(C_std, A_std^, B_std^, ctx)  # Uses defaults
            
            # Tensor network style: A[k,m] @ B[n,k] 
            # Need to explicitly set parameters:
            var A_tn2 = create_dynamic_tensor(ctx, List[Int](4, 3)^)
            var B_tn2 = create_dynamic_tensor(ctx, List[Int](5, 4)^)
            var C_tn2 = create_dynamic_tensor(ctx, List[Int](3, 5)^)
            dense_tensor_dot(C_tn2, A_tn2^, B_tn2^, ctx, ndim_mult=1,
                           axrange_A=True, axrange_B=False)
        ```
    
    Performance Notes:
        - Uses GPU-accelerated matrix multiplication (linalg.matmul)
        - Automatically copies non-contiguous tensors to contiguous layout
        - Employs tiled shared memory kernels for efficiency
        - Best performance with tile sizes 16-32 (Due to Warps) for float32 on modern GPUs
    
    Implementation Details:
        The function internally:
        1. Validates dimensions and contraction axes
        2. Saves shape information before transferring ownership
        3. Ensures tensors are contiguous (copies if needed) (TODO: NOT SURE ABOUT THIS COPYING MIGHT BE COSTLY)
        4. Flattens tensors to 2D matrices (non-contracted dims × contracted dims)
        5. Transposes if necessary to align contraction axes
        6. Calls GPU matrix multiplication kernel
        7. Result is written directly to output tensor C
    """
    var rank_A = len(A.shape)
    var rank_B = len(B.shape)
    if ndim_mult < 1 or rank_A < ndim_mult or rank_B < ndim_mult:
        raise Error("Invalid ndim_mult")

    # For standard tensor dot product with default parameters:
    # The convention depends on the use case:
    # - Standard linear algebra: A[m,k] @ B[k,n] = contract A's trailing with B's leading
    # - Tensor networks/chemistry: Often A[k,m] @ B[n,k] = contract A's leading with B's trailing
    # 
    # Current implementation: A's TRAILING axis × B's LEADING axis (standard linalg convention)
    # This means: A[..., k] @ B[k, ...] for any rank tensors
    var effective_axrange_B = axrange_B
    if not axrange_A and not axrange_B:
        # Default behavior: contract A's trailing axes with B's leading axes
        # This is the standard convention for matrix/tensor multiplication
        effective_axrange_B = True

    # Check contracting dims match
    # Important: When contracting A's trailing with B's leading, the order is REVERSED
    # Example: A[4, 3, 2, 1] contracts with B[1, 2, 3, 4] for ndim_mult=2
    #          i=0: A.shape[2]=2 must match B.shape[1]=2 ✓
    #          i=1: A.shape[3]=1 must match B.shape[0]=1 ✓
    #          Result shape: [4, 3] from A + [3, 4] from B = [4, 3, 3, 4]
    for i in range(ndim_mult):
        var ax_A: Int
        var ax_B: Int
        
        if axrange_A:
            # A uses leading axes: [0, 1, 2, ..., ndim_mult-1]
            ax_A = i
        else:
            # A uses trailing axes: [..., rank_A-ndim_mult, rank_A-ndim_mult+1, ..., rank_A-1]
            ax_A = rank_A - ndim_mult + i
        
        if effective_axrange_B:
            # B uses leading axes: [0, 1, 2, ..., ndim_mult-1]
            # When A is trailing and B is leading, we need to REVERSE the order
            if not axrange_A:
                # Reverse: A's last matches B's first
                ax_B = ndim_mult - 1 - i
            else:
                # Both leading: direct correspondence
                ax_B = i
        else:
            # B uses trailing axes: [..., rank_B-ndim_mult, rank_B-ndim_mult+1, ..., rank_B-1]
            if axrange_A:
                # A is leading, B is trailing: reverse order
                ax_B = rank_B - 1 - i
            else:
                # Both trailing: direct correspondence
                ax_B = rank_B - ndim_mult + i
        
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
    var A_contig = A^.copy_to_contiguous(ctx)
    var B_contig = B^.copy_to_contiguous(ctx)

    # Flatten non-contracted dimensions first
    var A_flat_step1 = A_contig^.flatten_dims(A_noncont_start, A_noncont_end, ctx)
    var B_flat_step1 = B_contig^.flatten_dims(B_noncont_start, B_noncont_end, ctx)
    
    # Now flatten the contracted dimensions
    # For A: if axrange_A is False (trailing), non-contracted are [0, rank-ndim], contracted are [rank-ndim, rank]
    #        After flattening non-contracted [0, rank-ndim), contracted dims are now at [1, 1+ndim_mult)
    # For A: if axrange_A is True (leading), non-contracted are [ndim, rank], contracted are [0, ndim]
    #        After flattening non-contracted [ndim, rank), contracted dims are still at [0, ndim_mult)
    var A_contract_start = 1 if not axrange_A else 0
    var A_contract_end = A_contract_start + ndim_mult
    
    # For B: if effective_axrange_B is True (leading), contracted are [0, ndim], non-contracted are [ndim, rank]
    #        After flattening non-contracted [ndim, rank), contracted dims are still at [0, ndim_mult)
    # For B: if effective_axrange_B is False (trailing), non-contracted are [0, rank-ndim], contracted are [rank-ndim, rank]
    #        After flattening non-contracted [0, rank-ndim), contracted dims are now at [1, 1+ndim_mult)
    var B_contract_start = 0 if effective_axrange_B else 1
    var B_contract_end = B_contract_start + ndim_mult
    
    var A_flat = A_flat_step1^.flatten_dims(A_contract_start, A_contract_end, ctx)  # Now 2D: m x k or k x m
    var B_flat = B_flat_step1^.flatten_dims(B_contract_start, B_contract_end, ctx)  # Now 2D: k x n or n x k

    # Transpose if leading (to make row-major inner contract)
    var trans_A = axrange_A
    var trans_B = not effective_axrange_B  # Flip for B if trailing ( shouldn't be the case for matrix matrix multiplication )
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
    
    # Verify C has the correct shape
    if len(C.shape) != len(C_shape_expected):
        raise Error("Output tensor C has wrong rank")
    for i in range(len(C_shape_expected)):
        if C.shape[i] != C_shape_expected[i]:
            raise Error("Output tensor C has wrong shape")

    # Calculate the 2D shape for matrix multiplication
    # m = product of A's non-contracted dimensions
    # n = product of B's non-contracted dimensions
    var m = A_flat.shape[0]  # Already flattened
    var n = B_flat.shape[1]  # Already flattened
    
    # Flatten C to 2D shape (m, n) for matrix multiplication
    # This creates a 2D "view" of C's storage without copying
    # For contiguous row-major 2D tensor (m, n), strides are [n, 1]
    var C_flat_shape = List[Int](m, n)
    var C_flat_stride = List[Int](n, 1)
    var C_flat = DynamicTensor[dtype](
        storage=C.storage,
        shape=C_flat_shape^,
        stride=C_flat_stride^
    )

    # Create NDBuffers (now 2D contiguous)
    var shape_Af = IndexList[2](A_flat.shape[0], A_flat.shape[1])
    var shape_Bf = IndexList[2](B_flat.shape[0], B_flat.shape[1])
    var shape_Cf = IndexList[2](C_flat.shape[0], C_flat.shape[1])

    var ndbuf_A = NDBuffer[dtype, 2, MutAnyOrigin](A_flat.storage.unsafe_ptr(), shape_Af)
    var ndbuf_B = NDBuffer[dtype, 2, MutAnyOrigin](B_flat.storage.unsafe_ptr(), shape_Bf)
    var ndbuf_C = NDBuffer[mut=True, dtype, 2, MutAnyOrigin](C_flat.storage.unsafe_ptr(), shape_Cf)

    # Call matmul (tiled shared mem kernel)
    # Result is written to C_flat, which shares storage with C
    # So C automatically gets the result in the correct ND shape
    linalg.matmul.matmul[target="gpu"](ndbuf_C, ndbuf_A, ndbuf_B, Optional(ctx))

    # Grid: 2D (ceildiv(n, tile), ceildiv(m, tile)); Blocks: 2D (tile, tile) threads; Warps: tile/32 per dim, load tiles to shared, accumulate; Threads: each owns C element, loops over k/tilesize.
    # Tile typically 16/32 for float32.



fn dense_tensor_qr[dtype: DType = DType.float32](
    var tensor: DynamicTensor[dtype],
    ctx: DeviceContext
) raises -> Tuple[DynamicTensor[dtype], DynamicTensor[dtype]]:
    """Compute the QR decomposition of a dense tensor using MAX API.
    
    Performs QR factorization where a matrix A is decomposed into:
    - Q: An orthogonal matrix (Q^T @ Q = I)
    - R: An upper triangular matrix
    Such that A = Q @ R
    
    Uses the Householder reflector method via MAX's linalg.qr_factorization,
    which is equivalent to LAPACK's geqrf + orgqr/ungqr approach.
    
    Parameters:
        dtype: The data type of the tensor elements (default: DType.float32).
    
    Args:
        tensor: Input dense tensor to decompose (must be 2D matrix).
        ctx: Device context for GPU operations.
    
    Returns:
        A tuple (Q, R) where:
        - Q is the orthogonal matrix (m × m for full QR)
        - R is the upper triangular matrix (m × n)
    
    Raises:
        Error: If the input tensor is not a 2D matrix.
    
    Note:
        The input tensor must be a 2D matrix. QR decomposition is not
        defined for higher-rank tensors in this implementation.
    
    Algorithm:
        1. Copy input matrix A (to preserve original)
        2. Call linalg.qr_factorization to compute Householder reflectors in-place
        3. Extract R from upper triangular part of factorized matrix
        4. Call linalg.form_q to generate orthogonal matrix Q
    
    References:
        MAX API: https://docs.modular.com/mojo/kernels/linalg/qr_factorization/qr_factorization/
        https://docs.modular.com/mojo/kernels/linalg/qr_factorization/form_q/
    """
    # Require 2D matrix
    var rank = len(tensor.shape)
    if rank != 2:
        raise Error("QR decomposition requires a 2D matrix, got rank " + String(rank))
    
    var m = tensor.shape[0]  # rows
    var n = tensor.shape[1]  # columns
    var k = min(m, n)
    
    var tensor_shape_copy = tensor.shape.copy()
    # Create a copy of input tensor for in-place factorization
    # (to preserve original tensor)
    var A_factorized = create_dynamic_tensor[dtype](ctx, tensor_shape_copy^, init_value=0.0)
    
    # Copy data from input tensor to A_factorized
    ctx.enqueue_copy(A_factorized.storage, tensor.storage)
    ctx.synchronize()
    
    # Allocate sigma vector for Householder scaling factors
    var sigma_shape = List[Int](k)
    var sigma = create_dynamic_tensor[dtype](ctx, sigma_shape^, init_value=0.0)
    
    # Create LayoutTensor views needed by MAX APIs
    # 2 Dimensional Row Major Layouts shape = 2 with stride = 1
    alias matrix_layout = Layout.row_major(2)
    alias sigma_layout = Layout.row_major(1)
    
    var sigma_shape_rt = RuntimeTuple[sigma_layout.shape](k)
    var sigma_stride_rt = RuntimeTuple[sigma_layout.stride](sigma.stride[0])
    var sigma_runtime_layout = RuntimeLayout[sigma_layout](
        shape=sigma_shape_rt,
        stride=sigma_stride_rt
    )
    # sigma tensor
    var sigma_tensor = LayoutTensor[
        mut=True,
        dtype,
        sigma_layout,
        MutAnyOrigin
    ](
        sigma.storage,
        runtime_layout=sigma_runtime_layout
    )
    
    var A_shape_rt = RuntimeTuple[matrix_layout.shape](m, n)
    var A_stride_rt = RuntimeTuple[matrix_layout.stride](A_factorized.stride[0], A_factorized.stride[1])
    var A_layout = RuntimeLayout[matrix_layout, linear_idx_type=_](shape=A_shape_rt, stride=A_stride_rt)
    # A tensor
    var A_tensor = LayoutTensor[
        mut=True,
        dtype,
        matrix_layout,
        MutAnyOrigin
    ](
        A_factorized.storage,
        runtime_layout=A_layout
    )
    
    # Step 1: Compute QR factorization in-place (Householder reflectors)
    # The function modifies A_tensor in-place to store Householder reflectors
    # and stores scaling factors in sigma_tensor
    qr_factorization(sigma_tensor, A_tensor)
    ctx.synchronize()
    
    # Step 2: Extract R from upper triangular part of A_factorized
    # After qr_factorization, the upper triangular part of A_factorized contains R
    var R = create_dynamic_tensor[dtype](ctx, List[Int](m, n), init_value=0.0)
    
    # Copy the upper triangular part via host memory
    var host_A = ctx.enqueue_create_host_buffer[dtype](m * n)
    ctx.enqueue_copy(host_A, A_factorized.storage)
    ctx.synchronize()
    
    var host_R = ctx.enqueue_create_host_buffer[dtype](m * n)
    for i in range(m):
        for j in range(n):
            var idx = i * n + j
            if i <= j:
                # Upper triangular part (including diagonal)
                host_R[idx] = host_A[idx]
            else:
                # Lower triangular part - set to zero
                host_R[idx] = Scalar[dtype](0.0)
    
    ctx.enqueue_copy(R.storage, host_R)
    ctx.synchronize()
    
    # Step 3: Form Q matrix from Householder reflectors
    var Q = create_dynamic_tensor[dtype](ctx, List[Int](m, m), init_value=0.0)
    
    # Create LayoutTensor view for Q (output of form_q)
    var Q_shape_rt = RuntimeTuple[matrix_layout.shape](m, m)
    var Q_stride_rt = RuntimeTuple[matrix_layout.stride](Q.stride[0], Q.stride[1])
    var Q_layout = RuntimeLayout[matrix_layout](shape=Q_shape_rt, stride=Q_stride_rt)
    var Q_tensor = LayoutTensor[mut=True, dtype, matrix_layout, MutAnyOrigin](
        Q.storage,
        runtime_layout=Q_layout
    )
    
    # Form the orthogonal matrix Q from the Householder reflectors in A and sigma
    # According to MAX API: form_q[dtype, element_layout](sigma, A, Q)
    form_q(sigma_tensor, A_tensor, Q_tensor)
    ctx.synchronize()
    
    return (Q, R)