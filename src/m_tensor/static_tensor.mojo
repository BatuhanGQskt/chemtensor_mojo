from memory import Pointer, AddressSpace, OwnedPointer
from layout import Layout, LayoutTensor, RuntimeLayout, IntTuple, RuntimeTuple
from collections.list import List
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, DeviceBuffer
from layout.tensor_builder import LayoutTensorBuild as tb
from layout.layout import DimList
from math import ceildiv
from main import MAX_RANK, list_to_dimlist

alias dtype = DType.float32

## General dense tensor structure for block sparse tensor networks
@fieldwise_init
struct StaticTensor[
    data_layout: Layout, 
](Writable):
    var data: LayoutTensor[
        mut=True, dtype, data_layout, MutableAnyOrigin
    ]  # Data array for tensor elements
    var dims: DimList  # Dynamic array of dimensions (runtime)
    var storage: DeviceBuffer[dtype]  # Dynamic GPU storage

    fn write_to[W: Writer](self, mut writer: W) -> None:
        # Minimal textual representation to avoid device-side serialization
        var m = self.dims.get[0]()
        var n = self.dims.get[1]()
        writer.write("StaticTensor[" )
        writer.write(m)
        writer.write(" x ")
        writer.write(n)
        writer.write("]")

    fn print_tensor(self, ctx: DeviceContext) raises -> None:
        var M = self.dims.get[0]()
        var N = self.dims.get[1]()
        var total = M * N
        var host_out = ctx.enqueue_create_host_buffer[dtype](total)
        ctx.enqueue_copy(host_out, self.storage)
        ctx.synchronize()

        print("Tensor (", M, "x", N, "):")
        for i in range(M):
            for j in range(N):
                var idx = i * N + j
                print("[", i, ",", j, "] = ", host_out[idx])
        

## GPU-optimized dot product for dense tensors (device-friendly view)
fn static_tensor_dot[layout: Layout](
    first: LayoutTensor[mut=False, dtype, layout, MutableAnyOrigin],
    other: LayoutTensor[mut=False, dtype, layout, MutableAnyOrigin],
    result: LayoutTensor[mut=True, dtype, layout, MutableAnyOrigin],
    M: Int, N: Int, K: Int,
) -> None:
    # TODO: Make this more general for N number of ranks if possible dynamic(not static 5)
    # TODO: Investigate complex and try to implement dot accordingly
    # TODO: kronecker
    # TODO: scalar add
    # TODO: SVD function (hard)
    var row = Int(block_dim.x * block_idx.x + thread_idx.x) #TODO: Make a version for coalescing
    var col = Int(block_dim.y * block_idx.y + thread_idx.y)

    if row < M and col < N and K > 0:
        var dst = first[row, 0] * other[0, col]
        for k in range(1, K):
            dst += first[row, k] * other[k, col]
        result[row, col] = dst



fn compute_strides(dims: List[Int], rank: Int) -> List[Int]:
    """Compute row-major strides for given dimensions."""
    var strides = List[Int]()
    var stride = 1
    
    # Compute strides in reverse order (row-major)
    for i in range(rank - 1, -1, -1):
        strides.insert(0, stride)
        stride *= dims[i]
    
    # Pad remaining with 1s
    for _ in range(MAX_RANK - rank):
        strides.append(1)
    
    return strides.copy()

fn compute_stride_and_dimlist_from_list(dims: List[Int], rank: Int) -> Tuple[DimList, DimList]:
    """Compute dimlist and strides from a list of dimensions."""
    var dim_list = list_to_dimlist(dims)
    var strides = list_to_dimlist(compute_strides(dims, rank))

    return Tuple[DimList, DimList](dim_list, strides)

## Explicit shape/stride variant for runtime-configurable layouts
fn create_static_tensor_with_stride[layout: Layout](ctx: DeviceContext, dims: DimList, stride_dims: DimList) raises -> StaticTensor[layout]:
    var x = dims.get[0]()
    var y = dims.get[1]()
    var sx = stride_dims.get[0]()
    var sy = stride_dims.get[1]()

    var size = x * y

    var host_storage = ctx.enqueue_create_host_buffer[dtype](size)
    for i in range(size):
        host_storage[i] = 42.0

    var device_storage = ctx.enqueue_create_buffer[dtype](size)
    ctx.enqueue_copy(device_storage, host_storage)

    var rt_shape = RuntimeTuple[layout.shape](x, y)
    var rt_stride = RuntimeTuple[layout.stride](sx, sy)
    var rt_layout = RuntimeLayout[layout](shape=rt_shape, stride=rt_stride)

    var tensor = LayoutTensor[mut=True, dtype, layout, MutableAnyOrigin](device_storage, runtime_layout=rt_layout)
    return StaticTensor[layout](tensor, dims, device_storage)

fn create_static_tensor[layout: Layout](ctx: DeviceContext, dims: List[Int], rank: Int) raises -> StaticTensor[layout]:
    var dims_dl, stride = compute_stride_and_dimlist_from_list(dims, rank)
    return create_static_tensor_with_stride[layout](ctx, dims_dl, stride)
