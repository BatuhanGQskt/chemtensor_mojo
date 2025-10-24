from memory import Pointer, AddressSpace, OwnedPointer
from io import Writer, Writable
from layout import Layout, LayoutTensor, RuntimeLayout, IntTuple, RuntimeTuple
from collections.list import List
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, DeviceBuffer
from layout.tensor_builder import LayoutTensorBuild as tb
from layout.layout import DimList
from math import ceildiv

alias dtype = DType.float32

## General dense tensor structure for block sparse tensor networks
@fieldwise_init
struct DenseTensor[
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
        writer.write("DenseTensor[" )
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
fn dense_tensor_dot[layout: Layout](
    first: LayoutTensor[mut=False, dtype, layout, MutableAnyOrigin],
    other: LayoutTensor[mut=False, dtype, layout, MutableAnyOrigin],
    result: LayoutTensor[mut=True, dtype, layout, MutableAnyOrigin],
    M: Int, N: Int, K: Int,
) -> None:
    var row = Int(block_dim.x * block_idx.x + thread_idx.x)
    var col = Int(block_dim.y * block_idx.y + thread_idx.y)

    if row < M and col < N and K > 0:
        var dst = first[row, 0] * other[0, col]
        for k in range(1, K):
            dst += first[row, k] * other[k, col]
        result[row, col] = dst

fn create_tensor[layout: Layout](ctx: DeviceContext, dims: DimList) raises -> DenseTensor[layout]:
    var x = dims.get[0]()
    var y = dims.get[1]()
    var size = x * y

    # Host buffer for initialization
    var host_storage = ctx.enqueue_create_host_buffer[dtype](size)
    for i in range(size):
        host_storage[i] = 42.0

    # Device buffer and copy
    var device_storage = ctx.enqueue_create_buffer[dtype](size)
    ctx.enqueue_copy(device_storage, host_storage)

    # Runtime layout (row-major)
    var rt_shape = RuntimeTuple[layout.shape](x, y)
    var rt_stride = RuntimeTuple[layout.stride](y, 1)
    var rt_layout = RuntimeLayout[layout](shape=rt_shape, stride=rt_stride)  # Named params fix positional error

    # LayoutTensor on device
    var tensor = LayoutTensor[mut=True, dtype, layout, MutableAnyOrigin](device_storage, runtime_layout=rt_layout)
    return DenseTensor[layout](tensor, dims, device_storage)

## Explicit shape/stride variant for runtime-configurable layouts
fn create_tensor_with_stride[layout: Layout](ctx: DeviceContext, dims: DimList, stride_dims: DimList) raises -> DenseTensor[layout]:
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
    return DenseTensor[layout](tensor, dims, device_storage)