from memory import Pointer, AddressSpace, OwnedPointer
from io import Writer, Writable
from layout import Layout, LayoutTensor, RuntimeLayout, IntTuple, RuntimeTuple
from collections.list import List
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, DeviceBuffer
from layout.tensor_builder import LayoutTensorBuild as tb
from layout.layout import DimList
from main import THREADS_PER_BLOCK, TPB, my_layout, res_layout, dynamic_layout
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
        writer.write(self.data)

    fn print_tensor(self) -> None:
        print("Tensor: ", self.data)

## GPU-optimized dot product for dense tensors
fn dense_tensor_dot(first_tensor: DenseTensor[dynamic_layout], other_tensor: DenseTensor[dynamic_layout], result_tensor: DenseTensor[dynamic_layout]) raises -> None:  # Removed unused size
    print("Dense tensor dot")
    # Naive approach for now: https://docs.modular.com/max/tutorials/custom-ops-matmul/
    var M = first_tensor.dims.get[0]()
    var N = other_tensor.dims.get[1]()
    var K = other_tensor.dims.get[0]()

    print("M: ", M, "N: ", N, "K: ", K)
    var row = Int(block_dim.x * block_idx.x + thread_idx.x)
    var col = Int(block_dim.y * block_idx.y + thread_idx.y)

    var dst_reg: Float32 = 0.0

    if row < M and col < N:
        for k_index in range(K):
            k_index += 1
            # dst_reg += first_tensor.data[row, k_index] * other_tensor.data[k_index, col]
            pass
    result_tensor.data[row, col] = dst_reg


fn create_tensor(ctx: DeviceContext, dims: DimList, layout: Layout) raises -> DenseTensor[dynamic_layout]:
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
    var rt_shape = RuntimeTuple[dynamic_layout.shape](x, y)
    var rt_stride = RuntimeTuple[dynamic_layout.stride](y, 1)
    var rt_layout = RuntimeLayout[dynamic_layout](shape=rt_shape, stride=rt_stride)  # Named params fix positional error

    # LayoutTensor on device
    var tensor = LayoutTensor[mut=True, dtype, dynamic_layout, MutableAnyOrigin](device_storage, runtime_layout=rt_layout)
    return DenseTensor[dynamic_layout](tensor, dims, device_storage)