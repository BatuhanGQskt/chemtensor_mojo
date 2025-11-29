from memory import Pointer, AddressSpace, OwnedPointer
from layout import Layout, LayoutTensor
from collections.list import List
from gpu import thread_idx, block_idx, block_dim, barrier
from layout.tensor_builder import LayoutTensorBuild as tb
from layout.layout import DimList

alias TPB = 5  # TODO: Cannot alocate dynamic memory in the shared memory space at dense_tensor_dot. Dynamic allocation not allow both for types and size which kills the point of optimizing tensor size and fitting it to the GPU space. This is from p13 of gpu quizes
alias THREADS_PER_BLOCK = (TPB, 4)
alias dtype = DType.float32
alias my_layout: Layout = Layout.row_major(THREADS_PER_BLOCK[0], THREADS_PER_BLOCK[1])


## General dense tensor structure for block sparse tensor networks
@fieldwise_init
struct DenseTensor[
    data_layout: Layout
]:  # TODO: Change this to more generic type after proper implementation [..., dtype: TypeDef like DType]
    var data: LayoutTensor[
        mut=True, dtype, data_layout, MutableAnyOrigin
    ]  # Data array for tensor elements
    var dims: DimList  # Dynamic array of dimensions (runtime)

    ## GPU-optimized dot product for dense tensors
    # @kernel # TODO: Change this to compiler.register after proper implementation like p18
    fn dense_tensor_dot(self, other: DenseTensor, result: DenseTensor, size: UInt) raises -> None: # [target: StaticString]
        # Naive approach for now: https://docs.modular.com/max/tutorials/custom-ops-matmul/
        var M = self.dims.get[0]()
        var N = other.dims.get[1]()
        var K = other.dims.get[0]()

        var row = block_dim.x * block_idx.x + thread_idx.x
        var col = block_dim.y * block_idx.y + thread_idx.y

        var dst_reg: Float32 = 0.0

        if row < UInt(M) and col < UInt(N):
            for k_index in range(K):
                dst_reg = dst_reg + self.data[row, k_index] * other.data[k_index, col]

        result.data[row, col] = dst_reg



## Helper function to allocate and initialize tensor
fn create_tensor(dims: DimList) -> DenseTensor[my_layout]:
    var size: Int = 1
    var index_list = dims.into_index_list[rank=2]()
    for i in range(index_list.size):
        size *= Int(index_list[i])

    var storage = InlineArray[Scalar[dtype], THREADS_PER_BLOCK[0] * THREADS_PER_BLOCK[1]](uninitialized=True)
    var tensor_5x4 = LayoutTensor[mut=True, dtype, my_layout, MutableAnyOrigin](storage)
    return DenseTensor[my_layout](tensor_5x4, dims)
