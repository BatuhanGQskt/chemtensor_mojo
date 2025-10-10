from gpu.host import DeviceContext
from sys import has_accelerator
from m_tensor.dense_tensor import create_tensor, dense_tensor_dot
from layout.layout import DimList, Layout
from gpu import thread_idx
from math import ceildiv

alias dtype = DType.float32
alias SIZE = 1024
alias TPB = 5  # TODO: Cannot alocate dynamic memory in the shared memory space at dense_tensor_dot. Dynamic allocation not allow both for types and size which kills the point of optimizing tensor size and fitting it to the GPU space. This is from p13 of gpu quizes
alias THREADS_PER_BLOCK = (TPB, 4)
alias BLOCK_PER_GRID = (1, 1)
alias my_layout: Layout = Layout.row_major(THREADS_PER_BLOCK[0], THREADS_PER_BLOCK[1])
alias res_layout: Layout = Layout.row_major(THREADS_PER_BLOCK[0], THREADS_PER_BLOCK[0])

alias dynamic_layout = Layout.row_major[2]()  # Rank-2 with unknown dims

# Push code with readme
# complete dot product working
# Look into the dynamic implementation

# For testing purposes
fn printing_kernel():
    print("GPU thread: [", thread_idx.x, thread_idx.y, thread_idx.z, "]")

def main():
    @parameter
    if not has_accelerator():
        print("No compatible GPU found")
    else:
        with DeviceContext() as ctx:
            print("Running on GPU")
            ctx.enqueue_function[printing_kernel](grid_dim=BLOCK_PER_GRID, block_dim=THREADS_PER_BLOCK)
            ctx.synchronize()

            dims: DimList = DimList(THREADS_PER_BLOCK[0], THREADS_PER_BLOCK[1])  # (5,4) for MxK
            other_dims: DimList = DimList(THREADS_PER_BLOCK[1], THREADS_PER_BLOCK[0])  # (4,5) for KxM
            res_dims: DimList = DimList(THREADS_PER_BLOCK[0], THREADS_PER_BLOCK[0])  # (5,5) for MxM for now. It will be MxK * KxN later

            first_tensor = create_tensor(ctx, dims, dynamic_layout)
            other_tensor = create_tensor(ctx, other_dims, dynamic_layout)
            result_tensor = create_tensor(ctx, res_dims, dynamic_layout)
            ctx.synchronize()
            # Compute runtime grid_dim to cover M x N
            var M = res_dims.get[0]()
            var N = res_dims.get[1]()
            var grid_dim = (ceildiv(M, THREADS_PER_BLOCK[0]), ceildiv(N, THREADS_PER_BLOCK[1]), 1)

            ctx.enqueue_function[dense_tensor_dot](first_tensor, other_tensor, result_tensor, grid_dim=grid_dim, block_dim=THREADS_PER_BLOCK)
            ctx.synchronize()

            result_tensor.print_tensor()  # Use result for output