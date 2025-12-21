from gpu.host import DeviceContext
from sys import has_accelerator
from m_tensor.dense_tensor import create_tensor, create_tensor_with_stride, dense_tensor_dot
from layout.layout import DimList, Layout
from gpu import thread_idx
from math import ceildiv
from time import perf_counter_ns


alias dtype = DType.float32
alias SIZE = 1024
alias TPB = 2  # TODO: Cannot alocate dynamic memory in the shared memory space at dense_tensor_dot. Dynamic allocation not allow both for types and size which kills the point of optimizing tensor size and fitting it to the GPU space. This is from p13 of gpu quizes
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
        var start_ns = perf_counter_ns()
        with DeviceContext() as ctx:
            print("Running on GPU")

            dims: DimList = DimList(THREADS_PER_BLOCK[0], THREADS_PER_BLOCK[1])  # (5,4) for MxK
            other_dims: DimList = DimList(THREADS_PER_BLOCK[1], THREADS_PER_BLOCK[0])  # (4,5) for KxM
            res_dims: DimList = DimList(THREADS_PER_BLOCK[0], THREADS_PER_BLOCK[0])  # (5,5) for MxM for now. It will be MxK * KxN later

            # Row-major strides derived at runtime: (leading_dim, 1)
            var stride_d0 = dims.get[1]()
            var stride_d1 = 1
            var stride_first: DimList = DimList(stride_d0, stride_d1)

            var stride_other_d0 = other_dims.get[1]()
            var stride_other_d1 = 1
            var stride_other: DimList = DimList(stride_other_d0, stride_other_d1)

            var stride_res_d0 = res_dims.get[1]()
            var stride_res_d1 = 1
            var stride_res: DimList = DimList(stride_res_d0, stride_res_d1)

            first_tensor = create_tensor_with_stride[dynamic_layout](ctx, dims, stride_first)
            other_tensor = create_tensor_with_stride[dynamic_layout](ctx, other_dims, stride_other)
            result_tensor = create_tensor_with_stride[dynamic_layout](ctx, res_dims, stride_res)
            ctx.synchronize()
            # Compute runtime grid_dim to cover M x N
            var M = res_dims.get[0]()
            var N = res_dims.get[1]()
            var K = other_dims.get[0]()
            var grid_dim = (ceildiv(M, THREADS_PER_BLOCK[0]), ceildiv(N, THREADS_PER_BLOCK[1]), 1)

            ctx.enqueue_function[dense_tensor_dot[dynamic_layout]](first_tensor.data, other_tensor.data, result_tensor.data, M, N, K, grid_dim=grid_dim, block_dim=THREADS_PER_BLOCK)
            ctx.synchronize()

            # Copy result back to host to print out
            var total = M * N
            var host_out = ctx.enqueue_create_host_buffer[dtype](total)
            ctx.enqueue_copy(host_out, result_tensor.storage)
            ctx.synchronize()

            # Use tensor printers (no duplication)
            print("A:")
            first_tensor.print_tensor(ctx)
            print("B:")
            other_tensor.print_tensor(ctx)

            print("Result (", M, "x", N, "):")
            for i in range(M):
                for j in range(N):
                    var idx = i * N + j
                    print("[", i, ",", j, "] = ", host_out[idx])
                    
            print("Completed running GPU")
        var finish_ns = perf_counter_ns()
        var res_ns = finish_ns - start_ns

        var res_s = Float64(res_ns) / 1_000_000_000.0
        print("Completed execution in ", res_s, " seconds (", res_ns, " ns)")