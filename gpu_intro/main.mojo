from sys import has_accelerator
from m_tensor.dense_tensor import create_tensor, create_tensor_with_stride, dense_tensor_dot
from layout.layout import DimList, Layout
from gpu import thread_idx
from gpu.host import DeviceContext
from math import ceildiv
from time import perf_counter_ns
from collections import optional, List
from python import Python

alias dtype = DType.float32
alias SIZE = 1024
alias MAX_RANK = 4
alias TPB = 16  # TODO: Cannot alocate dynamic memory in the shared memory space at dense_tensor_dot. Dynamic allocation not allow both for types and size which kills the point of optimizing tensor size and fitting it to the GPU space. This is from p13 of gpu quizes
alias THREADS_PER_BLOCK = (TPB, 2) # 16*2 = 32 for better memory utilization on Warps for threads/block
alias BLOCK_PER_GRID = (1, 1)
alias my_layout: Layout = Layout.row_major(THREADS_PER_BLOCK[0], THREADS_PER_BLOCK[1])
alias res_layout: Layout = Layout.row_major(THREADS_PER_BLOCK[0], THREADS_PER_BLOCK[0])

alias dynamic_layout = Layout.row_major[MAX_RANK]()  # Rank-4 we will pad lower-rank tensors

# Look into the dynamic implementation
fn get_dims_from_user(prompt: String, rank: Int) raises -> List[Int]:
    """Get dimensions from user input and pad to MAX_RANK."""
    var input = Python.import_module("builtins").input
    var int_fn = Python.import_module("builtins").int
    
    print(prompt)
    var dims = List[Int]()
    
    for i in range(rank):
        var dim_prompt = "  Enter dimension " + String(i) + ": "
        var dim_str = input(dim_prompt)
        var dim_val = int_fn(dim_str)
        var dim_int = Int(dim_val)
        dims.append(dim_int)
    
    # Pad remaining dimensions with 1
    for _ in range(MAX_RANK - rank):
        dims.append(1)
    
    return dims

def main():
    @parameter
    if not has_accelerator():
        print("No compatible GPU found")
    else:
        var ret = test_func(None)
        print("Ret val from test, ", ret)
        var rank = 2  # Runtime rank (e.g., 2 for matrix)
        
        # Since we are taking user input here, it is not related to computational comparison, so outside the device Context.
        # Option to use user input or default values
        var use_input = input("Use custom dimensions? (y/n): ")
        var str_fn = Python.import_module("builtins").str
        var dims: List[Int]
        var other_dims: List[Int]
        var res_dims: List[Int]
        
        if String(str_fn(use_input)).lower() == "y":
            dims = get_dims_from_user("Enter dimensions for first tensor (matrix A):", rank)
            other_dims = get_dims_from_user("Enter dimensions for second tensor (matrix B):", rank)
            # Result dimensions: [M, N] where A is [M, K] and B is [K, N]
            res_dims = List[Int]()
            res_dims.append(dims[0])  # M from first tensor
            res_dims.append(other_dims[1])  # N from second tensor
            for _ in range(MAX_RANK - rank):
                res_dims.append(1)
            
            print("First tensor dims: [", dims[0], ",", dims[1], "]")
            print("Second tensor dims: [", other_dims[0], ",", other_dims[1], "]")
            print("Result tensor dims: [", res_dims[0], ",", res_dims[1], "]")
        else:
            # Use default values
            dims = List[Int](5, 4, 1, 1)
            other_dims = List[Int](4, 5, 1, 1)
            res_dims = List[Int](5, 5, 1, 1)
            print("Using default dimensions: A=[5,4], B=[4,5], Result=[5,5]")
        
        var start_ns = perf_counter_ns()
        with DeviceContext() as ctx:
            print("Running on GPU")
            # Create tensors with fixed max-rank layout
            var first_tensor = create_tensor[my_layout](ctx, dims, rank)
            var other_tensor = create_tensor[my_layout](ctx, other_dims, rank)
            var result_tensor = create_tensor[my_layout](ctx, res_dims, rank)
            ctx.synchronize()

            # Compute grid dimensions
            var M = res_dims[0]
            var N = res_dims[1]
            var K = other_dims[0]
            var grid_dim = (ceildiv(M, THREADS_PER_BLOCK[0]), ceildiv(N, THREADS_PER_BLOCK[1]), 1)

            # Enqueue kernel
            ctx.enqueue_function[dense_tensor_dot[my_layout]](first_tensor.data, other_tensor.data, result_tensor.data, M, N, K, grid_dim=grid_dim, block_dim=THREADS_PER_BLOCK)
            ctx.synchronize()

            # Copy and print result
            var total = M * N
            var host_out = ctx.enqueue_create_host_buffer[dtype](total)
            ctx.enqueue_copy(host_out, result_tensor.storage)
            ctx.synchronize()

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


fn test_func(val: Optional[Int]) -> Int:
    var new_val = val.or_else(0)
    print("After or else", new_val)
    return val.or_else(0)