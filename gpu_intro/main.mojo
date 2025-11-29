from gpu.host import DeviceContext
from sys import has_accelerator
from m_tensor.dense_tensor import create_tensor
from layout.layout import DimList
from gpu import thread_idx

alias dtype = DType.float32
alias SIZE = 1024

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
            # number of qubits
            # number of gates
            ctx.enqueue_function[printing_kernel](grid_dim=1, block_dim=4)

            # Wait for the kernel to finish executing before handing back to CPU
            ctx.synchronize()
            dims: DimList = DimList(10, 10)
            first_tensor = create_tensor(dims, 2)
            second_tensor = create_tensor(dims, 2)
            out = create_tensor(dims, 2)

            ctx.enqueue_function[first_tensor.dense_tensor_dot](second_tensor, out, 100)
            print(out.data)