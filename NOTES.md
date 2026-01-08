# Issues about current implementation

- [x] DimList is dynamic, but to get values from dimList requires to use get method with a static integer. It is possible to use into_index_list, but then we need to know what rank we will be using which should also be a static value
    - DimList gets still static since get[static_int] is required. We need to move to dynamic list and somehow calculate size accordingly, but for the first version I will keep this as 2-rank list which also assumed in the dynamic_layout as 2-rank which is required too

- [ ] GPU related implementations requires Thread_per_block values to be implemented, so we can have LayoutTensor which requires static values as well.

- [x] ctx.enqueue_function requires to have only functions, so using structs with methods doesn't allow us to use DeviceContext which leads to unusable methods for the GPU. We can only implement functions for GPU.
    - Resolved as suggested moved to another function. Bad code design, but forced to do because language hasn't supported the method calls in enqueue_function DeviceContext function.

- I will try to solve these issues until 07/10/2025

Current issues (10/10/2025):
- [x] Still dense_tensor_dot not implemented completely. (Syntax and runtime issues takes all of my time to make progress)
- [ ] DimList still should change, but for different reason which is get using [static_int], but we want to use arbitrary size list which limits us.
- [x] Weird error:
```Bash
/gpu_intro/main.mojo:1:1: error: ptxas application ptx input, line 5085; error   : Illegal operand type to instruction 'st'
ptxas application ptx input, line 5086; error   : Illegal operand type to instruction 'st'
ptxas application ptx input, line 5087; error   : Illegal operand type to instruction 'st'
ptxas application ptx input, line 5085; error   : Unknown symbol 'func_retval0'
ptxas application ptx input, line 5086; error   : Unknown symbol 'func_retval0'
ptxas application ptx input, line 5087; error   : Unknown symbol 'func_retval0'
ptxas fatal   : Ptx assembly aborted due to errors
```


# Next Talk with Prof. Mendl
[x] Ask him to start officially
[x] TODO: Make this more general for N number of ranks if possible dynamic(not static 5)
[] TODO: Investigate complex
[x] TODO: try to implement dot accordingly

- TODO: kronecker
- TODO: scalar add
- TODO: SVD function (hard)

# Theoratical Parts
- GPU, Tensor
- Intro(what is thesis about) -> Preliminary (explain your method what is chemtensor, dense sparse mojo) -> Method (Explain exactly what I did, explain decisions why did you implement them, experiments) -> Conclusion ( what you did ) -> Future Work [ This can be splitted up. ]

Skip intro for now. -> Focus on Preliminary (explain what is tensor network, GPU programming MOJO, chemtensor (sparse, dense and logic behind reason))

## State module scaffolding (Nov 2025)
- Added `state/mps_state.mojo` with `MPSSite`, `MatrixProductState`, and helpers for zero/product states.
- Example usage:
```mojo
with DeviceContext() as ctx:
    var basis = List[Int](0, 1, 0)
    var psi = create_product_mps(ctx, 2, basis)
    psi.describe()
```