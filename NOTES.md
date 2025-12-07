# Issues about current implementation

- [x] DimList is dynamic, but to get values from dimList requires to use get method with a static integer. It is possible to use into_index_list, but then we need to know what rank we will be using which should also be a static value
    - DimList gets still static since get[static_int] is required. We need to move to dynamic list and somehow calculate size accordingly, but for the first version I will keep this as 2-rank list which also assumed in the dynamic_layout as 2-rank which is required too

- [ ] GPU related implementations requires Thread_per_block values to be implemented, so we can have LayoutTensor which requires static values as well.

- [x] ctx.enqueue_function requires to have only functions, so using structs with methods doesn't allow us to use DeviceContext which leads to unusable methods for the GPU. We can only implement functions for GPU.
    - Resolved as suggested moved to another function. Bad code design, but forced to do because language hasn't supported the method calls in enqueue_function DeviceContext function.

- I will try to solve these issues until 07/10/2025

Current issues (10/10/2025):
- [ ] Still dense_tensor_dot not implemented completely. (Syntax and runtime issues takes all of my time to make progress)
- [ ] DimList still should change, but for different reason which is get using [static_int], but we want to use arbitrary size list which limits us.
- [ ] Weird error:
```Bash
/gpu_intro/main.mojo:1:1: error: ptxas application ptx input, line 5085; error   : Illegal operand type to instruction 'st'
ptxas application ptx input, line 5086; error   : Illegal operand type to instruction 'st'
ptxas application ptx input, line 5087; error   : Illegal operand type to instruction 'st'
ptxas application ptx input, line 5085; error   : Unknown symbol 'func_retval0'
ptxas application ptx input, line 5086; error   : Unknown symbol 'func_retval0'
ptxas application ptx input, line 5087; error   : Unknown symbol 'func_retval0'
ptxas fatal   : Ptx assembly aborted due to errors
```
