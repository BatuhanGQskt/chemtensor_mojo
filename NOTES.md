# Issues about current implementation

[ ] DimList is dynamic, but to get values from dimList requires to use get method with a static integer. It is possible to use into_index_list, but then we need to know what rank we will be using which should also be a static value
[ ] GPU related implementations requires Thread_per_block values to be implemented, so we can have LayoutTensor which requires static values as well.
[x] ctx.enqueue_function requires to have only functions, so using structs with methods doesn't allow us to use DeviceContext which leads to unusable methods for the GPU. We can only implement functions for GPU.
    - Resolved as suggested moved to another function. Bad code design, but forced to do because language hasn't supported the method calls in enqueue_function DeviceContext function.

- I will try to solve these issues until 07/10/2025