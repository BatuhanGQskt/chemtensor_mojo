fn user_layout_setup():
    # Create RuntimeLayout from user configuration  
    print("Creating RuntimeLayout for rank", user_rank)
    
    # Create RuntimeLayout based on rank (supports ranks 1-4)
    if user_rank == 2:
        var rt_shape = RuntimeTuple[my_layout.shape](user_shape[0], user_shape[1])
        var rt_stride = RuntimeTuple[my_layout.stride](user_strides[0], user_strides[1])
        var user_layout = RuntimeLayout[my_layout](shape=rt_shape, stride=rt_stride)
        print("RuntimeLayout created successfully for 2D tensor")
    elif user_rank == 1:
        var rt_shape = RuntimeTuple[my_layout.shape](user_shape[0])
        var rt_stride = RuntimeTuple[my_layout.stride](user_strides[0])
        var user_layout = RuntimeLayout[my_layout](shape=rt_shape, stride=rt_stride)
        print("RuntimeLayout created successfully for 1D tensor")
    elif user_rank == 3:
        var rt_shape = RuntimeTuple[my_layout.shape](user_shape[0], user_shape[1], user_shape[2])
        var rt_stride = RuntimeTuple[my_layout.stride](user_strides[0], user_strides[1], user_strides[2])
        var user_layout = RuntimeLayout[my_layout](shape=rt_shape, stride=rt_stride)
        print("RuntimeLayout created successfully for 3D tensor")
    elif user_rank == 4:
        var rt_shape = RuntimeTuple[my_layout.shape](user_shape[0], user_shape[1], user_shape[2], user_shape[3])
        var rt_stride = RuntimeTuple[my_layout.stride](user_strides[0], user_strides[1], user_strides[2], user_strides[3])
        var user_layout = RuntimeLayout[my_layout](shape=rt_shape, stride=rt_stride)
        print("RuntimeLayout created successfully for 4D tensor")

    # Ask user which demo to run
    var choice = input("Choose demo:\n  1. Dynamic Tensor Demo (runtime rank/shape/stride)\n  2. Original Matrix Multiplication Demo\nEnter choice (1 or 2): ")
    var str_fn = Python.import_module("builtins").str
    
    if String(str_fn(choice)).strip() == "1":
        # Run dynamic tensor demo
        demo_dynamic_tensor()
        return
    