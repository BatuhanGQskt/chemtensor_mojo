@fieldwise_init
struct TensorAxisRange:
    alias LEADING = 0  # Mimics TENSOR_AXIS_RANGE_LEADING
    alias TRAILING = 1 # Mimics TENSOR_AXIS_RANGE_TRAILING
    alias NUM = 2      # Mimics TENSOR_AXIS_RANGE_NUM

    # Enforce valid axis values at compile time
    @staticmethod
    fn validate[axis: Int]() -> Bool:
        @parameter
        if axis not in (TensorAxisRange.LEADING, TensorAxisRange.TRAILING):
            constrained[False, "Axis must be LEADING (0) or TRAILING (1)"]()
        return True