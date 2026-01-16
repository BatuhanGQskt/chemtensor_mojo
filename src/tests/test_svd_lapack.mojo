from sys import has_accelerator
from gpu.host import DeviceContext
from collections.list import List

from src.m_tensor.dense_tensor import (
    create_dense_tensor_from_data,
    dense_tensor_svd_trunc,
)


fn main() raises:
    @parameter
    if not has_accelerator():
        print("No compatible GPU found - skipping LAPACK SVD smoke test")
        return

    with DeviceContext() as ctx:
        # 4x3 matrix with singular values [3, 2, 0]
        var data = List[Float64](
            2.0, 0.0, 0.0,
            0.0, 3.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        )
        var shape = List[Int](4, 3)
        var A = create_dense_tensor_from_data[DType.float64](ctx, data, shape^)

        var svd_result = dense_tensor_svd_trunc[DType.float64](A^, ctx, chi_max=2, eps_trunc=1e-12)
        var S = svd_result[1]
        var kept = svd_result[3]

        print("LAPACK SVD kept:", kept)
        print("Singular values:")
        S.print_tensor(ctx)

