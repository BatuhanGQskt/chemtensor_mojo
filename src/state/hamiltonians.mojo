from collections.list import List
from gpu.host import DeviceContext
from src.m_tensor.dense_tensor import create_dense_tensor_from_data, DenseTensor
from src.state.mpo_state import MPOSite, MatrixProductOperator


fn create_transverse_ising_mpo[dtype: DType = DType.float32](
    ctx: DeviceContext,
    num_sites: Int,
    J: Float64 = 1.0, # internal ferromagnetic coupling Ref: https://farside.ph.utexas.edu/teaching/329/lectures/node110.html
    h: Float64 = 0.5, # external field strength
) raises -> MatrixProductOperator[dtype]:
    """Create MPO for the transverse-field Ising model.
    
    Hamiltonian: H = -J * sum_i (Z_i Z_{i+1}) - h * sum_i X_i
    
    where Z and X are Pauli matrices:
        Z = [[1, 0], [0, -1]]
        X = [[0, 1], [1, 0]]
    
    This is implemented as an MPO with bond dimension 3 for bulk sites.
    
    Args:
        ctx: Device context.
        num_sites: Number of sites in the chain.
        J: Coupling strength (positive = ferromagnetic).
        h: Transverse field strength.
    
    Returns:
        MatrixProductOperator representing the Ising Hamiltonian.
    
    Example:
        ```mojo
        # Create Ising model with 10 sites
        var H = create_transverse_ising_mpo(ctx, 10, J=1.0, h=0.5)
        ```
    """
    if num_sites < 2:
        raise Error("Ising model requires at least 2 sites")
    
    var d = 2  # Physical dimension (qubit)
    var sites = List[MPOSite[dtype]](capacity=num_sites)
    
    # Local operators (Pauli matrices).
    #
    # ChemTensor's `construct_ising_1d_mpo(nsites, J, h, g)` uses the Pauli
    # operators (not spin-1/2) with the common convention:
    #   H = -J Σ Z_i Z_{i+1} - g Σ X_i - h Σ Z_i
    #
    # This helper implements the transverse-only variant (h_longitudinal = 0)
    # by calling `create_ising_1d_mpo` below.
    return create_ising_1d_mpo[dtype](ctx, num_sites, J=J, h_longitudinal=0.0, g_transverse=h)


fn create_ising_1d_mpo[dtype: DType = DType.float32](
    ctx: DeviceContext,
    num_sites: Int,
    J: Float64 = 1.0,
    h_longitudinal: Float64 = 0.0,
    g_transverse: Float64 = 1.0,
) raises -> MatrixProductOperator[dtype]:
    """Create MPO for the 1D Ising model with longitudinal + transverse fields.
    
    Matches ChemTensor `construct_ising_1d_mpo(nsites, J, h, g)`:
        H = -J * sum_i (Z_i Z_{i+1}) - g * sum_i X_i - h * sum_i Z_i
    with open boundary conditions.
    """
    if num_sites < 2:
        raise Error("Ising model requires at least 2 sites")

    var d = 2
    var sites = List[MPOSite[dtype]](capacity=num_sites)

    # Identity
    var I_data = List[Scalar[dtype]](
        Scalar[dtype](1.0), Scalar[dtype](0.0),
        Scalar[dtype](0.0), Scalar[dtype](1.0)
    )
    
    # Pauli X
    var X_data = List[Scalar[dtype]](
        Scalar[dtype](0.0), Scalar[dtype](1.0),
        Scalar[dtype](1.0), Scalar[dtype](0.0)
    )
    # Pauli Z
    var Z_data = List[Scalar[dtype]](
        Scalar[dtype](1.0), Scalar[dtype](0.0),
        Scalar[dtype](0.0), Scalar[dtype](-1.0)
    )
    
    # MPO structure (bond dimension 3):
    # Uses standard Upper Triangular form:
    # Row/Col 0: Start (I)
    # Row/Col 1: Site (Z)
    # Row/Col 2: End (H)
    #
    # W = [I   Z   -(g*X + h*Z)]
    #     [0   0   -J*Z        ]
    #     [0   0   I           ]
    #
    # First site (left boundary): W[0] with shape [1, 3, 2, 2] -> [I, Z, H_loc]
    # Bulk sites: W[i] with shape [3, 3, 2, 2]
    # Last site (right boundary): W[N-1] with shape [3, 1, 2, 2] -> [H_loc, -J*Z, I]^T
    
    # First site (left edge) - shape [1, 3, 2, 2]
    var W0_data = List[Scalar[dtype]](capacity=12)
    # Wl=0, Wr=0: I
    for val in I_data: W0_data.append(val)
    # Wl=0, Wr=1: Z
    for val in Z_data: W0_data.append(val)
    # Wl=0, Wr=2: on-site term  (-g*X - h*Z)
    for idx in range(4):
        var on_site = Scalar[dtype](-g_transverse) * X_data[idx] + Scalar[dtype](-h_longitudinal) * Z_data[idx]
        W0_data.append(on_site)
    
    var W0_temp = create_dense_tensor_from_data[dtype](
        ctx, W0_data^, List[Int](1, 3, d, d)
    )
    # Transpose to [Wl, d, d, Wr] = [1, 2, 2, 3]
    var W0_transposed = W0_temp.transpose(List[Int](0, 2, 3, 1), ctx)
    var W0_tensor = W0_transposed^.copy_to_contiguous(ctx)
    sites.append(MPOSite[dtype](W0_tensor^))
    
    # Bulk sites (middle) - shape [3, 3, 2, 2]
    for i in range(1, num_sites - 1):
        var W_bulk_data = List[Scalar[dtype]](capacity=36)
        
        # Row 0: [I, Z, -(g*X + h*Z)]
        for val in I_data: W_bulk_data.append(val)
        for val in Z_data: W_bulk_data.append(val)
        for idx in range(4):
            var on_site = Scalar[dtype](-g_transverse) * X_data[idx] + Scalar[dtype](-h_longitudinal) * Z_data[idx]
            W_bulk_data.append(on_site)
        
        # Row 1: [0, 0, -J*Z]
        for _ in range(8): W_bulk_data.append(Scalar[dtype](0.0))
        for val in Z_data: W_bulk_data.append(Scalar[dtype](-J) * val)
        
        # Row 2: [0, 0, I]
        for _ in range(8): W_bulk_data.append(Scalar[dtype](0.0))
        for val in I_data: W_bulk_data.append(val)
        
        var W_bulk_temp = create_dense_tensor_from_data[dtype](
            ctx, W_bulk_data^, List[Int](3, 3, d, d)
        )
        # Transpose to [Wl, d, d, Wr] = [3, 2, 2, 3]
        var W_bulk_transposed = W_bulk_temp.transpose(List[Int](0, 2, 3, 1), ctx)
        var W_bulk_tensor = W_bulk_transposed^.copy_to_contiguous(ctx)
        sites.append(MPOSite[dtype](W_bulk_tensor^))
    
    # Last site (right edge) - shape [3, 1, 2, 2]
    # Contract with [I, Z, H_prev] to get H_total
    var WN_data = List[Scalar[dtype]](capacity=12)
    # Wl=0, Wr=0: on-site term (-(g*X + h*Z))
    for idx in range(4):
        var on_site = Scalar[dtype](-g_transverse) * X_data[idx] + Scalar[dtype](-h_longitudinal) * Z_data[idx]
        WN_data.append(on_site)
    # Wl=1, Wr=0: -J*Z
    for val in Z_data: WN_data.append(Scalar[dtype](-J) * val)
    # Wl=2, Wr=0: I
    for val in I_data: WN_data.append(val)
    
    var WN_temp = create_dense_tensor_from_data[dtype](
        ctx, WN_data^, List[Int](3, 1, d, d)
    )
    # Transpose to [Wl, d, d, Wr] = [3, 2, 2, 1]
    var WN_transposed = WN_temp.transpose(List[Int](0, 2, 3, 1), ctx)
    var WN_tensor = WN_transposed^.copy_to_contiguous(ctx)
    sites.append(MPOSite[dtype](WN_tensor^))
    
    return MatrixProductOperator[dtype](sites^)


fn create_heisenberg_mpo[dtype: DType = DType.float32](
    ctx: DeviceContext,
    num_sites: Int,
    J: Float64 = 1.0,
) raises -> MatrixProductOperator[dtype]:
    """Create MPO for the Heisenberg XXX model.
    
    Hamiltonian: H = J * sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
    
    For J > 0, this is antiferromagnetic.
    
    Pauli matrices:
        X = [[0, 1], [1, 0]]
        Y = [[0, -i], [i, 0]]  (note: for real dtype, we can't represent this exactly)
        Z = [[1, 0], [0, -1]]
    
    Note: This implementation uses real arithmetic, so Y terms are approximated.
    For true Heisenberg, use complex dtype.
    
    Args:
        ctx: Device context.
        num_sites: Number of sites in the chain.
        J: Exchange coupling (positive = antiferromagnetic).
    
    Returns:
        MatrixProductOperator representing the Heisenberg Hamiltonian.
    """
    if num_sites < 2:
        raise Error("Heisenberg model requires at least 2 sites")
    
    var d = 2  # Physical dimension (qubit)
    var sites = List[MPOSite[dtype]](capacity=num_sites)
    
    # Pauli matrices
    var I_data = List[Scalar[dtype]](
        Scalar[dtype](1.0), Scalar[dtype](0.0),
        Scalar[dtype](0.0), Scalar[dtype](1.0)
    )
    
    var X_data = List[Scalar[dtype]](
        Scalar[dtype](0.0), Scalar[dtype](1.0),
        Scalar[dtype](1.0), Scalar[dtype](0.0)
    )
    
    var Z_data = List[Scalar[dtype]](
        Scalar[dtype](1.0), Scalar[dtype](0.0),
        Scalar[dtype](0.0), Scalar[dtype](-1.0)
    )
    
    # Pauli Y (imaginary, approximated for real dtype)
    # For real dtype, Y = [[0, -1], [1, 0]] (ignoring imaginary unit)
    var Y_data = List[Scalar[dtype]](
        Scalar[dtype](0.0), Scalar[dtype](-1.0),
        Scalar[dtype](1.0), Scalar[dtype](0.0)
    )
    
    # MPO structure (bond dimension 5):
    # W = [I    X    Y    Z    0  ]
    #     [0    0    0    0    J*X]
    #     [0    0    0    0    J*Y]
    #     [0    0    0    0    J*Z]
    #     [0    0    0    0    I  ]
    
    # First site - shape [1, 5, 2, 2] -> [I, X, Y, Z, 0]
    var W0_data = List[Scalar[dtype]](capacity=40)
    for val in I_data: W0_data.append(val)
    for val in X_data: W0_data.append(val)
    for val in Y_data: W0_data.append(val)
    for val in Z_data: W0_data.append(val)
    for _ in range(4): W0_data.append(Scalar[dtype](0.0))
    
    var W0_temp = create_dense_tensor_from_data[dtype](
        ctx, W0_data^, List[Int](1, 5, d, d)
    )
    # Transpose to [Wl, d, d, Wr]
    var W0_transposed = W0_temp.transpose(List[Int](0, 2, 3, 1), ctx)
    var W0_tensor = W0_transposed^.copy_to_contiguous(ctx)
    sites.append(MPOSite[dtype](W0_tensor^))
    
    # Bulk sites - shape [5, 5, 2, 2]
    for i in range(1, num_sites - 1):
        var W_bulk_data = List[Scalar[dtype]](capacity=100)
        
        # Row 0: [I, X, Y, Z, 0]
        for val in I_data: W_bulk_data.append(val)
        for val in X_data: W_bulk_data.append(val)
        for val in Y_data: W_bulk_data.append(val)
        for val in Z_data: W_bulk_data.append(val)
        for _ in range(4): W_bulk_data.append(Scalar[dtype](0.0))
        
        # Row 1: [0, 0, 0, 0, J*X]
        for _ in range(16): W_bulk_data.append(Scalar[dtype](0.0))
        for val in X_data: W_bulk_data.append(Scalar[dtype](J) * val)
        
        # Row 2: [0, 0, 0, 0, J*Y]
        for _ in range(16): W_bulk_data.append(Scalar[dtype](0.0))
        for val in Y_data: W_bulk_data.append(Scalar[dtype](J) * val)
        
        # Row 3: [0, 0, 0, 0, J*Z]
        for _ in range(16): W_bulk_data.append(Scalar[dtype](0.0))
        for val in Z_data: W_bulk_data.append(Scalar[dtype](J) * val)
        
        # Row 4: [0, 0, 0, 0, I]
        for _ in range(16): W_bulk_data.append(Scalar[dtype](0.0))
        for val in I_data: W_bulk_data.append(val)
        
        var W_bulk_temp = create_dense_tensor_from_data[dtype](
            ctx, W_bulk_data^, List[Int](5, 5, d, d)
        )
        # Transpose to [Wl, d, d, Wr]
        var W_bulk_transposed = W_bulk_temp.transpose(List[Int](0, 2, 3, 1), ctx)
        var W_bulk_tensor = W_bulk_transposed^.copy_to_contiguous(ctx)
        sites.append(MPOSite[dtype](W_bulk_tensor^))
    
    # Last site - shape [5, 1, 2, 2] -> [0, JX, JY, JZ, I]^T
    var WN_data = List[Scalar[dtype]](capacity=20)
    for _ in range(4): WN_data.append(Scalar[dtype](0.0))
    for val in X_data: WN_data.append(Scalar[dtype](J) * val)
    for val in Y_data: WN_data.append(Scalar[dtype](J) * val)
    for val in Z_data: WN_data.append(Scalar[dtype](J) * val)
    for val in I_data: WN_data.append(val)
    
    var WN_temp = create_dense_tensor_from_data[dtype](
        ctx, WN_data^, List[Int](5, 1, d, d)
    )
    # Transpose to [Wl, d, d, Wr]
    var WN_transposed = WN_temp.transpose(List[Int](0, 2, 3, 1), ctx)
    var WN_tensor = WN_transposed^.copy_to_contiguous(ctx)
    sites.append(MPOSite[dtype](WN_tensor^))
    
    return MatrixProductOperator[dtype](sites^)
