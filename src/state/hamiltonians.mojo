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
    
    # MPO structure (bond dimension 3, lower-triangular):
    # Channel mapping:
    #   ch0 = completed terms, ch1 = Z (pairs with -J*Z at receiver),
    #   ch2 = identity propagation in active channel.
    #
    # Bulk sites matrix structure [Wl, Wr]:
    #        Wr=0        Wr=1   Wr=2
    # Wl=0: [ I           0      0   ]
    # Wl=1: [-J*Z         0      0   ]
    # Wl=2: [-h*Z-g*X     Z      I   ]
    #
    # First site (left boundary): W[0] with shape [1, d, d, 3]
    # Bulk sites: W[i] with shape [3, d, d, 3]
    # Last site (right boundary): W[N-1] with shape [3, d, d, 1]
    
    # First site (left edge) - shape [Wl=1, d_in=2, d_out=2, Wr=3]
    # W[0] = [ -h*Z - g*X,  Z,  I ]
    var W0_data = List[Scalar[dtype]](capacity=12)
    
    for d_in in range(d):
        for d_out in range(d):
            var elem_idx = d_in * d + d_out
            
            var on_site = Scalar[dtype](-h_longitudinal) * Z_data[elem_idx] + Scalar[dtype](-g_transverse) * X_data[elem_idx]
            W0_data.append(on_site)
            W0_data.append(Z_data[elem_idx])
            W0_data.append(I_data[elem_idx])
    
    var W0_tensor = create_dense_tensor_from_data[dtype](
        ctx, W0_data^, List[Int](1, d, d, 3)
    )
    sites.append(MPOSite[dtype](W0_tensor^))
    
    # Bulk sites (middle) - shape [Wl=3, d_in=2, d_out=2, Wr=3]
    for i in range(1, num_sites - 1):
        var W_bulk_data = List[Scalar[dtype]](capacity=36)
        
        for wl in range(3):
            for d_in in range(d):
                for d_out in range(d):
                    var elem_idx = d_in * d + d_out
                    
                    if wl == 0:  # Row 0: [I, 0, 0]
                        W_bulk_data.append(I_data[elem_idx])
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(Scalar[dtype](0.0))
                    elif wl == 1:  # Row 1: [-J*Z, 0, 0]
                        W_bulk_data.append(Scalar[dtype](-J) * Z_data[elem_idx])
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(Scalar[dtype](0.0))
                    else:  # Row 2: [-h*Z-g*X, Z, I]
                        var on_site = Scalar[dtype](-h_longitudinal) * Z_data[elem_idx] + Scalar[dtype](-g_transverse) * X_data[elem_idx]
                        W_bulk_data.append(on_site)
                        W_bulk_data.append(Z_data[elem_idx])
                        W_bulk_data.append(I_data[elem_idx])
        
        var W_bulk_tensor = create_dense_tensor_from_data[dtype](
            ctx, W_bulk_data^, List[Int](3, d, d, 3)
        )
        sites.append(MPOSite[dtype](W_bulk_tensor^))
    
    # Last site (right edge) - shape [Wl=3, d_in=2, d_out=2, Wr=1]
    # W[-1] = [ I, -J*Z, -h*Z-g*X ]^T
    var WN_data = List[Scalar[dtype]](capacity=12)
    
    for wl in range(3):
        for d_in in range(d):
            for d_out in range(d):
                var elem_idx = d_in * d + d_out
                
                if wl == 0:  # I
                    WN_data.append(I_data[elem_idx])
                elif wl == 1:  # -J*Z
                    WN_data.append(Scalar[dtype](-J) * Z_data[elem_idx])
                else:  # -h*Z - g*X
                    var last_op = Scalar[dtype](-h_longitudinal) * Z_data[elem_idx] + Scalar[dtype](-g_transverse) * X_data[elem_idx]
                    WN_data.append(last_op)
    
    var WN_tensor = create_dense_tensor_from_data[dtype](
        ctx, WN_data^, List[Int](3, d, d, 1)
    )
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
    
    Uses raising/lowering operators S+/S- to correctly represent the
    XY interaction with real arithmetic:
        XX + YY = 2*(S+ S- + S- S+)
    where S+ = [[0,1],[0,0]] and S- = [[0,0],[1,0]] are both real.
    
    MPO structure (bond dimension 5):
        Row 0: [I,   S+,    S-,    Z,   0  ]
        Row 1: [0,    0,     0,    0,  2J*S-]
        Row 2: [0,    0,     0,    0,  2J*S+]
        Row 3: [0,    0,     0,    0,   J*Z ]
        Row 4: [0,    0,     0,    0,    I  ]
    
    Verification per bond: S+_i*(2J*S-_{i+1}) + S-_i*(2J*S+_{i+1}) + Z_i*(J*Z_{i+1})
                         = 2J*(S+S- + S-S+) + J*ZZ = J*(XX + YY + ZZ)  correct.
    
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
    
    # Identity  I = [[1, 0], [0, 1]]
    var I_data = List[Scalar[dtype]](
        Scalar[dtype](1.0), Scalar[dtype](0.0),
        Scalar[dtype](0.0), Scalar[dtype](1.0)
    )
    
    # Pauli Z  Z = [[1, 0], [0, -1]]
    var Z_data = List[Scalar[dtype]](
        Scalar[dtype](1.0), Scalar[dtype](0.0),
        Scalar[dtype](0.0), Scalar[dtype](-1.0)
    )
    
    # Raising operator  S+ = [[0, 1], [0, 0]]
    var Sp_data = List[Scalar[dtype]](
        Scalar[dtype](0.0), Scalar[dtype](1.0),
        Scalar[dtype](0.0), Scalar[dtype](0.0)
    )
    
    # Lowering operator  S- = [[0, 0], [1, 0]]
    var Sm_data = List[Scalar[dtype]](
        Scalar[dtype](0.0), Scalar[dtype](0.0),
        Scalar[dtype](1.0), Scalar[dtype](0.0)
    )
    
    # First site - shape [Wl=1, d_in=2, d_out=2, Wr=5]
    # Row vector: [I, S+, S-, Z, 0]
    var W0_data = List[Scalar[dtype]](capacity=20)
    
    for d_in in range(d):
        for d_out in range(d):
            var elem_idx = d_in * d + d_out
            
            # Wr=0: I
            W0_data.append(I_data[elem_idx])
            # Wr=1: S+
            W0_data.append(Sp_data[elem_idx])
            # Wr=2: S-
            W0_data.append(Sm_data[elem_idx])
            # Wr=3: Z
            W0_data.append(Z_data[elem_idx])
            # Wr=4: 0
            W0_data.append(Scalar[dtype](0.0))
    
    var W0_tensor = create_dense_tensor_from_data[dtype](
        ctx, W0_data^, List[Int](1, d, d, 5)
    )
    sites.append(MPOSite[dtype](W0_tensor^))
    
    # Bulk sites - shape [Wl=5, d_in=2, d_out=2, Wr=5]
    # Row 0: [I,   S+,    S-,    Z,    0   ]
    # Row 1: [0,    0,     0,    0,  2J*S-  ]
    # Row 2: [0,    0,     0,    0,  2J*S+  ]
    # Row 3: [0,    0,     0,    0,   J*Z   ]
    # Row 4: [0,    0,     0,    0,    I    ]
    for i in range(1, num_sites - 1):
        var W_bulk_data = List[Scalar[dtype]](capacity=100)
        
        for wl in range(5):
            for d_in in range(d):
                for d_out in range(d):
                    var elem_idx = d_in * d + d_out
                    
                    if wl == 0:  # Row 0: [I, S+, S-, Z, 0]
                        W_bulk_data.append(I_data[elem_idx])
                        W_bulk_data.append(Sp_data[elem_idx])
                        W_bulk_data.append(Sm_data[elem_idx])
                        W_bulk_data.append(Z_data[elem_idx])
                        W_bulk_data.append(Scalar[dtype](0.0))
                    elif wl == 1:  # Row 1: [0, 0, 0, 0, 2J*S-]
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(Scalar[dtype](2.0 * J) * Sm_data[elem_idx])
                    elif wl == 2:  # Row 2: [0, 0, 0, 0, 2J*S+]
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(Scalar[dtype](2.0 * J) * Sp_data[elem_idx])
                    elif wl == 3:  # Row 3: [0, 0, 0, 0, J*Z]
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(Scalar[dtype](J) * Z_data[elem_idx])
                    else:  # Row 4: [0, 0, 0, 0, I]
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(I_data[elem_idx])
        
        var W_bulk_tensor = create_dense_tensor_from_data[dtype](
            ctx, W_bulk_data^, List[Int](5, d, d, 5)
        )
        sites.append(MPOSite[dtype](W_bulk_tensor^))
    
    # Last site - shape [Wl=5, d_in=2, d_out=2, Wr=1]
    # Column vector: [0, 2J*S-, 2J*S+, J*Z, I]^T
    var WN_data = List[Scalar[dtype]](capacity=20)
    
    for wl in range(5):
        for d_in in range(d):
            for d_out in range(d):
                var elem_idx = d_in * d + d_out
                
                # Wr=0 (only one column for last site)
                if wl == 0:  # 0
                    WN_data.append(Scalar[dtype](0.0))
                elif wl == 1:  # 2J*S-
                    WN_data.append(Scalar[dtype](2.0 * J) * Sm_data[elem_idx])
                elif wl == 2:  # 2J*S+
                    WN_data.append(Scalar[dtype](2.0 * J) * Sp_data[elem_idx])
                elif wl == 3:  # J*Z
                    WN_data.append(Scalar[dtype](J) * Z_data[elem_idx])
                else:  # I
                    WN_data.append(I_data[elem_idx])
    
    var WN_tensor = create_dense_tensor_from_data[dtype](
        ctx, WN_data^, List[Int](5, d, d, 1)
    )
    sites.append(MPOSite[dtype](WN_tensor^))
    
    return MatrixProductOperator[dtype](sites^)


fn create_heisenberg_xxz_mpo[dtype: DType = DType.float32](
    ctx: DeviceContext,
    num_sites: Int,
    J: Float64 = 1.0,
    D: Float64 = 1.0,
    h: Float64 = 0.0,
) raises -> MatrixProductOperator[dtype]:
    """Create MPO for the Heisenberg XXZ model.
    
    Hamiltonian: H = -J * sum_i[(X_i*X_{i+1} + Y_i*Y_{i+1} + D*Z_i*Z_{i+1})] - h * sum_i(Z_i)
    
    This matches the C implementation in chemtensor/manual_tests/mpo.c.
    
    Uses raising/lowering operators S+/S- to avoid the sign bug that arises
    from approximating the complex Pauli Y with a real matrix.  The key
    identity is:  XX + YY = 2*(S+ S- + S- S+)  where
        S+ = [[0, 1], [0, 0]]   (raising)
        S- = [[0, 0], [1, 0]]   (lowering)
    Both are real, so no complex arithmetic is needed.
    
    MPO structure (bond dimension 5):
        First site: W[0] = [ -h*Z,  I,  S+,  S-,  D*Z ]
        Bulk sites: W[i] = [[   I,    0,  0,   0,    0   ],
                            [ -2J*S-, 0,  0,   0,    0   ],
                            [ -2J*S+, 0,  0,   0,    0   ],
                            [  -J*Z,  0,  0,   0,    0   ],
                            [  -h*Z,  I,  S+,  S-,  D*Z  ]]
        Last site:  W[N-1] = [[   I   ],
                              [ -2J*S- ],
                              [ -2J*S+ ],
                              [  -J*Z  ],
                              [  -h*Z  ]]
    
    Verification per bond (i, i+1):
        S+_i * (-2J*S-_{i+1}) + S-_i * (-2J*S+_{i+1}) + D*Z_i * (-J*Z_{i+1})
      = -2J*(S+S- + S-S+) - J*D*ZZ  =  -J*(XX + YY + D*ZZ)   correct.
    
    Args:
        ctx: Device context.
        num_sites: Number of sites in the chain.
        J: Exchange coupling strength.
        D: Anisotropy parameter (D=1 gives XXX model).
        h: Magnetic field strength.
    
    Returns:
        MatrixProductOperator representing the Heisenberg XXZ Hamiltonian.
    """
    if num_sites < 2:
        raise Error("Heisenberg model requires at least 2 sites")
    
    var d = 2  # Physical dimension (qubit)
    var sites = List[MPOSite[dtype]](capacity=num_sites)
    
    # Identity  I = [[1, 0], [0, 1]]
    var I_data = List[Scalar[dtype]](
        Scalar[dtype](1.0), Scalar[dtype](0.0),
        Scalar[dtype](0.0), Scalar[dtype](1.0)
    )
    
    # Pauli Z  Z = [[1, 0], [0, -1]]
    var Z_data = List[Scalar[dtype]](
        Scalar[dtype](1.0), Scalar[dtype](0.0),
        Scalar[dtype](0.0), Scalar[dtype](-1.0)
    )
    
    # Raising operator  S+ = [[0, 1], [0, 0]]
    var Sp_data = List[Scalar[dtype]](
        Scalar[dtype](0.0), Scalar[dtype](1.0),
        Scalar[dtype](0.0), Scalar[dtype](0.0)
    )
    
    # Lowering operator  S- = [[0, 0], [1, 0]]
    var Sm_data = List[Scalar[dtype]](
        Scalar[dtype](0.0), Scalar[dtype](0.0),
        Scalar[dtype](1.0), Scalar[dtype](0.0)
    )
    
    # First site - shape [Wl=1, d_in=2, d_out=2, Wr=5]
    # W[0] = [ -h*Z,  S+,  S-,  D*Z,  I ]
    #
    # Channel mapping (lower-triangular convention):
    #   ch0 = completed terms, ch1 = S+ (pairs with -2J*S- at receiver),
    #   ch2 = S- (pairs with -2J*S+), ch3 = D*Z (pairs with -J*Z),
    #   ch4 = identity propagation in active channel.
    var W0_data = List[Scalar[dtype]](capacity=20)
    
    for d_in in range(d):
        for d_out in range(d):
            var elem_idx = d_in * d + d_out
            
            W0_data.append(Scalar[dtype](-h) * Z_data[elem_idx])
            W0_data.append(Sp_data[elem_idx])
            W0_data.append(Sm_data[elem_idx])
            W0_data.append(Scalar[dtype](D) * Z_data[elem_idx])
            W0_data.append(I_data[elem_idx])
    
    var W0_tensor = create_dense_tensor_from_data[dtype](
        ctx, W0_data^, List[Int](1, d, d, 5)
    )
    sites.append(MPOSite[dtype](W0_tensor^))
    
    # Bulk sites - shape [Wl=5, d_in=2, d_out=2, Wr=5]
    # Row 0: [   I,    0,     0,     0,    0   ]
    # Row 1: [ -2J*S-, 0,     0,     0,    0   ]
    # Row 2: [ -2J*S+, 0,     0,     0,    0   ]
    # Row 3: [  -J*Z,  0,     0,     0,    0   ]
    # Row 4: [  -h*Z,  S+,    S-,   D*Z,   I   ]
    for i in range(1, num_sites - 1):
        var W_bulk_data = List[Scalar[dtype]](capacity=100)
        
        for wl in range(5):
            for d_in in range(d):
                for d_out in range(d):
                    var elem_idx = d_in * d + d_out
                    
                    if wl == 0:  # Row 0: [I, 0, 0, 0, 0]
                        W_bulk_data.append(I_data[elem_idx])
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(Scalar[dtype](0.0))
                    elif wl == 1:  # Row 1: [-2J*S-, 0, 0, 0, 0]
                        W_bulk_data.append(Scalar[dtype](-2.0 * J) * Sm_data[elem_idx])
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(Scalar[dtype](0.0))
                    elif wl == 2:  # Row 2: [-2J*S+, 0, 0, 0, 0]
                        W_bulk_data.append(Scalar[dtype](-2.0 * J) * Sp_data[elem_idx])
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(Scalar[dtype](0.0))
                    elif wl == 3:  # Row 3: [-J*Z, 0, 0, 0, 0]
                        W_bulk_data.append(Scalar[dtype](-J) * Z_data[elem_idx])
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(Scalar[dtype](0.0))
                    else:  # Row 4: [-h*Z, S+, S-, D*Z, I]
                        W_bulk_data.append(Scalar[dtype](-h) * Z_data[elem_idx])
                        W_bulk_data.append(Sp_data[elem_idx])
                        W_bulk_data.append(Sm_data[elem_idx])
                        W_bulk_data.append(Scalar[dtype](D) * Z_data[elem_idx])
                        W_bulk_data.append(I_data[elem_idx])
        
        var W_bulk_tensor = create_dense_tensor_from_data[dtype](
            ctx, W_bulk_data^, List[Int](5, d, d, 5)
        )
        sites.append(MPOSite[dtype](W_bulk_tensor^))
    
    # Last site - shape [Wl=5, d_in=2, d_out=2, Wr=1]
    # Column vector: [I, -2J*S-, -2J*S+, -J*Z, -h*Z]^T
    var WN_data = List[Scalar[dtype]](capacity=20)
    
    for wl in range(5):
        for d_in in range(d):
            for d_out in range(d):
                var elem_idx = d_in * d + d_out
                
                # Wr=0 (only one column for last site)
                if wl == 0:  # I
                    WN_data.append(I_data[elem_idx])
                elif wl == 1:  # -2J*S-
                    WN_data.append(Scalar[dtype](-2.0 * J) * Sm_data[elem_idx])
                elif wl == 2:  # -2J*S+
                    WN_data.append(Scalar[dtype](-2.0 * J) * Sp_data[elem_idx])
                elif wl == 3:  # -J*Z
                    WN_data.append(Scalar[dtype](-J) * Z_data[elem_idx])
                else:  # -h*Z
                    WN_data.append(Scalar[dtype](-h) * Z_data[elem_idx])
    
    var WN_tensor = create_dense_tensor_from_data[dtype](
        ctx, WN_data^, List[Int](5, d, d, 1)
    )
    sites.append(MPOSite[dtype](WN_tensor^))
    
    return MatrixProductOperator[dtype](sites^)


fn create_bose_hubbard_mpo[dtype: DType = DType.float32](
    ctx: DeviceContext,
    num_sites: Int,
    physical_dim: Int,  # d = max_occupancy + 1 (e.g., 3 for max 2 bosons per site)
    t: Float64 = 1.0,    # hopping strength
    u: Float64 = 1.0,    # on-site interaction strength
    mu: Float64 = 0.0    # chemical potential
) raises -> MatrixProductOperator[dtype]:
    """Create MPO for the 1D Bose-Hubbard model.
    
    Hamiltonian: H = -t*sum_i(b†_i b_{i+1} + h.c.) + (u/2)*sum_i(n_i(n_i-1)) - mu*sum_i(n_i)
    
    This is a real-valued Hamiltonian (no complex numbers needed for MPO representation).
    Bond dimension = 4 (similar structure to Ising model).
    
    MPO structure (bond dimension 4, lower-triangular):
        First site:  W[0] = [ (u/2)*n(n-1) - mu*n,  b†,  b,  I ]
        Bulk sites:  W[i] = [[        I,         0,   0,  0 ],
                             [       -t*b,       0,   0,  0 ],
                             [      -t*b†,       0,   0,  0 ],
                             [ (u/2)*n(n-1) - mu*n,  b†,  b,  I ]]
        Last site:   W[N-1] = [[       I        ],
                               [      -t*b      ],
                               [     -t*b†      ],
                               [ (u/2)*n(n-1) - mu*n ]]
    
    Args:
        ctx: Device context.
        num_sites: Number of sites in the chain.
        physical_dim: Local Hilbert space dimension (e.g., 3 for max 2 bosons).
        t: Hopping strength.
        u: On-site interaction strength.
        mu: Chemical potential.
    
    Returns:
        MatrixProductOperator representing the Bose-Hubbard Hamiltonian.
    """
    if num_sites < 2:
        raise Error("Bose-Hubbard model requires at least 2 sites")
    if physical_dim < 2:
        raise Error("Physical dimension must be >= 2")
    
    var d = physical_dim
    var sites = List[MPOSite[dtype]](capacity=num_sites)
    
    # Create bosonic operators
    # Identity: I[i,j] = delta_{i,j}
    var I_data = List[Scalar[dtype]](capacity=d * d)
    for i in range(d):
        for j in range(d):
            I_data.append(Scalar[dtype](1.0) if i == j else Scalar[dtype](0.0))
    
    # Creation operator: b†[i,j] = sqrt(j) * delta_{i,j+1}
    var b_dag_data = List[Scalar[dtype]](capacity=d * d)
    for i in range(d):
        for j in range(d):
            if j > 0 and i == j - 1:
                b_dag_data.append(Scalar[dtype](Float64(j) ** 0.5))
            else:
                b_dag_data.append(Scalar[dtype](0.0))
    
    # Annihilation operator: b[i,j] = sqrt(i) * delta_{i+1,j}
    var b_data = List[Scalar[dtype]](capacity=d * d)
    for i in range(d):
        for j in range(d):
            if i > 0 and j == i - 1:
                b_data.append(Scalar[dtype](Float64(i) ** 0.5))
            else:
                b_data.append(Scalar[dtype](0.0))
    
    # Number operator: n[i,j] = i * delta_{i,j}
    var n_data = List[Scalar[dtype]](capacity=d * d)
    for i in range(d):
        for j in range(d):
            n_data.append(Scalar[dtype](Float64(i)) if i == j else Scalar[dtype](0.0))
    
    # Interaction term: n(n-1)[i,j] = i*(i-1) * delta_{i,j}
    var n_n_minus_1_data = List[Scalar[dtype]](capacity=d * d)
    for i in range(d):
        for j in range(d):
            n_n_minus_1_data.append(Scalar[dtype](Float64(i * (i - 1))) if i == j else Scalar[dtype](0.0))
    
    # First site - shape [Wl=1, d, d, Wr=4]
    # W[0] = [ (u/2)*n(n-1) - mu*n,  b†,  b,  I ]
    var W0_data = List[Scalar[dtype]](capacity=d * d * 4)
    
    for i in range(d):
        for j in range(d):
            var idx = i * d + j
            
            # Wr=0: (u/2)*n(n-1) - mu*n
            W0_data.append(Scalar[dtype](u / 2.0) * n_n_minus_1_data[idx] + Scalar[dtype](-mu) * n_data[idx])
            # Wr=1: b†
            W0_data.append(b_dag_data[idx])
            # Wr=2: b
            W0_data.append(b_data[idx])
            # Wr=3: I
            W0_data.append(I_data[idx])
    
    var W0_tensor = create_dense_tensor_from_data[dtype](
        ctx, W0_data^, List[Int](1, d, d, 4)
    )
    sites.append(MPOSite[dtype](W0_tensor^))
    
    # Bulk sites - shape [Wl=4, d, d, Wr=4]
    for _ in range(1, num_sites - 1):
        var W_bulk_data = List[Scalar[dtype]](capacity=d * d * 4 * 4)
        
        for wl in range(4):
            for i in range(d):
                for j in range(d):
                    var idx = i * d + j
                    
                    if wl == 0:  # Row 0: [I, 0, 0, 0]
                        W_bulk_data.append(I_data[idx])
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(Scalar[dtype](0.0))
                    elif wl == 1:  # Row 1: [-t*b, 0, 0, 0]
                        W_bulk_data.append(Scalar[dtype](-t) * b_data[idx])
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(Scalar[dtype](0.0))
                    elif wl == 2:  # Row 2: [-t*b†, 0, 0, 0]
                        W_bulk_data.append(Scalar[dtype](-t) * b_dag_data[idx])
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(Scalar[dtype](0.0))
                        W_bulk_data.append(Scalar[dtype](0.0))
                    else:  # Row 3: [(u/2)*n(n-1) - mu*n, b†, b, I]
                        W_bulk_data.append(Scalar[dtype](u / 2.0) * n_n_minus_1_data[idx] + Scalar[dtype](-mu) * n_data[idx])
                        W_bulk_data.append(b_dag_data[idx])
                        W_bulk_data.append(b_data[idx])
                        W_bulk_data.append(I_data[idx])
        
        var W_bulk_tensor = create_dense_tensor_from_data[dtype](
            ctx, W_bulk_data^, List[Int](4, d, d, 4)
        )
        sites.append(MPOSite[dtype](W_bulk_tensor^))
    
    # Last site - shape [Wl=4, d, d, Wr=1]
    var WN_data = List[Scalar[dtype]](capacity=d * d * 4)
    
    for wl in range(4):
        for i in range(d):
            for j in range(d):
                var idx = i * d + j
                
                if wl == 0:  # I
                    WN_data.append(I_data[idx])
                elif wl == 1:  # -t*b
                    WN_data.append(Scalar[dtype](-t) * b_data[idx])
                elif wl == 2:  # -t*b†
                    WN_data.append(Scalar[dtype](-t) * b_dag_data[idx])
                else:  # (u/2)*n(n-1) - mu*n
                    WN_data.append(Scalar[dtype](u / 2.0) * n_n_minus_1_data[idx] + Scalar[dtype](-mu) * n_data[idx])
    
    var WN_tensor = create_dense_tensor_from_data[dtype](
        ctx, WN_data^, List[Int](4, d, d, 1)
    )
    sites.append(MPOSite[dtype](WN_tensor^))
    
    return MatrixProductOperator[dtype](sites^)
