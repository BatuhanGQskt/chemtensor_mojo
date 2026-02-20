"""DMRG gauge-invariant utilities — proper MPS/MPO tensor contractions.

Uses dense_tensor operations (expectation_value, variance, mps_overlap) from
environments. Do NOT compare raw MPS tensors entry-wise.
"""

from collections.list import List
from gpu.host import DeviceContext
from src.state.mps_state import MatrixProductState, MPSSite
from src.state.mpo_state import MatrixProductOperator, MPOSite, create_single_site_op_mpo, create_two_site_op_mpo
from src.state.environments import (
    expectation_value_normalized,
    variance,
    mps_overlap,
)


fn energy[dtype: DType](
    mps: MatrixProductState[dtype],
    mpo: MatrixProductOperator[dtype],
    ctx: DeviceContext,
) raises -> Float64:
    """Compute E = <psi|H|psi> / <psi|psi> via MPS-MPO-MPS contraction."""
    return expectation_value_normalized[dtype](mps, mpo, ctx)


fn variance_op[dtype: DType](
    mps: MatrixProductState[dtype],
    mpo: MatrixProductOperator[dtype],
    ctx: DeviceContext,
) raises -> Float64:
    """Compute var = <H^2> - <H>^2 via MPO composition and MPS contraction."""
    return variance[dtype](mps, mpo, ctx)


fn fidelity[dtype: DType](
    mps1: MatrixProductState[dtype],
    mps2: MatrixProductState[dtype],
    ctx: DeviceContext,
) raises -> Float64:
    """Compute F = |<psi1|psi2>| (absolute value for real, ignores global phase)."""
    var ov = mps_overlap[dtype](mps1, mps2, ctx)
    return (-ov) if ov < 0 else ov


# Pauli matrices (2x2 row-major): Z = [[1,0],[0,-1]], X = [[0,1],[1,0]]
# Sz = Z/2, Sx = X/2
fn _pauli_z_data[dtype: DType]() -> List[Scalar[dtype]]:
    return List[Scalar[dtype]](Scalar[dtype](1.0), Scalar[dtype](0.0), Scalar[dtype](0.0), Scalar[dtype](-1.0))


fn _pauli_x_data[dtype: DType]() -> List[Scalar[dtype]]:
    return List[Scalar[dtype]](Scalar[dtype](0.0), Scalar[dtype](1.0), Scalar[dtype](1.0), Scalar[dtype](0.0))


fn observe_sz[dtype: DType](
    mps: MatrixProductState[dtype],
    site_i: Int,
    ctx: DeviceContext,
) raises -> Float64:
    """Compute <Sz(i)> = (1/2) * <Z(i)> via single-site MPO expectation."""
    var Z_data = _pauli_z_data[dtype]()
    var Z_mpo = create_single_site_op_mpo[dtype](ctx, mps.num_sites(), site_i, Z_data)
    var val = expectation_value_normalized[dtype](mps, Z_mpo, ctx)
    return 0.5 * val


fn observe_sz_sz[dtype: DType](
    mps: MatrixProductState[dtype],
    site_i: Int,
    site_j: Int,
    ctx: DeviceContext,
) raises -> Float64:
    """Compute <Sz(i) Sz(j)> = (1/4) * <Z(i) Z(j)> via two-site MPO expectation."""
    var Z_data = _pauli_z_data[dtype]()
    var Z_mpo = create_two_site_op_mpo[dtype](ctx, mps.num_sites(), site_i, site_j, Z_data, Z_data)
    var val = expectation_value_normalized[dtype](mps, Z_mpo, ctx)
    return 0.25 * val


fn observe_sx_sx[dtype: DType](
    mps: MatrixProductState[dtype],
    site_i: Int,
    site_j: Int,
    ctx: DeviceContext,
) raises -> Float64:
    """Compute <Sx(i) Sx(j)> = (1/4) * <X(i) X(j)> via two-site MPO expectation."""
    var X_data = _pauli_x_data[dtype]()
    var X_mpo = create_two_site_op_mpo[dtype](ctx, mps.num_sites(), site_i, site_j, X_data, X_data)
    var val = expectation_value_normalized[dtype](mps, X_mpo, ctx)
    return 0.25 * val
