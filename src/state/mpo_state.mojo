from collections.list import List
from gpu.host import DeviceContext
from src.m_tensor.tensor_traits import TensorOps
from src.m_tensor.dense_tensor import (
    DenseTensor,
    create_dense_tensor,
    create_dense_tensor_from_data,
)
from src.m_tensor.block_sparse_tensor import BlockSparseTensor


# =============================================================================
# Convenience Type Aliases for Common Tensor Backends
# =============================================================================

alias DenseMPOSite = MPOSite[DType.float32, DenseTensor[DType.float32]]
"""Dense MPO site with float32 data type."""

alias DenseMPO = MatrixProductOperator[DType.float32, DenseTensor[DType.float32]]
"""Dense MPO with float32 data type."""

alias BlockSparseMPOSite = MPOSite[DType.float32, BlockSparseTensor[DType.float32]]
"""Block-sparse MPO site with float32 data type."""

alias BlockSparseMPO = MatrixProductOperator[DType.float32, BlockSparseTensor[DType.float32]]
"""Block-sparse MPO with float32 data type."""


struct MPOSite[dtype: DType, T: TensorOps](Writable, Movable, ImplicitlyCopyable):
    """Single site tensor inside an MPO (Matrix Product Operator).
    
    Each site is stored as a rank-4 tensor with layout
    [left_bond, phys_in, phys_out, right_bond] or [Wl, d_in, d_out, Wr].
    
    The tensor type T must implement TensorOps (plus Movable and Copyable),
    allowing both DenseTensor and BlockSparseTensor to be used.
    
    Parameters:
        dtype: The data type of tensor elements.
        T: The tensor type (must implement TensorOps).
    
    This convention matches the standard ChemTensor Python implementation:
    - Contract phys_in with ket physical index
    - phys_out becomes the new physical index
    """
    var tensor: T

    fn __init__(out self, var tensor: T):
        """Initialize an MPO site with a tensor."""
        self.tensor = tensor^
    
    fn __copyinit__(out self, other: Self):
        """Copy an MPO site."""
        self.tensor = other.tensor.copy()
    
    fn __moveinit__(out self, deinit other: Self):
        """Move an MPO site."""
        self.tensor = other.tensor^

    fn rank(self) -> Int:
        """Get the rank (number of dimensions) of the site tensor."""
        return self.tensor.get_rank()
    
    fn shape(self) -> List[Int]:
        """Get the shape of the site tensor."""
        return self.tensor.get_shape()
    
    fn left_bond_dim(self) raises -> Int:
        """Get the left bond dimension (first index of rank-4 tensor)."""
        self._assert_rank4()
        return self.tensor.get_shape_at(0)
    
    fn physical_in_dim(self) raises -> Int:
        """Get the physical input dimension (second index of rank-4 tensor)."""
        self._assert_rank4()
        return self.tensor.get_shape_at(1)
    
    fn physical_out_dim(self) raises -> Int:
        """Get the physical output dimension (third index of rank-4 tensor)."""
        self._assert_rank4()
        return self.tensor.get_shape_at(2)

    fn right_bond_dim(self) raises -> Int:
        """Get the right bond dimension (last index of rank-4 tensor)."""
        self._assert_rank4()
        return self.tensor.get_shape_at(3)
    
    fn _assert_rank4(self) raises -> None:
        """Validate that the tensor has rank 4."""
        var r = self.tensor.get_rank()
        if r != 4:
            raise Error(
                "MPOSite expects rank-4 tensors [Wl, d_in, d_out, Wr], got rank "
                + String(r)
            )
    
    fn write_to[W: Writer](self, mut writer: W) -> None:
        """Write tensor info to a writer (for Writable trait)."""
        var s = self.tensor.get_shape()
        writer.write("MPOSite[")
        for i in range(len(s)):
            if i > 0:
                writer.write(", ")
            writer.write(s[i])
        writer.write("]")


struct MatrixProductOperator[dtype: DType, T: TensorOps](Writable, Movable, ImplicitlyCopyable):
    """Matrix Product Operator (MPO) generic over tensor type.
    
    An MPO represents an operator on a many-body Hilbert space as a network
    of local tensors. Each site tensor has shape [Wl, d_in, d_out, Wr]:
    - Wl, Wr: left/right virtual bond dimensions (operator space)
    - d_in: input physical dimension (acts on ket)
    - d_out: output physical dimension (produces new ket)
    
    The tensor type T must implement the TensorOps trait, allowing both
    DenseTensor and BlockSparseTensor to be used.
    
    Parameters:
        dtype: The data type of tensor elements.
        T: The tensor type (must implement TensorOps).
    
    For Hermitian operators acting on real states, d_in == d_out.
    """
    var sites: List[MPOSite[dtype, T]]
    var physical_in_dim: Int
    var physical_out_dim: Int
    var length: Int
    var bond_dims: List[Int]
    
    fn __init__(out self, var sites: List[MPOSite[dtype, T]]) raises:
        if len(sites) == 0:
            raise Error("MatrixProductOperator requires at least one site tensor")
        
        var first_site: MPOSite[dtype, T] = sites[0]
        var phys_in = first_site.physical_in_dim()
        var phys_out = first_site.physical_out_dim()
        var bonds: List[Int] = List[Int](capacity=len(sites) + 1)
        bonds.append(first_site.left_bond_dim())
        
        for idx in range(len(sites)):
            var site = sites[idx]
            if site.physical_in_dim() != phys_in:
                raise Error("All MPO sites must have the same physical input dimension")
            if site.physical_out_dim() != phys_out:
                raise Error("All MPO sites must have the same physical output dimension")
            
            if idx > 0:
                var expected = sites[idx - 1].right_bond_dim()
                if site.left_bond_dim() != expected:
                    raise Error(
                        "Bond mismatch between sites "
                        + String(idx - 1)
                        + " and "
                        + String(idx)
                    )
            
            bonds.append(site.right_bond_dim())
        
        self.sites = sites^
        self.physical_in_dim = phys_in
        self.physical_out_dim = phys_out
        self.length = len(self.sites)
        self.bond_dims = bonds^
    
    fn __copyinit__(out self, existing: Self):
        self.sites = existing.sites.copy()
        self.physical_in_dim = existing.physical_in_dim
        self.physical_out_dim = existing.physical_out_dim
        self.length = existing.length
        self.bond_dims = existing.bond_dims.copy()
    
    fn num_sites(self) -> Int:
        return self.length
    
    fn bond_dimension(self, index: Int) -> Int:
        return self.bond_dims[index]
    
    fn site_shape(self, index: Int) -> List[Int]:
        return self.sites[index].shape()
    
    fn describe(self) -> None:
        print(
            "MatrixProductOperator(length=",
            self.length,
            ", physical_in=",
            self.physical_in_dim,
            ", physical_out=",
            self.physical_out_dim,
            ")",
        )
        
        var bond_str = String("  bond dims = [")
        for i in range(len(self.bond_dims)):
            if i > 0:
                bond_str += ", "
            bond_str += String(self.bond_dims[i])
        bond_str += "]"
        print(bond_str)
        
        for idx in range(self.length):
            var shape = self.sites[idx].shape()
            print(
                "  site ",
                idx,
                ": [",
                shape[0],
                ", ",
                shape[1],
                ", ",
                shape[2],
                ", ",
                shape[3],
                "]",
            )
    
    fn write_to[W: Writer](self, mut writer: W) -> None:
        writer.write("MatrixProductOperator(length=")
        writer.write(self.length)
        writer.write(", physical_in=")
        writer.write(self.physical_in_dim)
        writer.write(", physical_out=")
        writer.write(self.physical_out_dim)
        writer.write(", bond_dims=[")
        for i in range(len(self.bond_dims)):
            if i > 0:
                writer.write(", ")
            writer.write(self.bond_dims[i])
        writer.write("], sites=[")
        for idx in range(self.length):
            if idx > 0:
                writer.write(", ")
            var shape = self.sites[idx].shape()
            writer.write("(")
            writer.write(shape[0])
            writer.write(", ")
            writer.write(shape[1])
            writer.write(", ")
            writer.write(shape[2])
            writer.write(", ")
            writer.write(shape[3])
            writer.write(")")
        writer.write("])")


fn create_identity_mpo[dtype: DType = DType.float32](
    ctx: DeviceContext,
    num_sites: Int,
    physical_dim: Int,
) raises -> MatrixProductOperator[dtype, DenseTensor[dtype]]:
    """Create a DenseTensor-based MPO representing the identity operator.
    
    All bond dimensions are 1 (unentangled operator).
    Each site tensor is the identity matrix reshaped to [1, d, d, 1].
    
    Args:
        ctx: GPU device context.
        num_sites: Number of sites in the chain.
        physical_dim: Local Hilbert space dimension (e.g., 2 for qubits).
    
    Returns:
        DenseTensor-based MatrixProductOperator representing the identity.
    """
    if num_sites < 1:
        raise Error("num_sites must be >= 1")
    if physical_dim < 1:
        raise Error("physical_dim must be >= 1")
    
    var sites = List[MPOSite[dtype, DenseTensor[dtype]]](capacity=num_sites)
    
    for i in range(num_sites):
        var shape = List[Int](1, physical_dim, physical_dim, 1)
        var total_size = physical_dim * physical_dim
        var data = List[Scalar[dtype]](capacity=total_size)
        
        for p_in in range(physical_dim):
            for p_out in range(physical_dim):
                if p_in == p_out:
                    data.append(Scalar[dtype](1.0))
                else:
                    data.append(Scalar[dtype](0.0))
        
        var site_tensor = create_dense_tensor_from_data[dtype](ctx, data, shape^)
        sites.append(MPOSite[dtype, DenseTensor[dtype]](site_tensor^))
    
    return MatrixProductOperator[dtype, DenseTensor[dtype]](sites^)


fn create_single_site_op_mpo[dtype: DType = DType.float32](
    ctx: DeviceContext,
    num_sites: Int,
    site_idx: Int,
    op_data: List[Scalar[dtype]],
) raises -> MatrixProductOperator[dtype, DenseTensor[dtype]]:
    """Create DenseTensor-based MPO for single-site operator O_i (identity elsewhere).
    
    op_data is 2x2 row-major: [op[0,0], op[0,1], op[1,0], op[1,1]].
    """
    if num_sites < 1 or site_idx < 0 or site_idx >= num_sites:
        raise Error("create_single_site_op_mpo: invalid site_idx")
    if len(op_data) != 4:
        raise Error("create_single_site_op_mpo: op_data must be 4 elements (2x2)")

    var d = 2
    var sites = List[MPOSite[dtype, DenseTensor[dtype]]](capacity=num_sites)
    for i in range(num_sites):
        var shape = List[Int](1, d, d, 1)
        var data = List[Scalar[dtype]](capacity=4)
        if i == site_idx:
            for k in range(4):
                data.append(op_data[k])
        else:
            for p_in in range(d):
                for p_out in range(d):
                    data.append(Scalar[dtype](1.0) if p_in == p_out else Scalar[dtype](0.0))
        var site_tensor = create_dense_tensor_from_data[dtype](ctx, data, shape^)
        sites.append(MPOSite[dtype, DenseTensor[dtype]](site_tensor^))
    return MatrixProductOperator[dtype, DenseTensor[dtype]](sites^)


fn create_two_site_op_mpo[dtype: DType = DType.float32](
    ctx: DeviceContext,
    num_sites: Int,
    site_i: Int,
    site_j: Int,
    op_i_data: List[Scalar[dtype]],
    op_j_data: List[Scalar[dtype]],
) raises -> MatrixProductOperator[dtype, DenseTensor[dtype]]:
    """Create DenseTensor-based MPO for two-site operator O_i O_j (identity elsewhere).
    Assumes site_i < site_j.
    """
    if num_sites < 2 or site_i < 0 or site_j >= num_sites or site_i >= site_j:
        raise Error("create_two_site_op_mpo: invalid sites")
    if len(op_i_data) != 4 or len(op_j_data) != 4:
        raise Error("create_two_site_op_mpo: op data must be 4 elements each")

    var d = 2
    var sites = List[MPOSite[dtype, DenseTensor[dtype]]](capacity=num_sites)
    for i in range(num_sites):
        var shape = List[Int](1, d, d, 1)
        var data = List[Scalar[dtype]](capacity=4)
        if i == site_i:
            for k in range(4):
                data.append(op_i_data[k])
        elif i == site_j:
            for k in range(4):
                data.append(op_j_data[k])
        else:
            for p_in in range(d):
                for p_out in range(d):
                    data.append(Scalar[dtype](1.0) if p_in == p_out else Scalar[dtype](0.0))
        var site_tensor = create_dense_tensor_from_data[dtype](ctx, data, shape^)
        sites.append(MPOSite[dtype, DenseTensor[dtype]](site_tensor^))
    return MatrixProductOperator[dtype, DenseTensor[dtype]](sites^)
